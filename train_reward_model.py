import argparse
import numpy as np
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import wandb
from tqdm import tqdm

from reward_model import get_tokenizer, get_reward_model, get_ensemble
from reward_model_dataset import load_dataset, RewardModelDataset, RewardModelCollator, LABELS, USE_LIKERT
from utils import initialize_seeds, device, bool_type

def _compute_metrics(predictions, labels):
    invalid_label_idx = np.any(labels == 0.5, axis=1)
    predictions = predictions[~invalid_label_idx]
    labels = labels[~invalid_label_idx]
    if not USE_LIKERT:
        predictions[:, :1] = 1 - predictions[:, :1]
        labels[:, :1] = 1 - labels[:, :1]
    hard_predictions = predictions.copy()
    hard_predictions[hard_predictions >= 0] = 1
    hard_predictions[hard_predictions < 0] = 0
    auc_possible = np.any(labels == 1, axis=0) & np.any(labels == 0, axis=0)
    results = {
        "accuracy_all": round((hard_predictions == labels).sum().item() / (labels.shape[0] * labels.shape[1]), 3),
        "auc_all": round(roc_auc_score(labels, predictions, average="macro"), 3) if auc_possible.all() else None,
        "pearson_score": round(pearsonr(
            hard_predictions[:, 0] * hard_predictions.mean(axis=1),
            labels[:, 0] * labels.mean(axis=1)
        )[0], 2)
    }
    for label_idx, label in enumerate(LABELS):
        results[f"accuracy_{label}"] = round((hard_predictions[:, label_idx] == labels[:, label_idx]).sum().item() / labels.shape[0], 3)
        results[f"auc_{label}"] = round(roc_auc_score(labels[:, label_idx], predictions[:, label_idx]), 3) if auc_possible[label_idx] else None
        results[f"pearson_{label}"] = round(pearsonr(hard_predictions[:, label_idx], labels[:, label_idx])[0], 2)
    return results

def test_loop(model, collator, enc_dec, args):
    dataset = RewardModelDataset(*load_dataset("test", args.annotator), enc_dec, args.mmo, args.mmi)
    model = model.to(device)
    model.eval()
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
    predictions = []
    labels = []
    methods = []
    for batch in tqdm(dataloader):
        batch_predictions = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        ).logits.detach().cpu().numpy()
        batch_labels = batch["labels"].detach().cpu().numpy()
        batch_methods = batch["methods"]
        predictions.append(batch_predictions)
        labels.append(batch_labels)
        methods.append(batch_methods)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    methods = np.concatenate(methods)
    for method in np.unique(methods):
        method_idx = methods == method
        method_predictions = predictions[method_idx]
        method_labels = labels[method_idx]
        print(method)
        print(_compute_metrics(method_predictions, method_labels))
    print("All")
    print(_compute_metrics(predictions, labels))

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return _compute_metrics(predictions, labels)

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    parser.add_argument("--annotator", default="gpt-4", help="Which annotator's dataset to use for testing")
    parser.add_argument("--mmo", type=int, default=1, help="Number of mismatched feedbacks from across questions to include per gold sample")
    parser.add_argument("--mmi", type=bool_type, default=True, help="Include mismatched feedbacks from within the same question")
    parser.add_argument("--model_name", default="reward_model", help="Name of the model to train or test")
    parser.add_argument("--base_model", default="google/flan-t5-xl", help="Base pre-trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--wandb", action="store_true", help="Log performance to wandb")
    args = parser.parse_args()

    if args.wandb:
        wandb.init(
            project="feedback-gen",
            group="reward-model",
            config=args
        )

    use_t5_decoder = False
    enc_dec = ("t5" in args.base_model) and use_t5_decoder
    tokenizer = get_tokenizer(args.base_model)
    if args.test:
        model = get_ensemble(args.model_name.split(","), args.base_model, tokenizer, enc_dec, True)
    else:
        model = get_reward_model(args.base_model, args.base_model, tokenizer, enc_dec, False)
    collator = RewardModelCollator(tokenizer, enc_dec)

    if args.test:
        test_loop(model, collator, enc_dec, args)
    else:
        training_args = TrainingArguments(
            output_dir=args.model_name,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.wd,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            per_device_eval_batch_size=32,
            eval_accumulation_steps=4,
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
            report_to="wandb" if args.wandb else "none"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=RewardModelDataset(*load_dataset("train"), enc_dec, args.mmo, args.mmi),
            eval_dataset=RewardModelDataset(*load_dataset("val"), enc_dec, args.mmo, args.mmi),
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        trainer.train()
        # HACK: for some reason quantization makes it so that the original weights are updated rather than the custom weights
        # resulting in the custom modules not getting saved. Just copy over to load properly at test time.
        if "llama" in args.base_model or "meta-math" in args.base_model:
            trainer.model.score.modules_to_save.default.weight.data = trainer.model.score.original_module.weight.data
        trainer.save_model()

if __name__ == "__main__":
    main()
