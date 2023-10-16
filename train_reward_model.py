import argparse
import numpy as np
from transformers import TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score
import wandb

from reward_model import get_tokenizer, get_reward_model, get_ensemble
from reward_model_dataset import load_dataset, RewardModelDataset, RewardModelCollator, LABELS
from utils import initialize_seeds

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    invalid_label_idx = np.any(labels == 0.5, axis=1)
    predictions = predictions[~invalid_label_idx]
    labels = labels[~invalid_label_idx]
    hard_predictions = predictions.copy()
    hard_predictions[hard_predictions >= 0] = 1
    hard_predictions[hard_predictions < 0] = 0
    # TODO: handle likert
    # TODO: at test time, get accuracy across gen methods
    results = {
        "accuracy_all": (hard_predictions == labels).sum().item() / (labels.shape[0] * labels.shape[1]),
        "auc_all": roc_auc_score(labels, predictions, average="macro")
    }
    for label_idx, label in enumerate(LABELS):
        results[f"accuracy_{label}"] = (hard_predictions[:, label_idx] == labels[:, label_idx]).sum().item() / labels.shape[0]
        results[f"auc_{label}"] = roc_auc_score(labels[:, label_idx], predictions[:, label_idx])
    return results

def bool_type(x: str):
    return x != "0"

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    parser.add_argument("--annotator", default="gpt-4", help="Which annotator's dataset to use for testing")
    parser.add_argument("--mmo", type=int, default=1, help="Number of mismatched feedbacks from across questions to include per gold sample")
    parser.add_argument("--mmi", type=bool_type, default=True, help="Include mismatched feedbacks from within the same question")
    parser.add_argument("--model_name", default="reward_model", help="Name of the model to train or test")
    parser.add_argument("--base_model", default="xlnet-base-cased", help="Base pre-trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
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

    train_dataset = None if args.test else RewardModelDataset(*load_dataset("train"), enc_dec, args.mmo, args.mmi)
    eval_dataset = RewardModelDataset(*load_dataset("test", args.annotator), enc_dec, args.mmo, args.mmi) if args.test else RewardModelDataset(*load_dataset("val"), enc_dec, args.mmo, args.mmi)

    training_args = TrainingArguments(
        output_dir=args.model_name,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        optim="adafactor",
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RewardModelCollator(tokenizer, enc_dec),
        compute_metrics=compute_metrics
    )
    if args.test:
        print(trainer.evaluate())
    else:
        trainer.train()
        # HACK: for some reason quantization makes it so that the original weights are updated rather than the custom weights
        # resulting in the custom modules not getting saved. Just copy over to load properly at test time.
        if "llama" in args.base_model or "meta-math" in args.base_model:
            trainer.model.score.modules_to_save.default.weight.data = trainer.model.score.original_module.weight.data
        trainer.save_model()

if __name__ == "__main__":
    main()
