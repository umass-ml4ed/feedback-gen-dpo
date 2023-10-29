from typing import List
import argparse
import random
import pandas as pd
import torch

from reticl.training.train_reticl import train_reticl
from reticl.evaluate import evaluate
from reticl.models.generator import GeneratorCM
from reticl.data_loading.data_types import DataSample
from reticl.utils import TrainOptions
from reticl.constants import RLAlgorithm, SamplingMethod, Reward

from reward_model import get_ensemble, get_tokenizer
from create_reward_dataset import FEEDBACK_GEN_INSTRUCTION, FEEDBACK_GEN_INSTRUCTION_RUBRIC
from reward_model_dataset import format_input, USE_LIKERT
from utils import device, initialize_seeds

def load_data(split: str) -> List[dict]:
    return pd.read_csv(f"data/raw/eedi_expanded_{split}.csv").to_dict("records")

def get_data(split: str, options: TrainOptions):
    if split == "train":
        # Get training samples and corpus from train set
        train_data = load_data("train")
        random.Random(221).shuffle(train_data)
        if not options.train_size and not options.corpus_size:
            data = train_data
            corpus = None
        else:
            train_size = options.train_size or len(train_data) - options.corpus_size
            corpus_size = options.corpus_size or len(train_data) - train_size
            data = train_data[:train_size]
            corpus = train_data[train_size : train_size + corpus_size]
    else:
        # Get evaluation samples from split and corpus from train set
        if split == "dev":
            data = load_data("val")
        elif split == "test":
            data = load_data("test")
        if options.val_size:
            data = data[:options.val_size]
        corpus = load_data("train")
        if options.val_corpus_size:
            corpus = random.Random(221).sample(corpus, options.val_corpus_size)
    return data, corpus

def process_sample(sample: dict) -> DataSample:
    question = sample["question"]
    correct_answer = sample["correct_answer"]
    solution = sample["explanation"]
    distractor = sample["distractor"]
    feedback = sample["feedback"]
    return {
        "lm_context": f"Problem: {question}\nCorrect Answer: {correct_answer}\nSolution: {solution}\nIncorrect Answer: {distractor}\nFeedback:",
        "lm_label": f" {feedback}",
        "encoder_context": f"Problem: {question}\nCorrect Answer: {correct_answer}\nSolution: {solution}\nIncorrect Answer: {distractor}",
        "encoder_label": f"\nFeedback: {feedback}",
        "meta_data": sample,
    }

def get_check_correct_batch_fn(model_name: str, base_model: str, test: bool):
    use_t5_decoder = False
    enc_dec = ("t5" in base_model) and use_t5_decoder
    tokenizer = get_tokenizer(base_model)
    rm = get_ensemble(model_name.split(","), base_model, tokenizer, enc_dec, True)
    rm.eval()

    def check_correct_batch(src_meta_datas: List[dict], pred_texts: List[str]):
        nonlocal tokenizer, rm

        with torch.no_grad():
            batch_size = 4
            all_scores = []
            for batch_start_idx in range(0, len(pred_texts), batch_size):
                meta_data_batch = src_meta_datas[batch_start_idx : batch_start_idx + batch_size]
                pred_text_batch = pred_texts[batch_start_idx : batch_start_idx + batch_size]
                # TODO: handle encoder-decoder model
                inputs = tokenizer(
                    [format_input(meta_data, pred_text) for meta_data, pred_text in zip(meta_data_batch, pred_text_batch)],
                    padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                logits = rm(**inputs).logits
                scores = torch.sigmoid(logits)
                if not USE_LIKERT:
                    scores[:, :2] = 1 - scores[:, :2]
                if not test:
                    scores = scores[:, 0] * scores.mean(dim=1)
                all_scores.append(scores)
        return torch.concat(all_scores).cpu()

    return check_correct_batch

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser("Feedback RetICL Training")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--method", type=str, choices=["reticl", SamplingMethod.SIMILARITY.value, SamplingMethod.RANDOM.value], default="reticl")
    parser.add_argument("--model_name", type=str, default="feedback_reticl")
    parser.add_argument("--generator_model", type=str, default="gpt3")
    parser.add_argument("--openai_model", type=str, default="code-davinci-002")
    parser.add_argument("--rm_name", type=str, default="reward_model")
    parser.add_argument("--rm_base", type=str, default="google/flan-t5-xl")
    parser.add_argument("--include_rubric", action="store_true", help="Include rubric in prompt instruction")
    parser.add_argument("--reward", type=str, choices=[Reward.EXACT.value, Reward.EXACT_AND_PPL.value], default=Reward.EXACT.value)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--e_coef", type=float, default=0.4)
    parser.add_argument("--expl_decay_rate", type=float, default=1.0)
    parser.add_argument("--corpus_size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--gen_batch_size", type=int, default=10)
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    args = parser.parse_args()

    train_options = {
        "dataset": "feedback",
        "rl_algo": RLAlgorithm.PPO.value if args.method == "reticl" else None,
        "sm": SamplingMethod.SOFTMAX.value if args.method == "reticl" else args.method,
        "model_name": args.model_name if args.method == "reticl" else None,
        "generator_model": args.generator_model,
        "gpt3_model": args.openai_model,
        "reward": args.reward,
        "train_size": 0,
        "corpus_size": args.corpus_size,
        "e_coef": args.e_coef,
        "expl_decay_rate": args.expl_decay_rate,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "gen_batch_size": args.gen_batch_size,
        "val_size": 0 if args.test else 250,
        "max_gen_tokens": 300,
        "wandb": args.wandb
    }
    feedback_config = {
        "get_data": get_data,
        "process_sample": process_sample,
        "check_correct_batch": get_check_correct_batch_fn(args.rm_name, args.rm_base, False),
        "prompt_prefix": FEEDBACK_GEN_INSTRUCTION + (("\n\n" + FEEDBACK_GEN_INSTRUCTION_RUBRIC) if args.include_rubric else "")
    }

    with GeneratorCM(train_options):
        if args.test:
            evaluate(feedback_config, "test", train_options)
        else:
            train_reticl(feedback_config, "train", "dev", train_options)

if __name__ == "__main__":
    main()
