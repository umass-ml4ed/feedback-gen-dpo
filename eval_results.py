import argparse
import pandas as pd
import evaluate

from reward_model_dataset import LABELS, USE_LIKERT
from create_reward_dataset import annotate_with_llm
from train_reticl import get_check_correct_batch_fn
from llm import LLMCM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_filename", type=str, help="Result file to evaluate")
    parser.add_argument("--metric", type=str, choices=["llm", "rm", "rouge", "all"], help="Evaluation metric to use")
    parser.add_argument("--src", type=str, choices=["og", "icl", "reticl"], help="Source of result file")
    parser.add_argument("--rm_name", type=str, default="reward_model", help="Reward model name")
    parser.add_argument("--rm_base", type=str, default="xlnet-base-cased", help="Reward model base")

    parser.add_argument("--model", default="gpt-4", help="Inference model for LLM evaluation")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for LLM evaluation")
    parser.add_argument("--max_gen_tokens", type=int, default=300, help="Maximimum tokens to generate")

    args = parser.parse_args()

    # Load result data and normalize columns
    df = pd.read_csv(args.result_filename)
    if args.src == "reticl":
        og_df = pd.read_csv("data/raw/eedi_expanded_test.csv")
        og_df["feedback"] = df["pred"]
        df = og_df
    if args.src == "icl":
        df["feedback"] = df["generated_feedback"]

    df = df[:10]

    # Score feedback based on metric
    if args.metric == "llm":
        with LLMCM(args):
            annotate_with_llm(df)
            for label in LABELS:
                print(f"{label}: {df[label].mean():.4f}")
            if USE_LIKERT:
                overall = sum([df[label] for label in LABELS])
            else:
                overall = (1 - df["incorrect"]) + (1 - df["reveal"]) + df["suggestions"] + df["misconceptions"] + df["positive"]
            print(f"Overall: {overall.mean() / 5:.4f}")
    elif args.metric == "rm":
        check_correct_batch = get_check_correct_batch_fn(args.rm_name, args.rm_base, True)
        scores = check_correct_batch(df.to_dict("records"), df["feedback"].tolist())
        print(f"Overall: {scores.mean():.4f}")
    elif args.metric == "rouge":
        og_df = pd.read_csv("data/raw/eedi_expanded_test.csv")
        metric = evaluate.load("rouge")
        rouge = metric.compute(
            predictions=[sample["feedback"] for _, sample in df.iterrows()],
            references=[sample["feedback"] for _, sample in og_df.iterrows()],
            use_stemmer=True
        )
        print(rouge)

    # TODO: export file

if __name__ == "__main__":
    main()
