import argparse
import pandas as pd
import numpy as np
import evaluate

from reward_model_dataset import LABELS, USE_LIKERT
from create_reward_dataset import annotate_with_llm
from train_reticl import get_check_correct_batch_fn
from llm import LLMCM
from utils import initialize_seeds

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    parser.add_argument("result_filename", type=str, help="Result file to evaluate")
    parser.add_argument("--metric", type=str, choices=["llm", "rm", "ref"], help="Evaluation metric to use")
    parser.add_argument("--src", type=str, choices=["og", "icl", "reticl"], help="Source of result file")
    parser.add_argument("--rm_name", type=str, default="reward_model", help="Reward model name")
    parser.add_argument("--rm_base", type=str, default="google/flan-t5-xl", help="Reward model base")
    parser.add_argument("--model", default="gpt-4", help="Inference model for LLM evaluation")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for LLM evaluation")
    parser.add_argument("--max_gen_tokens", type=int, default=300, help="Maximimum tokens to generate")

    args = parser.parse_args()

    # Load result data and normalize columns
    df = pd.read_csv(args.result_filename)
    if args.src == "reticl":
        og_df = pd.read_csv("data/raw/eedi_expanded_test.csv")
        og_df["feedback"] = df["pred"]
        og_df["reticl_prompt"] = df["prompt"]
        df = og_df.copy()
    if args.src == "icl":
        df["feedback"] = df["generated_feedback"]

    # if args.metric == "llm":
    #     df = df.sample(n=50)

    labels = LABELS if USE_LIKERT else ["correct"] + LABELS[1:]

    # Score feedback based on metric
    if args.metric == "llm":
        with LLMCM(args):
            annotate_with_llm(df)
            if not USE_LIKERT:
                df["incorrect"] = 1 - df["incorrect"]
                df = df.rename(columns={"incorrect": "correct"})
                df["reveal"] = 1 - df["reveal"]
            for label in labels:
                print(f"{label}: {df[label].mean():.2f}")
            score = df[labels[0]] * sum([df[label] for label in labels]) / len(labels)
            print(f"Score: {score.mean():.2f}")
    elif args.metric == "rm":
        check_correct_batch = get_check_correct_batch_fn(args.rm_name, args.rm_base, True)
        scores = check_correct_batch(df.to_dict("records"), df["feedback"].tolist())
        scores = scores.round()
        for label_idx, label in enumerate(labels):
            df[label] = scores[:, label_idx]
            print(f"{label}: {scores[:, label_idx].mean():.2f}")
        score = scores[:, 0] * scores.mean(dim=1)
        print(f"Score: {score.mean():.2f}")
    elif args.metric == "ref":
        og_df = pd.read_csv("data/raw/eedi_expanded_test.csv")
        pred_feedbacks = [sample["feedback"] for _, sample in df.iterrows()]
        ref_feedbacks = [sample["feedback"] for _, sample in og_df.iterrows()]
        metric = evaluate.load("rouge")
        rouge = metric.compute(predictions=pred_feedbacks, references=ref_feedbacks)["rougeL"]
        metric = evaluate.load("bertscore")
        bertscore = np.array(metric.compute(
            predictions=pred_feedbacks, references=ref_feedbacks, model_type="microsoft/deberta-xlarge-mnli")["f1"]).mean()
        print(f"Rouge: {rouge:.2f}, BertScore: {bertscore:.2f}")

    # Export file
    if args.metric != "ref":
        df.to_csv(args.result_filename.replace(".csv", f"_eval_{args.metric}.csv"), index=False)

if __name__ == "__main__":
    main()
