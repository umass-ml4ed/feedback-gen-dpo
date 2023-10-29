from typing import List
import argparse
import re
import random
from itertools import combinations
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm
import evaluate

from llm import LLM, LLMCM
from reward_model_dataset import LABELS, USE_LIKERT
from feedback_dataset import get_raw_dataset, expand_rows
from utils import initialize_seeds

FEEDBACK_GEN_INSTRUCTION = "Given a math question, the correct answer, and an incorrect answer chosen by a student, " +\
    "write feedback for the incorrect answer that would help the student understand and correct their mistake. " +\
    "The feedback should be short, only one or two sentences."

FEEDBACK_GEN_INSTRUCTION_RUBRIC = "Additionally, keep the following requirements in mind:\n" +\
    "1. The feedback should not make any incorrect statements.\n" +\
    "2. The feedback should not directly reveal the answer to the question.\n" +\
    "3. The feedback should give suggestions to the student on how to improve their answer.\n" +\
    "4. The feedback should point out the misconception underlying the student's answer.\n" +\
    "5. The feedback should have a positive and encouraging tone."

LIKERT_LABELS = ["Accuracy", "Revealing", "Suggestions", "Misconception", "Positive"]
LIKERT_MAX_SCORES = [2, 2, 3, 3, 3]

def expand_dataset():
    df = get_raw_dataset()
    df = df.sample(frac=1)

    splits = [
        ("train", df.iloc[:int(len(df) * 0.6)]),
        ("val", df.iloc[int(len(df) * 0.6):int(len(df) * 0.8)]),
        ("test", df.iloc[int(len(df) * 0.8):])
    ]
    for split, split_df in splits:
        expand_rows(split_df).to_csv(f"data/raw/eedi_expanded_{split}.csv", index=False)

def feedback_prompt_input(row: pd.Series):
    return f"Question: {row['question'].strip()}\n" +\
        f"Correct Answer: {str(row['correct_answer']).strip()}\n" +\
        f"Incorrect Answer: {str(row['distractor']).strip()}"

def feedback_prompt_input_sol(row: pd.Series):
    return f"Problem: {row['question'].strip()}\n" +\
        f"Correct Answer: {str(row['correct_answer']).strip()}\n" +\
        f"Solution: {str(row['explanation']).strip()}\n" +\
        f"Incorrect Answer: {str(row['distractor']).strip()}"

def feedback_prompt_output(row: pd.Series):
    return f"Feedback: {row['feedback'].strip()}"

def generate_feedback_random(pool_df: pd.DataFrame, target_df: pd.DataFrame, k: int, prompt_input_fn, prompt_instruction: str, mask_diagonal: bool):
    prompts = []
    all_idxs = list(range(len(pool_df)))
    for row_idx, row in target_df.iterrows():
        if mask_diagonal:
            available_idxs = all_idxs[:row_idx] + all_idxs[row_idx + 1:]
        else:
            available_idxs = all_idxs
        example_idxs = random.sample(available_idxs, k)
        prompt = prompt_instruction + "\n\n"
        for example_idx in example_idxs:
            prompt += prompt_input_fn(pool_df.iloc[example_idx]) + "\n" + feedback_prompt_output(pool_df.iloc[example_idx]) + "\n\n"
        prompt += prompt_input_fn(row) + "\nFeedback:"
        prompts.append(prompt)
    outputs = [output.strip() for output in LLM.generate(prompts, show_progress=True)]
    return prompts, outputs

def encode_strings(encoder: SentenceTransformer, strings: List[str]):
    encodings = []
    batch_size = 10
    for batch_start_idx in tqdm(range(0, len(strings), batch_size)):
        batch = strings[batch_start_idx : batch_start_idx + batch_size]
        encodings.append(encoder.encode(batch, convert_to_tensor=True, normalize_embeddings=True))
    return torch.concat(encodings, dim=0)

def generate_feedback_knn(pool_df: pd.DataFrame, target_df: pd.DataFrame, k: int, encoder_model: str, prompt_input_fn, prompt_instruction: str, mask_diagonal: bool):
    encoder = SentenceTransformer(encoder_model)
    pool_inputs = [prompt_input_fn(row) for _, row in pool_df.iterrows()]
    pool_encodings = encode_strings(encoder, pool_inputs)
    if mask_diagonal:
        target_inputs = pool_inputs
        sim_matrix = pool_encodings @ pool_encodings.T
        sim_matrix[torch.eye(len(sim_matrix)).bool()] = 0
    else:
        target_inputs = [prompt_input_fn(row) for _, row in target_df.iterrows()]
        target_encodings = encode_strings(encoder, target_inputs)
        sim_matrix = target_encodings @ pool_encodings.T

    prompts = []
    for row_idx, _ in target_df.iterrows():
        example_idxs = torch.topk(sim_matrix[row_idx], k).indices
        example_idxs = torch.flip(example_idxs, dims=(0,))
        prompt = prompt_instruction + "\n\n"
        for example_idx in example_idxs.tolist():
            prompt += pool_inputs[example_idx] + "\n" + feedback_prompt_output(pool_df.iloc[example_idx]) + "\n\n"
        prompt += target_inputs[row_idx] + "\nFeedback:"
        prompts.append(prompt)
    outputs = [output.strip() for output in LLM.generate(prompts, show_progress=True)]
    return prompts, outputs

def generate_feedback_zs(df: pd.DataFrame, prompt_input_fn, prompt_instruction: str):
    prompts = [
        prompt_instruction + "\n\n" + prompt_input_fn(row) + "\nFeedback:"
        for _, row in df.iterrows()
    ]
    outputs = [output.strip() for output in LLM.generate(prompts, show_progress=True)]
    return prompts, outputs

def generate_feedback(method: str, args):
    prompt_input_fn = feedback_prompt_input_sol if args.include_sol else feedback_prompt_input
    prompt_instruction = FEEDBACK_GEN_INSTRUCTION
    if args.include_rubric:
        prompt_instruction += "\n\n" + FEEDBACK_GEN_INSTRUCTION_RUBRIC
    train_df = pd.read_csv("data/raw/eedi_expanded_train.csv")
    val_df = pd.read_csv("data/raw/eedi_expanded_val.csv")
    test_df = pd.read_csv("data/raw/eedi_expanded_test.csv")
    for split, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if args.split and split != args.split:
            continue
        filename = f"data/icl/feedback_{split}_{method}_{args.model}"
        if args.include_sol:
            filename += "_sol"
        if args.include_rubric:
            filename += "_rubric"
        if method == "random":
            prompts, outputs = generate_feedback_random(train_df, split_df, args.k, prompt_input_fn, prompt_instruction, split == "train")
            filename += f"_{args.k}"
        elif method == "knn":
            prompts, outputs = generate_feedback_knn(train_df, split_df, args.k, args.knn_model, prompt_input_fn, prompt_instruction, split == "train")
            filename += f"_{args.k}_{args.knn_model}"
        elif method == "zs":
            prompts, outputs = generate_feedback_zs(split_df, prompt_input_fn, prompt_instruction)
        split_df["prompt"] = prompts
        split_df["generated_feedback"] = outputs
        split_df.to_csv(f"{filename}.csv", index=False)

def compile_dataset(strategy: str):
    for split in ["train", "val", "test"]:
        dfs = []
        inputs = [
            ("knn", f"data/icl/feedback_{split}_knn_code-davinci-002_2_all-distilroberta-v1.csv"),
            ("random", f"data/icl/feedback_{split}_random_code-davinci-002_2.csv"),
            ("zs", f"data/icl/feedback_{split}_zs_gpt-3.5-turbo.csv")
        ]
        for method, filename in inputs:
            df = pd.read_csv(filename)
            df["method"] = method
            dfs.append(df)
        # Add gold feedbacks to be evaluated
        gold_df = dfs[0].copy()
        gold_df["method"] = "gold"
        gold_df["generated_feedback"] = gold_df["feedback"]
        dfs.append(gold_df)
        if strategy == "single":
            out_df = pd.concat(dfs)
            out_df = out_df[["qid", "question", "correct_answer", "explanation", "distractor", "generated_feedback", "method"]]
            out_df.rename(columns={"generated_feedback": "feedback"}, inplace=True)
        elif strategy == "h2h":
            import pdb; pdb.set_trace()
            methods = [method for method, _ in inputs] + ["gold"]
            result = []
            for i in range(len(dfs[0])):
                for matchup in combinations(range(len(methods)), 2):
                    if random.random() < 0.5:
                        matchup = (matchup[1], matchup[0])
                result.append({
                    "qid": dfs[0].iloc[i]["qid"],
                    "question": dfs[0].iloc[i]["question"],
                    "correct_answer": dfs[0].iloc[i]["correct_answer"],
                    "explanation": dfs[0].iloc[i]["explanation"],
                    "distractor": dfs[0].iloc[i]["distractor"],
                    "feedback_1": dfs[matchup[0]].iloc[i]["generated_feedback"],
                    "feedback_2": dfs[matchup[1]].iloc[i]["generated_feedback"],
                    "method_1": methods[matchup[0]],
                    "method_2": methods[matchup[1]],
                })
            out_df = pd.DataFrame(result)
        out_df.to_csv(f"data/compiled/feedback_{split}_{strategy}.csv", index=False)

def get_subset():
    df = pd.read_csv("data/compiled/feedback_train_single.csv")
    df.sample(n=10000).to_csv("data/compiled/feedback_train_single_subset.csv", index=False)
    df = pd.read_csv("data/compiled/feedback_val_single.csv")
    df.sample(n=1000).to_csv("data/compiled/feedback_val_single_subset.csv", index=False)
    df = pd.read_csv("data/compiled/feedback_test_single.csv")
    df.sample(n=1000).to_csv("data/compiled/feedback_test_single_subset.csv", index=False)

def get_single_annotation_prompt(question: str, correct_answer: str, explanation: str, distractor: str, feedback: str):
    return "Your job is to evaluate feedback given to students on math problems.\n\n" +\
        "Here is the question, the correct solution, the incorrect answer the student gave, and the feedback given to the student:\n" +\
        f"Question: {question}\n" +\
        f"Correct Answer: {correct_answer}\n" +\
        f"Solution: {explanation}\n" +\
        f"Incorrect Answer: {distractor}\n" +\
        f"Feedback: {feedback}\n\n" +\
        "For the following questions, provide a short explanation and then answer with \"Yes\" or \"No\":\n" +\
        "1. Does the feedback make any incorrect statements?\n" +\
        "2. Does the feedback directly reveal the answer to the question?\n" +\
        "3. Does the feedback give suggestions to the student on how to improve the answer?\n" +\
        "4. Does the feedback correctly point out the misconception underlying the student's answer?\n" +\
        "5. Does the feedback have a positive or encouraging tone?"

def get_likert_annotation_prompt(question: str, correct_answer: str, explanation: str, distractor: str, feedback: str):
    return "Your job is to evaluate feedback given to students on math problems.\n\n" +\
        "You will be given a question, the correct solution, the incorrect answer a student gave, " +\
        "and the feedback given to the student. Please evaluate the feedback for each of the following " +\
        "categories using a likert scale. For each category, first think step-by-step and then provide " +\
        "the final score using \"Score: <value>\".\n\n" +\
        "Accuracy: "+\
        "2 - the feedback only makes correct and accurate statements and correctly pertains to the question and student's answer. " +\
        "1 - the feedback makes inaccurate statements or does not pertain to the question or the answer that the student gave.\n" +\
        "Revealing: " +\
        "2 - the feedback does not directly mention the answer to the question (indirectly hinting at the answer is okay). " +\
        "1 - the feedback directly and clearly mentions the answer to the question.\n" +\
        "Suggestions: " +\
        "3 - the feedback gives a suggestion to the student that, when followed, will lead them to the correct answer. " +\
        "2 - the feedback does not explicitly provide a suggestion, but implicitly hints at something the student can do to reach the correct answer. " +\
        "1 - the feedback does not provide any suggestions OR the feedback provides a suggestion that will not help the student reach the correct answer OR the feedback provides a suggestion that does not pertain to the current question or student answer.\n" +\
        "Misconception: " +\
        "3 - the feedback correctly points out the error the student made. " +\
        "2 - the feedback does not explicitly point out the underlying misconception, but implicitly hints at the error the student made or partially identifies the misconception. " +\
        "1 - the feedback does not address the student's error OR identifies a misconception that does not reflect the actual error the student made.\n" +\
        "Positive: " +\
        "3 - the feedback has a positive or encouraging tone (even mildly positive counts). " +\
        "2 - the feedback is neutral in tone. " +\
        "1 - the feedback is blunt, impersonal, or not positive.\n\n" +\
        "Here is the question, the student's answer, and the feedback to evaluate::\n" +\
        f"Question: {question}\n" +\
        f"Correct Answer: {correct_answer}\n" +\
        f"Solution: {explanation}\n" +\
        f"Student Answer: {distractor}\n" +\
        f"Feedback: {feedback}"

def annotate_with_llm(df):
    prompt_fn = get_likert_annotation_prompt if USE_LIKERT else get_single_annotation_prompt
    prompts = [
        prompt_fn(
            sample["question"], str(sample["correct_answer"]), sample["explanation"], str(sample["distractor"]), str(sample["feedback"]))
        for _, sample in df.iterrows()
    ]
    predictions = LLM.generate(prompts, system_message="You are a math education expert.", show_progress=True)
    pred_labels = [[0.5] * len(predictions) for _ in range(len(LABELS))]
    for pred_idx, pred in enumerate(predictions):
        if USE_LIKERT:
            label_idx = 0
            for line in pred.split("\n"):
                score_match = re.findall(r"Score: (\d+)", line)
                if score_match:
                    score = int(score_match[0])
                    pred_labels[label_idx][pred_idx] = (score - 1) / (LIKERT_MAX_SCORES[label_idx] - 1)
                    label_idx += 1
            if label_idx < len(LABELS):
                print(f"Entry {pred_idx} missing annotation!")
        else:
            label_idx = -1
            for line in pred.split("\n"):
                if line.startswith(f"{label_idx + 2}."):
                    label_idx += 1
                if line.startswith(f"{label_idx + 1}. Yes") or line.endswith("Answer: Yes") or line.endswith("Answer: Yes."):
                    pred_labels[label_idx][pred_idx] = 1.
                elif line.startswith(f"{label_idx + 1}. No") or line.endswith("Answer: No") or line.endswith("Answer: No."):
                    pred_labels[label_idx][pred_idx] = 0.
            if any([pred_labels[label_idx][pred_idx] == 0.5 for label_idx in range(len(LABELS))]):
                print(f"Entry {pred_idx} missing annotation!")
    for pred_col, label in zip(pred_labels, LABELS):
        df[label] = pred_col
    df["prompt"] = prompts
    df["response"] = predictions

def annotate_all_with_llm(args):
    postfix = "lik" if USE_LIKERT else "bin"
    for split in ["train", "val", "test"]:
        if args.split and split != args.split:
            continue
        df = pd.read_csv(f"data/compiled/feedback_{split}_single_subset.csv")
        import pdb; pdb.set_trace()
        annotate_with_llm(df)
        df.to_csv(f"data/annotated/feedback_{split}_single_subset_annotated_{postfix}_{args.model}.csv", index=False)

def analyze_datasets():
    do_bertscore = False
    methods = ["gold", "random", "knn", "zs"]
    rouge_metric = evaluate.load("rouge")
    if do_bertscore:
        bertscore_metric = evaluate.load("bertscore")

    # Load data and filter out unlabeled entries
    postfix = "lik" if USE_LIKERT else "bin"
    anno_1_filenames = [f"data/annotated/feedback_{split}_single_subset_annotated_{postfix}_gpt-4.csv" for split in ["test"]]
    anno_2_filenames = [f"data/annotated/feedback_{split}_single_subset_annotated_{postfix}_manual.csv" for split in ["test"]]
    dfs = [
        pd.concat([pd.read_csv(filename) for filename in anno_1_filenames]),
        pd.concat([pd.read_csv(filename) for filename in anno_2_filenames])
    ]
    valid_idxs = ~dfs[0][LABELS[0]].isna()
    if len(dfs) == 2:
        valid_idxs &= ~dfs[1][LABELS[0]].isna()
    # Sorting is necessary to line up with original dataset after merging
    dfs = [df.loc[valid_idxs].sort_values(["qid", "distractor"], ignore_index=True) for df in dfs]

    # Flip labels
    labels = LABELS
    if not USE_LIKERT:
        labels[0] = "correct"
        for df in dfs:
            df["correct"] = 1 - df["incorrect"]
            df["reveal"] = 1 - df["reveal"]

    # Compute ROUGE-L and BERTScore
    og_df = pd.read_csv("data/raw/eedi_expanded_test.csv")
    og_df_dedup = og_df[~og_df[["qid","distractor"]].duplicated()]
    og_joined = dfs[0].merge(og_df_dedup, on=["qid", "distractor"])
    pred_feedbacks = og_joined["feedback_x"].to_numpy()
    og_feedbacks = og_joined["feedback_y"].to_numpy()
    rouge_all = np.array(rouge_metric.compute(
        predictions=pred_feedbacks, references=og_feedbacks, use_aggregator=False)["rougeL"])
    if do_bertscore:
        bertscore_all = np.array(bertscore_metric.compute(
            predictions=pred_feedbacks, references=og_feedbacks, model_type="microsoft/deberta-xlarge-mnli")["f1"])

    # Get means across datasets/labels/methods
    is_h2h = "feedback_1" in dfs[0].columns
    if is_h2h:
        print("Win averages:")
    else:
        print("Mean values:")
    df_to_label_to_method_to_mean = [{label: {} for label in labels + ["score", "rouge", "bertscore"]} for _ in range(2)]
    for df_idx, df in enumerate(dfs):
        for label in labels:
            means = []
            for method in methods:
                if is_h2h:
                    wins = (df["method_1"] == method & df[label] == 0) | (df["method_2"] == method & df[label] == 1)
                    means.append(f"{method}: {wins.mean():.2f}")
                else:
                    score = df[df["method"] == method][label].mean()
                    df_to_label_to_method_to_mean[df_idx][label][method] = score
                    means.append(f"{method}: {score:.2f}")
            if not is_h2h:
                scores = df[~df[label].isna()][label]
                means.append(f"all: {scores.mean():.2f}")
            print(f"{label}: {', '.join(means)}")
        means = []
        rouges = []
        bertscores = []
        for method in methods:
            df_method = df[df["method"] == method]
            score = (df_method[labels[0]] * sum([df_method[label] for label in labels]) / len(labels)).mean()
            df_to_label_to_method_to_mean[df_idx]["score"][method] = score
            means.append(f"{method}: {score:.2f}")
            rouge = rouge_all[df["method"] == method].mean()
            df_to_label_to_method_to_mean[df_idx]["rouge"][method] = rouge
            rouges.append(f"{method}: {rouge:.2f}")
            if do_bertscore:
                bertscore = bertscore_all[df["method"] == method].mean()
                df_to_label_to_method_to_mean[df_idx]["bertscore"][method] = bertscore
                bertscores.append(f"{method}: {bertscore:.2f}")
        scores = df[labels[0]] * sum([df[label] for label in labels]) / len(labels)
        means.append(f"all: {scores.mean():.2f}")
        print(f"Score: {', '.join(means)}")
        rouges.append(f"all: {rouge_all.mean():.2f}")
        print(f"ROUGE-L: {', '.join(rouges)}")
        if do_bertscore:
            bertscores.append(f"all: {bertscore_all.mean():.2f}")
            print(f"BERTScore: {', '.join(bertscores)}")
        print("\n")

    # Get agreement across datasets per label
    if len(dfs) == 2:
        print("Agreement:")

        method_masks = [dfs[0]["method"] == method for method in methods] + [np.ones(len(dfs[0])).astype(bool)]
        for method, mask in zip(methods + ["All"], method_masks):
            print("\nMethod:", method)
            annos = [df.loc[mask] for df in dfs]
            for label in labels:
                pearson = pearsonr(annos[0][label], annos[1][label])[0]
                kappa = cohen_kappa_score(annos[0][label], annos[1][label])
                print(f"{label}: Kappa: {kappa:.2f}, Pearson: {pearson:.2f}")

            anno_scores = [df[labels[0]] * sum([df[label] for label in labels]) / len(labels) for df in annos]
            print(f"Score Correlation: {pearsonr(anno_scores[0], anno_scores[1])[0]:.2f}")
            rouge = rouge_all[mask]
            print(f"ROUGE-L Correlation: {pearsonr(anno_scores[1], rouge)[0]:.2f}")
            if do_bertscore:
                bertscore = bertscore_all[mask]
                print(f"BERTScore Correlation: {pearsonr(anno_scores[1], bertscore)[0]:.2f}")

        print("\nSystem:")
        for label in labels + ["score"]:
            corr = pearsonr(
                list(df_to_label_to_method_to_mean[0][label].values()),
                list(df_to_label_to_method_to_mean[1][label].values()))[0]
            print(f"{label}: {corr:.2f}")
        corr = pearsonr(
            list(df_to_label_to_method_to_mean[1]['score'].values()),
            list(df_to_label_to_method_to_mean[1]['rouge'].values()))[0]
        print(f"rouge-l: {corr:.2f}")
        if do_bertscore:
            corr = pearsonr(
                list(df_to_label_to_method_to_mean[1]['score'].values()),
                list(df_to_label_to_method_to_mean[1]['bertscore'].values()))[0]
            print(f"bertscore: {corr:.2f}")

        perfect = np.array([True] * len(dfs[0]))
        for label in labels:
            perfect &= dfs[0][label] == dfs[1][label]
        print(f"Perfect: {perfect.sum()} / {len(dfs[0])}")
        print_imperfect = True
        if print_imperfect:
            for (_, a1_row), (_, a2_row) in zip(dfs[0][~perfect].iterrows(), dfs[1][~perfect].iterrows()):
                print()
                print(a1_row["question"])
                print(a1_row["distractor"])
                print(a1_row["feedback"])
                print([a1_row[label] for label in labels])
                print([a2_row[label] for label in labels])

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser("Feedback Reward Dataset")
    # Modes
    parser.add_argument("--expand", action="store_true", help="Expand dataset to have one feedback per data sample and split into train/val/test")
    parser.add_argument("--generate", type=str, choices=["random", "knn", "zs"], help="Generate feedback using specified method")
    parser.add_argument("--compile", type=str, choices=["single", "h2h"], help="Create feedback reward dataset from generated feedback files. Choose between single entry per feedback or head-to-head matchups between methods.")
    parser.add_argument("--subset", action="store_true", help="Compute subset of compiled dataset")
    parser.add_argument("--annotate", action="store_true", help="Annotate feedback reward dataset using LLM")
    parser.add_argument("--analyze", action="store_true", help="Get annotation statistics and agreement on dataset(s)")
    # Params
    parser.add_argument("--split", type=str, help="Only run on specified split if given")
    parser.add_argument("--model", type=str, default="code-davinci-002", help="Inference model for feedback generation or annotation")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for feedback generation or annotation")
    parser.add_argument("--max_gen_tokens", type=int, default=300, help="Maximimum tokens to generate")
    parser.add_argument("--k", type=int, default=2, help="Number of examples to use for few-shot feedback generation")
    parser.add_argument("--knn_model", default="all-distilroberta-v1", help="S-BERT model to use for example encoding for similarity search")
    parser.add_argument("--include_sol", action="store_true", help="Include solution in feedback generation prompt")
    parser.add_argument("--include_rubric", action="store_true", help="Include rubric in feedback generation prompt")

    args = parser.parse_args()
    if args.expand:
        expand_dataset()
    elif args.generate:
        with LLMCM(args):
            generate_feedback(args.generate, args)
    elif args.compile:
        compile_dataset(args.compile)
    elif args.subset:
        get_subset()
    elif args.annotate:
        with LLMCM(args):
            annotate_all_with_llm(args)
    elif args.analyze:
        analyze_datasets()

if __name__ == "__main__":
    main()
