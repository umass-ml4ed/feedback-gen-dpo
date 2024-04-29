import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
import evaluate

from reward_model_dataset import LABELS, USE_LIKERT

methods = ["gold", "random", "knn", "zs"]

def get_stats(dfs, labels, rouge_all, bertscore_all):
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
            if bertscore_all is not None:
                bertscore = bertscore_all[df["method"] == method].mean()
                df_to_label_to_method_to_mean[df_idx]["bertscore"][method] = bertscore
                bertscores.append(f"{method}: {bertscore:.2f}")
        scores = df[labels[0]] * sum([df[label] for label in labels]) / len(labels)
        means.append(f"all: {scores.mean():.2f}")
        print(f"Score: {', '.join(means)}")
        rouges.append(f"all: {rouge_all.mean():.2f}")
        print(f"ROUGE-L: {', '.join(rouges)}")
        if bertscore_all is not None:
            bertscores.append(f"all: {bertscore_all.mean():.2f}")
            print(f"BERTScore: {', '.join(bertscores)}")
        print("\n")

    return df_to_label_to_method_to_mean

def get_agreement(dfs, labels, rouge_all, bertscore_all, df_to_label_to_method_to_mean):
    print("Agreement:")
    method_masks = [dfs[0]["method"] == method for method in methods] + [np.ones(len(dfs[0])).astype(bool)]
    for method, mask in zip(methods + ["All"], method_masks):
        print("\nMethod:", method)
        annos = [df.loc[mask] for df in dfs]
        metrics_over_labels = []
        for label in labels:
            kappa = cohen_kappa_score(annos[0][label], annos[1][label])
            pearson = pearsonr(annos[0][label], annos[1][label])[0]
            acc = (annos[0][label] == annos[1][label]).mean()
            prec, rec, f1, _ = precision_recall_fscore_support(annos[1][label], annos[0][label], average="binary")
            metrics_over_labels.append([kappa, pearson, acc, prec, rec, f1])
            print(f"{label}: Kappa: {kappa:.2f}, Pearson: {pearson:.2f}, Acc: {acc:.2f}, Prec: {prec:.2f}, Rec: {rec:.2f}, F1: {f1:.2f}")
        avg_mol = np.array(metrics_over_labels).mean(axis=0)
        print(f"Mean: Kappa {avg_mol[0]:.2f}:, Pearson {avg_mol[1]:.2f}:, Acc: {avg_mol[2]:.2f}, Prec: {avg_mol[3]:.2f}, Rec: {avg_mol[4]:.2f}, F1: {avg_mol[5]:.2f}")

        anno_scores = [df[labels[0]] * sum([df[label] for label in labels]) / len(labels) for df in annos]
        print(f"Score Correlation: {pearsonr(anno_scores[0], anno_scores[1])[0]:.2f}")
        rouge = rouge_all[mask]
        print(f"ROUGE-L Correlation: {pearsonr(anno_scores[1], rouge)[0]:.2f}")
        if bertscore_all is not None:
            bertscore = bertscore_all[mask]
            print(f"BERTScore Correlation: {pearsonr(anno_scores[1], bertscore)[0]:.2f}")

    do_system = False
    if do_system:
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
        if bertscore_all is not None:
            corr = pearsonr(
                list(df_to_label_to_method_to_mean[1]['score'].values()),
                list(df_to_label_to_method_to_mean[1]['bertscore'].values()))[0]
            print(f"bertscore: {corr:.2f}")

    perfect = np.array([True] * len(dfs[0]))
    for label in labels:
        perfect &= dfs[0][label] == dfs[1][label]
    print(f"Perfect: {perfect.sum()} / {len(dfs[0])}")
    print_imperfect = False
    if print_imperfect:
        for (_, a1_row), (_, a2_row) in zip(dfs[0][~perfect].iterrows(), dfs[1][~perfect].iterrows()):
            print()
            print(a1_row["question"])
            print(a1_row["distractor"])
            print(a1_row["feedback"])
            print([a1_row[label] for label in labels])
            print([a2_row[label] for label in labels])

def analyze_datasets(pred_anno: str, gold_anno: str, do_bertscore: bool):
    rouge_metric = evaluate.load("rouge")
    if do_bertscore:
        bertscore_metric = evaluate.load("bertscore")

    # Load data
    postfix = "lik" if USE_LIKERT else "bin"
    name_to_file_map = {
        "ours": f"data/annotated/feedback_test_single_subset_annotated_{postfix}_manual.csv",
        "gpt4": f"data/annotated/feedback_test_single_subset_annotated_{postfix}_gpt-4.csv",
        "external": f"data/annotated/feedback_test_single_subset_annotated_{postfix}_external.csv"
    }
    dfs = [pd.read_csv(name_to_file_map[pred_anno]), pd.read_csv(name_to_file_map[gold_anno])]
    df_to_external = [pred_anno == "external", gold_anno == "external"]

    # Only consider rows where both dfs are labeled
    dfs = [df[df[LABELS[1]].notna()] for df in dfs]
    if len(dfs) == 2:
        merge_keys = ["qid", "distractor", "feedback"]
        dfs[0] = dfs[0].merge(dfs[1][merge_keys], on=merge_keys, how="inner")
        dfs[1] = dfs[1].merge(dfs[0][merge_keys], on=merge_keys, how="inner")
    # Sorting is necessary to line up with original dataset after merging
    dfs = [df.sort_values(["qid", "distractor"], ignore_index=True) for df in dfs]

    # Flip labels
    labels = LABELS
    if not USE_LIKERT:
        labels[0] = "correct"
        for df, external in zip(dfs, df_to_external):
            if external:
                for label in labels:
                    df[label] = df[label].astype(int)
            else:
                df["correct"] = 1 - df["incorrect"]
                df["reveal"] = 1 - df["reveal"]

    # Compute ROUGE-L and BERTScore
    og_df = pd.read_csv("data/raw/eedi_expanded_test.csv")
    og_df_dedup = og_df[~og_df[["qid", "distractor"]].duplicated()]
    og_joined = dfs[0].merge(og_df_dedup, on=["qid", "distractor"])
    pred_feedbacks = og_joined["feedback_x"].to_numpy()
    og_feedbacks = og_joined["feedback_y"].to_numpy()
    rouge_all = np.array(rouge_metric.compute(
        predictions=pred_feedbacks, references=og_feedbacks, use_aggregator=False)["rougeL"])
    if do_bertscore:
        bertscore_all = np.array(bertscore_metric.compute(
            predictions=pred_feedbacks, references=og_feedbacks, model_type="microsoft/deberta-xlarge-mnli")["f1"])
    else:
        bertscore_all = None

    # Get means across datasets/labels/methods
    df_to_label_to_method_to_mean = get_stats(dfs, labels, rouge_all, bertscore_all)

    # Get agreement across datasets per label
    if len(dfs) == 2:
        get_agreement(dfs, labels, rouge_all, bertscore_all, df_to_label_to_method_to_mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_anno", type=str, choices=["ours", "external", "gpt4"], help="Set for predicted annotations")
    parser.add_argument("gold_anno", type=str, choices=["ours", "external", "gpt4"], help="Set for ground-truch annotations")
    parser.add_argument("--do_bertscore", action="store_true", help="Compute BERTScore")
    args = parser.parse_args()

    analyze_datasets(args.pred_anno, args.gold_anno, args.do_bertscore)
