from typing import Union
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

from utils import device

USE_LIKERT = False

if USE_LIKERT:
    LABELS = ["accurate", "reveal", "suggestions", "misconceptions", "positive"]
else:
    LABELS = ["incorrect", "reveal", "suggestions", "misconceptions", "positive"]

def load_dataset(split: str, annotator: str = "gpt-4"):
    postfix = "lik" if USE_LIKERT else "bin"
    dataset = pd.read_csv(f"data/annotated/feedback_{split}_single_subset_annotated_{postfix}_{annotator}.csv")
    dataset = dataset[~dataset[LABELS[0]].isna()]
    src_data = pd.read_csv(f"data/raw/eedi_expanded_{split}.csv")
    return dataset, src_data

def format_input(row: Union[dict, pd.Series], feedback: str = None):
    if feedback is None:
        feedback = str(row["feedback"])
    return "Given a question, its solution, the incorrect answer a student gave, and the feedback given to the student, evaluate the feedback.\n" +\
        f"Question: {row['question'].strip()}\n" +\
        f"Solution: {str(row['explanation']).strip()}\n" +\
        f"Incorrect Answer: {str(row['distractor']).strip()}\n" +\
        f"Feedback: {str(feedback).strip()}"
    # return f"Question: {row['question'].strip()}\n" +\
    #     f"Correct Answer: {str(row['correct_answer']).strip()}\n" +\
    #     f"Incorrect Answer: {str(row['distractor']).strip()}\n" +\
    #     f"Feedback: {str(feedback).strip()}"

def format_input_enc(row: Union[dict, pd.Series]):
    return "Given a question, its solution, the incorrect answer a student gave, and the feedback given to the student, evaluate the feedback.\n" +\
        f"Question: {row['question'].strip()}\n" +\
        f"Solution: {str(row['explanation']).strip()}\n" +\
        f"Incorrect Answer: {str(row['distractor']).strip()}"

def format_input_dec(row: Union[dict, pd.Series], feedback: str = None):
    if feedback is None:
        feedback = str(row["feedback"])
    return f"Feedback: {feedback.strip()}"

class RewardModelDataset(Dataset):
    def __init__(self, data: pd.DataFrame, src_data: pd.DataFrame, enc_dec: bool, mismatch_rate: int, use_mismatch_intra: bool):
        super().__init__()
        # TODO: handle likert
        # additionally, make it so that inaccurate feedbacks get all negative scores
        # might also want to round up .5 suggestions to 1 since hints can be good (look at this)

        # Add mismatched feedback from across questions
        for _ in range(mismatch_rate):
            # mismatch_samples = data[data["method"] == "gold"].copy()
            mismatch_samples = src_data.copy()
            mismatch_samples["method"] = "mismatch_outer"
            mismatch_samples["feedback"] = mismatch_samples["feedback"].sample(frac=1).values
            mismatch_samples["incorrect"] = 1.0
            for label in LABELS[1:]:
                mismatch_samples[label] = 0.0
            data = pd.concat([data, mismatch_samples])
        # Add mismatched feedback within questions
        if use_mismatch_intra:
            idxs = np.arange(len(src_data))
            for offset in range(1, 3):
                mismatch_samples = src_data.copy()
                mismatch_samples["method"] = "mismatch_inner"
                target_idxs = idxs - idxs % 3 + (idxs + offset) % 3
                mismatch_samples["feedback"] = src_data.iloc[target_idxs]["feedback"].values
                mismatch_samples["incorrect"] = 1.0
                for label in LABELS[1:]:
                    mismatch_samples[label] = 0.0
                data = pd.concat([data, mismatch_samples])
        self.data = [
            {
                "input": format_input_enc(row) if enc_dec else format_input(row),
                "decoder_input": format_input_dec(row) if enc_dec else None,
                "label": torch.Tensor([row[label] for label in LABELS]),
                "method": row["method"]
            }
            for _, row in data.iterrows()
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class RewardModelCollator:
    def __init__(self, tokenizer, enc_dec: bool):
        self.tokenizer = tokenizer
        self.enc_dec = enc_dec

    def __call__(self, batch):
        tokenized_inputs = self.tokenizer(
            [sample["input"] for sample in batch], padding=True, truncation=True, return_tensors="pt"
        )
        result = {
            **tokenized_inputs,
            "labels": torch.stack([sample["label"] for sample in batch]),
            "methods": [sample["method"] for sample in batch]
        }
        if self.enc_dec:
            tokenized_decoder_inputs = self.tokenizer(
                [sample["decoder_input"] for sample in batch],
                padding=True, truncation=True, return_tensors="pt"
            )
            result["decoder_input_ids"] = tokenized_decoder_inputs.input_ids
            result["decoder_attention_mask"] = tokenized_decoder_inputs.attention_mask
        return result
