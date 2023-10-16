import re
import pandas as pd
import pyarrow as pa
from datasets import Dataset
from pylatexenc.latex2text import LatexNodes2Text

latex_parser = LatexNodes2Text()
whitespace_re = re.compile(r"\s+")

def clean_whitespace(text: str):
    return whitespace_re.sub(" ", text.strip())

def parse_text(text: str):
    return clean_whitespace(latex_parser.latex_to_text(text))

def get_raw_dataset():
    df = pd.read_csv("data/pp_eedi_data_0912.csv")
    valid = df["Script Clean"] | (df["Cleaned"] & ~df["Discuss Flag"])
    valid = valid & ~df["Image Needed"]
    df = df[valid]
    # Convert LaTeX to unicode
    # df["question"] = df["question"].apply(parse_text)
    # for i in range(1, 5):
    #     df[f"Answer{i}"] = df[f"Answer{i}"].apply(parse_text)
    #     df[f"Explanation{i}"] = df[f"Explanation{i}"].apply(clean_whitespace)
    return df

def expand_rows(df: pd.DataFrame):
    result = []
    for row_idx, row in df.iterrows():
        correct_answer, explanation = None, None
        distractors = []
        for i in range(1, 5):
            try:
                if i == int(row["CorrectAnswer"]):
                    correct_answer = row[f"Answer{i}"]
                    explanation = row[f"Explanation{i}"]
                else:
                    distractors.append((row[f"Answer{i}"], row[f"Explanation{i}"]))
            except Exception as e:
                print(e)
        if not correct_answer:
            print(f"No correct answer for {row_idx} - skipping")
            continue
        for distractor, feedback in distractors:
            result.append(
                {
                    "qid": row["id"],
                    "question": row["question"],
                    "construct_id": row["ConstructId"],
                    "correct_answer": correct_answer,
                    "explanation": explanation,
                    "distractor": distractor,
                    "feedback": feedback,
                }
            )
    return pd.DataFrame(result)

def extract_feedback(output: str):
    return re.search(r"Feedback: (.*)$", output).group(1).strip()

def load_pd_dataset(filename: str):
    return expand_rows(pd.read_csv(filename))

def load_hf_dataset(filename: str):
    return Dataset(pa.Table.from_pandas(load_pd_dataset(filename)))
