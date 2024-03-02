import pandas as pd
from transformers import AutoTokenizer

def load_dataset(split: str):
    return pd.read_csv(f"data/raw/eedi_expanded_{split}.csv")

def main():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    for split in ["train", "val", "test"]:
        dataset = load_dataset(split)
        token_lens = []
        for _, row in dataset.iterrows():
            token_lens.append(len(tokenizer(row["feedback"].strip()).input_ids))
        print(f"{split}: {pd.Series(token_lens).max()}")

if __name__ == "__main__":
    main()
