# download_dataset.py

from datasets import load_dataset
import pandas as pd

def download_dataset():
    dataset = load_dataset("ucirvine/sms_spam", split="train")
    df = pd.DataFrame({
        "label": [0 if l == 'ham' else 1 for l in dataset["label"]],
        "text": dataset["sms"]
    })
    df.to_csv("data/spam.csv", index=False)
    print("Saved", len(df), "rows to data/spam.csv")

if __name__ == "__main__":
    download_dataset()
