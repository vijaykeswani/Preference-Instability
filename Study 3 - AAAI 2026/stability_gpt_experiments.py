import warnings
warnings.filterwarnings("ignore")

import random
import itertools
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from scipy.stats import f_oneway, pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Follow previous pipeline to clean and prepare the data
df = pd.read_csv('../data/clean_data_qualtrics_participants.csv')
df = df.reset_index(drop=True)
df = df[df.attfailed == 0]
ids = [id for id in df.tid.unique() if df[df.tid == id]['session_id'].nunique() >= 3]
df = df[df.tid.isin(ids)]
feats = ['dep', 'alco', 'crim', 'life',
         'years_waiting', 'work_hours', 'obesity', 'reject_chance']
for f in feats:
    df[f + "_diff"] = df["l_" + f] - df["r_" + f]
diff_cols = [f + "_diff" for f in feats]
l_cols = ["l_" + f for f in feats]
df_diff = df[["tid"] + diff_cols + ["chosen", "query_num", "created_at", "session_number"]]
df_diff = df_diff[df_diff["reject_chance_diff"] != 100]
df_diff = df_diff[df_diff["reject_chance_diff"] != -100]
df_diff.to_csv('clean_data_qualtrics.csv', index=False)

# Convert the differences into a readable format for LLMs
def make_prompt(row):
    parts = []

    if row["life_diff"] != 0:
        parts.append(
            f"Patient A is expected to have {abs(row['life_diff'])} more life year{'s' if abs(row['life_diff']) != 1 else ''} gained due to a successful kidney transplant than Patient B."
            if row["life_diff"] > 0 else
            f"Patient B is expected to have {abs(row['life_diff'])} more life year{'s' if abs(row['life_diff']) != 1 else ''} gained due to a successful kidney transplant than Patient A."
        )

    if row["years_waiting_diff"] != 0:
        parts.append(
            f"Patient A has waited {abs(row['years_waiting_diff'])} more year{'s' if abs(row['years_waiting_diff']) != 1 else ''} on the transplant list than Patient B."
            if row["years_waiting_diff"] > 0 else
            f"Patient B has waited {abs(row['years_waiting_diff'])} more year{'s' if abs(row['years_waiting_diff']) != 1 else ''} on the transplant list than Patient A."
        )

    if row["work_hours_diff"] != 0:
        parts.append(
            f"Patient A is expected to work {abs(row['work_hours_diff'])} more hours per week after the transplant than Patient B."
            if row["work_hours_diff"] > 0 else
            f"Patient B is expected to work {abs(row['work_hours_diff'])} more hours per week after the transplant than Patient A."
        )

    if row["reject_chance_diff"] != 0:
        parts.append(
            f"Patient A has a {abs(row['reject_chance_diff'])}% lower chance of organ rejection than Patient B."
            if row["reject_chance_diff"] < 0 else
            f"Patient B has a {abs(row['reject_chance_diff'])}% lower chance of organ rejection than Patient A."
        )

    if row["dep_diff"] != 0:
        parts.append(
            f"Patient A has {abs(row['dep_diff'])} more dependents than Patient B."
            if row["dep_diff"] > 0 else
            f"Patient B has {abs(row['dep_diff'])} more dependents than Patient A."
        )

    if row["alco_diff"] != 0:
        parts.append(
            f"Patient A drinks {abs(row['alco_diff'])} fewer alcoholic drinks per day prediagnosis than Patient B."
            if row["alco_diff"] < 0 else
            f"Patient B drinks {abs(row['alco_diff'])} fewer alcoholic drinks per day prediagnosis than Patient A."
        )

    if row["crim_diff"] != 0:
        parts.append(
            f"Patient A has committed {abs(row['crim_diff'])} fewer past serious crimes than Patient B."
            if row["crim_diff"] < 0 else
            f"Patient B has committed {abs(row['crim_diff'])} fewer past serious crimes than Patient A."
        )

    if row["obesity_diff"] != 0:
        parts.append(
            f"Patient A is {abs(row['obesity_diff'])} level(s) of obesity (from underweight, normal weight, overweight, obese, or severely obese) lower than Patient B."
            if row["obesity_diff"] < 0 else
            f"Patient B is {abs(row['obesity_diff'])} level(s) of obesity (from underweight, normal weight, overweight, obese, or severely obese) lower than Patient A."
        )

    return "Choose which patient should receive the single available kidney: " + " ".join(parts) + " Who should receive the kidney?"

# Create a new DataFrame with the prompts and labels
df_diff["text"] = df_diff.apply(make_prompt, axis=1)
df_diff = df_diff.rename(columns={"chosen": "label"})
df = df_diff[["tid", "session_number", "created_at", "text", "label"]]
df.to_csv("llm/llm_readable_data.csv", index=False)

# Split the data into training and testing sets
train_rows = []
test_rows = []
for tid, group in df.groupby("tid"):
    train_part, test_part = train_test_split(
        group,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    train_rows.append(train_part)
    test_rows.append(test_part)

train_df = pd.concat(train_rows).reset_index(drop=True)
test_df = pd.concat(test_rows).reset_index(drop=True)

train_df.to_csv("llm/llm_train_data.csv", index=False)
test_df.to_csv("llm/llm_test_data.csv", index=False)

## AGGREGATE ANALYSIS (GPT2)
## Trained on aggregated 50% of each participant's data, tested on the other 50%

df = pd.read_csv("llm/llm_train_data.csv")
dataset = Dataset.from_pandas(df)

split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset   = split["test"]

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

def tokenize(batch):
    toks = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    toks["labels"] = batch["label"]
    return toks

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text", "tid", "session_number", "created_at"])
val_dataset   = val_dataset.map(tokenize, batched=True, remove_columns=["text", "tid", "session_number", "created_at"])

for ds in (train_dataset, val_dataset):
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="./gpt2_output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    seed=42,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()

test_dataset = pd.read_csv("llm/llm_test_data.csv")
original_dataset = test_dataset.copy()
test_dataset = Dataset.from_pandas(test_dataset)

tokenized_ds = test_dataset.map(tokenize, batched=True)
tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

outputs = trainer.predict(tokenized_ds)

y_true = outputs.label_ids
y_pred = np.argmax(outputs.predictions, axis=1)

original_dataset["GPT2_Pred"] = y_pred
original_dataset.to_csv("./data/ext_data/GPT2_aggregate_predictions.csv", index=False)

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall    = recall_score(y_true, y_pred, average='binary')
f1        = f1_score(y_true, y_pred, average='binary')

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save the accuracy of each predictions for each participant
df = pd.read_csv("exports/GPT2_aggregate_predictions.csv")
df["Correct"] = df["label"] == df["GPT2_Pred"]
accuracy_dataset = (
    df
    .groupby("tid")["Correct"]
    .mean()
    .reset_index()
    .rename(columns={"Correct": "accuracy"})
)
accuracy_dataset = accuracy_dataset.sort_values(by="accuracy", ascending=False)
accuracy_dataset.to_csv("./data/ext_data/GPT2_accuracy_aggregate.csv", index=False)
