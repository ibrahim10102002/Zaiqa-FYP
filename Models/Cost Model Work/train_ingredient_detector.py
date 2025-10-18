import os, json, torch, numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

DATA_DIR = "Scrapping/cost_estimates"
COST_FILE = "Ingredients-Costs/Combined_Cost_List.json"
MODEL_DIR = "models/ingredient_detector"

class RecipeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item["labels"] = torch.FloatTensor(self.labels[idx])
        return item

def load_recipes():
    data = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".json"):
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                data.extend(json.load(f))
    return data

def main():
    print("📦 Loading data...")
    recipes = load_recipes()
    with open(COST_FILE, "r", encoding="utf-8") as f:
        cost_items = json.load(f)

    ingredients_vocab = sorted(set(i["item"].lower().strip() for i in cost_items))
    print(f"✅ {len(ingredients_vocab)} ingredients loaded.")

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    texts, labels = [], []

    for r in recipes:
        text = (r.get("ingredients", "") + " " + r.get("instructions", "")).lower()
        matched = [i for i in ingredients_vocab if i in text]
        labels.append(matched)
        texts.append(text)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    dataset = RecipeDataset(texts, y, tokenizer)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir="./models/tmp",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=20,
        save_strategy="epoch",
        disable_tqdm=False,
        do_eval=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    print("🚀 Training started (CPU mode)...")
    trainer.train()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, "ingredient_labels.json"), "w", encoding="utf-8") as f:
        json.dump(mlb.classes_.tolist(), f, ensure_ascii=False, indent=2)

    print(f"✅ Model saved at {MODEL_DIR}")

if __name__ == "__main__":
    main()
