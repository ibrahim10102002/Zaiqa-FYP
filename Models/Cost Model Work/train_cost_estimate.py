import os, json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import torch

ING_MODEL_DIR = "models/ingredient_detector"
COST_ESTIMATE_DIR = "Scrapping/cost_estimates"
SAVE_DIR = "models/cost_estimator_v2"

def load_ingredient_model():
    tokenizer = BertTokenizer.from_pretrained(ING_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(ING_MODEL_DIR)
    with open(os.path.join(ING_MODEL_DIR, "ingredient_labels.json"), "r", encoding="utf-8") as f:
        labels = json.load(f)
    return model, tokenizer, labels

def encode_ingredients(model, tokenizer, labels, text):
    model.eval()
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().numpy()
    return probs

def load_training_data(model, tokenizer, labels):
    X, y = [], []
    for file in os.listdir(COST_ESTIMATE_DIR):
        if file.endswith(".json"):
            with open(os.path.join(COST_ESTIMATE_DIR, file), "r", encoding="utf-8") as f:
                recipes = json.load(f)
            for r in recipes:
                text = (r.get("ingredients", "") + " " + r.get("instructions", "")).lower()
                features = encode_ingredients(model, tokenizer, labels, text)
                meta = np.array([
                    float(r.get("prepTime", 0)),
                    float(r.get("cookTime", 0)),
                    float(r.get("totalTime", 0)),
                    float(r.get("calories", 0))
                ])
                X.append(np.concatenate([features, meta]))
                y.append(r.get("estimated_cost_v4", {}).get("avg", 0))
    return np.array(X), np.array(y)

def main():
    print("📦 Loading ingredient model...")
    model, tokenizer, labels = load_ingredient_model()

    print("📊 Preparing data...")
    X, y = load_training_data(model, tokenizer, labels)
    print(f"✅ Loaded {len(y)} samples with {X.shape[1]} features.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    print("🚀 Training cost estimator...")
    reg.fit(X_scaled, y)

    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump({"model": reg, "scaler": scaler, "labels": labels}, os.path.join(SAVE_DIR, "xgb_cost_model.pkl"))
    print(f"✅ Model saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()
