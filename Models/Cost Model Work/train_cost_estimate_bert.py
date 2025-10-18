import json
import os
import torch
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Paths
COST_EST_DIR = Path("Scrapping/cost_estimates")
MODEL_DIR = Path("models/cost_estimator_bert_v3")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load multilingual BERT
print("📦 Loading multilingual BERT model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
bert_model.eval()

def embed_text(text):
    """Generate 768-dim embedding using mean pooling."""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def load_training_data():
    X, y = [], []
    print("📊 Loading training data from cost_estimates...")
    for file in tqdm(list(COST_EST_DIR.glob("*.json"))):
        try:
            recipes = json.load(open(file, "r", encoding="utf-8"))
            for r in recipes:
                if "ingredients" in r and "estimated_cost_v4" in r:
                    ing = r.get("ingredients", "")
                    instr = r.get("instructions", "")
                    cat = r.get("category", "")
                    text = f"{cat}. {ing}. {instr}"
                    cost = r["estimated_cost_v4"].get("avg")
                    if cost and isinstance(cost, (int, float)):
                        X.append(text)
                        y.append(float(cost))
        except Exception as e:
            print(f"⚠️ Skipping {file.name}: {e}")
    return X, y

print("📂 Gathering dataset...")
texts, costs = load_training_data()
print(f"✅ Loaded {len(texts)} recipes for training.")

# Generate embeddings
print("🔢 Generating embeddings...")
embeddings = np.vstack([embed_text(t) for t in tqdm(texts)])

# Split data
X_train, X_val, y_train, y_val = train_test_split(embeddings, costs, test_size=0.15, random_state=42)

# Train regressor
print("🚀 Training XGBoost Regressor on BERT embeddings...")
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror"
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"📊 Validation MAE: {mae:.2f}, R²: {r2:.3f}")

# Save
joblib.dump(model, MODEL_DIR / "cost_estimator_bert.pkl")
print(f"✅ Saved new BERT-based model to {MODEL_DIR}/cost_estimator_bert.pkl")
