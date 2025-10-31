import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

VALIDATED_DATA_PATH = "Scrapping/validated_dataset_high_confidence.json"  # produced from prepare_dataset.py
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


print("Loading validated dataset...")
with open(VALIDATED_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ensure it's list of recipes
if isinstance(data, dict):
    data = list(data.values())

# Extract usable rows
records = []
for r in data:
    name = r.get("name") or r.get("recipe_name") or "Unknown Recipe"
    ingredients_text = r.get("ingredients") or r.get("ingredients_text") or ""
    cost_info = r.get("estimated_cost_v4") or {}
    estimated_cost = cost_info.get("avg") or cost_info.get("average") or 0.0
    matched = list((r.get("matched_ingredients_v4") or {}).keys())

    if ingredients_text and estimated_cost > 0:
        records.append({
            "name": name,
            "ingredients_text": ingredients_text,
            "matched_ingredients": ", ".join(matched),
            "estimated_cost": estimated_cost
        })

df = pd.DataFrame(records)
print(f"=>Loaded {len(df)} high-confidence recipes for training.")

if len(df) < 10:
    raise ValueError("!Not enough recipes to train a model. Check your dataset filtering!")

#feature preparation
print("=>Building text features...")
df["text"] = df["name"].astype(str) + " " + df["ingredients_text"].astype(str)
y = df["estimated_cost"].astype(float).values

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df["text"]).toarray()

# Scale features (optional)
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42 #80% train, 20% test
)

print("===>Training RandomForestRegressor on high-confidence data<===")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)


preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"MAE: {mae:.2f}") #es se avg abs diff bw predicted & actual cost ka pata chala
print(f"R²: {r2:.3f}") #es se model kitna acha fit hua data pe ka pata chala


joblib.dump(model, os.path.join(MODEL_DIR, "cost_estimator_v2.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer_v2.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_v2.pkl"))

print("\nSaved:")
print(f" - models/cost_estimator_v2.pkl")
print(f" - models/vectorizer_v2.pkl")
print(f" - models/scaler_v2.pkl")
print("\nTraining complete on validated dataset.")
