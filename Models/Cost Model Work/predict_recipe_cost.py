import os
import json
import re
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModel
import torch

# ================================
# CONFIG
# ================================
BASE_DIR = Path(__file__).resolve().parent
COST_FILE = BASE_DIR / "Ingredients-Costs" / "Combined_Cost_List.json"
RECIPE_DIR = BASE_DIR / "Scrapping" / "cleaned_jsons"
MODEL_PATH = BASE_DIR / "models" / "cost_estimator_bert_v3" / "cost_estimator_bert.pkl"
OUTPUT_DIR = BASE_DIR / "AI_cost_predictions"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ================================
# LOAD MODELS
# ================================
def load_models():
    print("📦 Loading multilingual BERT model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")

    print("📦 Loading cost estimator...")
    cost_estimator = joblib.load(MODEL_PATH)
    if isinstance(cost_estimator, dict):
        cost_estimator = cost_estimator.get("model", cost_estimator)
        print("✅ Extracted model from dict.")
    print(f"✅ Model loaded successfully. Expecting {cost_estimator.n_features_in_} features.")
    return tokenizer, bert_model, cost_estimator

# ================================
# LOAD COST MAP
# ================================
def load_cost_map():
    with open(COST_FILE, "r", encoding="utf-8") as f:
        items = json.load(f)

    cost_map = {}
    for entry in items:
        try:
            name = re.sub(r"[\(\)\/]", " ", str(entry["item"])).strip().lower()
            name = re.sub(r"[^a-zA-Z\u0600-\u06FF\s]", " ", name)  # Keep Urdu too
            name = re.sub(r"\s+", " ", name)

            nums = re.findall(r"\d+", str(entry["Price"]))
            if len(nums) >= 2:
                low, high = float(nums[0]), float(nums[-1])
            elif len(nums) == 1:
                low, high = float(nums[0]), float(nums[0])
            else:
                continue

            avg = (low + high) / 2
            cost_map[name] = {
                "price_low": low,
                "price_avg": avg,
                "price_high": high,
                "RateUnit": entry.get("RateUnit", ""),
                "quantity": entry.get("quantity", "")
            }
        except Exception as e:
            print(f"⚠ Skipped item due to parsing error: {e}")
    print(f"💰 Loaded {len(cost_map)} cost items.")
    return cost_map

# ================================
# HELPERS
# ================================
def clean_text(txt):
    return re.sub(r"[^a-zA-Z\u0600-\u06FF\s]", " ", str(txt)).lower().strip()

def estimate_qty(ing_text):
    ing_text = str(ing_text).lower()
    qty, unit = 1.0, "unit"
    num_match = re.search(r"(\d+(\.\d+)?)", ing_text)
    if num_match:
        qty = float(num_match.group(1))

    grams = 100.0 * qty
    if "kg" in ing_text:
        grams = qty * 1000
        unit = "kg"
    elif "g" in ing_text:
        grams = qty
        unit = "g"
    elif "tsp" in ing_text:
        grams = qty * 5
        unit = "tsp"
    elif "tbsp" in ing_text:
        grams = qty * 15
        unit = "tbsp"
    elif "cup" in ing_text:
        grams = qty * 240
        unit = "cup"
    elif "piece" in ing_text or "pieces" in ing_text:
        grams = qty * 100
        unit = "pieces"
    elif "whole" in ing_text:
        grams = 1000
        unit = "whole"
    elif "as required" in ing_text:
        grams = 100
        unit = "as required"
    return qty, unit, grams

def find_best_match(ingredient, cost_map):
    ingredient = str(ingredient)
    best_match, best_score = None, 0
    for item in cost_map.keys():
        try:
            score = fuzz.token_sort_ratio(str(ingredient), str(item))
            if score > best_score:
                best_match, best_score = item, score
        except Exception:
            continue
    return best_match, best_score

def heuristic_estimate(ingredients, cost_map):
    matched = {}
    total = 0.0
    tokens = re.split(r"[,\n]", ingredients)

    for raw_ing in tokens:
        clean_ing = clean_text(raw_ing)
        if not clean_ing or len(clean_ing) < 3:
            continue

        best_match, score = find_best_match(clean_ing, cost_map)
        if not best_match or score < 75:
            continue

        qty, unit, grams = estimate_qty(raw_ing)
        rate = cost_map[best_match]
        price = rate["price_avg"]

        # Special logic for meats
        if any(word in clean_ing for word in ["chicken", "beef", "mutton", "fish"]):
            if "whole" in raw_ing:
                grams = 1000
            elif "breast" in raw_ing or "piece" in raw_ing:
                grams = 500 * qty
            else:
                grams = 1000 * qty

        contrib = round(price * (grams / 1000), 2)
        matched[best_match] = {
            **rate,
            "qty": qty,
            "unit": unit,
            "grams": grams,
            "cost_contrib": contrib,
        }
        total += contrib

    return matched, round(total, 2)

def ai_corrected_cost(bert_model, tokenizer, cost_estimator, recipe_text, base_cost):
    inputs = tokenizer(recipe_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        embeddings = bert_model(**inputs).last_hidden_state.mean(dim=1).numpy()
    ai_cost = float(cost_estimator.predict(embeddings)[0])
    final_cost = 0.6 * base_cost + 0.4 * ai_cost
    return round(final_cost, 2), ai_cost

# ================================
# MAIN PIPELINE
# ================================
def main():
    tokenizer, bert_model, cost_estimator = load_models()
    cost_map = load_cost_map()
    recipe_files = [f for f in RECIPE_DIR.glob("*.json")][:6]
    results = []

    for recipe_file in tqdm(recipe_files, desc="Predicting"):
        try:
            data = json.load(open(recipe_file, "r", encoding="utf-8"))
            recipe = data[0] if isinstance(data, list) else data

            name = recipe.get("name", "unknown")
            ingredients = recipe.get("ingredients", "")
            category = recipe.get("category", "unknown")

            matched_ings, heuristic_cost = heuristic_estimate(ingredients, cost_map)
            combined_text = (recipe.get("name", "") + " " + ingredients + " " + recipe.get("instructions", "")).strip()
            final_cost, ai_cost = ai_corrected_cost(bert_model, tokenizer, cost_estimator, combined_text, heuristic_cost)

            results.append({
                "name": name,
                "category": category,
                "ingredients": ingredients,
                "estimated_cost_ai": ai_cost,
                "final_estimated_cost": final_cost,
                "ingredient_based_estimate": heuristic_cost,
                "matched_ingredients": matched_ings
            })
        except Exception as e:
            print(f"⚠ Error in {recipe_file.name}: {e}")

    out_path = OUTPUT_DIR / "BBQ_test_predictions.json"
    json.dump(results, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(results)} predictions to {out_path}")

if __name__ == "__main__":
    main()
