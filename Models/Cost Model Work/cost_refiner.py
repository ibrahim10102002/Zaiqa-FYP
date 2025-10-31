import os
import json
import re
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
COST_FILE = r"Ingredients-Costs\Combined_Cost_List.json"
RECIPES_DIR = r"Scrapping\cleaned_jsons"
OUTPUT_DIR = r"Scrapping\cost_estimates"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# UNIT CONVERSIONS
# ==============================
UNIT_CONVERSIONS = {
    "kg": 1000, "g": 1, "gram": 1, "grams": 1,
    "tsp": 5, "tbsp": 15, "cup": 240,
    "piece": 100, "pieces": 100, "clove": 5,
    "pinch": 0.5, "bunch": 50, "dozen": 12,
    "ml": 1, "ltr": 1000, "liter": 1000
}

DEFAULT_GRAMS = 100  # when unit missing
AS_REQUIRED_GRAMS = 5  # for "as required" cases

# ==============================
# HELPERS
# ==============================

def extract_quantity_and_unit(text):
    """Extracts approximate quantity (float) and unit from ingredient string."""
    text = text.lower().replace("½", "0.5").replace("¼", "0.25").strip()

    if "as required" in text:
        return 1.0, "as required"

    match = re.search(r"(\d+(\.\d+)?)(\s*(to|-)\s*\d+(\.\d+)?)?\s*([a-zA-Z]+)?", text)
    if not match:
        return 1.0, "unit"

    qty = float(match.group(1))
    if match.group(3):
        upper = re.search(r"\d+(\.\d+)?$", match.group(3))
        if upper:
            qty = (qty + float(upper.group(0))) / 2
    unit = (match.group(6) or "unit").strip().lower()
    return qty, unit


def normalize_to_grams(qty, unit):
    """Converts qty+unit into grams/ml equivalents."""
    unit = unit.lower().strip()
    if "as required" in unit:
        return AS_REQUIRED_GRAMS
    if unit in UNIT_CONVERSIONS:
        return qty * UNIT_CONVERSIONS[unit]
    # fallback
    return qty * DEFAULT_GRAMS


def compute_cost(price_per_kg, qty_in_grams):
    """Convert per-kg/litre price to proportional cost."""
    return (price_per_kg / 1000) * qty_in_grams


def extract_ingredient_list(ingredient_text):
    """Split long text into separate ingredient chunks."""
    text = ingredient_text.split("Ingredients for")[-1]
    text = re.sub(r"recipe.*?:", "", text, flags=re.I)
    parts = re.split(r"(?<=\d)|(?= [A-Z])|,|;", text)
    cleaned = [p.strip().lower() for p in parts if len(p.strip()) > 2]
    return cleaned


def parse_price_range(price_str):
    """Extract numeric price range and compute average."""
    price_str = price_str.lower().replace("rs", "").replace(",", "")
    nums = re.findall(r"\d+\.?\d*", price_str)
    if not nums:
        return 0.0, 0.0, 0.0
    nums = [float(n) for n in nums]
    if len(nums) == 1:
        return nums[0], nums[0], nums[0]
    low, high = min(nums), max(nums)
    avg = (low + high) / 2
    return low, avg, high


# ==============================
# MAIN
# ==============================

def main():
    print("->Loading cost data...")
    with open(COST_FILE, "r", encoding="utf-8") as f:
        cost_items = json.load(f)

    normalized_costs = []
    for item in cost_items:
        name = item.get("item", "").lower().strip()
        low, avg, high = parse_price_range(item.get("Price", "0"))
        normalized_costs.append({
            "name": name,
            "low_price": low,
            "avg_price": avg,
            "high_price": high,
            "unit": item.get("RateUnit", "per kg").lower()
        })

    print(f"->Loaded {len(normalized_costs)} cost items.")

    recipe_files = [f for f in os.listdir(RECIPES_DIR) if f.endswith(".json")]
    print(f"-> Found {len(recipe_files)} recipe files.\n")

    for file in recipe_files:
        path = os.path.join(RECIPES_DIR, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                recipes = json.load(f)
        except Exception as e:
            print(f"⚠ Error reading {file}: {e}")
            continue

        if not isinstance(recipes, list):
            recipes = [recipes]

        for recipe in recipes:
            ingredients_text = recipe.get("ingredients", "")
            if not ingredients_text:
                continue

            ingredients = extract_ingredient_list(ingredients_text)
            matched = {}
            total_min = total_max = total_avg = 0.0

            for ing in ingredients:
                for cost_item in normalized_costs:
                    if cost_item["name"] in ing:
                        qty, unit = extract_quantity_and_unit(ing)
                        grams = normalize_to_grams(qty, unit)

                        cost_min = compute_cost(cost_item["low_price"], grams)
                        cost_avg = compute_cost(cost_item["avg_price"], grams)
                        cost_max = compute_cost(cost_item["high_price"], grams)

                        matched[cost_item["name"]] = {
                            "ingredient_text": ing,
                            "qty": qty,
                            "unit": unit,
                            "grams": grams,
                            "price_low": cost_item["low_price"],
                            "price_avg": cost_item["avg_price"],
                            "price_high": cost_item["high_price"],
                            "cost_range": [round(cost_min, 2), round(cost_max, 2)],
                            "cost_contrib_avg": round(cost_avg, 2)
                        }

                        total_min += cost_min
                        total_max += cost_max
                        total_avg += cost_avg
                        break

            recipe["matched_ingredients_v4"] = matched
            recipe["estimated_cost_v4"] = {
                "min": round(total_min, 2),
                "avg": round(total_avg, 2),
                "max": round(total_max, 2)
            }

        out_path = os.path.join(OUTPUT_DIR, file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(recipes, f, indent=2, ensure_ascii=False)

        print(f"-> Processed → {file} | saved at {out_path}")

    print("\n-> Enhanced cost estimation completed successfully!")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()
