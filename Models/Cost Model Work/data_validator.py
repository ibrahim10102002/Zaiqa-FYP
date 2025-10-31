import os
import json
import csv
from statistics import mean

# ==============================
# CONFIGURATION
# ==============================
INPUT_DIR = r"Scrapping\cost_estimates"
OUTPUT_JSON = r"Scrapping\validated_dataset_high_confidence.json"
REPORT_CSV = r"Scrapping\validation_report.csv"

MIN_CONF_SCORE = 60          # minimum match score to keep ingredient
MAX_SINGLE_INGR_RATIO = 0.7  # ingredient cost % threshold
MIN_COST = 10.0              # min recipe cost in Rs
MAX_COST = 10000.0           # max recipe cost in Rs

# ==============================
# VALIDATION UTILITIES
# ==============================
def calc_confidence(recipe):
    """Compute heuristic confidence based on ingredient count and cost spread."""
    matched = recipe.get("matched_ingredients_v4", {})
    est = recipe.get("estimated_cost_v4", {})
    n_ing = len(matched)
    if n_ing == 0:
        return 0
    avg = est.get("avg", 0)
    min_c, max_c = est.get("min", 0), est.get("max", 0)
    if avg <= 0:
        return 0
    spread = (max_c - min_c) / avg if avg else 0
    base = 60 + min(n_ing * 4, 20)  # + up to 20 points for ingredient richness
    penalty = min(spread * 20, 20)  # penalize wide cost range
    score = max(0, min(100, base - penalty))
    return round(score, 1)


def validate_and_clean(recipe):
    """Check recipe validity and clean invalid ingredients."""
    matched = recipe.get("matched_ingredients_v4", {})
    est = recipe.get("estimated_cost_v4", {})

    if not matched or not est:
        return None, "missing_data"

    avg_cost = est.get("avg", 0.0)
    if avg_cost < MIN_COST or avg_cost > MAX_COST:
        return None, "outlier_cost"

    # Drop weak matches
    cleaned = {
        k: v for k, v in matched.items()
        if v.get("_match_score", 0) >= MIN_CONF_SCORE
    }
    if not cleaned:
        return None, "low_match_scores"

    # Detect dominant ingredient
    total_avg = sum(v.get("cost_contrib_avg", 0.0) for v in cleaned.values())
    if total_avg == 0:
        return None, "zero_total_cost"

    dominant = max(cleaned.items(), key=lambda x: x[1].get("cost_contrib_avg", 0))
    if dominant[1]["cost_contrib_avg"] / total_avg > MAX_SINGLE_INGR_RATIO:
        # Remove dominant ingredient
        del cleaned[dominant[0]]
        total_avg = sum(v.get("cost_contrib_avg", 0.0) for v in cleaned.values())
        if total_avg == 0:
            return None, "dominant_removed_empty"

    # Recalculate total cost
    min_total = sum(v.get("cost_range", [0, 0])[0] for v in cleaned.values())
    avg_total = sum(v.get("cost_contrib_avg", 0.0) for v in cleaned.values())
    max_total = sum(v.get("cost_range", [0, 0])[1] for v in cleaned.values())

    recipe["matched_ingredients_v4"] = cleaned
    recipe["estimated_cost_v4"] = {
        "min": round(min_total, 2),
        "avg": round(avg_total, 2),
        "max": round(max_total, 2)
    }

    conf = calc_confidence(recipe)
    recipe["confidence_score"] = conf

    if conf < 50:
        return None, "low_confidence"

    return recipe, "valid"


# ==============================
# MAIN VALIDATION
# ==============================
def main():
    print("Starting dataset validation...")
    valid_recipes, report_rows = [], []

    recipe_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    print(f"Found {len(recipe_files)} files in {INPUT_DIR}")

    for file in recipe_files:
        path = os.path.join(INPUT_DIR, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                recipes = json.load(f)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        if not isinstance(recipes, list):
            recipes = [recipes]

        for rec in recipes:
            name = rec.get("name", "Unknown Recipe")
            cleaned, status = validate_and_clean(rec)
            report_rows.append({"file": file, "recipe_name": name, "status": status})
            if cleaned:
                valid_recipes.append(cleaned)

    # Write validated dataset
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(valid_recipes, f, indent=2, ensure_ascii=False)

    # Write validation report
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "recipe_name", "status"])
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Validation complete: {len(valid_recipes)} valid recipes saved to {OUTPUT_JSON}")
    print(f"Report saved to {REPORT_CSV}")


if __name__ == "__main__":
    main()
