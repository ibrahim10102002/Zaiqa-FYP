import os
import json
import pandas as pd


COST_ESTIMATES_DIR = r"Scrapping\cost_estimates"
OUTPUT_DATASET = r"training\recipe_cost_dataset.csv"
os.makedirs(os.path.dirname(OUTPUT_DATASET), exist_ok=True)


def main():
    all_recipes = []
    files = [f for f in os.listdir(COST_ESTIMATES_DIR) if f.endswith(".json")]

    print(f"Found {len(files)} cost estimate files.\n")

    for file in files:
        path = os.path.join(COST_ESTIMATES_DIR, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        # Ensure it's a list of recipes
        if not isinstance(data, list):
            data = [data]

        for recipe in data:
            name = recipe.get("name") or recipe.get("title") or "Unknown Recipe"
            ingredients_text = recipe.get("ingredients", "")
            matched = recipe.get("matched_ingredients_v4", {})
            estimated = recipe.get("estimated_cost_v4", {})

            matched_names = list(matched.keys())
            est_min = estimated.get("min", 0.0)
            est_avg = estimated.get("avg", 0.0)
            est_max = estimated.get("max", 0.0)

            # If cost_refiner didn’t find anything, skip
            if not matched_names or est_avg == 0.0:
                continue

            all_recipes.append({
                "recipe_name": name,
                "ingredients_text": ingredients_text,
                "matched_ingredients": ", ".join(matched_names),
                "estimated_cost_min": est_min,
                "estimated_cost_avg": est_avg,
                "estimated_cost_max": est_max,
                "num_matched_ingredients": len(matched_names)
            })

    if not all_recipes:
        print("No valid recipes found with cost data.")
        return

    df = pd.DataFrame(all_recipes)
    df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8-sig")

    print(f"\nDataset prepared successfully!")
    print(f"Saved to: {OUTPUT_DATASET}")
    print(f"Total recipes included: {len(df)}")


if __name__ == "__main__":
    main()
