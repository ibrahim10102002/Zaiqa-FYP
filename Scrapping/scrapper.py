import os, csv, re
import json5
from bs4 import BeautifulSoup

SAVE_DIR = "saved_pages"
OUTPUT_CSV = "recipes.csv"

def clean_text(text):
    if isinstance(text, list):
        text = " ".join(map(str, text))
    return re.sub(r"\s+", " ", str(text).replace("Â", "").replace("\xa0", " ")).strip()

def preclean_json(raw: str) -> str:
    """
    Clean raw JSON-LD text before parsing.
    Fixes unescaped newlines, smart quotes, and stray characters.
    """
    # Replace Windows-style newlines inside quotes with \n
    raw = re.sub(r'(?<!\\)\n', '\\n', raw)

    # Normalize fancy quotes
    raw = raw.replace("“", "\"").replace("”", "\"").replace("’", "'")

    # Remove stray control characters
    raw = re.sub(r'[\x00-\x1F\x7F]', '', raw)

    return raw.strip()

def extract_recipe_from_json(data):
    """Recursively search for a Recipe object inside JSON-LD"""
    if isinstance(data, dict):
        if data.get("@type") == "Recipe":
            return data
        for v in data.values():
            found = extract_recipe_from_json(v)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = extract_recipe_from_json(item)
            if found:
                return found
    return None

def parse_json_ld(content):
    """Extract JSON-LD recipe data from HTML content"""
    soup = BeautifulSoup(content, "html.parser")
    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        raw_json = script.string
        if not raw_json:
            continue
        try:
            data = json5.loads(raw_json)
            recipe = extract_recipe_from_json(data)
            if recipe:
                return {
                    "name": clean_text(recipe.get("name", "")),
                    "ingredients": [clean_text(i) for i in recipe.get("recipeIngredient", [])],
                    "instructions": [clean_text(i) for i in recipe.get("recipeInstructions", [])],
                    "prepTime": recipe.get("prepTime", ""),
                    "cookTime": recipe.get("cookTime", ""),
                    "totalTime": recipe.get("totalTime", ""),
                    "recipeYield": recipe.get("recipeYield", ""),
                    "calories": clean_text(recipe.get("nutrition", {}).get("calories", "")),
                    "youtube_link": recipe.get("video", {}).get("contentUrl", ""),
                }
        except Exception as e:
            snippet = raw_json[:200].replace("\n", " ")  # safe preview
            print(f"⚠ Error parsing JSON in file: {snippet}... ({e})")
            continue
    return None


def parse_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_json_ld(content)

def main():
    recipes = []
    for file in os.listdir(SAVE_DIR):
        if file.startswith("recipe") and file.endswith(".txt"):
            filepath = os.path.join(SAVE_DIR, file)
            recipe = parse_file(filepath)
            if recipe:
                recipes.append(recipe)
            else:
                print(f"No recipe data found in {filepath}")

    if not recipes:
        print("No recipes parsed.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "name", "ingredients", "instructions", "prepTime", "cookTime",
            "totalTime", "recipeYield", "calories", "youtube_link"
        ])
        writer.writeheader()
        for r in recipes:
            r["ingredients"] = " | ".join(r["ingredients"])
            r["instructions"] = " | ".join(r["instructions"])
            writer.writerow(r)

    print(f"Saved {len(recipes)} recipes to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
