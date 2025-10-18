# estimate_recipe_costs_v2.py
import os
import json
import re
from itertools import islice
from fuzzywuzzy import fuzz, process
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPES_FOLDER = os.path.join(BASE_DIR, "Scrapping", "cleaned_jsons")
COST_FILE = os.path.join(BASE_DIR, "Ingredients-Costs", "Combined_Cost_List.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "recipes_with_costs.json")

# Config: tune these if needed
FUZZY_WEIGHT = 0.6
OVERLAP_WEIGHT = 0.4
MATCH_THRESHOLD = 82   # combined score threshold to accept a match
MAX_NGRAM = 3          # consider 1..3 word ngrams
DEBUG_RECIPES = 5      # print debug info for first N recipes

STOPWORDS = set([
    "for","to","as","and","or","with","of","on","in","the","a","an","per","each",
    "cup","cups","tbsp","tsp","tablespoon","teaspoon","piece","pieces","kg","g","gram",
    "grams","kg.","kg", "slice", "slices", "bunch", "chopped", "chopped", "fresh",
    "diced", "large", "small", "medium", "required", "taste", "optional"
])

def normalize(text):
    if not text:
        return ""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)           # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(text):
    t = normalize(text)
    tokens = [tok for tok in t.split() if tok not in STOPWORDS]
    return tokens

def build_cost_index(cost_file):
    with open(cost_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cost_entries = []
    name_to_price = {}
    for entry in data:
        # adapt to your Combined_Cost_List fields; support both "item" and "Product"
        raw_name = entry.get("item") or entry.get("Product") or entry.get("ProductName") or ""
        raw_price = entry.get("Price") or entry.get("RateValue") or entry.get("PriceRs") or ""
        name = normalize(raw_name)
        # average numeric value from price strings like "RS 595 - 850" or "595"
        nums = re.findall(r"\d+(?:\.\d+)?", str(raw_price))
        avg_price = float(nums[0]) if len(nums) == 1 else (sum(map(float, nums))/len(nums) if nums else 0.0)
        tokens = set(tokenize(name))
        cost_entries.append({
            "orig_name": raw_name,
            "name": name,
            "tokens": tokens,
            "price": avg_price
        })
        name_to_price[name] = avg_price
    return cost_entries, name_to_price

def ngrams(tokens, n):
    if not tokens:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def candidate_cost_items(cost_entries, ng_tokens):
    """Return cost entries that share at least one token with ng_tokens"""
    ng_set = set(ng_tokens)
    candidates = []
    for e in cost_entries:
        if e["tokens"] & ng_set:
            candidates.append(e)
    return candidates

def token_overlap_score(ng_tokens, cost_tokens):
    if not ng_tokens or not cost_tokens:
        return 0.0
    inter = len(set(ng_tokens) & set(cost_tokens))
    union = len(set(ng_tokens) | set(cost_tokens))
    return inter / union if union else 0.0

def match_ngram_to_cost(ngram, cost_entries, top_k=5):
    """
    Try to match ngram to cost entries.
    Returns best match dict with combined score and extra details or None.
    """
    ng_norm = normalize(ngram)
    ng_tokens = [t for t in ng_norm.split() if t not in STOPWORDS]
    if not ng_tokens:
        return None

    # Restrict search space
    candidates = candidate_cost_items(cost_entries, ng_tokens)
    if not candidates:
        # small fallback: search entire list but will be lower-confidence
        candidates = cost_entries

    best = None
    for e in candidates:
        # compute fuzzy ratio on names (use partial_ratio for short ngrams)
        f = fuzz.token_sort_ratio(ng_norm, e["name"]) if len(ng_norm.split())>1 else fuzz.ratio(ng_norm, e["name"])
        overlap = token_overlap_score(ng_tokens, e["tokens"])
        combined = FUZZY_WEIGHT * f + OVERLAP_WEIGHT * (overlap*100)  # overlap scaled to 0..100
        if not best or combined > best["combined"]:
            best = {
                "candidate_name": e["name"],
                "orig_name": e["orig_name"],
                "price": e["price"],
                "fuzzy": f,
                "overlap": overlap,
                "combined": combined
            }
    if best and best["combined"] >= MATCH_THRESHOLD:
        return best
    return None

def extract_qty_unit(ing_text):
    # Try to extract a numeric qty and unit (basic)
    m = re.search(r"(\d+(\.\d+)?)\s*(kg|g|grams|gram|piece|pieces|pcs|cup|tbsp|tsp|litre|ltr)?", ing_text.lower())
    if m:
        q = float(m.group(1))
        u = m.group(3) or "unit"
        return q, u
    return None, None

def match_ingredients_to_costs(ingredient_text, cost_entries):
    """Main worker: returns matched entries (name->price) and total cost."""
    tokens = tokenize(ingredient_text)
    # build all ngrams (prefer longer ngrams)
    matched = {}
    used_spans = set()  # indices consumed (to avoid re-matching sub-ngrams)
    for n in range(MAX_NGRAM, 0, -1):
        ngs = ngrams(tokens, n)
        for idx, ng in enumerate(ngs):
            # idx corresponds to token position idx..idx+n-1
            span = (idx, idx+n-1)
            # skip if overlapping with already matched span
            if any(not (span[1] < s[0] or span[0] > s[1]) for s in used_spans):
                continue
            best = match_ngram_to_cost(ng, cost_entries)
            if best:
                # accept, mark span used
                used_spans.add(span)
                # determine qty scaling (very basic)
                q, u = extract_qty_unit(ng)  # try quantity in the ngram itself
                # fallback look ahead/backwards in token string for numeric near the ngram
                if q is None:
                    # look within the surrounding ingredient_text for numbers near ngram
                    qmatch = re.search(r"(\d+(\.\d+)?)\s*(?:kg|g|pcs|pieces|piece|cup|tbsp|tsp|ltr|litre)?\s*(?:%s)" % re.escape(ng), ingredient_text, flags=re.IGNORECASE)
                    if qmatch:
                        q = float(qmatch.group(1))
                        u = qmatch.group(3) if len(qmatch.groups())>=3 else "unit"
                # default qty 1
                if q is None:
                    q = 1.0

                # scale price: if unit indicates kg/gram, we assume cost entries are per kg (if price>0)
                price = best["price"] or 0.0
                unit = u or "unit"
                # convert if unit is grams
                if unit and unit.startswith("g"):
                    scaled_price = price * (q/1000.0)
                elif unit and unit in ("kg", "kgs"):
                    scaled_price = price * q
                else:
                    # for piece/cup/unit: treat as one unit => use price as-is (fallback)
                    scaled_price = price * q

                matched[best["candidate_name"]] = {
                    "orig_name": best["orig_name"],
                    "price": price,
                    "matched_ngram": ng,
                    "fuzzy": best["fuzzy"],
                    "overlap": best["overlap"],
                    "combined": best["combined"],
                    "qty": q,
                    "unit": unit,
                    "cost_contrib": round(scaled_price, 2)
                }
    total = round(sum(v["cost_contrib"] for v in matched.values()), 2)
    return total, matched

def parse_ingredient_field(field):
    """If ingredients are a string, produce a normalized text to analyze."""
    if not field:
        return ""
    if isinstance(field, list):
        return " ".join(field)
    return str(field)

def main():
    cost_entries, _ = build_cost_index(COST_FILE)
    print(f"Loaded {len(cost_entries)} cost entries.")

    files = [f for f in os.listdir(RECIPES_FOLDER) if f.endswith(".json")]
    all_out = []
    debug_count = 0

    for file in tqdm(files):
        path = os.path.join(RECIPES_FOLDER, file)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print("Failed to load", file, e)
            continue

        recipes = data if isinstance(data, list) else [data]
        for recipe in recipes:
            ing_raw = parse_ingredient_field(recipe.get("ingredients", ""))
            if not ing_raw.strip():
                recipe["estimated_cost_v2"] = 0.0
                recipe["matched_ingredients_v2"] = {}
                all_out.append(recipe)
                continue

            total, matched = match_ingredients_to_costs(ing_raw, cost_entries)
            recipe["estimated_cost_v2"] = total
            # convert matched dict for JSON dumping
            recipe["matched_ingredients_v2"] = matched

            all_out.append(recipe)

            # debug
            if debug_count < DEBUG_RECIPES:
                print("\n==== DEBUG RECIPE ====")
                print("Name:", recipe.get("name") or recipe.get("title") or file)
                print("Ingredients raw:", ing_raw)
                print("Estimated cost:", total)
                print("Matches:")
                for k,v in matched.items():
                    print(f"  - {k} (orig: {v['orig_name']}) -> contrib {v['cost_contrib']} Rs (fuzzy {v['fuzzy']}, overlap {v['overlap']:.2f})")
                debug_count += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfh:
        json.dump(all_out, outfh, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_out)} recipes with estimated costs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
