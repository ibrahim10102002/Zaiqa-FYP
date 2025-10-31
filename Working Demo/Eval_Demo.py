import psycopg2
import joblib
import numpy as np
from getpass import getpass
from sentence_transformers import util
from torch.nn.functional import normalize
from chatbot_spec_extractor import RecipeDiscoveryChatbot

# DATABASE CONFIGURATION
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "102044",
    "host": "localhost",
    "port": "5432"
}

print("====Loading AI model components====\n")

# ---- load cost estimator/vectorizer/scaler (joblib) ----
MODEL_BASE = "models/cost_estimator_v2"
model = joblib.load(f"{MODEL_BASE}/cost_estimator_v2.pkl")
vectorizer = joblib.load(f"{MODEL_BASE}/vectorizer_v2.pkl")
scaler = joblib.load(f"{MODEL_BASE}/scaler_v2.pkl")
print("=>Model, vectorizer, and scaler loaded.\n")

# ---- load ingredient_model safely (may contain torch tensors) ----
ING_MODEL_PATH = "models/Ingredient_model/ingredient_model.pkl"
print("=>Loading ingredient-based recommendation model...")

ingredient_model = None
try:
    # primary attempt using joblib (works for many sklearn / python objects)
    ingredient_model = joblib.load(ING_MODEL_PATH)
    print("=>Ingredient model loaded via joblib.\n")
except Exception as e_joblib:
    # If joblib loading fails (commonly due to CUDA tensors), try torch.load with map_location
    import torch
    try:
        print("!! joblib.load failed, attempting torch.load with map_location='cpu' ...")
        loaded = torch.load(ING_MODEL_PATH, map_location=torch.device("cpu"))

        # If the object is a SentenceTransformer instance, use it directly
        from sentence_transformers import SentenceTransformer
        if isinstance(loaded, SentenceTransformer):
            ingredient_model = loaded
            print("=>Ingredient model loaded via torch.load (SentenceTransformer instance).\n")
        else:
            # If it's a dict or other object, try to inspect it
            if isinstance(loaded, dict):
                # common-case: someone pickled a dict containing model under a key like 'model'
                if "model" in loaded and isinstance(loaded["model"], SentenceTransformer):
                    ingredient_model = loaded["model"]
                    print("=>Ingredient model extracted from dict (key 'model').\n")
                else:
                    # not a ready-to-use SentenceTransformer object
                    print("⚠ Loaded object is a dict but does not contain a SentenceTransformer instance.")
                    print("   You should re-save the ingredient model using SentenceTransformer.save() to a folder,")
                    print("   then load with: SentenceTransformer('models/Ingredient_model').")
                    raise e_joblib
            else:
                # unknown object type => can't safely use it
                print("⚠ torch.load returned an object of type:", type(loaded))
                print("   If this is a SentenceTransformer state dict, re-save your model using SentenceTransformer.save()")
                raise e_joblib
    except Exception as e_torch:
        # both attempts failed -> raise original joblib exception with helpful notes
        print("\nERROR: Failed to load ingredient_model via both joblib and torch with CPU mapping.")
        print("Reason (joblib):", repr(e_joblib))
        print("Reason (torch fallback):", repr(e_torch))
        print("\nRecommended fixes:")
        print("  1) If you trained a SentenceTransformer, re-save it (on the machine where it was trained) like this:")
        print("       model.save('models/Ingredient_model')  # saves a folder with tokenizer & pytorch_model.bin")
        print("     Then on this machine load it with:")
        print("       from sentence_transformers import SentenceTransformer")
        print("       ingredient_model = SentenceTransformer('models/Ingredient_model')")
        print("  2) If you only have a .pkl that contains torch tensors saved with CUDA, re-create it on CPU with:")
        print("       torch.save(model, 'ingredient_model_cpu.pkl', _use_new_zipfile_serialization=True)")
        print("     or load it on the GPU machine and re-save with map_location='cpu'.")
        raise

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def create_user(username, email, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id;",
        (username, email, password)
    )
    user_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return user_id

def authenticate_user(email, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash FROM users WHERE email=%s;", (email,))
    row = cur.fetchone()
    conn.close()
    if row and row[2] == password:
        return {"id": row[0], "username": row[1]}
    return None

def get_recipes_by_category(category):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, ingredients, category FROM recipes WHERE category ILIKE %s;", (f"%{category}%",))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_all_recipes():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, ingredients, category FROM recipes;")
    rows = cur.fetchall()
    conn.close()
    return rows

def get_recipe_details(recipe_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, category, ingredients, instructions, prep_time, cook_time,
               total_time, recipe_yield, calories, youtube_link
        FROM recipes WHERE id=%s;
    """, (recipe_id,))
    recipe = cur.fetchone()
    conn.close()
    return recipe

def save_recipe(user_id, recipe_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_saved_recipes (user_id, recipe_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING;
    """, (user_id, recipe_id))
    conn.commit()
    conn.close()

def unsave_recipe(user_id, recipe_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM user_saved_recipes WHERE user_id=%s AND recipe_id=%s;", (user_id, recipe_id))
    conn.commit()
    conn.close()

def get_saved_recipes(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT r.id, r.name, r.category
        FROM recipes r
        JOIN user_saved_recipes s ON r.id = s.recipe_id
        WHERE s.user_id = %s;
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def list_all_users(current_user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users WHERE id != %s;", (current_user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def follow_user(follower_id, following_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_follows (follower_id, following_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING;
    """, (follower_id, following_id))
    conn.commit()
    conn.close()

def unfollow_user(follower_id, following_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM user_follows WHERE follower_id=%s AND following_id=%s;", (follower_id, following_id))
    conn.commit()
    conn.close()

def get_followers(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT u.username
        FROM user_follows f
        JOIN users u ON f.follower_id = u.id
        WHERE f.following_id = %s;
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

def get_following(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT u.id, u.username
        FROM user_follows f
        JOIN users u ON f.following_id = u.id
        WHERE f.follower_id = %s;
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_following_saved_recipes(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT u.username, r.id, r.name, r.category
        FROM user_follows f
        JOIN users u ON f.following_id = u.id
        JOIN user_saved_recipes s ON u.id = s.user_id
        JOIN recipes r ON s.recipe_id = r.id
        WHERE f.follower_id = %s;
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def predict_cost(ingredients_text):
    features = vectorizer.transform([ingredients_text])
    scaled = scaler.transform(features.toarray())
    predicted_cost = model.predict(scaled)[0]
    return max(predicted_cost, 0.0)


def find_recipes_by_ingredients_manual():
    """Manual ingredient-based recipe finder."""
    print("\nIngredient-based Recipe Finder")
    user_input = input("Enter ingredients with quantities (e.g., '2 tomatoes, 1 chicken, 3 capsicum'): ").strip()

    ingredients_list = [i.strip() for i in user_input.split(",") if i.strip()]
    if not ingredients_list:
        print("Please enter at least one ingredient.")
        return

    combined_input = " ".join(ingredients_list)

    try:
        user_embedding = ingredient_model.encode(combined_input, convert_to_tensor=True)
        user_embedding = normalize(user_embedding.unsqueeze(0), p=2, dim=1)
    except Exception as e:
        print(f"Error encoding input: {e}")
        return

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, category, ingredients FROM recipes;")
    recipes = cur.fetchall()
    conn.close()

    if not recipes:
        print("No recipes found in database.")
        return

    recipe_texts = [r[3] for r in recipes]
    recipe_embeddings = ingredient_model.encode(recipe_texts, convert_to_tensor=True)
    recipe_embeddings = normalize(recipe_embeddings, p=2, dim=1)

    similarities = (user_embedding @ recipe_embeddings.T)[0]
    sorted_indices = similarities.argsort(descending=True)

    print("\nTop Matching Recipes:")
    shown = 0
    for idx in sorted_indices:
        score = float(similarities[idx])
        percent = score * 100
        if percent < 40:
            continue
        rid, name, cat, ing = recipes[int(idx)]
        shown += 1
        print(f"{shown}. {name} ({cat}) -> similarity: {percent:.1f}%")
        if shown >= 10:
            break

    if shown == 0:
        print("No close matches found.")
        return

    sub = input("\nEnter recipe number to view: ")
    if sub.isdigit():
        sub = int(sub)
        if 1 <= sub <= shown:
            rid = recipes[int(sorted_indices[sub - 1])][0]
            view_recipe(rid)
        else:
            print("Invalid selection.")


def find_recipes_by_chatbot_specs(user_id):
    """AI-powered recipe finder using chatbot specification extractor."""
    print("\n" + "="*60)
    print("ZAIQA AI Recipe Finder Chatbot")
    print("="*60)
    chatbot = RecipeDiscoveryChatbot()

    print("Welcome to Zaiqa Recipe Discovery Chatbot!")
    print("Tell me what you'd like to cook, and I'll extract all the details.")
    print("Type 'done' when finished, 'summary' to see extracted specs, or 'reset' to start over.\n")

    while True:
        try:
            user_input = input("You: ").strip().lower()

            if user_input == "done":
                print("\nFinalizing your recipe specifications...")
                chatbot.print_summary()
                break

            elif user_input == "summary":
                chatbot.print_summary()
                continue

            elif user_input == "reset":
                chatbot.reset()
                print("Chatbot reset. Let's start fresh!\n")
                continue

            elif user_input in {"quit", "exit"}:
                print("Thanks for using Zaiqa! Goodbye!")
                break

            elif not user_input:
                print("Please enter something or type 'help' for commands.\n")
                continue

            elif user_input == "help":
                print("\nAvailable commands:")
                print("  'done'    - Finalize and show extracted specifications")
                print("  'summary' - Show current extracted specifications")
                print("  'reset'   - Start a new conversation")
                print("  'quit'    - Exit the chatbot\n")
                continue

            response = chatbot.chat(user_input)
            print(f"Bot: {response}\n")

        except KeyboardInterrupt:
            print("\n\nThanks for using Zaiqa! Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}. Please try again.\n")

    specs = chatbot.get_specifications_for_models()

    print("\nFinding recipes based on your specifications...\n")
    
    ingredient_text = " ".join([ing["name"] for ing in specs["ingredient_finder"]["ingredients"]])
    max_budget = specs["cost_finder"]["max_budget"]
    
    try:
        user_embedding = ingredient_model.encode(ingredient_text, convert_to_tensor=True)
        user_embedding = normalize(user_embedding.unsqueeze(0), p=2, dim=1)
    except Exception as e:
        print(f"Error encoding ingredients: {e}")
        return

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, category, ingredients FROM recipes;")
    recipes = cur.fetchall()
    conn.close()

    if not recipes:
        print("No recipes found in database.")
        return

    recipe_texts = [r[3] for r in recipes]
    recipe_embeddings = ingredient_model.encode(recipe_texts, convert_to_tensor=True)
    recipe_embeddings = normalize(recipe_embeddings, p=2, dim=1)

    similarities = (user_embedding @ recipe_embeddings.T)[0]
    sorted_indices = similarities.argsort(descending=True)

    matching_recipes = []
    for idx in sorted_indices:
        score = float(similarities[idx])
        percent = score * 100
        
        if percent < 40:
            continue
            
        rid, name, cat, ing = recipes[int(idx)]
        cost = predict_cost(ing)
        
        if max_budget and cost > max_budget:
            continue
        
        allergies = specs["ingredient_finder"]["exclude_allergies"]
        if allergies:
            ing_lower = ing.lower()
            if any(allergy.lower() in ing_lower for allergy in allergies):
                continue
        
        matching_recipes.append((rid, name, cat, ing, cost, percent))
        
        if len(matching_recipes) >= 10:
            break

    if not matching_recipes:
        print("No recipes match your specifications. Try adjusting your requirements.")
        return

    print("Top Matching Recipes:")
    print("-" * 60)
    for i, (rid, name, cat, ing, cost, percent) in enumerate(matching_recipes, 1):
        print(f"{i}. {name} ({cat})")
        print(f"   Similarity: {percent:.1f}% | Estimated Cost: Rs.{cost:.2f}")
        print()

    while True:
        sub = input("Enter recipe number to view details, 's' to save a recipe, or 'b' to go back: ").strip().lower()
        
        if sub == 'b':
            break
        
        if sub == 's':
            save_num = input("Enter recipe number to save: ").strip()
            if save_num.isdigit():
                save_idx = int(save_num)
                if 1 <= save_idx <= len(matching_recipes):
                    rid = matching_recipes[save_idx - 1][0]
                    save_recipe(user_id, rid)
                    print(f"Recipe saved successfully!\n")
                else:
                    print("Invalid recipe number.\n")
            continue
        
        if sub.isdigit():
            sub_idx = int(sub)
            if 1 <= sub_idx <= len(matching_recipes):
                rid = matching_recipes[sub_idx - 1][0]
                view_recipe(rid)
                
                sav = input("\nSave this recipe? (y/n): ").lower()
                if sav == "y":
                    save_recipe(user_id, rid)
                    print("Recipe saved!\n")
            else:
                print("Invalid selection.\n")

def register_flow():
    print("\nRegister New User")
    username = input("Username: ")
    email = input("Email: ")
    password = getpass("Password: ")
    user_id = create_user(username, email, password)
    print(f"Registered successfully! (User ID: {user_id})")
    return {"id": user_id, "username": username}

def login_flow():
    print("\nLogin")
    email = input("Email: ")
    password = getpass("Password: ")
    user = authenticate_user(email, password)
    if user:
        print(f"Welcome back, {user['username']}!")
        return user
    else:
        print("Invalid credentials.")
        return None

def view_recipe(recipe_id):
    recipe = get_recipe_details(recipe_id)
    if not recipe:
        print("Recipe not found.")
        return

    name, category, ingredients, instructions, prep, cook, total, yield_, cal, yt = recipe[1:]
    cost = predict_cost(ingredients)

    print("\n" + "="*60)
    print("RECIPE DETAILS")
    print("="*60)
    print(f"Name: {name}")
    print(f"Category: {category}")
    print(f"Prep Time: {prep} min | Cook Time: {cook} min | Total: {total} min")
    print(f"Servings: {yield_} | Calories: {cal}")
    print(f"YouTube: {yt}")
    print(f"\nEstimated Cost: Rs.{cost:.2f}")
    print(f"\nIngredients:\n{ingredients}")
    print(f"\nInstructions:\n{instructions}")
    print("="*60 + "\n")


def main_menu(user):
    while True:
        print(f"\nWelcome, {user['username']}!")
        print("="*60)
        print("1. View Saved Recipes")
        print("2. Browse Recipes by Category")
        print("3. Find Recipes Under a Budget")
        print("4. View Feed (Saved Recipes from People You Follow)")
        print("5. Social (Follow / Unfollow Users)")
        print("6. Find Recipes by Ingredients (AI-Powered)")
        print("7. Find Recipes with AI Chatbot")
        print("8. Logout")
        print("9. Exit")
        print("="*60)
        choice = input("Choose: ").strip()

        if choice == "1":
            saved = get_saved_recipes(user["id"])
            if not saved:
                print("No saved recipes.")
                continue
            for i, r in enumerate(saved, 1):
                print(f"{i}. {r[1]} ({r[2]})")
            sub = input("Enter number to view or 'b' to go back: ")
            if sub.isdigit():
                rid = saved[int(sub)-1][0]
                view_recipe(rid)
                uns = input("Unsave this recipe? (y/n): ").lower()
                if uns == "y":
                    unsave_recipe(user["id"], rid)
                    print("Recipe unsaved.")

        elif choice == "2":
            category = input("Enter category: ")
            recipes = get_recipes_by_category(category)
            if not recipes:
                print("No recipes found in that category.")
                continue
            for i, (rid, name, ing, cat) in enumerate(recipes, 1):
                cost = predict_cost(ing)
                print(f"{i}. {name} ({cat}) -> Rs.{cost:.2f}")
            sub = input("Enter recipe number to view: ")
            if sub.isdigit():
                rid = recipes[int(sub)-1][0]
                view_recipe(rid)
                sav = input("Save this recipe? (y/n): ").lower()
                if sav == "y":
                    save_recipe(user["id"], rid)
                    print("Recipe saved!")

        elif choice == "3":
            budget = float(input("Enter your budget (Rs): "))
            all_recipes = get_all_recipes()
            affordable = []
            for rid, name, ing, cat in all_recipes:
                cost = predict_cost(ing)
                if cost <= budget:
                    affordable.append((rid, name, cat, cost))
            if not affordable:
                print("No recipes found within that budget.")
                continue
            print(f"\nRecipes under Rs.{budget:.0f}:")
            for i, (rid, name, cat, cost) in enumerate(affordable, 1):
                print(f"{i}. {name} ({cat}) -> Rs.{cost:.2f}")
            sub = input("Enter number to view: ")
            if sub.isdigit():
                rid = affordable[int(sub)-1][0]
                view_recipe(rid)
                sav = input("Save this recipe? (y/n): ").lower()
                if sav == "y":
                    save_recipe(user["id"], rid)
                    print("Recipe saved!")

        elif choice == "4":
            feed = get_following_saved_recipes(user["id"])
            if not feed:
                print("No recipes saved by people you follow.")
                continue
            print("\nRecipes Saved by People You Follow:")
            for i, (uname, rid, name, cat) in enumerate(feed, 1):
                print(f"{i}. {name} ({cat}) — saved by {uname}")
            sub = input("Enter number to view recipe or 'b' to go back: ")
            if sub.isdigit():
                rid = feed[int(sub)-1][1]
                view_recipe(rid)

        elif choice == "5":
            print("\nSOCIAL MENU")
            print("1. View All Users")
            print("2. View My Followers")
            print("3. View Who I Follow")
            print("4. Go Back")
            sub_choice = input("Choose: ")

            if sub_choice == "1":
                users = list_all_users(user["id"])
                for i, (uid, uname) in enumerate(users, 1):
                    print(f"{i}. {uname}")
                act = input("Enter number to (f)ollow / (u)nfollow or 'b' to go back: ").lower()
                if act.isdigit():
                    target_id = users[int(act)-1][0]
                    mode = input("Follow or Unfollow (f/u): ").lower()
                    if mode == "f":
                        follow_user(user["id"], target_id)
                        print("Followed successfully.")
                    elif mode == "u":
                        unfollow_user(user["id"], target_id)
                        print("Unfollowed.")
            elif sub_choice == "2":
                followers = get_followers(user["id"])
                print("\nYour Followers:")
                print("\n".join(followers) if followers else "No followers yet.")
            elif sub_choice == "3":
                following = get_following(user["id"])
                print("\nYou Follow:")
                print("\n".join([u[1] for u in following]) if following else "You're not following anyone.")
            elif sub_choice == "4":
                continue

        elif choice == "6":
            find_recipes_by_ingredients_manual()

        elif choice == "7":
            find_recipes_by_chatbot_specs(user["id"])

        elif choice == "8":
            print("Logged out.")
            break

        elif choice == "9":
            print("Goodbye!")
            exit()

        else:
            print("Invalid choice.")


def main():
    print("="*60)
    print("WELCOME TO ZAIQA CLI")
    print("="*60)
    while True:
        print("\n1. Register")
        print("2. Login")
        print("3. Exit")
        opt = input("Choose: ")
        if opt == "1":
            user = register_flow()
            main_menu(user)
        elif opt == "2":
            user = login_flow()
            if user:
                main_menu(user)
        elif opt == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()