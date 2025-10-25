# =========================================================
# recipe_cli_demo_v3.py
# AI-powered Recipe Recommender + Social Feed CLI
# =========================================================

import psycopg2
import joblib
import numpy as np
from getpass import getpass

# ===============================
# 🔧 Database Configuration
# ===============================
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "102044",
    "host": "localhost",
    "port": "5432"
}

# ===============================
# 🧠 Load Model Components
# ===============================
print("📦 Loading AI model components...")
model = joblib.load("models/cost_estimator_v2/cost_estimator_v2.pkl")
vectorizer = joblib.load("models/cost_estimator_v2/vectorizer_v2.pkl")
scaler = joblib.load("models/cost_estimator_v2/scaler_v2.pkl")
print("✅ Model, vectorizer, and scaler loaded.\n")

# ===============================
# 🗄️ Database Helper Functions
# ===============================
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
    if row and row[2] == password:  # NOTE: replace with bcrypt in production
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

# ===============================
# 👥 Social Features (User Follows)
# ===============================
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
    return rows  # [(id, username)]

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
    return rows  # [(username, recipe_id, recipe_name, category)]

# ===============================
# 🤖 AI Cost Prediction
# ===============================
def predict_cost(ingredients_text):
    features = vectorizer.transform([ingredients_text])
    scaled = scaler.transform(features.toarray())
    predicted_cost = model.predict(scaled)[0]
    return max(predicted_cost, 0.0)

# ===============================
# 🧑 User Flow Functions
# ===============================
def register_flow():
    print("\n📝 Register New User")
    username = input("Username: ")
    email = input("Email: ")
    password = getpass("Password: ")
    user_id = create_user(username, email, password)
    print(f"✅ Registered successfully! (User ID: {user_id})")
    return {"id": user_id, "username": username}

def login_flow():
    print("\n🔐 Login")
    email = input("Email: ")
    password = getpass("Password: ")
    user = authenticate_user(email, password)
    if user:
        print(f"✅ Welcome back, {user['username']}!")
        return user
    else:
        print("❌ Invalid credentials.")
        return None

def view_recipe(recipe_id):
    recipe = get_recipe_details(recipe_id)
    if not recipe:
        print("❌ Recipe not found.")
        return

    name, category, ingredients, instructions, prep, cook, total, yield_, cal, yt = recipe[1:]
    cost = predict_cost(ingredients)

    print("\n🍽 RECIPE DETAILS")
    print(f"Name: {name}")
    print(f"Category: {category}")
    print(f"Prep Time: {prep} min | Cook Time: {cook} min | Total: {total} min")
    print(f"Servings: {yield_} | Calories: {cal}")
    print(f"YouTube: {yt}")
    print(f"\n🧾 Estimated Cost (AI): Rs.{cost:.2f}")
    print(f"\nIngredients:\n{ingredients}\n")
    print(f"Instructions:\n{instructions}\n")

# ===============================
# 🧭 Main Menu (Now with Social Feed)
# ===============================
def main_menu(user):
    while True:
        print(f"\n🏠 Welcome, {user['username']}!")
        print("1. View Saved Recipes")
        print("2. Browse Recipes by Category")
        print("3. Find Recipes Under a Budget")
        print("4. View Feed (Saved Recipes from People You Follow)")
        print("5. Social (Follow / Unfollow Users)")
        print("6. Logout")
        choice = input("Choose: ")

        if choice == "1":
            saved = get_saved_recipes(user["id"])
            if not saved:
                print("📭 No saved recipes.")
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
                    print("🗑️ Recipe unsaved.")

        elif choice == "2":
            category = input("Enter category: ")
            recipes = get_recipes_by_category(category)
            if not recipes:
                print("❌ No recipes found in that category.")
                continue
            for i, (rid, name, ing, cat) in enumerate(recipes, 1):
                cost = predict_cost(ing)
                print(f"{i}. {name} ({cat}) → Rs.{cost:.2f}")
            sub = input("Enter recipe number to view: ")
            if sub.isdigit():
                rid = recipes[int(sub)-1][0]
                view_recipe(rid)
                sav = input("Save this recipe? (y/n): ").lower()
                if sav == "y":
                    save_recipe(user["id"], rid)
                    print("💾 Recipe saved!")

        elif choice == "3":
            budget = float(input("Enter your budget (Rs): "))
            all_recipes = get_all_recipes()
            affordable = []
            for rid, name, ing, cat in all_recipes:
                cost = predict_cost(ing)
                if cost <= budget:
                    affordable.append((rid, name, cat, cost))
            if not affordable:
                print("❌ No recipes found within that budget.")
                continue
            print(f"\n🍽 Recipes under Rs.{budget:.0f}:")
            for i, (rid, name, cat, cost) in enumerate(affordable, 1):
                print(f"{i}. {name} ({cat}) → Rs.{cost:.2f}")
            sub = input("Enter number to view: ")
            if sub.isdigit():
                rid = affordable[int(sub)-1][0]
                view_recipe(rid)
                sav = input("Save this recipe? (y/n): ").lower()
                if sav == "y":
                    save_recipe(user["id"], rid)
                    print("💾 Recipe saved!")

        elif choice == "4":
            feed = get_following_saved_recipes(user["id"])
            if not feed:
                print("📭 No recipes saved by people you follow.")
                continue
            print("\n📰 Recipes Saved by People You Follow:")
            for i, (uname, rid, name, cat) in enumerate(feed, 1):
                print(f"{i}. {name} ({cat}) — saved by {uname}")
            sub = input("Enter number to view recipe or 'b' to go back: ")
            if sub.isdigit():
                rid = feed[int(sub)-1][1]
                view_recipe(rid)

        elif choice == "5":
            print("\n👥 SOCIAL MENU")
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
                        print("✅ Followed successfully.")
                    elif mode == "u":
                        unfollow_user(user["id"], target_id)
                        print("🗑️ Unfollowed.")
            elif sub_choice == "2":
                followers = get_followers(user["id"])
                print("\n👤 Your Followers:")
                print("\n".join(followers) if followers else "No followers yet.")
            elif sub_choice == "3":
                following = get_following(user["id"])
                print("\n➡️ You Follow:")
                print("\n".join([u[1] for u in following]) if following else "You're not following anyone.")
            elif sub_choice == "4":
                continue

        elif choice == "6":
            print("👋 Logged out.")
            break
        else:
            print("❌ Invalid choice.")

# ===============================
# 🚀 Main Entry Point
# ===============================
def main():
    print("🍳 Welcome to the AI Recipe Recommender + Social Feed CLI!\n")
    while True:
        print("1. Register")
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
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option.")

if __name__ == "__main__":
    main()
