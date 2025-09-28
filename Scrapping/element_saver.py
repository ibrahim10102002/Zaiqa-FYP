from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, os
from bs4 import BeautifulSoup
import re

BASE_URL = "https://www.masala.tv/recipe/bbq-recipes/" #enter the desired URL
SAVE_DIR = "saved_pages"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Paths to Brave and ChromeDriver (adjust these to your system) ---
brave_path = r"C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe" #enter path of the browser you use
chromedriver_path = r"C:/Users/Ibrahim Shahid/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe" #enter path where you installed the chromedriver

options = webdriver.ChromeOptions()
options.binary_location = brave_path
# options.add_argument("--headless")  # comment if you want to watch

service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=options)

def save_page_html(url, filename):
    """Visit URL, wait for JSON or fallback, save HTML to file"""
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "script[type='application/ld+json']"))
        )
    except:
        time.sleep(2)

    html = driver.page_source
    with open(os.path.join(SAVE_DIR, filename), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved {filename}")
    return html

def get_recipe_links_from_html(html):
    """Extract all recipe URLs from category page inside col-lg-8 col-md-12"""
    soup = BeautifulSoup(html, "html.parser")

    container = soup.select_one("div.super_container section.main-content div.wrapper div.container div.recipe-main-wrapper div.col div.row div.col-lg-8.col-md-12 div.row")
    if not container:
        return []

    recipe_links = []
    for post in container.find_all("div", class_=re.compile(r"post-")):
        a_tag = post.find("a", href=True)
        if a_tag:
            recipe_links.append(a_tag["href"])
    return recipe_links

# --- figure out where to resume ---
existing_recipes = [f for f in os.listdir(SAVE_DIR) if f.startswith("recipe") and f.endswith(".txt")]
if existing_recipes:
    last_recipe_num = max(int(re.sub(r"\D", "", f)) for f in existing_recipes)
else:
    last_recipe_num = 0

existing_pages = [f for f in os.listdir(SAVE_DIR) if f.startswith("desserts_page") and f.endswith(".txt")]
if existing_pages:
    last_page_num = max(int(re.sub(r"\D", "", f)) for f in existing_pages)
else:
    last_page_num = 0

recipe_counter = last_recipe_num + 1
page_num = last_page_num + 1   # 👈 auto-resume

while True:
    url = BASE_URL if page_num == 1 else f"{BASE_URL}page/{page_num}/"
    driver.get(url)
    time.sleep(2)

    if "Nothing Found" in driver.page_source or "404" in driver.title:
        print("🚫 No more pages found. Stopping.")
        break

    cat_html = save_page_html(url, f"desserts_page{page_num}.txt")

    recipe_links = get_recipe_links_from_html(cat_html)
    print(f"➡ Found {len(recipe_links)} recipes on page {page_num}")

    for link in recipe_links:
        filename = f"recipe{recipe_counter}.txt"
        if not os.path.exists(os.path.join(SAVE_DIR, filename)):  # don’t overwrite
            save_page_html(link, filename)
        recipe_counter += 1

    page_num += 1

driver.quit()
print("Finished saving all BBQ pages and recipes.")
