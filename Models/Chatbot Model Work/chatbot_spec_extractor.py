"""
UseCase3: Recipe Finder Chatbot - Specification Extractor
Pure specification extractor with NO hardcoded data.
Dynamically extracts and structures user input for recipe finder models.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class SpecificationExtractor:
    """
    Dynamically extracts user preferences into structured specifications.
    No hardcoded ingredient/cuisine/restriction lists - purely data-driven.
    """

    def __init__(self):
        self.specifications = {
            "ingredients": [],  # [{"name": "chicken", "quantity": 500, "unit": "g"}, ...]
            "budget": {
                "min": None,
                "max": None,
                "currency": "PKR"
            },
            "cooking_time": {
                "min": None,
                "max": None,
                "unit": "minutes"
            },
            "cuisine_types": [],  # Whatever user mentions
            "spice_level": None,  # Whatever user specifies
            "dietary_restrictions": [],  # Whatever user mentions
            "allergies": [],  # Whatever user mentions
            "age_group": None,  # Whatever user specifies
            "skill_level": None,  # Whatever user specifies
            "meal_type": [],  # Whatever user mentions
            "health_conditions": [],  # Whatever user mentions
            "context": {
                "weather": None,
                "time_of_day": None,
                "occasion": None,
                "day_of_week": datetime.now().strftime("%A")
            },
            "preferences": [],  # Any general preferences mentioned
            "user_input_raw": ""
        }
        self.conversation_history = []
        self.addressed_categories = set()

    def extract_from_input(self, user_input: str) -> Tuple[Dict, str]:
        """
        Extract specifications from user input.
        Returns: (specifications_dict, response_message)
        """
        self.specifications["user_input_raw"] = user_input
        self.conversation_history.append({"role": "user", "content": user_input})

        if user_input.lower().strip() == "none":
            self.addressed_categories.add("allergies")
            self.addressed_categories.add("dietary_restrictions")

        # Extract all components dynamically
        self._extract_ingredients(user_input)
        self._extract_budget(user_input)
        self._extract_cooking_time(user_input)
        self._extract_cuisine(user_input)
        self._extract_spice_level(user_input)
        self._extract_dietary_restrictions(user_input)
        self._extract_allergies(user_input)
        self._extract_age_group(user_input)
        self._extract_skill_level(user_input)
        self._extract_meal_type(user_input)
        self._extract_health_conditions(user_input)
        self._extract_context(user_input)
        self._extract_preferences(user_input)

        response = self._generate_response()
        self.conversation_history.append({"role": "assistant", "content": response})

        return self.specifications, response

    def _extract_ingredients(self, text: str):
        """
        Dynamically extract ANY ingredient mentioned with quantities.
        Uses regex to find patterns like "500g chicken", "2 cups rice", etc.
        Filter out currency and time units from ingredients
        """
        blacklist_words = {
            "pkr", "rupees", "rupee", "rs", "taka", "dollar", "dollars",
            "mins", "minutes", "minute", "hours", "hour", "hrs", "hr",
            "seconds", "second", "secs", "sec", "days", "day"
        }

        # Pattern: number + unit + ingredient name
        # Examples: "500g chicken", "2 cups rice", "1kg beef"
        quantity_pattern = r'(\d+(?:\.\d+)?)\s*(g|kg|ml|l|cup|cups|tbsp|tsp|piece|pieces|lb|oz|gram|grams|kilogram|liter|liters|milliliter|milliliters)?\s+([a-zA-Z\s]+?)(?:\s+and|\s+with|,|$)'

        matches = re.finditer(quantity_pattern, text, re.IGNORECASE)
        for match in matches:
            qty = float(match.group(1))
            unit = match.group(2) or "piece"
            ingredient_name = match.group(3).strip().lower()

            if ingredient_name and ingredient_name not in blacklist_words:
                self.specifications["ingredients"].append({
                    "name": ingredient_name,
                    "quantity": qty,
                    "unit": unit
                })

        # Also extract ingredients without quantities (e.g., "I have chicken and rice")
        # Look for common ingredient patterns without numbers
        ingredient_pattern = r'\b(have|got|with|using|need)\s+([a-zA-Z\s,and]+?)(?:\s+and|\s+with|,|$)'
        matches = re.finditer(ingredient_pattern, text, re.IGNORECASE)
        for match in matches:
            ingredients_text = match.group(2)
            # Split by "and" or comma
            items = re.split(r'\s+and\s+|,', ingredients_text)
            for item in items:
                item = item.strip().lower()
                if item and len(item) > 2 and item not in blacklist_words:
                    # Check if this ingredient is already added with quantity
                    if not any(ing["name"] == item for ing in self.specifications["ingredients"]):
                        self.specifications["ingredients"].append({
                            "name": item,
                            "quantity": None,
                            "unit": "available"
                        })

    def _extract_budget(self, text: str):
        """
        Dynamically extract budget from ANY mention of money/cost.
        Looks for patterns like "500 rupees", "under 300", "budget 1000", etc.
        """
        # Pattern: number + currency/cost keywords
        budget_pattern = r'(\d+)\s*(rupees|rs|pkr|rupee)?'

        # Look for budget-related keywords
        budget_keywords = ["budget", "rupees", "rs", "pkr", "under", "within", "afford", "cost", "price", "spend"]

        if any(keyword in text.lower() for keyword in budget_keywords):
            matches = re.findall(budget_pattern, text, re.IGNORECASE)
            if matches:
                amounts = [int(match[0]) for match in matches]
                if amounts:
                    # If "under" or "within" mentioned, it's max budget
                    if any(word in text.lower() for word in ["under", "within", "max", "maximum"]):
                        self.specifications["budget"]["max"] = max(amounts)
                    else:
                        self.specifications["budget"]["max"] = max(amounts)
                    self.addressed_categories.add("budget")

    def _extract_cooking_time(self, text: str):
        """
        Dynamically extract cooking time from ANY time mention.
        Looks for patterns like "30 minutes", "1 hour", "quick", etc.
        """
        # Pattern: number + time unit
        time_pattern = r'(\d+)\s*(minute|min|hour|hr|second|sec)?'

        time_keywords = ["quick", "fast", "minutes", "min", "hour", "hours", "time", "ready", "within"]

        if any(keyword in text.lower() for keyword in time_keywords):
            matches = re.findall(time_pattern, text, re.IGNORECASE)
            if matches:
                times = []
                for match in matches:
                    time_val = int(match[0])
                    unit = match[1].lower() if match[1] else "minute"

                    # Convert to minutes
                    if "hour" in unit or "hr" in unit:
                        time_val *= 60
                    elif "sec" in unit:
                        time_val = time_val / 60

                    times.append(time_val)

                if times:
                    self.specifications["cooking_time"]["max"] = int(max(times))
                    self.addressed_categories.add("cooking_time")

    def _extract_cuisine(self, text: str):
        """
        Dynamically extract ANY cuisine type mentioned.
        No predefined list - extracts whatever user says.
        """
        # Common cuisine keywords to look for
        cuisine_keywords = [
            "pakistani", "desi", "indian", "chinese", "continental", "western",
            "italian", "turkish", "thai", "mexican", "japanese", "korean",
            "fusion", "modern", "traditional", "authentic", "street food"
        ]

        text_lower = text.lower()
        for cuisine in cuisine_keywords:
            if cuisine in text_lower:
                if cuisine not in self.specifications["cuisine_types"]:
                    self.specifications["cuisine_types"].append(cuisine)

    def _extract_spice_level(self, text: str):
        """
        Dynamically extract spice level from user description.
        Looks for keywords like "spicy", "mild", "hot", etc.
        """
        text_lower = text.lower()

        # Map spice keywords to levels
        spice_keywords = {
            "very_hot": ["very spicy", "very hot", "teekha", "bohot teekha", "extra spicy", "extremely spicy"],
            "hot": ["spicy", "hot", "teekha", "mirch", "peppery"],
            "medium": ["medium", "moderate", "normal", "balanced"],
            "mild": ["mild", "light", "not spicy", "no spice", "bland"]
        }

        for level, keywords in spice_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                self.specifications["spice_level"] = level
                break

    def _extract_dietary_restrictions(self, text: str):
        """
        Dynamically extract ANY dietary restriction mentioned.
        No predefined list - extracts whatever user specifies.
        """
        text_lower = text.lower()

        # Common dietary restriction keywords
        restriction_keywords = [
            "vegetarian", "vegan", "gluten-free", "gluten free", "low-fat", "low fat",
            "low-sodium", "low sodium", "halal", "kosher", "no meat", "no dairy",
            "no animal", "plant-based", "organic", "sugar-free"
        ]

        for restriction in restriction_keywords:
            if restriction in text_lower:
                if restriction not in self.specifications["dietary_restrictions"]:
                    self.specifications["dietary_restrictions"].append(restriction)
                self.addressed_categories.add("dietary_restrictions")

    def _extract_allergies(self, text: str):
        """
        Dynamically extract ANY allergy mentioned.
        No predefined list - extracts whatever user mentions.
        """
        text_lower = text.lower()

        # Common allergy keywords
        allergy_keywords = [
            "nuts", "peanuts", "almonds", "cashews", "dairy", "milk", "cheese",
            "lactose", "shellfish", "shrimp", "crab", "lobster", "eggs", "egg",
            "soy", "soya", "sesame", "til", "fish", "wheat", "gluten", "pollen"
        ]

        for allergy in allergy_keywords:
            if allergy in text_lower:
                if allergy not in self.specifications["allergies"]:
                    self.specifications["allergies"].append(allergy)
                self.addressed_categories.add("allergies")

    def _extract_age_group(self, text: str):
        """
        Dynamically extract age group from user mention.
        """
        text_lower = text.lower()

        age_keywords = {
            "kids": ["kids", "children", "child", "baby", "toddler", "kids-friendly"],
            "elderly": ["elderly", "old", "senior", "grandpa", "grandma", "aged"],
            "adult": ["adult", "adults", "grown-up"],
            "mixed": ["family", "everyone", "all ages", "mixed"]
        }

        for group, keywords in age_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                self.specifications["age_group"] = group
                break

    def _extract_skill_level(self, text: str):
        """
        Dynamically extract cooking skill level.
        """
        text_lower = text.lower()

        skill_keywords = {
            "beginner": ["beginner", "easy", "simple", "new to cooking", "first time"],
            "intermediate": ["intermediate", "medium", "normal", "experienced"],
            "advanced": ["advanced", "complex", "difficult", "expert", "professional"]
        }

        for level, keywords in skill_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                self.specifications["skill_level"] = level
                break

    def _extract_meal_type(self, text: str):
        """
        Dynamically extract meal type from user mention.
        """
        text_lower = text.lower()

        meal_keywords = {
            "breakfast": ["breakfast", "subah", "morning meal", "brunch"],
            "lunch": ["lunch", "dopahar", "midday"],
            "dinner": ["dinner", "raat", "evening meal", "supper"],
            "snack": ["snack", "nashta", "appetizer", "starter"],
            "dessert": ["dessert", "sweet", "mithai", "pudding"]
        }

        for meal, keywords in meal_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if meal not in self.specifications["meal_type"]:
                    self.specifications["meal_type"].append(meal)

    def _extract_health_conditions(self, text: str):
        """
        Dynamically extract health conditions mentioned.
        """
        text_lower = text.lower()

        health_keywords = [
            "diabetic", "diabetes", "blood sugar", "low sodium", "blood pressure",
            "hypertension", "low-fat", "cholesterol", "high protein", "protein rich",
            "low-carb", "keto", "heart disease", "kidney", "liver"
        ]

        for condition in health_keywords:
            if condition in text_lower:
                if condition not in self.specifications["health_conditions"]:
                    self.specifications["health_conditions"].append(condition)

    def _extract_context(self, text: str):
        """
        Dynamically extract contextual information.
        """
        text_lower = text.lower()

        # Weather
        weather_keywords = {
            "rainy": ["rain", "rainy", "barsat", "wet"],
            "sunny": ["sunny", "sun", "dhoop", "bright"],
            "hot": ["hot", "garam", "heat"],
            "cold": ["cold", "thandi", "winter", "freezing"]
        }

        for weather, keywords in weather_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                self.specifications["context"]["weather"] = weather
                break

        # Time of day
        time_keywords = {
            "morning": ["morning", "breakfast", "subah", "dawn"],
            "afternoon": ["afternoon", "dopahar", "midday"],
            "evening": ["evening", "shaam", "sunset"],
            "night": ["night", "raat", "late"]
        }

        for time, keywords in time_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                self.specifications["context"]["time_of_day"] = time
                break

        # Occasion
        occasion_keywords = {
            "ramadan": ["ramadan", "ramazan", "sehri", "iftar"],
            "eid": ["eid", "eidi"],
            "cricket": ["cricket", "match", "psl"],
            "party": ["party", "gathering", "guests", "celebration"]
        }

        for occasion, keywords in occasion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                self.specifications["context"]["occasion"] = occasion
                break

    def _extract_preferences(self, text: str):
        """
        Dynamically extract general preferences mentioned.
        """
        text_lower = text.lower()

        preference_keywords = [
            "quick", "fast", "ready", "hurry", "comfort", "cozy", "warm",
            "healthy", "fit", "diet", "nutrition", "traditional", "desi",
            "authentic", "fusion", "modern", "new", "simple", "easy"
        ]

        for pref in preference_keywords:
            if pref in text_lower:
                if pref not in self.specifications["preferences"]:
                    self.specifications["preferences"].append(pref)

    def _generate_response(self) -> str:
        """Generate response asking for missing critical information."""
        missing = []

        if not self.specifications["ingredients"]:
            missing.append("ingredients you have")

        if self.specifications["budget"]["max"] is None and "budget" not in self.addressed_categories:
            missing.append("your budget")

        if self.specifications["cooking_time"]["max"] is None and "cooking_time" not in self.addressed_categories:
            missing.append("how much time you have")

        if (not self.specifications["dietary_restrictions"] and
            not self.specifications["allergies"] and
            "allergies" not in self.addressed_categories and
            "dietary_restrictions" not in self.addressed_categories):
            missing.append("any dietary restrictions or allergies")

        if missing:
            return f"Got it! To find the perfect recipe, could you also tell me about: {', '.join(missing)}?"
        else:
            return "Perfect! I have all the information I need. Let me find the best recipes for you!"

    def get_specifications(self) -> Dict:
        """Return the extracted specifications."""
        return self.specifications

    def get_specifications_for_models(self) -> Dict:
        """
        Return specifications formatted for different recipe finder models.
        This is the output that will be sent to ingredient-based, cost-based, etc. finders.
        """
        return {
            "ingredient_finder": {
                "ingredients": self.specifications["ingredients"],
                "exclude_allergies": self.specifications["allergies"],
                "dietary_restrictions": self.specifications["dietary_restrictions"]
            },
            "cost_finder": {
                "max_budget": self.specifications["budget"]["max"],
                "ingredients": self.specifications["ingredients"]
            },
            "time_finder": {
                "max_cooking_time": self.specifications["cooking_time"]["max"],
                "skill_level": self.specifications["skill_level"]
            },
            "cuisine_finder": {
                "cuisines": self.specifications["cuisine_types"],
                "spice_level": self.specifications["spice_level"]
            },
            "context_finder": {
                "weather": self.specifications["context"]["weather"],
                "time_of_day": self.specifications["context"]["time_of_day"],
                "occasion": self.specifications["context"]["occasion"]
            },
            "health_finder": {
                "health_conditions": self.specifications["health_conditions"],
                "age_group": self.specifications["age_group"]
            },
            "all_specifications": self.specifications
        }

    def print_specifications(self):
        """Pretty print the extracted specifications."""
        print("\n" + "="*60)
        print("EXTRACTED RECIPE SPECIFICATIONS")
        print("="*60)
        print(json.dumps(self.specifications, indent=2))
        print("\n" + "="*60)
        print("SPECIFICATIONS FOR MODELS")
        print("="*60)
        print(json.dumps(self.get_specifications_for_models(), indent=2))
        print("="*60 + "\n")


class RecipeDiscoveryChatbot:
    """
    Main chatbot interface for recipe discovery.
    Collects user preferences and outputs structured specifications.
    """

    def __init__(self):
        self.extractor = SpecificationExtractor()
        self.conversation_turns = 0

    def chat(self, user_input: str) -> str:
        """Process user input and return response."""
        self.conversation_turns += 1
        specs, response = self.extractor.extract_from_input(user_input)
        return response

    def get_final_specifications(self) -> Dict:
        """Get the final extracted specifications."""
        return self.extractor.get_specifications()

    def get_specifications_for_models(self) -> Dict:
        """Get specifications formatted for different recipe finder models."""
        return self.extractor.get_specifications_for_models()

    def print_summary(self):
        """Print a summary of extracted specifications."""
        self.extractor.print_specifications()

    def reset(self):
        """Reset the chatbot for a new conversation."""
        self.extractor = SpecificationExtractor()
        self.conversation_turns = 0
      
    def is_complete(self) -> bool:
        """
        Marks chatbot as complete once the user explicitly says 'done' or similar.
        Does not depend on whether all details are filled.
        """
        if not self.extractor.conversation_history:
            return False

        # Check if the last user input indicates completion
        last_input = self.extractor.conversation_history[-2]["content"].lower() if len(self.extractor.conversation_history) >= 2 else ""
        completion_keywords = ["done", "finish", "complete", "that’s all", "that's all", "ok done", "i’m done", "im done"]

        return any(keyword in last_input for keyword in completion_keywords)




# Example usage
if __name__ == "__main__":
    chatbot = RecipeDiscoveryChatbot()

    print("🍳 Welcome to Zaiqa Recipe Discovery Chatbot!")
    print("Tell me what you'd like to cook, and I'll extract all the details.")
    print("Type 'done' when finished, 'summary' to see extracted specs, or 'reset' to start over.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            # Handle special commands
            if user_input.lower() == "done":
                print("\nFinalizing your recipe specifications...")
                chatbot.print_summary()
                break

            elif user_input.lower() == "summary":
                chatbot.print_summary()
                continue

            elif user_input.lower() == "reset":
                chatbot.reset()
                print("Chatbot reset. Let's start fresh!\n")
                continue

            elif user_input.lower() == "quit" or user_input.lower() == "exit":
                print("Thanks for using Zaiqa! Goodbye!")
                break

            elif not user_input:
                print("Please enter something or type 'help' for commands.\n")
                continue

            elif user_input.lower() == "help":
                print("\nAvailable commands:")
                print("  'done'    - Finalize and show extracted specifications")
                print("  'summary' - Show current extracted specifications")
                print("  'reset'   - Start a new conversation")
                print("  'quit'    - Exit the chatbot\n")
                continue

            # Process user input
            response = chatbot.chat(user_input)
            print(f"Bot: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Zaiqa! Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}. Please try again.\n")
