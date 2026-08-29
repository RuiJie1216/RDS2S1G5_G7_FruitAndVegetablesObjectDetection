"""
nutrition_api.py
Nutrition data source: USDA FoodData Central API only (no local database).

Performance strategy (this is the important part for a responsive GUI):
    1. In-memory cache      -> a food already looked up this session is instant.
    2. On-disk cache (JSON) -> persists between runs, so re-analyzing the same
                                photo, or common foods across sessions, never
                                re-hits the network.
    3. Parallel requests    -> when a photo has multiple detected foods, all
                                *uncached* foods are fetched concurrently with
                                a thread pool instead of one-by-one, so total
                                wait time is close to the slowest single call,
                                not the sum of all calls.
    4. Short timeout        -> a single slow/hanging request can't freeze the
                                whole batch; it just fails gracefully for that
                                one item.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

USDA_API_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

CACHE_FILE = os.path.join(os.path.dirname(__file__), "nutrition_cache.json")
REQUEST_TIMEOUT = 4       # seconds - fail fast rather than freeze the GUI
MAX_WORKERS = 6           # parallel API calls per batch

_NUTRIENT_MAP = {
    "Energy": "calories",
    "Protein": "protein",
    "Total lipid (fat)": "fat",
    "Carbohydrate, by difference": "carbs",
    "Fiber, total dietary": "fiber",
    "Sugars, total including NLEA": "sugar",
    "Vitamin C, total ascorbic acid": "vitamin_c",
}

# Simple rule-based health tags used by summarize_meal(); adjust thresholds as needed.
HEALTH_TAGS = {
    "high_protein": lambda t: t["protein"] >= 25,
    "high_carb": lambda t: t["carbs"] >= 60,
    "high_fat": lambda t: t["fat"] >= 25,
    "high_sugar": lambda t: t["sugar"] >= 15,
    "high_fiber": lambda t: t["fiber"] >= 8,
    "high_vitamin_c": lambda t: t["vitamin_c"] >= 40,
}

_EMPTY_NUTRITION = {
    "calories": 0, "protein": 0, "fat": 0, "carbs": 0,
    "fiber": 0, "sugar": 0, "vitamin_c": 0,
    "is_composite": False, "ingredients": [], "source": "not found",
}

# ---------------------------------------------------------------------------
# Cache: in-memory dict backed by a JSON file on disk
# ---------------------------------------------------------------------------
_cache = {}


def _load_cache():
    global _cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            _cache = {}


def _save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # caching is a performance optimization, never crash the app over it


_load_cache()


# ---------------------------------------------------------------------------
# Single API call
# ---------------------------------------------------------------------------
def _fetch_from_api(food_key: str) -> dict:
    """Query USDA FoodData Central for one food. Returns a normalized dict."""
    query = food_key.replace("_", " ")
    try:
        params = {"api_key": USDA_API_KEY, "query": query, "pageSize": 1}
        resp = requests.get(USDA_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        foods = data.get("foods", [])
        if not foods:
            return dict(_EMPTY_NUTRITION, display_name=query.title())

        food = foods[0]
        result = dict(_EMPTY_NUTRITION)
        result["display_name"] = food.get("description", query).title()
        result["source"] = "USDA API"

        for n in food.get("foodNutrients", []):
            key = _NUTRIENT_MAP.get(n.get("nutrientName"))
            if key:
                result[key] = n.get("value", 0)

        # Branded foods sometimes carry a raw "ingredients" text field
        # (e.g. "RICE, EGG, CARROT, SOY SAUCE") - use it when present so
        # composite dishes can still show a breakdown without a local DB.
        ingredients_text = food.get("ingredients")
        if ingredients_text:
            parts = [p.strip().title() for p in ingredients_text.split(",") if p.strip()]
            result["ingredients"] = parts[:8]
            result["is_composite"] = len(parts) > 1

        return result

    except requests.RequestException:
        return dict(_EMPTY_NUTRITION, display_name=query.title(), source="API error")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_nutrition(food_key: str) -> dict:
    """Single-item lookup, cache-first."""
    if food_key in _cache:
        return _cache[food_key]
    result = _fetch_from_api(food_key)
    _cache[food_key] = result
    _save_cache()
    return result


def get_nutrition_batch(food_keys: list) -> dict:
    """
    Look up several foods at once. Cached items return instantly; only the
    *uncached* ones are fetched, and those are fetched in parallel so a photo
    with several detected items doesn't wait on each API call sequentially.
    """
    uncached = [k for k in dict.fromkeys(food_keys) if k not in _cache]

    if uncached:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(uncached))) as pool:
            futures = {pool.submit(_fetch_from_api, k): k for k in uncached}
            for future in as_completed(futures):
                key = futures[future]
                _cache[key] = future.result()
        _save_cache()

    return {k: _cache[k] for k in food_keys}


def summarize_meal(detected_foods: list) -> dict:
    """
    Aggregate nutrition for a whole meal (list of detections) and produce a
    health verdict + suitable/caution audience. Uses get_nutrition_batch so
    all lookups for one photo happen in a single parallel batch.
    """
    food_keys = [d["food_key"] for d in detected_foods]
    nutrition_by_key = get_nutrition_batch(food_keys)

    totals = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0,
              "fiber": 0, "sugar": 0, "vitamin_c": 0}
    items = []

    for det in detected_foods:
        info = dict(nutrition_by_key[det["food_key"]])
        info["confidence"] = det.get("confidence", 0)
        items.append(info)
        for k in totals:
            totals[k] += info.get(k, 0)

    tags = [name for name, cond in HEALTH_TAGS.items() if cond(totals)]

    score = 10
    if "high_fat" in tags:
        score -= 2
    if "high_sugar" in tags:
        score -= 2
    if "high_carb" in tags and totals["fiber"] < 5:
        score -= 1
    if "high_fiber" in tags or "high_vitamin_c" in tags:
        score += 1
    score = max(0, min(10, score))

    suitable_for, caution_for = [], []
    if totals["protein"] >= 25:
        suitable_for.append("athletes / muscle building")
    if totals["carbs"] < 40 and totals["fat"] < 15:
        suitable_for.append("weight management")
    if totals["vitamin_c"] >= 30:
        suitable_for.append("people needing more vitamin C")
    if "high_sugar" in tags or "high_fat" in tags:
        caution_for.append("diabetic / high cholesterol individuals")
    if totals["carbs"] >= 60:
        caution_for.append("low-carb diet followers")

    verdict = "balanced and healthy" if score >= 7 else \
              ("acceptable, some caution advised" if score >= 4 else "not very healthy")

    return {
        "items": items,
        "totals": totals,
        "tags": tags,
        "score": round(score, 1),
        "verdict": verdict,
        "suitable_for": suitable_for or ["general population"],
        "caution_for": caution_for or [],
    }


if __name__ == "__main__":
    import time

    demo = [
        {"food_key": "grilled_chicken", "confidence": 0.94},
        {"food_key": "steamed_rice", "confidence": 0.89},
        {"food_key": "broccoli", "confidence": 0.91},
    ]

    start = time.time()
    result = summarize_meal(demo)
    print(f"First run (cold cache) took {time.time() - start:.2f}s")
    print(json.dumps(result, indent=2))

    start = time.time()
    summarize_meal(demo)
    print(f"Second run (warm cache) took {time.time() - start:.4f}s")
