"""
tools.py — VibeTrip tool functions used by agents
"""

import json
import os
import httpx
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# In-memory store (replace with a real DB for production)
# ---------------------------------------------------------------------------
# Structure: { user_id: { trips: {...}, vibe_profile: {...} } }
USER_DATA: dict = {}

# ---------------------------------------------------------------------------
# Seed data — real trip from the user, pre-loaded for demo
# ---------------------------------------------------------------------------
SEED_TRIPS = {
    "Japan 2025": {
        "start_date": "2025-10-15",
        "end_date":   "2025-10-18",
        "created_at": "2025-10-15T16:00:00",
        "checkins": [
            {
                "place_name": "DoubleTree by Hilton Tokyo Ariake",
                "category":   "accommodation",
                "price_tier": "mid",
                "transport":  "taxi",
                "date":       "2025-10-15",
                "notes":      "Checked in at 4pm after long flight, nice hotel in Ariake area"
            },
            {
                "place_name": "7-Eleven convenience store",
                "category":   "food",
                "price_tier": "budget",
                "transport":  "walk",
                "date":       "2025-10-15",
                "notes":      "Grabbed instant ramen and rice ball for a simple solo dinner — classic convenience store Japan experience"
            },
            {
                "place_name": "Haneda Airport pickup",
                "category":   "transport",
                "price_tier": "free",
                "transport":  "transit",
                "date":       "2025-10-15",
                "notes":      "Picked up girlfriend at airport around 11pm, took transit back to hotel"
            },
            {
                "place_name": "Tonkatsu Marushichi - Fukagawa Fudo",
                "category":   "food",
                "price_tier": "mid",
                "transport":  "transit",
                "date":       "2025-10-16",
                "notes":      "Katsudon restaurant — juicy and delicious, took subway to get here, left hotel around noon"
            },
            {
                "place_name": "Park near Tokyo Tower",
                "category":   "sightseeing",
                "price_tier": "free",
                "transport":  "transit",
                "date":       "2025-10-16",
                "notes":      "Chill vibe, walked around the park with views of Tokyo Tower, relaxed afternoon"
            },
            {
                "place_name": "Joto Curry",
                "category":   "food",
                "price_tier": "budget",
                "transport":  "transit",
                "date":       "2025-10-16",
                "notes":      "Really delicious Japanese curry for dinner, took subway back to hotel after"
            },
            {
                "place_name": "7-Eleven convenience store (dessert run)",
                "category":   "food",
                "price_tier": "budget",
                "transport":  "walk",
                "date":       "2025-10-16",
                "notes":      "Grabbed desserts at the convenience store before bed — konbini sweets are underrated"
            },
            {
                "place_name": "Ramen Jazzy Beats",
                "category":   "food",
                "price_tier": "budget",
                "transport":  "transit",
                "date":       "2025-10-17",
                "notes":      "Delicious ramen in a restaurant under the subway rail — nice, clean, great atmosphere. Left hotel around noon."
            },
            {
                "place_name": "Gotokuji Temple",
                "category":   "culture",
                "price_tier": "free",
                "transport":  "transit",
                "date":       "2025-10-17",
                "notes":      "Saw the temple and the famous lucky cat (maneki-neko) dolls — unique and charming spot"
            },
            {
                "place_name": "Pokemon Center Tokyo DX",
                "category":   "entertainment",
                "price_tier": "free",
                "transport":  "transit",
                "date":       "2025-10-17",
                "notes":      "Browsed cute Pokemon merchandise, fun and lively atmosphere"
            },
            {
                "place_name": "Nenotsu",
                "category":   "food",
                "price_tier": "budget",
                "transport":  "transit",
                "date":       "2025-10-17",
                "notes":      "Small but cozy udon noodle restaurant, great noodles, wonderful vibe — last meal before heading to airport"
            },
        ]
    },
    "Los Angeles 2025": {
        "start_date": "2025-07-04",
        "end_date":   "2025-07-07",
        "created_at": "2025-07-04T15:00:00",
        "checkins": [
            # Day 1 — Arrival + Silver Lake
            {
                "place_name": "Freehand Los Angeles",
                "category":   "accommodation",
                "price_tier": "mid",
                "transport":  "taxi",
                "date":       "2025-07-04",
                "notes":      "Checked in around 3pm, cool boutique hotel in the middle of everything, rooftop pool"
            },
            {
                "place_name": "Sqirl",
                "category":   "food",
                "price_tier": "budget",
                "transport":  "walk",
                "date":       "2025-07-04",
                "notes":      "Famous LA breakfast spot — ricotta toast and sorrel pesto rice bowl, worth the short wait, casual local crowd"
            },
            {
                "place_name": "Sunset Boulevard stroll",
                "category":   "sightseeing",
                "price_tier": "free",
                "transport":  "walk",
                "date":       "2025-07-04",
                "notes":      "Walked through Silver Lake, great street art and indie shops, very chill vibe"
            },
            {
                "place_name": "Tacos 1986",
                "category":   "food",
                "price_tier": "budget",
                "transport":  "walk",
                "date":       "2025-07-04",
                "notes":      "Late night street-style tacos, super affordable and delicious, long line but moved fast — very local"
            },
            # Day 2 — Venice Beach + Culver City
            {
                "place_name": "Venice Beach Boardwalk",
                "category":   "sightseeing",
                "price_tier": "free",
                "transport":  "car",
                "date":       "2025-07-05",
                "notes":      "Morning walk along the boardwalk, watched street performers and skaters at the skate park, laid-back energy"
            },
            {
                "place_name": "Gjusta Bakery",
                "category":   "food",
                "price_tier": "mid",
                "transport":  "walk",
                "date":       "2025-07-05",
                "notes":      "Incredible pastries and smoked fish sandwiches, industrial-chic space — one of LA's best"
            },
            {
                "place_name": "Abbot Kinney Boulevard",
                "category":   "shopping",
                "price_tier": "mid",
                "transport":  "walk",
                "date":       "2025-07-05",
                "notes":      "Browsed indie boutiques and concept stores, very LA aesthetic — bought a vintage tee from a small shop"
            },
            {
                "place_name": "Gjelina",
                "category":   "food",
                "price_tier": "mid",
                "transport":  "walk",
                "date":       "2025-07-05",
                "notes":      "Dinner at Venice staple — wood-fired pizzas and seasonal small plates, great outdoor patio atmosphere"
            },
            # Day 3 — DTLA + Arts District
            {
                "place_name": "Grand Central Market",
                "category":   "food",
                "price_tier": "budget",
                "transport":  "transit",
                "date":       "2025-07-06",
                "notes":      "Bustling downtown market, grabbed egg tostada from Egg Slut and fresh juice, tons of options, lively atmosphere"
            },
            {
                "place_name": "The Broad Museum",
                "category":   "museum",
                "price_tier": "free",
                "transport":  "walk",
                "date":       "2025-07-06",
                "notes":      "Free contemporary art museum — Jeff Koons and Cindy Sherman exhibits were standouts, beautiful building"
            },
            {
                "place_name": "Arts District street art walk",
                "category":   "culture",
                "price_tier": "free",
                "transport":  "walk",
                "date":       "2025-07-06",
                "notes":      "Self-guided walk through murals and street art, very photogenic and creative energy"
            },
            {
                "place_name": "Bavel",
                "category":   "food",
                "price_tier": "mid",
                "transport":  "walk",
                "date":       "2025-07-06",
                "notes":      "Middle Eastern-inspired dinner in the Arts District — hummus, lamb flatbread, amazing cocktails, lively scene"
            },
            # Day 4 — Griffith Park + departure
            {
                "place_name": "Griffith Observatory",
                "category":   "sightseeing",
                "price_tier": "free",
                "transport":  "car",
                "date":       "2025-07-07",
                "notes":      "Morning hike up to the observatory, stunning panoramic views of LA and the Hollywood sign, not too crowded early"
            },
            {
                "place_name": "Café Tropical",
                "category":   "food",
                "price_tier": "budget",
                "transport":  "car",
                "date":       "2025-07-07",
                "notes":      "Cuban-style Silver Lake café, great café con leche and pastelitos, beloved local institution, cash only"
            },
            {
                "place_name": "Canter's Deli",
                "category":   "food",
                "price_tier": "budget",
                "transport":  "car",
                "date":       "2025-07-07",
                "notes":      "Classic LA Jewish deli open since 1931, pastrami sandwich and matzo ball soup before heading to LAX"
            },
        ]
    },
}


def _user(user_id: str) -> dict:
    if user_id not in USER_DATA:
        USER_DATA[user_id] = {"trips": {}, "vibe_profile": None, "planned_trips": []}
    # Ensure planned_trips key exists for older sessions
    if "planned_trips" not in USER_DATA[user_id]:
        USER_DATA[user_id]["planned_trips"] = []
    return USER_DATA[user_id]


def seed_demo_user(user_id: str = "demo") -> None:
    """Pre-load real trip data for the demo user on server startup."""
    if user_id not in USER_DATA:
        USER_DATA[user_id] = {"trips": {}, "vibe_profile": None, "planned_trips": []}
    USER_DATA[user_id]["trips"] = SEED_TRIPS
    if "planned_trips" not in USER_DATA[user_id]:
        USER_DATA[user_id]["planned_trips"] = []


def save_planned_trip(user_id: str, itinerary: dict) -> dict:
    """
    Save an AI-planned itinerary to the user's planned trips list.

    Args:
        user_id:   Identifier for the user session.
        itinerary: The full itinerary JSON object from the planner.

    Returns:
        Confirmation dict.
    """
    data = _user(user_id)
    itinerary["saved_at"] = datetime.utcnow().isoformat()
    data["planned_trips"].append(itinerary)
    return {"status": "saved", "destination": itinerary.get("destination", "Unknown")}


def get_planned_trips(user_id: str) -> dict:
    """
    Retrieve all AI-planned itineraries for a user.

    Args:
        user_id: Identifier for the user session.

    Returns:
        List of planned itinerary objects.
    """
    data = _user(user_id)
    return {"planned_trips": data.get("planned_trips", [])}


# ---------------------------------------------------------------------------
# Trip logging tools
# ---------------------------------------------------------------------------

def save_checkin(user_id: str, trip_name: str, place_name: str, category: str,
                 price_tier: str, transport: str, date: str = "", notes: str = "") -> dict:
    """
    Save a manual check-in to the user's current trip log.

    Args:
        user_id:    Identifier for the user session.
        trip_name:  Name/destination of the trip (e.g. "Japan 2025").
        place_name: Name of the place checked in to.
        category:   One of: food, museum, sightseeing, culture, nature, nightlife,
                    shopping, wellness, entertainment, accommodation, transport.
        price_tier: One of: free, budget, mid, luxury.
        transport:  How the user got there: walk, transit, taxi, bike, car.
        date:       Date of visit in YYYY-MM-DD format (optional).
        notes:      Optional personal note about the visit.

    Returns:
        Confirmation dict with the saved check-in.
    """
    data = _user(user_id)
    if trip_name not in data["trips"]:
        data["trips"][trip_name] = {"checkins": [], "created_at": datetime.utcnow().isoformat()}

    checkin = {
        "place_name": place_name,
        "category":   category,
        "price_tier": price_tier,
        "transport":  transport,
        "date":       date or datetime.utcnow().strftime("%Y-%m-%d"),
        "notes":      notes,
        "timestamp":  datetime.utcnow().isoformat(),
    }
    data["trips"][trip_name]["checkins"].append(checkin)
    return {"status": "saved", "trip": trip_name, "checkin": checkin}


def get_trip_history(user_id: str) -> dict:
    """
    Retrieve all past trips and their check-ins for a user.

    Args:
        user_id: Identifier for the user session.

    Returns:
        Dict of all trips with their check-ins.
    """
    data = _user(user_id)
    return {"trips": data["trips"]}


def get_vibe_profile(user_id: str) -> dict:
    """
    Retrieve the stored vibe profile for a user (if it exists).

    Args:
        user_id: Identifier for the user session.

    Returns:
        The vibe profile dict, or null if not yet generated.
    """
    data = _user(user_id)
    return {"vibe_profile": data["vibe_profile"]}


def save_vibe_profile(user_id: str, profile: dict) -> dict:
    """
    Persist the inferred vibe profile for a user.

    Args:
        user_id: Identifier for the user session.
        profile: Structured vibe profile dict produced by the Vibe Analyzer.

    Returns:
        Confirmation dict.
    """
    data = _user(user_id)
    data["vibe_profile"] = profile
    return {"status": "saved", "vibe_profile": profile}


# ---------------------------------------------------------------------------
# Place enrichment / search tools
# ---------------------------------------------------------------------------

def search_places(destination: str, query: str, vibe_tags: list[str]) -> dict:
    """
    Search for real places at a destination that match given vibe tags,
    using the Google Places Text Search API.

    Args:
        destination: City or region to search in (e.g. "Barcelona, Spain").
        query:       What to search for (e.g. "local tapas bar").
        vibe_tags:   List of vibe descriptors to bias the search (e.g. ["budget", "local", "walkable"]).

    Returns:
        List of places with name, address, rating, price level, and types.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        # Return mock data if no API key configured
        return _mock_places(destination, query)

    full_query = f"{query} in {destination} {' '.join(vibe_tags)}"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": full_query, "key": api_key}

    try:
        resp = httpx.get(url, params=params, timeout=10)
        results = resp.json().get("results", [])[:5]
        places = []
        for r in results:
            places.append({
                "name": r.get("name"),
                "address": r.get("formatted_address"),
                "rating": r.get("rating"),
                "price_level": r.get("price_level"),
                "types": r.get("types", [])[:3],
            })
        return {"places": places}
    except Exception as e:
        return {"error": str(e), "places": []}


def _mock_places(destination: str, query: str) -> dict:
    """Fallback mock places when no API key is set."""
    return {
        "places": [
            {"name": f"Local spot near {destination}", "address": destination,
             "rating": 4.5, "price_level": 1, "types": ["restaurant"]},
            {"name": f"Hidden gem in {destination}", "address": destination,
             "rating": 4.3, "price_level": 2, "types": ["cafe"]},
        ],
        "note": "Mock data — set GOOGLE_PLACES_API_KEY for real results"
    }


def get_transport_options(origin: str, destination_area: str, preferred_mode: str) -> dict:
    """
    Return transport recommendations between areas in a city.

    Args:
        origin:           Starting neighborhood or landmark.
        destination_area: Target neighborhood or landmark.
        preferred_mode:   User's preferred mode from vibe profile (walk, transit, taxi, bike).

    Returns:
        Ranked transport options with estimated cost and time.
    """
    # In production this would call Google Directions or Rome2Rio API
    options = {
        "walk":    {"mode": "Walking",           "est_time": "15–30 min", "est_cost": "Free",    "vibe": "exploratory"},
        "transit": {"mode": "Metro / Bus",        "est_time": "10–20 min", "est_cost": "$1–3",    "vibe": "local"},
        "bike":    {"mode": "Bike share",         "est_time": "10–25 min", "est_cost": "$2–5",    "vibe": "active"},
        "taxi":    {"mode": "Taxi / Ride-share",  "est_time": "5–15 min",  "est_cost": "$10–20",  "vibe": "convenient"},
    }
    primary = options.get(preferred_mode, options["transit"])
    alternatives = [v for k, v in options.items() if k != preferred_mode]
    return {
        "origin": origin,
        "destination": destination_area,
        "recommended": primary,
        "alternatives": alternatives[:2],
    }


# ---------------------------------------------------------------------------
# Weather tool (Open-Meteo archive API — free, no API key required)
# ---------------------------------------------------------------------------

# Monthly climate mock data used as fallback if API is unreachable
_CLIMATE_FALLBACK = {
    "barcelona": {1:(12,2.5),2:(13,2.1),3:(15,2.8),4:(17,3.5),5:(21,2.9),6:(25,1.2),
                  7:(28,0.6),8:(28,1.1),9:(24,3.2),10:(20,3.8),11:(15,3.1),12:(12,2.6)},
    "tokyo":     {1:(6,1.5), 2:(7,1.8), 3:(11,3.5),4:(16,3.8),5:(21,3.9),6:(24,5.8),
                  7:(28,4.2),8:(30,3.8),9:(25,5.5),10:(19,3.1),11:(13,2.1),12:(8,1.3)},
    "paris":     {1:(6,1.5), 2:(7,1.4), 3:(11,1.7),4:(14,2.1),5:(18,2.3),6:(22,1.8),
                  7:(24,1.5),8:(24,1.6),9:(20,1.8),10:(15,2.2),11:(10,2.1),12:(7,1.8)},
    "new york":  {1:(2,2.8), 2:(4,2.6), 3:(9,3.1), 4:(15,3.2),5:(21,3.1),6:(26,2.8),
                  7:(29,2.9),8:(28,2.7),9:(23,2.8),10:(17,2.9),11:(11,3.0),12:(5,3.2)},
}


def get_weather(city: str, month: int) -> dict:
    """
    Fetch historical climate averages for a city in a given month
    using the Open-Meteo geocoding + archive API (no key required).

    Args:
        city:  City name (e.g. "Barcelona", "Tokyo").
        month: Month number 1–12.

    Returns:
        Dict with avg_temp_c, avg_rain_mm_day, summary, and packing_tips.
    """
    month_name = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"][month - 1]

    try:
        # Step 1 — geocode city to lat/lon
        geo_resp = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10
        )
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            raise ValueError(f"City not found: {city}")

        loc     = geo_data["results"][0]
        lat     = loc["latitude"]
        lon     = loc["longitude"]
        name    = loc.get("name", city)
        country = loc.get("country", "")

        # Step 2 — pull 3 years of same-month data from archive API
        import datetime as dt
        current_year = dt.date.today().year
        all_temps, all_precip = [], []

        for year in range(current_year - 4, current_year - 1):
            start = f"{year}-{month:02d}-01"
            # last day of month
            if month == 12:
                end = f"{year}-12-31"
            else:
                end = f"{year}-{month+1:02d}-01"
                # subtract one day via string (simple approach)
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                end = f"{year}-{month:02d}-{last_day}"

            resp = httpx.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude":  lat,
                    "longitude": lon,
                    "start_date": start,
                    "end_date":   end,
                    "daily":     ["temperature_2m_mean", "precipitation_sum"],
                    "timezone":  "auto",
                },
                timeout=15
            )
            daily = resp.json().get("daily", {})
            all_temps  += [t for t in daily.get("temperature_2m_mean", []) if t is not None]
            all_precip += [p for p in daily.get("precipitation_sum",   []) if p is not None]

        if not all_temps:
            raise ValueError("No data returned")

        avg_temp = round(sum(all_temps)  / len(all_temps),  1)
        avg_rain = round(sum(all_precip) / len(all_precip), 1)

    except Exception:
        # Graceful fallback to hardcoded monthly averages
        key = city.lower().strip()
        fallback = _CLIMATE_FALLBACK.get(key)
        if fallback and month in fallback:
            avg_temp, avg_rain = fallback[month]
            name, country = city.title(), ""
        else:
            # Generic temperate default
            avg_temp, avg_rain = 18.0, 2.5
            name, country = city.title(), ""

    # Build plain-English summary for the LLM
    if avg_temp < 5:       temp_desc = "cold"
    elif avg_temp < 14:    temp_desc = "cool"
    elif avg_temp < 22:    temp_desc = "mild and pleasant"
    elif avg_temp < 28:    temp_desc = "warm"
    else:                  temp_desc = "hot"

    if avg_rain < 2:       rain_desc = "dry"
    elif avg_rain < 4:     rain_desc = "occasionally rainy"
    else:                  rain_desc = "quite rainy"

    summary = (
        f"{name}{', ' + country if country else ''} in {month_name} is typically "
        f"{temp_desc} (avg {avg_temp}°C) and {rain_desc} (avg {avg_rain} mm/day)."
    )

    # Practical packing tips
    tips = []
    if avg_temp < 10:
        tips.append("Pack a warm coat and thermal layers")
    elif avg_temp < 18:
        tips.append("Bring a light jacket — evenings can be chilly")
    else:
        tips.append("Light clothing works well; a layer for evenings is enough")

    if avg_rain > 4:
        tips.append("Pack a compact umbrella or waterproof jacket")
        tips.append("On rainy days, prioritise indoor spots: museums, markets, cozy restaurants")
    elif avg_rain > 2:
        tips.append("Occasional showers possible — a packable rain layer is handy")
    else:
        tips.append("Mostly dry — great for outdoor sightseeing and long walks")

    if avg_temp > 28:
        tips.append("Avoid midday sun — plan outdoor activities for morning or late afternoon")
        tips.append("Stay hydrated and wear sunscreen")

    return {
        "city":            name,
        "country":         country,
        "month":           month_name,
        "avg_temp_c":      avg_temp,
        "avg_rain_mm_day": avg_rain,
        "summary":         summary,
        "packing_tips":    tips,
    }


# ---------------------------------------------------------------------------
# Tool registry (for agent tool_config)
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    {
        "name": "save_checkin",
        "description": "Save a manual check-in to the user's trip log.",
        "parameters": {
            "type": "object",
            "properties": {
                "trip_name":  {"type": "string"},
                "place_name": {"type": "string"},
                "category":   {"type": "string", "enum": [
                    "food", "museum", "sightseeing", "culture", "nature",
                    "nightlife", "shopping", "wellness", "entertainment",
                    "accommodation", "transport"
                ]},
                "price_tier": {"type": "string", "enum": ["free", "budget", "mid", "luxury"]},
                "transport":  {"type": "string", "enum": ["walk", "transit", "taxi", "bike", "car"]},
                "date":       {"type": "string", "description": "Date of visit in YYYY-MM-DD format"},
                "notes":      {"type": "string"},
            },
            "required": ["trip_name", "place_name", "category", "price_tier", "transport"],
        },
    },
    {
        "name": "get_trip_history",
        "description": "Retrieve all past trips and check-ins for the user.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_vibe_profile",
        "description": "Retrieve the stored vibe profile for the user.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "save_vibe_profile",
        "description": "Persist the inferred vibe profile after analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "object",
                    "description": "Structured vibe profile with keys: pace, budget_style, interests, transport_preference, social_style, vibe_summary, tags",
                }
            },
            "required": ["profile"],
        },
    },
    {
        "name": "search_places",
        "description": "Search for real places at a destination matching the user's vibe.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {"type": "string"},
                "query":       {"type": "string"},
                "vibe_tags":   {"type": "array", "items": {"type": "string"}},
            },
            "required": ["destination", "query", "vibe_tags"],
        },
    },
    {
        "name": "get_transport_options",
        "description": "Get transport recommendations between areas matching the user's preferred mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin":           {"type": "string"},
                "destination_area": {"type": "string"},
                "preferred_mode":   {"type": "string"},
            },
            "required": ["origin", "destination_area", "preferred_mode"],
        },
    },
    {
        "name": "get_weather",
        "description": "Fetch historical climate averages for a city in a given month. Call this at the start of trip planning to tailor recommendations to the weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "city":  {"type": "string", "description": "City name e.g. 'Barcelona'"},
                "month": {"type": "integer", "description": "Month number 1–12"},
            },
            "required": ["city", "month"],
        },
    },
]


def dispatch_tool(name: str, args: dict, user_id: str) -> str:
    """Route a tool call to the correct function and return JSON string result."""
    args["user_id"] = user_id  # inject user context

    fn_map = {
        "save_checkin":          save_checkin,
        "get_trip_history":      get_trip_history,
        "get_vibe_profile":      get_vibe_profile,
        "save_vibe_profile":     save_vibe_profile,
        "search_places":         search_places,
        "get_transport_options": get_transport_options,
        "get_weather":           get_weather,
    }

    fn = fn_map.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})

    # Remove user_id from args for functions that don't accept it
    if name in ("search_places", "get_transport_options", "get_weather"):
        args.pop("user_id", None)

    try:
        result = fn(**args)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})