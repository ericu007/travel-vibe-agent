"""
app.py — VibeTrip FastAPI server
Multi-agent travel style tracker and trip planner.

Agents:
  1. Orchestrator   — routes intent to the right specialist
  2. Logger Agent   — handles trip check-ins (saves to memory)
  3. Vibe Analyzer  — deep-research pattern: synthesizes past trips → traveler profile
  4. Planner Agent  — generates itinerary grounded in vibe profile + live place search
  5. Critic Agent   — generator-critic pattern: reviews itinerary against profile, flags mismatches
"""

import json
import os
import re
import asyncio
import warnings
from typing import AsyncGenerator

# Suppress Vertex AI SDK deprecation warning (migration to google-genai is optional)
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tools import TOOL_DECLARATIONS, dispatch_tool, seed_demo_user

load_dotenv()

app = FastAPI(title="VibeTrip")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pre-load demo trip data on startup
seed_demo_user("demo")

VERTEX_PROJECT  = os.getenv("VERTEX_PROJECT", "")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
MODEL           = "gemini-2.5-flash"

# Initialise Vertex AI once at startup
vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)

# ---------------------------------------------------------------------------
# System prompts — each agent has a focused persona
# ---------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """You are the VibeTrip Orchestrator. Your only job is to read the user's message and classify their intent into exactly one of these categories:

INTENTS:
- LOG_CHECKIN: User wants to log a place they visited on a trip (mentions a place name, food spot, attraction, hotel, etc.)
- ANALYZE_VIBE: User wants to understand their travel style or see their traveler profile
- PLAN_TRIP: User wants to plan a trip to a new destination
- VIEW_TRIPS: User wants to see their past trips or check-in history
- LOAD_DEMO: User wants to load sample/demo trip data to try the app (says things like "load demo", "use sample data", "load example trips", "show me a demo")
- GENERAL: Greetings, questions about how VibeTrip works, or anything else

Respond ONLY with a JSON object in this exact format:
{"intent": "<INTENT>", "summary": "<one sentence describing what the user wants>"}

Examples:
User: "I just had amazing ramen at Ichiran in Tokyo, adding it to my Japan trip"
{"intent": "LOG_CHECKIN", "summary": "User wants to log a ramen restaurant visit in Tokyo to their Japan trip"}

User: "What kind of traveler am I based on my trips?"
{"intent": "ANALYZE_VIBE", "summary": "User wants their travel vibe profile analyzed from past trips"}

User: "Plan me 5 days in Barcelona"
{"intent": "PLAN_TRIP", "summary": "User wants a 5-day itinerary for Barcelona"}

User: "Plan me a luxury trip to Paris"
{"intent": "PLAN_TRIP", "summary": "User wants a luxury 4-day itinerary for Paris"}

User: "Can you plan something more adventurous and outdoorsy in New Zealand?"
{"intent": "PLAN_TRIP", "summary": "User wants an adventure/outdoorsy trip to New Zealand"}

User: "Plan me a romantic weekend in Rome"
{"intent": "PLAN_TRIP", "summary": "User wants a romantic weekend itinerary for Rome"}

User: "Show me all my trips"
{"intent": "VIEW_TRIPS", "summary": "User wants to see their logged trip history"}

User: "load demo trips"
{"intent": "LOAD_DEMO", "summary": "User wants to load sample trip data to explore the app"}

User: "Hey, how does this app work?"
{"intent": "GENERAL", "summary": "User is asking about VibeTrip features"}
"""

LOGGER_PROMPT = """You are the VibeTrip Logger Agent. You help users log places they visited during their travels.

Your job:
1. Extract trip details from the user's message (trip name, place name, category, price tier, how they got there)
2. Call save_checkin to save it
3. Confirm warmly and ask if they want to log anything else

Categories: food, culture, nature, nightlife, shopping, transport, accommodation
Price tiers: free, budget (under $15), mid ($15–50), luxury ($50+)
Transport: walk, transit, taxi, bike, car

If the user doesn't mention price or transport, make a reasonable inference and mention it.
If you're unsure about the trip name, ask: "Which trip should I log this under?"

Always be warm and conversational — you're their travel journal companion.

Escape hatch: If you cannot determine the place or trip name, ask a clarifying question instead of guessing.
"""

VIBE_ANALYZER_PROMPT = """You are the VibeTrip Vibe Analyzer — a deep research agent that infers a user's travel personality from their logged trips.

Your process (follow these steps in order):
1. Call get_trip_history to retrieve all past trips and check-ins
2. Analyze patterns across ALL trips: categories visited, price tiers, transport modes, pacing, types of places
3. Call save_vibe_profile to persist your findings
4. Present the profile to the user in an engaging, personal way

The profile you save must include these keys:
- pace: "slow" | "moderate" | "fast"
- budget_style: "backpacker" | "budget-conscious" | "mid-range" | "luxury"
- interests: list of top 3 (e.g. ["food", "culture", "nature"])
- transport_preference: "walk" | "transit" | "mix" | "taxi"
- social_style: "solo explorer" | "small group" | "social"
- vibe_summary: one punchy sentence describing their travel style
- tags: list of 4–6 short vibe tags (e.g. ["foodie", "off-the-beaten-path", "budget", "walkable"])

Be specific — reference actual places they visited. Don't give generic descriptions.
If there are fewer than 3 check-ins total, tell the user you need more trip data for an accurate profile.

Escape hatch: If trip data is ambiguous or contradictory, note the contradiction and explain your reasoning.
"""

PLANNER_PROMPT = """You are the VibeTrip Planner Agent. You create hyper-personalized travel itineraries grounded in the user's actual travel style and real weather data.

Your process (follow in order):
1. Call get_vibe_profile to retrieve the user's traveler profile
2. If no profile exists, tell the user to log some past trips first so you can learn their style
3. Call get_weather for the destination and travel month — use this to shape the entire itinerary
4. Search for real places using search_places — make 3–5 searches targeting their specific interests and vibe tags
5. Call get_transport_options to recommend how to get between key areas
6. Output the itinerary as a JSON object (see format below) — no markdown, no prose outside the JSON

Important rules:
- Match price tier to their budget_style — never recommend luxury spots to a backpacker
- Match transport to their transport_preference
- Reference their actual past trips in the personalisation_note field
- If weather is rainy or cold, weight itinerary toward indoor spots
- If weather is warm and sunny, prioritise outdoor sightseeing and walking routes

Output ONLY a valid JSON object in this exact format (no markdown fences, no text before or after):
{
  "destination": "Barcelona, Spain",
  "duration_days": 4,
  "travel_month": "October",
  "weather_summary": "Mild and pleasant, avg 20°C, occasionally rainy",
  "packing_tips": ["Light jacket for evenings", "Packable umbrella"],
  "vibe_tags": ["foodie", "budget-friendly", "urban adventurer"],
  "personalisation_note": "Since you loved the ramen spots in Japan and street food in LA...",
  "days": [
    {
      "day": 1,
      "title": "Gothic Quarter & El Born",
      "theme": "Culture & Local Food",
      "slots": [
        {
          "time_of_day": "Morning",
          "place_name": "Barri Gòtic",
          "category": "sightseeing",
          "description": "Wander the medieval lanes and find the Barcelona Cathedral",
          "why_it_fits": "Active explorers love the endless discoveries here",
          "transport": "walk",
          "est_cost": "Free",
          "image_search_query": "Gothic Quarter Barcelona narrow streets"
        },
        {
          "time_of_day": "Lunch",
          "place_name": "El Xampanyet",
          "category": "food",
          "description": "Classic Catalan tapas bar packed with locals, house cava is famous",
          "why_it_fits": "Budget foodie staple — exactly your kind of spot",
          "transport": "walk",
          "est_cost": "$10–15",
          "image_search_query": "El Xampanyet tapas bar Barcelona"
        }
      ]
    }
  ],
  "what_to_buy": ["Local olive oil from Boqueria", "Espadrilles from El Born"],
  "hidden_gems": ["Carrer del Parlament in Sant Antoni for local bar-hopping", "Bunkers del Carmel for the best panoramic view of Barcelona"]
}

Escape hatch: If you cannot determine the month, ask the user before proceeding.
"""

CRITIC_PROMPT = """You are the VibeTrip Critic Agent. You review draft itineraries and flag mismatches against the user's travel profile.

You will receive:
- The user's vibe profile
- A draft itinerary

Your job:
1. Check each day for mismatches: wrong price tier, wrong transport mode, tourist traps, pacing issues
2. Output a JSON list of flags, then a revised recommendation for each

Flag format:
{"day": 1, "item": "place name", "issue": "description of mismatch", "suggestion": "better alternative"}

After flagging, present a friendly summary to the user:
- How well the itinerary matches their profile (score out of 10)
- List of flagged items for their review (human-in-the-loop: ask them to approve or swap each one)
- Overall verdict

Be constructive, not critical. Frame flags as improvements, not mistakes.
Escape hatch: If the itinerary is a strong match with no issues, say so clearly and approve it.
"""

GENERAL_PROMPT = """You are VibeTrip, a travel companion that learns your travel style and plans personalized trips.

Explain what you can do:
- 📍 Log places from past trips (just tell me where you went)
- 🧠 Analyze your travel vibe from your logged trips
- 🗺️ Plan a personalized trip to any new destination based on your style
- 📋 View all your past trips and check-ins

If the user is new or has no trips logged yet, warmly offer them two options:
1. Load sample trip data (a Japan trip and an LA trip) so they can explore the app right away — tell them to just say "load demo trips" or click the option
2. Start logging their own real trips

Be friendly, brief, and encouraging.
"""

# ---------------------------------------------------------------------------
# Vertex AI helpers
# ---------------------------------------------------------------------------

def _build_vertex_tools(tool_declarations: list) -> list[Tool]:
    """Convert our tool declaration dicts into Vertex AI Tool objects."""
    fn_decls = []
    for t in tool_declarations:
        fn_decls.append(FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
        ))
    return [Tool(function_declarations=fn_decls)]


def _history_to_contents(messages: list) -> list[Content]:
    """Convert OpenAI-style message dicts to Vertex AI Content objects."""
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        text = m.get("content") or ""
        if text:
            contents.append(Content(role=role, parts=[Part.from_text(text)]))
    return contents


async def run_agent(system: str, messages: list, user_id: str,
                    use_tools: bool = True, force_json: bool = False) -> AsyncGenerator[str, None]:
    """
    Run a single Vertex AI agent with an agentic tool-calling loop.
    Yields SSE-formatted strings for streaming to the frontend.
    Uses asyncio.to_thread so the blocking Vertex SDK call doesn't freeze the event loop.
    """
    vertex_tools = _build_vertex_tools(TOOL_DECLARATIONS) if use_tools else None

    gen_config = GenerationConfig(
        temperature=0.7,
        response_mime_type="application/json" if force_json else "text/plain",
    )

    model = GenerativeModel(
        MODEL,
        system_instruction=system,
        tools=vertex_tools,
        generation_config=gen_config,
    )

    history_contents = _history_to_contents(messages[:-1])
    last_msg = messages[-1]["content"] if messages else ""
    chat = model.start_chat(history=history_contents)
    current_message = last_msg

    for round_num in range(8):
        yield f"data: {json.dumps({'type': 'heartbeat', 'round': round_num})}\n\n"

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(chat.send_message, current_message),
                timeout=90.0
            )
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Request timed out after 90s — please try again.'})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        candidate = response.candidates[0]
        has_tool_calls = False
        text_parts = []
        tool_results = []

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            fn_name = getattr(fc, "name", None)
            if not fn_name:
                continue

            has_tool_calls = True
            fn_args = dict(fc.args) if fc.args else {}

            yield f"data: {json.dumps({'type': 'tool_call', 'tool': fn_name, 'args': fn_args})}\n\n"

            result = await asyncio.to_thread(dispatch_tool, fn_name, fn_args, user_id)

            yield f"data: {json.dumps({'type': 'tool_result', 'tool': fn_name, 'result': result[:300]})}\n\n"

            tool_results.append(Part.from_function_response(
                name=fn_name,
                response={"content": json.loads(result)},
            ))

        if text_parts:
            combined = "\n".join(text_parts)
            # Strip markdown fences and extract JSON if present
            combined = _extract_json(combined)
            yield f"data: {json.dumps({'type': 'text', 'content': combined})}\n\n"

        if not has_tool_calls:
            break

        current_message = tool_results if len(tool_results) > 1 else tool_results[0]

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


def _extract_json(text: str) -> str:
    """Strip markdown fences and extract the first JSON object if present."""
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    # If it looks like JSON, validate and return clean
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            return json.dumps(parsed)  # normalized, no extra whitespace
        except json.JSONDecodeError:
            # Find the last complete } in case of partial output
            last_brace = text.rfind("}")
            if last_brace > 0:
                candidate = text[:last_brace + 1]
                try:
                    parsed = json.loads(candidate)
                    return json.dumps(parsed)
                except Exception:
                    pass
    return text


# ---------------------------------------------------------------------------
# Orchestrator — routes to the right specialist
# ---------------------------------------------------------------------------

async def orchestrate(user_message: str, history: list, user_id: str) -> AsyncGenerator[str, None]:
    """
    Step 1: Ask orchestrator to classify intent (no streaming, fast call).
    Step 2: Route to appropriate specialist agent and stream response.
    """
    # Fast orchestration call — no tools, low temperature for deterministic routing
    orch_model = GenerativeModel(
        MODEL,
        system_instruction=ORCHESTRATOR_PROMPT,
        generation_config=GenerationConfig(temperature=0.1),
    )
    try:
        orch_response = await asyncio.wait_for(
            asyncio.to_thread(orch_model.generate_content, user_message),
            timeout=30.0
        )
        raw = orch_response.text.strip()
    except Exception:
        raw = '{"intent": "GENERAL"}'

    # Parse intent — handle markdown fences
    try:
        clean = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("```").strip()
        intent_obj = json.loads(clean)
        intent = intent_obj.get("intent", "GENERAL")
    except Exception:
        intent = "GENERAL"

    yield f"data: {json.dumps({'type': 'intent', 'intent': intent})}\n\n"

    # Build message history for specialist
    messages = history + [{"role": "user", "content": user_message}]

    # Route to specialist
    specialist_map = {
        "LOG_CHECKIN":   LOGGER_PROMPT,
        "ANALYZE_VIBE":  VIBE_ANALYZER_PROMPT,
        "PLAN_TRIP":     PLANNER_PROMPT,
        "VIEW_TRIPS":    None,   # handled inline below
        "LOAD_DEMO":     None,   # handled inline below
        "GENERAL":       GENERAL_PROMPT,
    }

    # Handle LOAD_DEMO inline
    if intent == "LOAD_DEMO":
        from tools import USER_DATA, SEED_TRIPS
        data = USER_DATA.setdefault(user_id, {"trips": {}, "vibe_profile": None})
        data["trips"] = dict(SEED_TRIPS)
        trip_names = list(SEED_TRIPS.keys())
        checkin_counts = [len(SEED_TRIPS[t]["checkins"]) for t in trip_names]
        summary_lines = [f"**{trip_names[i]}** — {checkin_counts[i]} check-ins" for i in range(len(trip_names))]
        msg = (
            "✅ Demo trips loaded! I've added the following trips to your profile:\n\n"
            + "\n".join(f"- {l}" for l in summary_lines)
            + "\n\nYou can now:\n- Say **\"analyze my vibe\"** to see your travel style\n"
            "- Say **\"plan me a trip to Barcelona\"** (or any city!) to get a personalized itinerary"
        )
        yield f"data: {json.dumps({'type': 'intent', 'intent': 'LOAD_DEMO'})}\n\n"
        yield f"data: {json.dumps({'type': 'text', 'content': msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # Handle VIEW_TRIPS inline — emit structured JSON for rich frontend rendering
    if intent == "VIEW_TRIPS":
        from tools import USER_DATA
        user_data = USER_DATA.get(user_id, {})
        trips = user_data.get("trips", {})
        if not trips:
            no_trips_msg = 'You have no trips logged yet! Say "load demo trips" to explore a sample, or start logging your own.'
            yield f"data: {json.dumps({'type': 'intent', 'intent': 'VIEW_TRIPS'})}\n\n"
            yield f"data: {json.dumps({'type': 'text', 'content': no_trips_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        # Emit as a special trip_journal event for rich rendering
        yield f"data: {json.dumps({'type': 'intent', 'intent': 'VIEW_TRIPS'})}\n\n"
        yield f"data: {json.dumps({'type': 'trip_journal', 'trips': trips})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # If user wants to plan or analyze but has no trips yet — intercept and offer demo
    if intent in ("PLAN_TRIP", "ANALYZE_VIBE"):
        from tools import USER_DATA, SEED_TRIPS
        user_data = USER_DATA.get(user_id, {})
        has_trips = bool(user_data.get("trips"))
        if not has_trips:
            msg = (
                "I'd love to help with that! To personalize your trip based on your travel style, "
                "I first need some data about your past travels.\n\n"
                "Would you like to:\n"
                "- 🗂️ **Load demo trips** — I'll add sample Japan & LA trips to show you how it works "
                "(just say **\"load demo trips\"**)\n"
                "- ✏️ **Log your own trips** — tell me about places you've visited and I'll build your profile\n\n"
                "Which would you prefer?"
            )
            yield f"data: {json.dumps({'type': 'intent', 'intent': intent})}\n\n"
            yield f"data: {json.dumps({'type': 'text', 'content': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

    system = specialist_map.get(intent, GENERAL_PROMPT)
    use_tools = intent != "GENERAL"

    if intent == "PLAN_TRIP":
        async for chunk in run_planner(messages, user_id):
            yield chunk
        return

    async for chunk in run_agent(system, messages, user_id, use_tools=use_tools):
        yield chunk


# ---------------------------------------------------------------------------
# Two-phase planner — separates tool calls from JSON generation
# ---------------------------------------------------------------------------

PLANNER_GATHER_PROMPT = """You are a travel data gatherer. Your ONLY job is to call tools to collect data needed for trip planning.

First, read the user's message carefully for any vibe override — phrases like:
- "luxury trip", "budget trip", "adventure trip", "romantic trip", "foodie trip"
- "more relaxed", "fast-paced", "off the beaten path", "touristy is fine"
- Any specific style that differs from their usual profile

Call these tools in order:
1. get_vibe_profile — get the user's stored travel style
2. get_weather — get weather for the destination and month the user mentioned
3. search_places — search 3-4 times for places matching the destination. If the user specified a vibe override, use that for searches instead of their stored profile (e.g. if they said "luxury", search for "luxury hotels Barcelona" not budget ones)
4. get_transport_options — get transport recommendations

After calling all tools, output a single line: DONE
Do not write any itinerary. Do not explain anything. Just call tools then say DONE.
"""

PLANNER_JSON_PROMPT = """You are a travel itinerary formatter. You will receive collected travel data and must output ONLY a JSON object.

Important: Read the user's original request carefully.
- If they specified a vibe override (e.g. "luxury trip", "romantic getaway", "adventure trip", "budget trip", "foodie focused"), PRIORITIZE that over their stored profile.
- If no override, use their stored vibe profile to personalize.
- The vibe_tags in your output should reflect the ACTUAL vibe of the trip (stored profile or override).
- The personalisation_note should acknowledge if you're doing something different: e.g. "Switching it up from your usual budget style — here's a luxury take on Barcelona!"

SLOT REQUIREMENTS PER DAY — this is mandatory:
Each day MUST have at least 5 slots in this order:
1. Morning — a sightseeing, culture, museum, or nature activity
2. Lunch — a specific restaurant or food spot (category: food)
3. Afternoon — a sightseeing, culture, shopping, or entertainment activity
4. Dinner — a specific restaurant or food spot (category: food), different from lunch
5. Evening — optional nightlife, dessert spot, bar, or evening activity
Breakfast is optional but encouraged — add it before Morning if relevant (e.g. a famous local cafe or market)

Food slots must name SPECIFIC restaurants or food spots, not generic descriptions like "find a local restaurant".
Reference the user's past food experiences to guide choices (e.g. if they loved ramen and street food, suggest similar casual spots).

Rules:
- Output ONLY valid JSON. No markdown. No prose. No explanation. No backticks.
- Start your response with { and end with }
- Follow this exact schema:

{
  "destination": "Barcelona, Spain",
  "duration_days": 4,
  "travel_month": "June",
  "weather_summary": "Warm and sunny, avg 25°C",
  "packing_tips": ["Sunscreen", "Light clothing"],
  "vibe_tags": ["foodie", "budget-friendly"],
  "personalisation_note": "Since you loved ramen spots in Japan and street food in LA...",
  "days": [
    {
      "day": 1,
      "title": "Gothic Quarter & Gaudí",
      "theme": "Culture & Food",
      "slots": [
        {
          "time_of_day": "Breakfast",
          "place_name": "Federal Café",
          "category": "food",
          "description": "Trendy all-day brunch spot in Eixample, great coffee and avocado toast",
          "why_it_fits": "Casual local vibe matches your konbini breakfast style in Japan",
          "transport": "walk",
          "est_cost": "$8–12",
          "image_search_query": "Federal Cafe Barcelona brunch"
        },
        {
          "time_of_day": "Morning",
          "place_name": "Sagrada Família",
          "category": "culture",
          "description": "Gaudí's unfinished masterpiece — book ahead for early entry",
          "why_it_fits": "Unique cultural landmark like Gotokuji Temple in Tokyo",
          "transport": "transit",
          "est_cost": "$25",
          "image_search_query": "Sagrada Familia Barcelona interior stained glass"
        },
        {
          "time_of_day": "Lunch",
          "place_name": "Bar del Pla",
          "category": "food",
          "description": "Classic Catalan tapas bar in El Born, famous for ham croquettes and local wine",
          "why_it_fits": "Budget-friendly local spot, similar energy to Joto Curry in Tokyo",
          "transport": "transit",
          "est_cost": "$12–18",
          "image_search_query": "Bar del Pla tapas Barcelona"
        },
        {
          "time_of_day": "Afternoon",
          "place_name": "Gothic Quarter streets",
          "category": "sightseeing",
          "description": "Wander the medieval lanes, find hidden plazas and the Barcelona Cathedral",
          "why_it_fits": "Love of urban wandering like your Sunset Boulevard stroll in LA",
          "transport": "walk",
          "est_cost": "Free",
          "image_search_query": "Gothic Quarter Barcelona narrow streets"
        },
        {
          "time_of_day": "Dinner",
          "place_name": "El Xampanyet",
          "category": "food",
          "description": "Legendary cava bar and tapas spot in El Born, packed with locals since 1929",
          "why_it_fits": "Local institution vibe like Canter's Deli in LA — unpretentious and packed",
          "transport": "walk",
          "est_cost": "$15–22",
          "image_search_query": "El Xampanyet Barcelona cava bar tapas"
        },
        {
          "time_of_day": "Evening",
          "place_name": "El Born waterfront walk",
          "category": "sightseeing",
          "description": "Stroll along the port and Barceloneta promenade as the sun sets",
          "why_it_fits": "Relaxed end to the day, like your Venice Beach evening walks",
          "transport": "walk",
          "est_cost": "Free",
          "image_search_query": "Barcelona port sunset promenade evening"
        }
      ]
    }
  ],
  "what_to_buy": ["Local olive oil from Boqueria", "Espadrilles from El Born"],
  "hidden_gems": ["Bunkers del Carmel for the best panoramic view of Barcelona"]
}
"""


async def run_planner(messages: list, user_id: str) -> AsyncGenerator[str, None]:
    """
    Two-phase planner:
    Phase 1 — gather data using tools (no JSON output constraint)
    Phase 2 — generate clean JSON from gathered data (no tools, JSON only)
    """
    yield f"data: {json.dumps({'type': 'intent', 'intent': 'PLAN_TRIP'})}\n\n"

    # ── Pre-load user context directly (don't rely on model calling the tool) ──
    from tools import USER_DATA, get_trip_history, get_vibe_profile
    trip_data   = get_trip_history(user_id)
    vibe_data   = get_vibe_profile(user_id)
    vibe_profile = vibe_data.get("vibe_profile")
    trips        = trip_data.get("trips", {})

    # If no vibe profile yet, synthesize a quick summary from raw trip data
    if not vibe_profile and trips:
        all_checkins = []
        for t in trips.values():
            all_checkins.extend(t.get("checkins", []))
        cats    = {}
        budgets = {}
        transports = {}
        for c in all_checkins:
            cats[c["category"]]       = cats.get(c["category"], 0) + 1
            budgets[c["price_tier"]]  = budgets.get(c["price_tier"], 0) + 1
            transports[c["transport"]]= transports.get(c["transport"], 0) + 1
        top_cat   = max(cats,       key=cats.get)       if cats       else "food"
        top_budget= max(budgets,    key=budgets.get)    if budgets    else "budget"
        top_trans = max(transports, key=transports.get) if transports else "transit"
        vibe_profile = {
            "interests":             [top_cat],
            "budget_style":          top_budget,
            "transport_preference":  top_trans,
            "vibe_summary":          f"Traveler who enjoys {top_cat}, prefers {top_budget} spending and gets around by {top_trans}",
            "tags":                  [top_cat, top_budget, top_trans],
            "pace":                  "moderate",
            "social_style":          "small group",
        }

    last_msg = messages[-1]["content"] if messages else ""

    # ── Phase 1: gather live data with tools ──────────────────
    vertex_tools = _build_vertex_tools(TOOL_DECLARATIONS)
    gather_model = GenerativeModel(
        MODEL,
        system_instruction=PLANNER_GATHER_PROMPT,
        tools=vertex_tools,
        generation_config=GenerationConfig(temperature=0.2),
    )

    history_contents = _history_to_contents(messages[:-1])
    chat = gather_model.start_chat(history=history_contents)
    current_message = last_msg
    gathered_context = []

    for round_num in range(8):
        yield f"data: {json.dumps({'type': 'heartbeat', 'round': round_num})}\n\n"
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(chat.send_message, current_message),
                timeout=60.0
            )
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        candidate = response.candidates[0]
        has_tool_calls = False
        tool_results = []

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                if "DONE" in part.text:
                    break

            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            fn_name = getattr(fc, "name", None)
            if not fn_name:
                continue

            has_tool_calls = True
            fn_args = dict(fc.args) if fc.args else {}

            yield f"data: {json.dumps({'type': 'tool_call', 'tool': fn_name, 'args': fn_args})}\n\n"

            result = await asyncio.to_thread(dispatch_tool, fn_name, fn_args, user_id)
            gathered_context.append(f"[{fn_name}]\n{result}")

            yield f"data: {json.dumps({'type': 'tool_result', 'tool': fn_name, 'result': result[:200]})}\n\n"

            tool_results.append(Part.from_function_response(
                name=fn_name,
                response={"content": json.loads(result)},
            ))

        if not has_tool_calls:
            break

        current_message = tool_results if len(tool_results) > 1 else tool_results[0]

    # ── Phase 2: generate JSON from gathered data ─────────────
    yield f"data: {json.dumps({'type': 'heartbeat', 'round': 99})}\n\n"

    # Build rich context string for the JSON model
    trip_names = list(trips.keys())
    place_names = []
    for t in trips.values():
        place_names += [c["place_name"] for c in t.get("checkins", [])[:5]]

    context_summary = f"""USER REQUEST: {last_msg}

USER VIBE PROFILE:
{json.dumps(vibe_profile, indent=2) if vibe_profile else "No profile yet — use trip data below"}

PAST TRIPS: {", ".join(trip_names) if trip_names else "None"}
PLACES THEY VISITED: {", ".join(place_names) if place_names else "None"}

LIVE GATHERED DATA:
{chr(10).join(gathered_context) if gathered_context else "No live data gathered"}

Use the vibe profile and past trips to personalise the itinerary. Reference specific past places in personalisation_note.
If the user requested a specific vibe (luxury, adventure, romantic, etc.), prioritise that over the stored profile."""

    json_model = GenerativeModel(
        MODEL,
        system_instruction=PLANNER_JSON_PROMPT,
        generation_config=GenerationConfig(
            temperature=0.7,
            response_mime_type="application/json",
        ),
    )

    try:
        json_response = await asyncio.wait_for(
            asyncio.to_thread(json_model.generate_content, context_summary),
            timeout=90.0
        )
        raw_json = json_response.text.strip()
        clean = _extract_json(raw_json)

        # Auto-save to planned trips
        try:
            parsed = json.loads(clean)
            from tools import save_planned_trip
            save_planned_trip(user_id, parsed)
        except Exception:
            pass  # Don't fail the whole response if saving fails

        yield f"data: {json.dumps({'type': 'text', 'content': clean})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'Failed to generate itinerary: {str(e)}'})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"

async def run_critic(itinerary: str, user_id: str) -> AsyncGenerator[str, None]:
    """
    Generator-critic pattern: reviews the draft itinerary against the user's vibe profile.
    Human-in-the-loop: surfaces flagged items to the user for approval.
    """
    from tools import get_vibe_profile
    profile_result = get_vibe_profile(user_id)
    profile = profile_result.get("vibe_profile")

    if not profile:
        yield f"data: {json.dumps({'type': 'text', 'content': 'No vibe profile found — skipping critic review.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    critic_messages = [{
        "role": "user",
        "content": f"Vibe Profile:\n{json.dumps(profile, indent=2)}\n\nDraft Itinerary:\n{itinerary}\n\nPlease review this itinerary against my travel profile and flag any mismatches."
    }]

    async for chunk in run_agent(CRITIC_PROMPT, critic_messages, user_id, use_tools=False):
        yield chunk


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_message: str = body.get("message", "")
    history: list = body.get("history", [])
    user_id: str = body.get("user_id", "default")

    return StreamingResponse(
        orchestrate(user_message, history, user_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/critic")
async def critic(request: Request):
    body = await request.json()
    itinerary: str = body.get("itinerary", "")
    user_id: str = body.get("user_id", "default")

    return StreamingResponse(
        run_critic(itinerary, user_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/trips/{user_id}")
async def get_trips(user_id: str):
    from tools import get_trip_history, get_vibe_profile
    return {
        "trips": get_trip_history(user_id),
        "vibe_profile": get_vibe_profile(user_id),
    }


@app.get("/planned-trips/{user_id}")
async def get_planned_trips_route(user_id: str):
    from tools import get_planned_trips
    return get_planned_trips(user_id)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "project": VERTEX_PROJECT, "location": VERTEX_LOCATION}