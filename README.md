# VibeTrip — Your Travel Style, Learned

An AI-powered travel agent that learns your travel personality from past trips and uses it to plan hyper-personalized itineraries for new destinations. Log where you've been, analyze your vibe, and get a trip plan that actually matches how you travel.

**Live URL:** `https://travel-vibe-agent-203861284458.us-central1.run.app`  
**Model:** `gemini-2.5-flash` via Vertex AI  
**Stack:** FastAPI · Python 3.11 · Vertex AI SDK · Vanilla JS + HTML

---

## Project Structure

```
├── app.py          # FastAPI server, all agent definitions, orchestration logic
├── tools.py        # Tool functions, in-memory data store, seed trip data
├── index.html      # Single-page web UI with SSE streaming and rich card rendering
├── static/         # Static file directory served at /static
├── pyproject.toml  # Dependencies (managed with uv)
└── .env            # Environment variables (not committed)
```

---

## Running Locally

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- A Google Cloud project with Vertex AI API enabled
- GCP credentials configured locally (`gcloud auth application-default login`)

### Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/travel-vibe-agent.git
cd travel-vibe-agent

# Install dependencies
uv sync

# Create .env file
cp .env.example .env
# Fill in your values (see below)
```

### Environment Variables

Create a `.env` file in the project root:

```env
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1
GOOGLE_PLACES_API_KEY=your-key-here   # optional — falls back to mock data
```

### Run

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Then open `http://localhost:8080` in your browser.

### Quick Demo Flow

1. Click **"Load demo trips"** in the welcome screen or sidebar — seeds Japan 2025 and Los Angeles 2025 trip data
2. Say **"Analyze my travel vibe"** — the Vibe Analyzer reads all past trips and builds your traveler profile
3. Say **"Plan me a 4 day trip to Barcelona for June"** — generates a personalized itinerary as visual journey cards
4. Click **"Review it"** in the Critic panel — the Generator-Critic agent checks the itinerary against your profile and flags mismatches
5. Say **"Plan me a creator trip to Tokyo"** — switches to Creator Mode with photo spots, content tips, and aesthetic food picks
6. View saved plans anytime from the **"✈ Planned trips"** section in the sidebar

---

## Deploying to GCP

```bash
# Build and deploy to Cloud Run
gcloud run deploy travel-vibe-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --set-env-vars VERTEX_PROJECT=your-project-id,VERTEX_LOCATION=us-central1
```

---

## Class Concepts Implemented

### 1. Orchestrator + Router Pattern
**File:** `app.py`  
**Functions:** `orchestrate()`, `ORCHESTRATOR_PROMPT`

A lightweight Orchestrator agent runs first on every user message. It classifies intent into one of six categories (`LOG_CHECKIN`, `ANALYZE_VIBE`, `PLAN_TRIP`, `VIEW_TRIPS`, `LOAD_DEMO`, `GENERAL`) using a low-temperature call (0.1), then routes to the appropriate specialist. Simple intents like `VIEW_TRIPS` and `LOAD_DEMO` are handled inline without an LLM call at all, saving latency and tokens.

```
User message
     │
     ▼
Orchestrator (classifies intent)
     │
     ├── LOG_CHECKIN   → Logger Agent        (app.py: LOGGER_PROMPT)
     ├── ANALYZE_VIBE  → Vibe Analyzer Agent (app.py: VIBE_ANALYZER_PROMPT)
     ├── PLAN_TRIP     → run_planner()       (app.py: run_planner)
     ├── VIEW_TRIPS    → inline handler      (app.py: orchestrate)
     └── GENERAL       → General Agent       (app.py: GENERAL_PROMPT)
```

---

### 2. Multi-Agent Architecture (Splitting Work Across Agents)
**File:** `app.py`  
**Functions:** `run_agent()`, `run_planner()`, `run_critic()`  
**Prompts:** `LOGGER_PROMPT`, `VIBE_ANALYZER_PROMPT`, `PLANNER_GATHER_PROMPT`, `PLANNER_JSON_PROMPT`, `CRITIC_PROMPT`

Five distinct specialist agents each have a focused system prompt, their own tool access, and a single responsibility:

| Agent | Responsibility | Tools |
|---|---|---|
| Logger Agent | Save check-ins from user messages | `save_checkin` |
| Vibe Analyzer | Synthesize travel style from trip history | `get_trip_history`, `save_vibe_profile` |
| Planner (Gather phase) | Collect live destination data | `get_vibe_profile`, `get_weather`, `search_places`, `get_transport_options` |
| Planner (JSON phase) | Format collected data as structured itinerary | none (JSON output only) |
| Critic Agent | Review itinerary against traveler profile | none |

Each agent is invoked via `run_agent()` which implements an agentic tool-calling loop (up to 8 rounds) using `asyncio.to_thread()` to keep FastAPI's event loop unblocked.

---

### 3. Handoffs vs. Agent-as-Tool
**File:** `app.py`  
**Functions:** `orchestrate()`, `run_planner()`

The project demonstrates both inter-agent communication patterns:

- **Handoff pattern** — the Orchestrator fully hands off control to a specialist via `run_agent()`. The specialist takes over the conversation, calls its own tools, and streams the response directly to the user. Used for `LOG_CHECKIN`, `ANALYZE_VIBE`, and `GENERAL`.

- **Agent-as-tool pattern** — the two-phase Planner (`run_planner()`) calls the gather phase as a sub-process, collects all tool results, then passes the entire context as input to the JSON formatter phase. The gather agent is effectively a tool used by the orchestration layer. This is also why `get_vibe_profile` and `save_vibe_profile` are exposed as callable tools — the Vibe Analyzer invokes them mid-reasoning rather than receiving them as pre-loaded context.

---

### 4. Human-in-the-Loop Checkpoints
**File:** `app.py`, `index.html`  
**Functions:** `run_critic()`, `CRITIC_PROMPT`  
**Frontend:** `offerCriticReview()`, `runCritic()`, `approvePlan()`

After the Planner generates an itinerary, the UI surfaces a **Critic Agent review panel** — the user can approve or skip it. If they choose to review, the Critic Agent (`run_critic()`) reads the draft itinerary and the stored vibe profile, then flags any mismatches (wrong price tier, tourist traps, pacing issues). Flagged items are presented back to the user with an **approve/regenerate choice** before the plan is finalized. The itinerary is not saved until the user explicitly approves.

This implements the class concept of pausing the agent loop at sensitive decision points to require human confirmation, rather than acting autonomously.

---

### 5. Deep Research Pattern
**File:** `app.py`  
**Functions:** `run_planner()`, `PLANNER_GATHER_PROMPT`, `VIBE_ANALYZER_PROMPT`

Two agents implement the deep research pattern — multi-step analysis before producing a final answer:

**Vibe Analyzer** — does not answer immediately. It first calls `get_trip_history`, then reasons across all check-ins (categories, price tiers, transport modes, pacing, notes) before synthesizing a structured traveler profile. It surfaces specific patterns rather than generic summaries.

**Two-phase Planner** — explicitly separates research from synthesis:
- *Phase 1 (Gather):* Calls `get_vibe_profile` → `get_weather` → 3–4 `search_places` calls → `get_transport_options`. Collects all data before forming any opinion.
- *Phase 2 (Synthesize):* Receives the full gathered context and produces the structured JSON itinerary in a single focused call with `response_mime_type: application/json` enforced.

This separation prevents the model from generating an itinerary before it has sufficient grounding data.

---

### 6. Generator-Critic Pattern
**File:** `app.py`  
**Functions:** `run_planner()`, `run_critic()`, `CRITIC_PROMPT`

The Planner (generator) produces a draft itinerary. A separate Critic Agent then independently reviews it against the user's vibe profile, scoring the match and flagging specific items. This is the same generator-critic pattern from class — one agent drafts, a second evaluates, and the human decides whether to accept or iterate.

The Critic checks for: price tier mismatches, transport mode inconsistencies, tourist-trap recommendations that contradict the user's "local spots" preference, and pacing issues relative to the user's historical pace.

---

### 7. Persistent Memory
**File:** `tools.py`  
**Functions:** `_user()`, `save_checkin()`, `save_vibe_profile()`, `save_planned_trip()`, `get_trip_history()`, `get_vibe_profile()`, `get_planned_trips()`

User data persists across the entire session in an in-memory key-value store (`USER_DATA` dict in `tools.py`). Three types of memory are maintained per user:

- **Trip memory** — every check-in is stored with category, price tier, transport, date, and personal notes
- **Vibe profile memory** — the synthesized traveler profile (pace, budget style, interests, transport preference, tags) persists and is reused across planning requests without re-analysis
- **Planned trip memory** — every successfully generated itinerary is auto-saved and viewable from the sidebar

In production this would be replaced with a persistent store (PostgreSQL, Redis, or Firestore).

---

### 8. Tool Calling
**File:** `tools.py`, `app.py`  
**Functions:** `dispatch_tool()`, `TOOL_DECLARATIONS`, `_build_vertex_tools()`

Eight tools are registered and callable by agents at runtime:

| Tool | File | Purpose |
|---|---|---|
| `save_checkin` | `tools.py` | Persist a place visit with category, price tier, transport, date, notes |
| `get_trip_history` | `tools.py` | Retrieve all past trips and check-ins for a user |
| `get_vibe_profile` | `tools.py` | Read the stored traveler profile |
| `save_vibe_profile` | `tools.py` | Write the synthesized traveler profile after analysis |
| `save_planned_trip` | `tools.py` | Auto-save a completed itinerary to the user's planned trips list |
| `get_planned_trips` | `tools.py` | Retrieve all AI-planned itineraries for sidebar display |
| `search_places` | `tools.py` | Query Google Places API (or mock fallback) for real venues |
| `get_transport_options` | `tools.py` | Return transport recommendations matched to user's preferred mode |
| `get_weather` | `tools.py` | Fetch historical climate averages from Open-Meteo API — used at planning time |

Tool declarations are defined in `TOOL_DECLARATIONS` (`tools.py`) and converted to Vertex AI `FunctionDeclaration` objects by `_build_vertex_tools()` (`app.py`). Tool dispatch is handled by `dispatch_tool()` which routes by name and injects `user_id` context. The in-memory store (`USER_DATA` dict, `tools.py`) is keyed by `user_id` and holds three namespaces per user: `trips`, `vibe_profile`, and `planned_trips`.

---

### 9. Constrained / Structured Output
**File:** `app.py`  
**Functions:** `run_planner()`, `run_creator_planner()`, `_extract_json()`, `PLANNER_JSON_PROMPT`, `CREATOR_JSON_PROMPT`

Both the Planner and Creator Planner use a two-phase approach to enforce JSON output. Phase 2 uses `response_mime_type: "application/json"` in Vertex AI's `GenerationConfig` — this is intentionally separated from Phase 1 (tool calling) because Vertex AI does not support both simultaneously. The `_extract_json()` post-processor strips any residual markdown fences as a safety net. The frontend detects JSON responses by checking for `{` and `"days"` and renders them as rich journey cards.

---

### 10. Creator Trip Mode (Specialist Agent Variant)
**File:** `app.py`  
**Functions:** `run_creator_planner()`, `CREATOR_GATHER_PROMPT`, `CREATOR_JSON_PROMPT`  
**Intent:** `CREATOR_TRIP` (classified by `ORCHESTRATOR_PROMPT`)

A dedicated trip planning mode for content creators and influencers, implemented as a fully separate two-phase agent with its own gather and JSON prompts. The orchestrator classifies requests mentioning photography, Instagram, aesthetic food, or influencer travel as `CREATOR_TRIP` and routes to `run_creator_planner()`.

The Creator gather phase runs 5–6 targeted searches for photogenic spots, Instagram-famous locations, aesthetic cafes, hidden gem photo locations, and visually stunning food. The JSON phase produces itinerary slots with three additional fields not present in regular trips:

- `best_time_to_shoot` — optimal lighting window (golden hour, blue hour, soft morning light)
- `content_tip` — specific framing, angle, and composition advice per location
- `crowd_level` — expected crowd density at the recommended shooting time

The frontend renders Creator trips with a distinct dark purple `#1a1a2e` theme and a `📸 CREATOR MODE` badge, plus additional sections for Top Photo Spots, Aesthetic Food Picks, and a Content Calendar summary.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve the web UI |
| `POST` | `/chat` | Main chat endpoint — streams SSE events |
| `POST` | `/critic` | Run critic agent on a draft itinerary |
| `GET` | `/trips/{user_id}` | Get past trips and vibe profile for a user |
| `GET` | `/planned-trips/{user_id}` | Get all AI-planned itineraries for a user |
| `GET` | `/health` | Health check with model and project info |

---

## Architecture Diagram

```
Browser (index.html)
        │  POST /chat (SSE stream)
        ▼
FastAPI (app.py)
        │
        ▼
   orchestrate()
   ┌─────────────────────────────────────────────────────┐
   │  ORCHESTRATOR_PROMPT (Gemini 2.5 Flash, t=0.1)      │
   │  Classifies intent → routes to specialist            │
   └────────┬────────────────────────────────────────────┘
            │
    ┌───────┴──────┬──────────────┬───────────────┬──────────────┐
    ▼              ▼              ▼               ▼              ▼
Logger Agent  Vibe Analyzer  run_planner()  run_creator_   Critic Agent
(run_agent)   (run_agent)                  planner()      (run_critic)
    │              │          ┌──────────┐  ┌──────────┐
    ▼              ▼          │ Phase 1  │  │ Phase 1  │
tools.py      tools.py        │ Gather   │  │ Creator  │
save_checkin  get_trip_       │ (tools)  │  │ Gather   │
              history         │          │  │ (tools)  │
              save_vibe_      │ Phase 2  │  │          │
              profile         │ JSON LLM │  │ Phase 2  │
                              └────┬─────┘  │ JSON LLM │
                                   │        └────┬─────┘
                              tools.py           │
                         search_places      tools.py
                         get_weather        search_places
                         get_transport_     get_weather
                         options            save_planned_trip
                         save_planned_trip
```

---

## Known Limitations

- **In-memory storage** — all user data resets on server restart. Production would use a persistent database (PostgreSQL, Redis, or Firestore).
- **Images** — activity photos come from [Picsum Photos](https://picsum.photos) (consistent random photos seeded by place name). Production would use Google Places Photos API for real venue imagery.
- **Location tracking** — check-ins are manual. A production app would use GPS from a native mobile client.
- **Creator photo spots** — photo location data comes from Google Places text search. A production version would integrate with a dedicated photography spots API or curated database.