import os
import json
import re
import requests
from contextlib import asynccontextmanager
from dotenv import load_dotenv  # type: ignore
from fastapi import FastAPI, Query  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from vector_store import VectorStore

# Load environment variables from .env file
load_dotenv()

store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    # Single global Qdrant client connection during startup
    store = VectorStore()
    yield
    # Safely release Qdrant local database lock on shutdown
    if store and hasattr(store, "qdrant"):
        store.qdrant.close()


app = FastAPI(lifespan=lifespan)


def parse_model_json(content: str, fallback_context: dict) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Attempt 1: Standard json.loads with strict=False
    try:
        data = json.loads(cleaned, strict=False)
        if isinstance(data, dict) and "answer" in data:
            return {
                "is_relevant": bool(data.get("is_relevant", True)),
                "answer": str(data.get("answer", "")).strip(),
                "selected_timestamp_seconds": int(data.get("selected_timestamp_seconds", fallback_context.get("start", 0))),
                "excerpt": str(data.get("excerpt", "")).strip(),
            }
    except Exception:
        pass

    # Attempt 2: Regex extraction of JSON fields
    try:
        answer_match = re.search(r'"answer"\s*:\s*"(.*?)"\s*,\s*"selected_timestamp_seconds"', cleaned, re.DOTALL)
        ts_match = re.search(r'"selected_timestamp_seconds"\s*:\s*(\d+)', cleaned)
        excerpt_match = re.search(r'"excerpt"\s*:\s*"(.*?)"\s*\}', cleaned, re.DOTALL)

        if answer_match:
            ans = answer_match.group(1).replace("\\n", "\n").replace('\\"', '"').strip()
            ts = int(ts_match.group(1)) if ts_match else int(fallback_context.get("start", 0))
            exc = excerpt_match.group(1).replace("\\n", "\n").replace('\\"', '"').strip() if excerpt_match else ""
            return {
                "is_relevant": True,
                "answer": ans,
                "selected_timestamp_seconds": ts,
                "excerpt": exc,
            }
    except Exception:
        pass

    # Fallback if no valid JSON structure could be extracted
    return {
        "is_relevant": True,
        "answer": cleaned.replace("**", "").strip(),
        "selected_timestamp_seconds": int(fallback_context.get("start", 0)),
        "excerpt": fallback_context.get("text", "")[:150],
    }


def synthesize_answer(query: str, contexts: list, target_title: str) -> dict:
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()

    default_fail = {
        "is_relevant": False,
        "answer": "GROQ API Key missing. Please check your .env file.",
        "selected_timestamp_seconds": 0,
        "excerpt": "",
    }

    if not groq_key:
        return default_fail

    if not contexts:
        return {
            "is_relevant": False,
            "answer": "No relevant video transcript segments were found for this question.",
            "selected_timestamp_seconds": 0,
            "excerpt": "",
        }

    formatted_contexts = []
    for i, c in enumerate(contexts, start=1):
        start_sec = int(c.get("start", 0))
        end_sec = int(c.get("end", 0))
        m, s = start_sec // 60, start_sec % 60
        text = c.get("text", "").strip()
        formatted_contexts.append(
            f"Snippet [{i}] | Start Time: {start_sec}s ({m:02d}:{s:02d}) | End Time: {end_sec}s\nTranscript: \"{text}\""
        )

    context_str = "\n\n".join(formatted_contexts)

    prompt = f"""You are an expert Data Structures & Algorithms (DSA) AI Assistant and Tutor.
Provide a clear, direct, to-the-point answer based strictly on the provided video transcript snippets.

CRITICAL INSTRUCTIONS FOR ACCURACY:
1. BRIEF & IN POINTS: Format the answer as 3 to 5 clear, bulleted points. Keep each point focused and directly explaining the technical solution.
2. STRICT GROUNDING: Use ONLY facts explicitly stated in the transcript snippets.
3. TIMESTAMP SELECTION RULES:
   - IGNORE INTRO BANTER: Ignore introductory greetings, channel intros, small talk, or generic warm-ups (e.g., 'Hello kids welcome', 'Today my voice is hoarse'). Do NOT pick 0s if it is just a greeting!
   - SELECT TECHNICAL START: Find the exact snippet where the actual technical explanation, problem breakdown, or solution algorithm begins. Set 'selected_timestamp_seconds' to the exact start time integer (in seconds) of that specific snippet.
4. SINGLE EXCERPT: Set 'excerpt' to a 1-sentence concise summary or key quote from that exact selected snippet.
5. LANGUAGE MATCHING: Match the language of the user's question (English, Hinglish, or Hindi).

OUTPUT FORMAT:
Respond strictly with a valid JSON object matching this schema:
{{
  "is_relevant": true,
  "answer": "- **Step 1**: Concise explanation\\n- **Step 2**: Concise explanation\\n- **Step 3**: Concise explanation",
  "selected_timestamp_seconds": 263,
  "excerpt": "Concise key quote from transcript..."
}}
If the snippets do not contain information to answer the question, set "is_relevant": false, "answer": "This specific topic is not covered in the indexed video transcript.", "selected_timestamp_seconds": 0, "excerpt": "".

VIDEO TITLE: {target_title}

TRANSCRIPT SNIPPETS:
{context_str}

USER QUESTION:
{query}

Respond strictly with valid JSON:"""

    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 650,
            },
            timeout=15,
        )

        data = res.json()
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"].get("content", "").strip()
            return parse_model_json(content, contexts[0])

        err_msg = data.get("error", {}).get("message", "Unknown error")
        return {
            "is_relevant": False,
            "answer": f"Groq API Error: {err_msg}",
            "selected_timestamp_seconds": 0,
            "excerpt": "",
        }
    except Exception as e:
        return {
            "is_relevant": False,
            "answer": f"Failed to communicate with LLM engine: {e}",
            "selected_timestamp_seconds": 0,
            "excerpt": "",
        }


@app.get("/api/search")
def api_search(q: str = Query(...)):
    if store is None:
        return {
            "is_relevant": False,
            "summary": "Vector store not initialized.",
            "video_id": None,
            "title": None,
            "start": 0,
            "start_formatted": "00:00",
            "description": "",
        }

    query_vector = store.encoder.encode(q).tolist()

    # Query vector DB for top 25 candidate hits
    response = store.qdrant.query_points(
        collection_name=store.collection_name,
        query=query_vector,
        limit=25,
    )

    hits_with_scores = [(point.payload, point.score) for point in response.points]

    if not hits_with_scores:
        return {
            "is_relevant": False,
            "summary": "No relevant video segment was found for this question.",
            "video_id": None,
            "title": None,
            "start": 0,
            "start_formatted": "00:00",
            "description": "",
        }

    # Video Relevance Reranking: Calculate cumulative relevance score per video
    video_scores = {}
    video_hits_map = {}
    video_title_map = {}

    for payload, score in hits_with_scores:
        v_id = payload.get("video_id")
        if not v_id:
            continue

        if v_id not in video_scores:
            video_scores[v_id] = 0.0
            video_hits_map[v_id] = []
            video_title_map[v_id] = payload.get("title", "DSA Video")

        video_hits_map[v_id].append(payload)
        # Add score with density weight for multiple relevant hits
        video_scores[v_id] += float(score)

    # Select the video with highest cumulative relevance score
    best_video_id = max(video_scores.keys(), key=lambda vid: video_scores[vid])
    best_title = video_title_map[best_video_id]

    # Isolate candidate snippets from this single video (sorted by timestamp)
    single_video_hits = sorted(
        video_hits_map[best_video_id][:6],
        key=lambda h: int(h.get("start", 0))
    )

    # Synthesize answer and pinpoint exact timestamp using LLM
    synthesis = synthesize_answer(q, single_video_hits, best_title)

    # Determine exact start timestamp in seconds
    chosen_start = synthesis.get("selected_timestamp_seconds")
    valid_starts = [int(h.get("start", 0)) for h in single_video_hits]

    # Filter out 0s intro timestamp if a non-zero technical segment is available in candidate hits
    non_zero_starts = [s for s in valid_starts if s > 40]
    if (chosen_start <= 40 or chosen_start not in valid_starts) and non_zero_starts:
        chosen_start = non_zero_starts[0]
    elif chosen_start not in valid_starts:
        chosen_start = valid_starts[0] if valid_starts else 0

    m, s = chosen_start // 60, chosen_start % 60
    start_formatted = f"{m:02d}:{s:02d}"

    # Determine description corresponding strictly to the 1 chosen video segment
    matching_snippet = next((h for h in single_video_hits if int(h.get("start", 0)) == chosen_start), single_video_hits[0])
    description = synthesis.get("excerpt") or matching_snippet.get("text", "")

    return {
        "is_relevant": synthesis.get("is_relevant", True),
        "summary": synthesis.get("answer", "No answer available."),
        "video_id": best_video_id,
        "title": best_title,
        "start": chosen_start,
        "start_formatted": start_formatted,
        "description": description,
    }


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DSA Video Knowledge Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; }
        .prose p { margin-bottom: 0.6rem; line-height: 1.6; color: #cbd5e1; }
        .prose ul { list-style-type: none; padding-left: 0; margin-bottom: 0.75rem; }
        .prose li { position: relative; padding-left: 1.5rem; margin-bottom: 0.5rem; line-height: 1.55; color: #e2e8f0; }
        .prose li::before { content: "•"; position: absolute; left: 0.3rem; top: 0; color: #818cf8; font-weight: bold; font-size: 1.2em; }
        .prose code { background-color: #1e293b; color: #38bdf8; padding: 0.15rem 0.4rem; border-radius: 0.25rem; font-family: monospace; font-size: 0.875em; border: 1px solid #334155; }
        .prose strong { color: #818cf8; font-weight: 700; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-4 md:p-8 max-w-7xl mx-auto selection:bg-indigo-500 selection:text-white">
    
    <!-- Top Header -->
    <header class="mb-8 border-b border-slate-800/80 pb-5">
        <div class="flex items-center justify-between">
            <div>
                <h1 class="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent flex items-center gap-3">
                    <span>⚡</span> DSA Video Knowledge Assistant
                </h1>
                <p class="text-slate-400 text-sm mt-1">Instant DSA problem answers backed by exact video timestamp playback</p>
            </div>
            <span class="hidden md:inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold">
                <span class="w-2 h-2 rounded-full bg-indigo-400 animate-ping"></span> Live RAG Engine
            </span>
        </div>
    </header>
    
    <!-- Search Bar -->
    <div class="relative mb-8">
        <div class="flex gap-3">
            <div class="relative flex-1">
                <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-slate-500">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                </div>
                <input type="text" id="qInput" placeholder="Ask a DSA doubt e.g., How does sliding window pattern work?" 
                       class="w-full bg-slate-900 border border-slate-800 rounded-xl pl-11 pr-4 py-3.5 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 transition-all text-base shadow-inner">
            </div>
            <button onclick="search()" class="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white px-8 py-3.5 rounded-xl font-semibold transition-all shadow-lg shadow-indigo-600/25 flex items-center gap-2 text-base active:scale-95">
                <span>Search</span>
            </button>
        </div>
    </div>
    
    <!-- Status Loading Indicator -->
    <div id="status" class="hidden mb-6 p-4 rounded-xl bg-slate-900/90 border border-indigo-500/30 text-indigo-300 text-sm animate-pulse flex items-center gap-3 shadow-lg">
        <svg class="animate-spin h-5 w-5 text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <span>Searching video transcripts & pinpointing exact solution timestamp...</span>
    </div>

    <!-- Main Content Layout -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <!-- Left Column: Answer & Video Description -->
        <div class="lg:col-span-5 flex flex-col gap-6">
            
            <!-- AI Answer Box -->
            <div id="aiBox" class="bg-slate-900 p-6 rounded-2xl border border-indigo-500/20 shadow-2xl hidden relative overflow-hidden backdrop-blur-sm">
                <div class="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 rounded-full blur-2xl pointer-events-none"></div>
                <div class="flex items-center gap-2.5 mb-4 border-b border-slate-800 pb-3">
                    <div class="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 font-bold">💡</div>
                    <div>
                        <h2 class="text-indigo-400 font-bold text-lg leading-none">AI Answer</h2>
                        <span class="text-slate-500 text-xs font-medium">Synthesized from video solution</span>
                    </div>
                </div>
                <div id="aiText" class="prose text-slate-200 text-sm leading-relaxed"></div>
            </div>

            <!-- Target Video Info Badge -->
            <div id="badgeBox" class="bg-slate-900/90 p-4 rounded-xl border border-slate-800 shadow-md hidden flex items-center justify-between">
                <div class="flex items-center gap-3 overflow-hidden">
                    <span class="bg-indigo-600 text-white font-mono px-3 py-1 rounded-lg text-xs font-extrabold flex items-center gap-1 shadow-sm shrink-0" id="tsTag">
                        <span>▶</span> <span id="tsVal">00:00</span>
                    </span>
                    <span id="vTitleBadge" class="text-sm font-semibold text-slate-200 truncate">Video Title</span>
                </div>
            </div>

            <!-- Single Video Description Box -->
            <div id="descBox" class="bg-slate-900 p-5 rounded-2xl border border-slate-800 shadow-xl hidden">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-slate-400 text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                        <span>📝</span> Solution Segment Excerpt
                    </h3>
                </div>
                <p id="descText" class="text-sm text-slate-300 italic leading-relaxed bg-slate-950/80 p-4 rounded-xl border-l-4 border-indigo-500 border-y border-r border-slate-800/80"></p>
            </div>

        </div>

        <!-- Right Column: Embedded Player -->
        <div class="lg:col-span-7">
            <div class="sticky top-6 bg-slate-900 p-4 rounded-2xl border border-slate-800 shadow-2xl">
                <div class="flex items-center justify-between mb-3 px-1">
                    <h2 id="vTitle" class="text-sm font-bold truncate text-indigo-300 flex items-center gap-2">
                        <span class="text-red-500">▶</span> <span class="truncate">Search a question to play video</span>
                    </h2>
                    <span id="timeDisplay" class="text-xs bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 px-3 py-1 rounded-full font-mono font-semibold hidden">00:00</span>
                </div>
                <div class="aspect-video bg-black rounded-xl overflow-hidden relative shadow-2xl border border-slate-950">
                    <iframe id="player" class="w-full h-full" src="" 
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                            referrerpolicy="strict-origin-when-cross-origin"
                            allowfullscreen></iframe>
                </div>
            </div>
        </div>

    </div>

    <script>
        function playVideo(videoId, startSec, title, formattedTime) {
            const player = document.getElementById('player');
            const vTitle = document.getElementById('vTitle');
            const timeDisplay = document.getElementById('timeDisplay');
            
            if (!videoId) {
                player.src = "";
                vTitle.innerText = "No matching video found";
                timeDisplay.classList.add('hidden');
                return;
            }

            const startInt = Math.floor(startSec || 0);
            vTitle.innerHTML = `<span class="text-red-500">▶</span> <span class="truncate">${title}</span>`;
            timeDisplay.innerText = `Timestamp: ${formattedTime}`;
            timeDisplay.classList.remove('hidden');

            const embedUrl = `https://www.youtube-nocookie.com/embed/${videoId}?start=${startInt}&autoplay=1&enablejsapi=1`;
            player.src = embedUrl;
        }

        async function search() {
            const q = document.getElementById('qInput').value.trim();
            if(!q) return;

            // Hide boxes & show status
            document.getElementById('aiBox').classList.add('hidden');
            document.getElementById('badgeBox').classList.add('hidden');
            document.getElementById('descBox').classList.add('hidden');
            
            const status = document.getElementById('status');
            status.classList.remove('hidden');

            try {
                const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
                const data = await res.json();

                status.classList.add('hidden');

                // Render Markdown Answer safely
                document.getElementById('aiText').innerHTML = marked.parse(data.summary || "");
                document.getElementById('aiBox').classList.remove('hidden');

                if (data.video_id) {
                    // Update Badge
                    document.getElementById('tsVal').innerText = data.start_formatted;
                    document.getElementById('vTitleBadge').innerText = data.title;
                    document.getElementById('badgeBox').classList.remove('hidden');

                    // Update Single Video Description
                    document.getElementById('descText').innerText = `"${data.description}"`;
                    document.getElementById('descBox').classList.remove('hidden');

                    // Play Video at exact start timestamp
                    playVideo(data.video_id, data.start, data.title, data.start_formatted);
                } else {
                    playVideo(null, 0, "", "");
                }
            } catch (err) {
                status.classList.add('hidden');
                document.getElementById('aiText').innerText = "An error occurred while fetching the answer.";
                document.getElementById('aiBox').classList.remove('hidden');
            }
        }

        document.getElementById('qInput').addEventListener('keypress', function (e) {
            if (e.key === 'Enter') search();
        });
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)