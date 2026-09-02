"""
CareerPath AI  🎓
A student career-guidance chatbot.

What it does:
  1. Student fills in a short profile (course, subjects, interests, skills, strengths).
  2. An AI model suggests career paths, a learning roadmap, and courses.
  3. It searches the live web (DuckDuckGo, no API key needed) for real internship
     listings and shows clickable links.
  4. A chat tab lets the student ask follow-up questions with their profile as context.

WHICH AI MODEL IS USED IS A BACKEND DECISION (see CONFIG below).
The end user never picks a model. By default the app uses the FREE Llama model on
Hugging Face. If whoever deploys the app adds an OpenAI (paid) key, it uses that
instead. Nothing about the model shows up in the user interface.
"""

import streamlit as st

# ==================================================================
# BACKEND CONFIG  — the person deploying edits THIS section only.
# ==================================================================
# Which AI to use. Choose one:
#   "auto"   -> use OpenAI or Claude if that paid key is configured,
#               otherwise fall back to the FREE Llama model. (recommended)
#   "llama"  -> always use the free Llama model on Hugging Face.
#   "openai" -> always use OpenAI (paid).
#   "claude" -> always use Claude / Anthropic (paid).
PREFERRED_PROVIDER = "auto"

# Model names for each provider (change only if you know what you're doing).
OPENAI_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-sonnet-4-6"
LLAMA_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"
# ==================================================================


# ------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="CareerPath AI", page_icon="🎓", layout="wide")


# ------------------------------------------------------------------
# KEY + PROVIDER HELPERS
# ------------------------------------------------------------------
def _secret(name: str) -> str:
    """Read a key from Streamlit secrets. Returns '' if not set."""
    try:
        return st.secrets[name]
    except Exception:
        return ""


def resolve_provider() -> str:
    """Decide which provider to use, purely on the backend."""
    pref = PREFERRED_PROVIDER.lower().strip()
    if pref in ("openai", "claude", "llama"):
        return pref
    # "auto": prefer a configured paid key, else fall back to free Llama.
    if _secret("OPENAI_API_KEY"):
        return "openai"
    if _secret("ANTHROPIC_API_KEY"):
        return "claude"
    return "llama"


# ------------------------------------------------------------------
# MODEL CALLERS  (one per provider) — all return a plain string.
# ------------------------------------------------------------------
def call_openai(api_key, system_prompt, messages):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    full = [{"role": "system", "content": system_prompt}] + messages
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, messages=full, temperature=0.7, max_tokens=2000)
    return resp.choices[0].message.content


def call_claude(api_key, system_prompt, messages):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=2000, system=system_prompt, messages=messages)
    return resp.content[0].text


def call_llama(api_key, system_prompt, messages):
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=api_key)
    full = [{"role": "system", "content": system_prompt}] + messages
    resp = client.chat_completion(
        messages=full, model=LLAMA_MODEL, max_tokens=2000, temperature=0.7)
    return resp.choices[0].message.content


def ask_model(system_prompt, messages):
    """Send a request to whichever provider the backend resolved to.

    Returns (answer_text, error_text); exactly one is non-empty.
    """
    provider = resolve_provider()
    try:
        if provider == "openai":
            key = _secret("OPENAI_API_KEY")
            if not key:
                return "", "OpenAI is selected but OPENAI_API_KEY is not set in secrets."
            return call_openai(key, system_prompt, messages), ""

        if provider == "claude":
            key = _secret("ANTHROPIC_API_KEY")
            if not key:
                return "", "Claude is selected but ANTHROPIC_API_KEY is not set in secrets."
            return call_claude(key, system_prompt, messages), ""

        # default: free Llama on Hugging Face
        key = _secret("HF_API_KEY")
        if not key:
            return "", ("No AI key is configured. Add a free HF_API_KEY (Hugging Face) "
                        "in secrets, or an OPENAI_API_KEY to use OpenAI.")
        return call_llama(key, system_prompt, messages), ""
    except Exception as e:
        return "", f"Something went wrong calling the model:\n\n{e}"


# ------------------------------------------------------------------
# LIVE INTERNSHIP SEARCH  (free, no API key)
# ------------------------------------------------------------------
def search_internships(query, max_results=8):
    """Search the web for internships. Returns a list of result dicts."""
    try:
        from ddgs import DDGS            # newer package name
    except ImportError:
        from duckduckgo_search import DDGS  # older name, just in case

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(r)
    except Exception as e:
        st.error(f"Search failed: {e}")
    return results


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("🎓 CareerPath AI")
st.write("Tell me about yourself and I'll suggest career paths, a learning "
         "roadmap, courses, and find live internship openings for you.")

# Memory that survives between clicks.
st.session_state.setdefault("profile_text", "")
st.session_state.setdefault("guidance", "")
st.session_state.setdefault("chat", [])

tab_profile, tab_intern, tab_chat = st.tabs(
    ["1️⃣ Career Guidance", "2️⃣ Find Internships", "3️⃣ Chat with Advisor"])


# ------------------------------------------------------------------
# TAB 1 — CAREER GUIDANCE
# ------------------------------------------------------------------
with tab_profile:
    st.subheader("Your profile")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name (optional)")
        course = st.text_input("Current course / degree",
                               placeholder="e.g. B.Tech Computer Science, 2nd year")
        subjects = st.text_input("Favourite subjects",
                                 placeholder="e.g. Maths, Databases, Statistics")
    with col2:
        interests = st.text_input("Interests / hobbies",
                                  placeholder="e.g. gaming, data, design, finance")
        skills = st.text_input("Skills you already have",
                               placeholder="e.g. Python, Excel, basic SQL")
        strengths = st.text_input("Your strengths",
                                  placeholder="e.g. problem solving, communication")

    if st.button("✨ Get my career guidance", type="primary"):
        profile_text = (
            f"Name: {name or 'N/A'}\n"
            f"Course: {course}\n"
            f"Favourite subjects: {subjects}\n"
            f"Interests: {interests}\n"
            f"Skills: {skills}\n"
            f"Strengths: {strengths}\n"
        )
        st.session_state["profile_text"] = profile_text

        system_prompt = (
            "You are a friendly, practical career counselor for students. "
            "Based on the student's profile, respond in clean Markdown with these "
            "sections and nothing else:\n"
            "## Recommended Career Paths\n"
            "List 3-5 careers. For each: one line on *why it fits this student*.\n"
            "## Learning Roadmap\n"
            "A phased, step-by-step plan (Beginner -> Intermediate -> Advanced).\n"
            "## Recommended Courses\n"
            "Specific courses with the platform name (Coursera, Udemy, freeCodeCamp, etc.).\n"
            "## Skill Gaps to Close\n"
            "What the student is missing for their top career, as a short checklist.\n"
            "## Internship Keywords\n"
            "3-5 short search phrases to look for internships (comma separated)."
        )
        user_msg = [{"role": "user",
                     "content": f"Here is the student's profile:\n\n{profile_text}"}]

        with st.spinner("Thinking about your future..."):
            answer, err = ask_model(system_prompt, user_msg)

        if err:
            st.error(err)
        else:
            st.session_state["guidance"] = answer

    if st.session_state["guidance"]:
        st.divider()
        st.markdown(st.session_state["guidance"])


# ------------------------------------------------------------------
# TAB 2 — INTERNSHIP FINDER
# ------------------------------------------------------------------
with tab_intern:
    st.subheader("Find real internships")
    st.caption("Searches the live web and returns real listing links.")

    c1, c2 = st.columns([2, 1])
    with c1:
        field = st.text_input("Role / field",
                              placeholder="e.g. data science, UI/UX, marketing")
    with c2:
        location = st.text_input("Location", placeholder="e.g. Bengaluru, remote")

    if st.button("🔎 Search internships", type="primary"):
        if not field.strip():
            st.warning("Type a role or field first.")
        else:
            query = f"{field} internship {location} apply".strip()
            with st.spinner("Searching the web..."):
                results = search_internships(query, max_results=8)

            if not results:
                st.info("No results found. Try different words.")
            else:
                st.success(f"Found {len(results)} listings:")
                for r in results:
                    title = r.get("title", "Untitled")
                    link = r.get("href", "")
                    snippet = r.get("body", "")
                    st.markdown(f"### [{title}]({link})")
                    st.write(snippet)
                    st.divider()


# ------------------------------------------------------------------
# TAB 3 — CHAT WITH ADVISOR
# ------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask me anything about your career")
    if not st.session_state["profile_text"]:
        st.info("Tip: fill in your profile on Tab 1 first so I can give personal advice.")

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Type your question...")
    if user_q:
        st.session_state["chat"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        system_prompt = (
            "You are a helpful student career advisor. Keep answers practical and "
            "encouraging. Use the student's profile below for context.\n\n"
            f"STUDENT PROFILE:\n{st.session_state['profile_text'] or 'Not provided yet.'}"
        )
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, err = ask_model(system_prompt, st.session_state["chat"])
            if err:
                st.error(err)
            else:
                st.markdown(answer)
                st.session_state["chat"].append({"role": "assistant", "content": answer})
