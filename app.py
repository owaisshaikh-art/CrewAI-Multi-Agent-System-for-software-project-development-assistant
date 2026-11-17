# ============================================================
# CrewAI Multi-Agent Project Developer (Token-Optimized)
# Keep this file ONLY ONCE in app.py.
# ============================================================

# Compatibility workaround: run before importing CrewAI.
try:
    import crewai.llms.cache as crew_cache
    crew_cache.mark_cache_breakpoint = lambda msg: msg
except (ImportError, AttributeError):
    pass

import os
import re
import json
import time
import zipfile
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM


# ------------------------- CONFIG ----------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def normalize_groq_model_name(raw_value):
    if not raw_value:
        return DEFAULT_GROQ_MODEL

    value = raw_value.strip().replace("groq/", "").replace("openai/", "")
    if not value:
        return DEFAULT_GROQ_MODEL

    # Groq native models are passed without the provider prefix.
    if value in {"gpt-oss-120b", "gpt-oss-20b"}:
        return DEFAULT_GROQ_MODEL

    return value


MODEL_NAME = normalize_groq_model_name(os.getenv("GROQ_API_MODEL", DEFAULT_GROQ_MODEL))

# Keep each response below Groq's per-request output-token allowance.
MAX_OUTPUT_TOKENS = 350
LLM_RETRIES = 2

st.set_page_config(
    page_title="CrewAI Multi-Agent Project Developer",
    page_icon="🤖",
    layout="wide"
)


# ------------------------- UI STYLE --------------------------

st.markdown("""
<style>
.main-title {text-align:center;color:#2446a8;font-size:32px;font-weight:700;}
.subtitle {text-align:center;color:gray;font-size:16px;}
.stButton > button {background:#2446a8;color:white;border-radius:8px;min-height:42px;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🤖 CrewAI Multi-Agent System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Token-optimized software project development assistant</div>',
    unsafe_allow_html=True
)
st.divider()
st.write(
    "Six specialized agents create requirements, architecture, prototype code, "
    "tests, documentation and a project report."
)


# ---------------------- GROQ CONFIG ---------------------------

def check_groq_configuration():
    if not GROQ_API_KEY:
        return False, "GROQ_API_KEY is missing. Add it to the .env file."
    if not GROQ_API_KEY.startswith("gsk_"):
        return False, "The Groq API key format looks incorrect."
    if not MODEL_NAME:
        return False, "GROQ_API_MODEL is missing."
    return True, "Groq configuration detected."


def get_llm():
    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing.")
    return LLM(
        model=f"groq/{MODEL_NAME}",
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=MAX_OUTPUT_TOKENS,
        additional_params={"num_retries": LLM_RETRIES},
    )


# ------------------------- AGENTS -----------------------------

def create_agents(llm):
    common = {
        "llm": llm,
        "verbose": False,
        "allow_delegation": False,
        "max_iter": 2,
    }

    return {
        "requirements": Agent(
            role="Software Requirements Analyst",
            goal="Create concise, complete software requirements.",
            backstory="You are a practical business analyst and SRS specialist.",
            **common
        ),
        "architecture": Agent(
            role="Software Architect",
            goal="Design a simple, secure and implementable architecture.",
            backstory="You are an experienced application and database architect.",
            **common
        ),
        "code": Agent(
            role="Senior Software Developer",
            goal="Produce a small, readable working prototype.",
            backstory="You write maintainable code with validation and error handling.",
            **common
        ),
        "testing": Agent(
            role="QA Engineer",
            goal="Create focused test cases and pytest examples.",
            backstory="You specialize in unit, functional and integration testing.",
            **common
        ),
        "documentation": Agent(
            role="Technical Writer",
            goal="Write clear setup and usage documentation.",
            backstory="You create beginner-friendly README files.",
            **common
        ),
        "report": Agent(
            role="Academic Project Report Specialist",
            goal="Write a concise academic software project report.",
            backstory="You prepare accurate technical project reports.",
            **common
        )
    }


# -------------------------- TASKS -----------------------------

def create_tasks(agents, data):
    title = data["title"]
    desc = data["description"]
    language = data["language"]
    database = data["database"]
    features = data["features"]

    req = Task(
        description=f"""
Create a concise SRS for this project. Keep it under 220 words.
Project: {title}
Description: {desc}
Language: {language}; Database: {database}
Features: {features}
Use short bullets for the problem, objectives, users, core requirements,
constraints and outcome. Do not repeat the input.
""",
        expected_output="A brief SRS in Markdown, under 220 words.",
        agent=agents["requirements"]
    )

    arch = Task(
        description=f"""
Design a minimal architecture for {title}. Keep it under 180 words.
Stack: {language}; database: {database}.
Features: {features}
List the main components, data flow, essential data entities and one security
consideration. Use brief bullets.
""",
        expected_output="Brief architecture in Markdown, under 180 words.",
        agent=agents["architecture"],
        context=[req]
    )

    code = Task(
        description=f"""
Create a minimal prototype for {title}, using {language} and {database}.
Required features: {features}
Return only the smallest runnable example and essential files. Use a short
file tree, label each code block with its filename, and include basic input
validation. Omit optional features and lengthy explanations.
Do not claim that code was run or tested.
""",
        expected_output="A minimal prototype with filenames and source code.",
        agent=agents["code"],
        context=[arch]
    )

    testing = Task(
        description=f"""
Create a compact test plan for {title}: four test cases in a table with ID,
scenario and expected result, plus a tiny pytest example for the core logic.
State that tests are proposed, not executed.
""",
        expected_output="A brief test plan and pytest example.",
        agent=agents["testing"],
        context=[code]
    )

    docs = Task(
        description=f"""
Write a short README for {title} with purpose, stack, install/run steps and
one usage note. Use {language} and {database}. Do not invent credentials or
URLs. Keep it under 180 words.
""",
        expected_output="A README in Markdown, under 180 words.",
        agent=agents["documentation"],
        context=[arch]
    )

    report = Task(
        description=f"""
Write a concise academic summary for {title}, under 220 words.
Description: {desc}
Include the problem, objectives, solution, proposed testing, limitations and
future scope in short sections.
Do not invent measured results, citations or completed testing.
""",
        expected_output="A short academic project summary in Markdown.",
        agent=agents["report"],
        context=[req, testing]
    )

    return [
        ("requirements", req),
        ("architecture", arch),
        ("code", code),
        ("testing", testing),
        ("documentation", docs),
        ("report", report)
    ]


# ---------------------- RUN CREW ------------------------------

def run_project_crew(data):
    llm = get_llm()
    agents = create_agents(llm)
    named_tasks = create_tasks(agents, data)

    crew = Crew(
        agents=list(agents.values()),
        tasks=[task for _, task in named_tasks],
        process=Process.sequential,
        verbose=False
    )

    crew.kickoff()

    outputs = {}
    for name, task in named_tasks:
        task_output = task.output
        if task_output is not None:
            raw = getattr(task_output, "raw", None)
            outputs[name] = str(raw if raw is not None else task_output)
        else:
            outputs[name] = "No output generated."
    return outputs


# ---------------------- SAVE OUTPUTS --------------------------

def save_outputs(title, outputs):
    safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", title).strip("_") or "project"
    folder = OUTPUT_DIR / safe_title
    folder.mkdir(parents=True, exist_ok=True)

    for name, content in outputs.items():
        (folder / f"{name}.md").write_text(str(content), encoding="utf-8")

    (folder / "project_outputs.json").write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return folder


def make_zip(folder):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in folder.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(folder))
    buffer.seek(0)
    return buffer.getvalue()


# --------------------------- SIDEBAR --------------------------

api_ok, api_message = check_groq_configuration()

with st.sidebar:
    st.title("⚙️ Settings")
    if api_ok:
        st.success(api_message)
    else:
        st.error(api_message)
    st.caption(f"Model: {MODEL_NAME}")
    st.caption(f"Maximum output tokens per task: {MAX_OUTPUT_TOKENS}")
    st.divider()
    st.subheader("Six AI agents")
    for label in [
        "📋 Requirements", "🏗️ Architecture", "💻 Code",
        "🧪 Testing", "📚 Documentation", "📊 Report"
    ]:
        st.write(label)


# ------------------------- INPUT FORM -------------------------

st.header("📝 Project Information")

with st.form("project_form"):
    project_title = st.text_input(
        "Project Title",
        placeholder="Example: AI Career Guidance System"
    )
    description = st.text_area("Project Description", height=100)
    col1, col2 = st.columns(2)

    with col1:
        language = st.selectbox(
            "Programming Language",
            ["Python", "Java", "JavaScript", "C++", "PHP", "Other"]
        )
    with col2:
        database = st.selectbox(
            "Database",
            ["SQLite", "MySQL", "PostgreSQL", "MongoDB", "Firebase", "None"]
        )

    features = st.text_area(
        "Required Features",
        height=110,
        placeholder="User registration\nLogin\nDashboard\nReports"
    )
    submitted = st.form_submit_button(
        "🚀 Generate Project",
        use_container_width=True
    )


# ------------------------ GENERATION --------------------------

if submitted:
    if not project_title.strip():
        st.error("Enter a project title.")
    elif not description.strip():
        st.error("Enter a project description.")
    elif not features.strip():
        st.error("Enter the required features.")
    elif not api_ok:
        st.error(api_message)
    else:
        data = {
            "title": project_title.strip(),
            "description": description.strip(),
            "language": language,
            "database": database,
            "features": features.strip()
        }

        try:
            with st.spinner(
                "Six agents are generating concise outputs. "
                "This may take a few minutes..."
            ):
                outputs = run_project_crew(data)
                folder = save_outputs(project_title, outputs)

            st.session_state["project_outputs"] = outputs
            st.session_state["project_folder"] = str(folder)
            st.session_state["project_title"] = project_title.strip()
            st.success("All six agent tasks completed.")
            st.balloons()

        except Exception as exc:
            message = str(exc)
            is_rate_limit = (
                "ratelimit" in type(exc).__name__.lower()
                or "rate_limit" in message.lower()
                or re.search(
                    r"\b(?:otpm|tpm)\b|output tokens per minute",
                    message,
                    re.IGNORECASE,
                )
                is not None
            )
            if is_rate_limit:
                retry_match = re.search(
                    r"try again in\s+(\d+(?:\.\d+)?)\s*s",
                    message,
                    re.IGNORECASE,
                )
                retry_seconds = (
                    max(1, int(float(retry_match.group(1)) + 0.99))
                    if retry_match
                    else 60
                )
                st.error(
                    f"Groq's output-token limit was reached. Wait about "
                    f"{retry_seconds} seconds, then submit again. The app "
                    f"automatically retries transient LLM requests up to "
                    f"{LLM_RETRIES} times and limits each response to "
                    f"{MAX_OUTPUT_TOKENS} tokens. If the limit persists, "
                    "shorten the description/features or use a model with a "
                    "higher quota."
                )
            else:
                st.error(
                    f"Generation failed ({type(exc).__name__}). "
                    "Check your Groq API key and model setting, then try again."
                )


# ------------------------- RESULTS ----------------------------

if "project_outputs" in st.session_state:
    outputs = st.session_state["project_outputs"]
    st.divider()
    st.header("📂 Generated Results")
    st.write(f"Project: **{st.session_state['project_title']}**")

    names = [
        "requirements", "architecture", "code",
        "testing", "documentation", "report"
    ]
    labels = [
        "📋 Requirements", "🏗️ Architecture", "💻 Source Code",
        "🧪 Testing", "📚 Documentation", "📊 Report"
    ]

    tabs = st.tabs(labels)
    for tab, name in zip(tabs, names):
        with tab:
            content = outputs.get(name, "No output available.")
            st.markdown(content)
            st.download_button(
                f"Download {name.title()}",
                data=content,
                file_name=f"{name}.md",
                mime="text/markdown",
                key=f"download_{name}"
            )

    st.divider()
    st.subheader("📦 Download all generated documents")
    folder = Path(st.session_state["project_folder"])
    st.download_button(
        "📥 Download ZIP",
        data=make_zip(folder),
        file_name="generated_project_documents.zip",
        mime="application/zip",
        use_container_width=True
    )
    st.info(
        "The ZIP contains six Markdown deliverables and a JSON summary. "
        "The code agent's prototype is included in code.md; it is not "
        "automatically executed."
    )


# ---------------------------- FOOTER --------------------------

st.divider()
st.markdown(
    "<div style='text-align:center;'><b>CrewAI Multi-Agent System "
    "for Project Development</b><br>Python | Streamlit | CrewAI | Groq</div>",
    unsafe_allow_html=True
)
