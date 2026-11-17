import os
from pathlib import Path
from crewai import Agent, Crew, Process, Task, LLM
from dotenv import load_dotenv

load_dotenv()

def check_groq_configuration():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        return False, "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
    return True, "Groq API key configured."

DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
MAX_OUTPUT_TOKENS = 350
LLM_RETRIES = 2


def normalize_groq_model_name(raw_value):
    if not raw_value:
        return DEFAULT_GROQ_MODEL

    value = raw_value.strip().replace("groq/", "").replace("openai/", "")
    if not value:
        return DEFAULT_GROQ_MODEL

    if value in {"gpt-oss-120b", "gpt-oss-20b"}:
        return DEFAULT_GROQ_MODEL

    return value


def make_llm():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        raise RuntimeError("GROQ_API_KEY missing. Configure the .env file.")
    model = normalize_groq_model_name(os.getenv("GROQ_API_MODEL", DEFAULT_GROQ_MODEL))
    return LLM(
        model=f"groq/{model}",
        api_key=key,
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")),
        max_tokens=MAX_OUTPUT_TOKENS,
        additional_params={"num_retries": LLM_RETRIES},
    )

def run_project_crew(project, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = make_llm()
    brief = f"""TITLE: {project['title']}
DESCRIPTION: {project['description']}
LANGUAGE: {project['language']} | DATABASE: {project['database']}
TYPE: {project['project_type']} | USERS: {project['target_users']}
COMPLEXITY: {project['complexity']}
FEATURES: {', '.join(project['features'])}
CONSTRAINTS: {project['constraints']}"""

    def agent(role, goal, backstory):
        return Agent(role=role, goal=goal, backstory=backstory, llm=llm,
                     verbose=False, allow_delegation=False, max_iter=2)

    req = agent("Requirements Analysis Agent", "Define complete, testable functional and non-functional requirements.",
                "You are a senior business analyst who writes traceable requirements.")
    arch = agent("Solution Architect Agent", "Design modules, architecture, data model and interfaces.",
                 "You are a practical software architect.")
    dev = agent("Code Generation Agent", "Produce a small coherent prototype and clearly labeled source files.",
                "You are a careful developer who writes maintainable code.")
    qa = agent("Testing Agent", "Create a traceable test strategy, test cases and automated test examples.",
               "You are a QA engineer focused on edge cases and reproducibility.")
    docs = agent("Documentation Agent", "Write accurate installation, developer and user documentation.",
                 "You are a technical writer.")
    report = agent("Project Report Agent", "Prepare a formal academic project report based on previous outputs.",
                   "You are an academic software project report writer.")

    t1 = Task(description=f"""Analyze this project: {brief}
In concise bullets, state scope, assumptions, users, core functional and non-functional
requirements, acceptance criteria and out-of-scope items. Keep under 220 words.""",
              expected_output="Structured requirements specification in Markdown.", agent=req)
    t2 = Task(description=f"""Design a minimal architecture for: {brief}
Use previous requirements. Summarize components, data flow, essential data entities and one
security/error-handling note. Keep under 180 words.""",
              expected_output="Architecture and database design in Markdown.", agent=arch, context=[t1])
    t3 = Task(description=f"""Generate the smallest useful prototype for: {brief}
Follow prior requirements and architecture. Return only essential files in labeled code blocks,
with basic validation. Omit optional features and lengthy explanation. Do not claim it was executed.
For a library system include add/view books, register students, issue and return books.""",
              expected_output="Prototype file tree and source code.", agent=dev, context=[t1, t2])
    t4 = Task(description=f"""Create a compact test plan for: {brief}
Use previous outputs. Include four test cases with IDs and expected results, plus a tiny
pytest/equivalent example. State that tests are proposed, not executed.""",
              expected_output="Test plan and automated test examples.", agent=qa, context=[t1, t2, t3])
    t5 = Task(description=f"""Write a short README for: {brief}
Use prior outputs. Include purpose, stack, install/run steps and one usage note. Do not claim
features are implemented unless shown in the prototype. Keep under 180 words.""",
              expected_output="Complete README-style documentation.", agent=docs, context=[t1, t2, t3, t4])
    t6 = Task(description=f"""Write a concise academic summary for: {brief}
Use prior outputs. Cover problem, objectives, solution, proposed testing, limitations and future
scope in short sections. Use placeholders for student/college details. Never invent test results.
Keep under 220 words.""",
              expected_output="Complete academic project report in Markdown.", agent=report,
              context=[t1, t2, t3, t4, t5])

    crew = Crew(agents=[req, arch, dev, qa, docs, report],
                tasks=[t1, t2, t3, t4, t5, t6], process=Process.sequential, verbose=True)
    crew.kickoff()
    tasks = {"requirements": t1, "architecture": t2, "code": t3,
             "testing": t4, "documentation": t5, "report": t6}
    names = {"requirements": "01_requirements.md", "architecture": "02_architecture.md",
             "code": "03_prototype_code.md", "testing": "04_testing.md",
             "documentation": "05_documentation.md", "report": "06_project_report.md"}
    result = {}
    for key, task in tasks.items():
        raw = task.output.raw if task.output else "No output returned."
        result[key] = str(raw)
        (output_dir / names[key]).write_text(result[key], encoding="utf-8")
    return result
