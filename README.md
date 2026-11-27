# FORGEAI — Feature 1: Intelligent Intake Interviewer

Extracted and adapted from Mozilla AI's [`agent-factory`](https://github.com/mozilla-ai/agent-factory) patterns (Pydantic v2 structured schemas, Jinja2 instruction templates, LiteLLM provider-agnostic completion, and validation pipelines).

## What Feature 1 Implements

- **Brief Parser** (`src/forgeai/intake/brief_parser.py`): Extracts domain hints, target audience, candidate topics, capabilities, constraints, and ambiguity signals.
- **Document Analyzer** (`src/forgeai/intake/document_analyzer.py`): Scans folders containing `.pdf`, `.docx`, `.txt`, and `.csv` files.
- **Requirement Extraction Prompts** (`src/forgeai/intake/prompts.py`): Jinja2 prompts for extraction, clarification (max 5 questions), and answer refinement.
- **Structured Requirement Schema** (`src/forgeai/intake/schemas.py`): `AgentRequirements` matching `requirements.json`.
- **Clarification Question Generator** (`src/forgeai/intake/clarifier.py`): Generates at most 5 targeted clarification questions when the brief is ambiguous.
- **Requirement Validator** (`src/forgeai/intake/validator.py`): Validates required fields, non-overlapping topics, source grounding, completeness score, and the 5-question maximum.
- **Orchestrator & CLI** (`src/forgeai/intake/interviewer.py`, `src/forgeai/intake/cli.py`): Exports validated `requirements.json`.

## Quick Start

Install in editable mode:

```bash
pip install -e .[dev]
```

Run against the HR example:

```bash
python -m forgeai.intake.cli \
  --brief examples/hr_assistant/brief.txt \
  --docs examples/hr_assistant/docs \
  --config examples/hr_assistant/config.json \
  --output examples/hr_assistant/requirements.json
```

Run interactively to answer clarification questions and refine `requirements.json`:

```bash
python -m forgeai.intake.cli \
  --brief examples/hr_assistant/brief.txt \
  --docs examples/hr_assistant/docs \
  --config examples/hr_assistant/config.json \
  --output examples/hr_assistant/requirements.json \
  --interactive
```

Run tests:

```bash
pytest -v
```

---

## Feature 2: Document Ingestion & Knowledge Processing

Pipeline: `Documents -> File Loader -> Text Extraction -> Cleaning & Normalization -> Chunking -> Metadata Attachment -> Embeddings -> Vector Database -> Agent Retrieval`

Supported formats: **PDF** (`.pdf`), **DOCX** (`.docx`), **TXT** (`.txt`), **CSV** (`.csv`), and **Markdown** (`.md`, `.markdown`).

Ingest documents into `knowledge_manifest.json` + searchable SQLite vector store (`knowledge_base.sqlite`):

```bash
python -m forgeai.knowledge.cli ingest \
  --docs examples/hr_assistant/docs \
  --out examples/hr_assistant/kb
```

Query the knowledge base:

```bash
python -m forgeai.knowledge.cli query \
  --manifest examples/hr_assistant/kb/knowledge_manifest.json \
  --q "How many days of annual paid leave do employees receive?"
```

---

## Feature 3: Groundability Guard

Maps requested topics from `requirements.json` against the ingested `knowledge_manifest.json` + vector store to classify each topic as **Supported (`Available`)**, **Partial (`Partial`)**, or **Unsupported (`Missing`)**, and emits explicit refusal guardrails.

```bash
python -m forgeai.groundability.cli \
  --requirements examples/hr_assistant/requirements.json \
  --manifest examples/hr_assistant/kb/knowledge_manifest.json \
  --topics "Leave" "Attendance" "Salary" "Promotions" "Medical insurance" "Retirement benefits" \
  --out examples/hr_assistant/groundability
```

Outputs:
- `groundability_report.md` (`GROUNDABILITY REPORT`)
- `groundability_report.json` (`GroundabilityReport`)
- `supported_topic_config.json` (`SupportedTopicConfig` with `refusal_policy` & `restrictions`)

---

## Feature 4: Agent Generator

Synthesizes the final validated `agent.json` blueprint combining `requirements.json`, `supported_topic_config.json`, and `knowledge_manifest.json`:

```bash
python -m forgeai.generator.cli \
  --requirements examples/hr_assistant/requirements.json \
  --groundability examples/hr_assistant/groundability/supported_topic_config.json \
  --manifest examples/hr_assistant/kb/knowledge_manifest.json \
  --model gpt-4o-mini \
  --no-llm \
  --out examples/hr_assistant/agent.json
```

---

## Feature 5: Tool Registry & Tool Selection

Centralized registry (`src/forgeai/tools/registry.py`) and dynamic selector (`src/forgeai/tools/selector.py`) covering:
- **Essential**: `document_retriever`, `calculator`, `csv_lookup`
- **Recommended**: `date_tool`
- **Optional**: `http_api`, `web_search`

List registry or run tool selection with explicit per-tool reasons:

```bash
python -m forgeai.tools.cli list
python -m forgeai.tools.cli select \
  --requirements examples/hr_assistant/requirements.json \
  --groundability examples/hr_assistant/groundability/supported_topic_config.json \
  --manifest examples/hr_assistant/kb/knowledge_manifest.json \
  --no-llm \
  --out examples/hr_assistant/tools
```

---

## Features 6 & 7: Automatic Evaluation Dataset Generator & Hybrid Evaluation Judge

Generate >= 20 document-driven test cases (`evals.json`: 8 answerable, 4 multi-document, 3 edge-case, 2 ambiguous, 3 unsupported), execute `agent.json`, and score via deterministic checks + optional LLM judge:

```bash
python -m forgeai.eval.cli run-all \
  --agent examples/hr_assistant/agent.json \
  --manifest examples/hr_assistant/kb/knowledge_manifest.json \
  --no-llm \
  --out examples/hr_assistant/eval
```

Outputs: `evals.json`, `eval_results.json`, and `scorecard.md`.

---

## Features 8, 9 & 10: Critic & Optimizer Loop, Honest Final Report & Agent Runtime

Run the Critic & Optimizer Loop (max 3 iterations):

```bash
python -m forgeai.optimizer.cli \
  --agent examples/hr_assistant/agent.json \
  --evals examples/hr_assistant/eval/evals.json \
  --no-llm \
  --out examples/hr_assistant/eval
```

Or run the entire **10-feature FORGEAI pipeline** end-to-end:

```bash
python -m forgeai.pipeline \
  --brief examples/hr_assistant/brief.txt \
  --docs examples/hr_assistant/docs \
  --config examples/hr_assistant/config.json \
  --topics "Salary" "Promotions" "Medical insurance" "Retirement benefits" \
  --no-llm \
  --out examples/hr_assistant/run_output
```

Chat with the generated agent (Feature 10):

```bash
python -m forgeai.runtime.cli --agent examples/hr_assistant/agent.json --no-llm
```

---

## Feature 11 & Production Enhancements: Two Complete Demos

- **Demo A — HR Assistant**: `demos/demo_a_hr_assistant/` (`employee_handbook.pdf`, `leave_policy.docx`, `attendance_policy.txt`, `hr_faq.md`)
- **Demo B — Procurement / Ship-Spares Assistant**: `demos/demo_b_procurement_assistant/` (`spare_parts_catalog.csv`, `supplier_directory.pdf`, `engine_specifications.docx`, `procurement_policy.md`)

Run both demos end-to-end (producing `requirements.json`, `kb/`, `groundability/`, `tools/`, `agent.json`, `eval/`, `versions/`, `final_report.md`, `dashboard.html`, `trace_log.json`, and `export_bundle/`):

```bash
python -m forgeai.pipeline --brief demos/demo_a_hr_assistant/brief.txt --docs demos/demo_a_hr_assistant/docs --config demos/demo_a_hr_assistant/config.json --topics "Salary" "Personal leave balance" "Promotions" --no-llm --out demos/demo_a_hr_assistant/output

python -m forgeai.pipeline --brief demos/demo_b_procurement_assistant/brief.txt --docs demos/demo_b_procurement_assistant/docs --config demos/demo_b_procurement_assistant/config.json --topics "Inventory" "Part numbers" "Specifications" "Supplier information" "Live port customs clearance status" "Execute purchase order in ERP" --no-llm --out demos/demo_b_procurement_assistant/output
```

---

## Interactive 7-Tab UI Dashboard (`app.py`)

Launch the Streamlit dashboard (4 KPI cards + 7 tabs: `01 · Project Intake` through `07 · Test Your Agent`):

```bash
streamlit run app.py
```

---

## Deployment

### Option 1: Streamlit Community Cloud (Fastest, Free)
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Select repository `SachiPrasad/FORGEAI`, branch `main`, and main file path `app.py`.
3. *(Optional)* In **Advanced settings -> Secrets**, add `OPENAI_API_KEY` or `GEMINI_API_KEY` for Live LLM mode (without keys, the app runs in Deterministic Grounded Fallback mode).
4. Click **Deploy**.

### Option 2: Render (Docker Blueprint)
1. Connect `SachiPrasad/FORGEAI` on [dashboard.render.com](https://dashboard.render.com/) using `render.yaml` (`Dockerfile`).
2. Optionally set `OPENAI_API_KEY` / `GEMINI_API_KEY` in Environment Variables and deploy.

### Option 3: Local / Cloud Docker Container
```bash
docker build -t forgeai .
docker run -p 8501:8501 --env-file .env forgeai
```










