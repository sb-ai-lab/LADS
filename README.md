
<div align="center">
  <img src="./image/lads.jpg" alt="Logo" width="200">
  <h1 align="center">LADS — LLM-Powered AutoDS Agent</h1>
</div>

<div align="center">

<img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
<img src="https://img.shields.io/badge/uv-package%20manager-blueviolet.svg" alt="uv">
<a href="https://github.com/deepbiolab/LADS/blob/main/LICENSE"><img src="https://img.shields.io/badge/LICENSE-BSD--3--Clause-green"></a>

</div>

An agentic system for tabular data tasks that combines **LLM-driven code generation** with **AutoGluon AutoML** in a clean two-path architecture.

---

## Architecture

```
User input
  └─ code_router
       ├─ NO  → direct answer (no code needed)
       └─ YES → automl_router
                    ├─ AUTOGLUON → autogluon_config → autogluon_executor → report
                    └─ LLM path  → plan → generate → execute → validate → improve → final report
```

**Two execution paths:**
- **AutoGluon path** — mention "autogluon" or "automl" and the agent configures and runs `TabularPredictor` automatically
- **LLM codegen path** — everything else; the agent iteratively generates, executes, validates, and improves Python code (up to `max_improvements` rounds)

**Project layout:**

```
LADS/
├── app.py              # Streamlit entry point
├── pyproject.toml      # Dependencies (uv)
├── config.yml          # Runtime config (model, limits)
├── app/                # UI layer (Streamlit)
│   ├── agent_handler.py
│   ├── ui_components.py
│   ├── fragments.py
│   ├── session_state.py
│   ├── data_handlers.py
│   └── media_utils.py
└── src/                # Core logic layer
    ├── config.py       # Pydantic config models + load_config()
    ├── llm_factory.py  # Multi-provider LLM factory
    ├── state.py        # LangGraph AgentState
    ├── prompts.py      # All prompt templates + load_prompt()
    ├── builder.py      # LangGraph graph definition
    ├── nodes.py        # LLM-driven nodes
    ├── executor.py     # Code execution nodes (local / E2B / AutoGluon)
    └── backends/       # AutoML plugin system
        ├── base.py
        ├── registry.py
        ├── autogluon_backend.py
        └── flaml_backend.py  # (future)
```

---

## Quick Start

**1. Clone**

```bash
git clone https://github.com/deepbiolab/LADS.git
cd LADS
```

**2. Install dependencies with [uv](https://docs.astral.sh/uv/)**

```bash
# Core dependencies
uv sync

# Optional: E2B sandboxed code execution
uv sync --extra e2b

# Optional: AutoGluon backend
uv sync --extra autogluon
```

> Don't have uv? Install it with `pip install uv` or see [docs.astral.sh/uv](https://docs.astral.sh/uv/).

**3. Configure API keys**

```bash
cp .env_example .env
```

Edit `.env` and fill in your API key:

```env
OPENAI_API_KEY=sk-...
# Optional
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
E2B_API_KEY=...
```

**4. (Optional) Edit `config.yml`**

```yaml
llm:
  provider: openai        # openai | anthropic | groq | ollama
  model_name: gpt-4.5
  base_url:               # custom endpoint (leave blank for default)

general:
  max_improvements: 5     # max LLM improvement iterations
  code_generation_config: local   # local | e2b
  max_code_execution_time: 600    # seconds
```

**5. Run**

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## UI

<img src="./image/AutoDS-UI.png" align="center">

The interface has two panels:

- **Right panel** — technical pipeline steps (node-by-node progress)
- **Left panel** — plain-language summary for non-expert users

Supported file formats: CSV, XLSX, Parquet.

---

## LLM Providers

LADS uses OpenAI natively and falls back to [LiteLLM](https://github.com/BerriAI/litellm) for everything else:

| Provider | `provider` value | Notes |
|---|---|---|
| OpenAI | `openai` | Default |
| Anthropic | `anthropic` | Requires `ANTHROPIC_API_KEY` |
| Groq | `groq` | Requires `GROQ_API_KEY` |
| Ollama (local) | `ollama` | Set `base_url: http://localhost:11434` |

Per-node model overrides are supported via `model_overrides` in `config.yml`.

---

## Adding an AutoML Backend

1. Create `src/backends/your_backend.py` implementing `AbstractAutoMLBackend`
2. Register it in `src/backends/registry.py`
3. Add a node in `src/executor.py` and wire edges in `src/builder.py`

---

## License

Distributed under the BSD 3-Clause License. See [`LICENSE`](./LICENSE) for more information.
