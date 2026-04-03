<p align="center">
  <h1 align="center">🎨 Visualization Agent</h1>
  <p align="center">
    <strong>Production-Grade Plotly Spec Generator Microservice</strong>
    <br />
    A backend service that automatically aggregates data and generates intelligent Plotly JSON specifications using Groq LLM + a Rule-based engine. Designed specifically for integration with the Orchestrator UI.
    <br /><br />
    <a href="#-quick-start">Quick Start</a> · <a href="#-api-endpoints">API Docs</a> · <a href="#-features">Features</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Groq_Llama_3.3-F55036?logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Charts-Plotly-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-Passing-brightgreen?logo=pytest&logoColor=white" />
</p>

---

## 💡 What Is This?

**Visualization Agent** is a specialized, backend-only FastAPI microservice built to run within a larger multi-agent data analysis platform. It does **not** serve HTML or frontend dashboards. Instead, its sole purpose is to:

1. **Receive** unstructured or raw tabular data.
2. **Aggregate** the data automatically using `pandas` (aware of the target chart type).
3. **Analyze** the data using a dual recommendation system:
   - 🧠 **Rule-based engine** — instant, zero-cost, covers common patterns.
   - 🤖 **Groq LLM (Llama 3.3 70B)** — picks optimal charts and formats complex layouts.
4. **Generate** a clean, valid, and deeply customized Plotly JSON specification.
5. **Return** the JSON payload back to the orchestrator frontend (React/Next.js) for seamless rendering.

---

## ✨ Features

| Feature | Description |
|---------|------------|
| 🎯 **Auto-Insights** | AI analyzes tabular data and automatically suggests the 2 best visualizations. |
| 📊 **10 Chart Types** | Bar, line, scatter, pie (donut), histogram, heatmap, box, treemap, waterfall, area. |
| 🛡️ **Spec Sanitization** | All LLM output is strictly sanitized to guarantee crash-free React Plotly rendering. |
| 🧮 **Smart Aggregation** | The backend automatically groups by category/date before sending data to the LLM. |
| 🔄 **Sequential Generation** | Prevents LLM rate-limiting by generating multiple charts sequentially. |
| 🔌 **Orchestrator Ready** | Native `/run` endpoint designed to seamlessly process tasks from an upstream orchestrator agent. |
| 🖼️ **Fallback PNG Rendering** | Optional internal integration with Kaleido to produce base64 PNG fallback images. |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Groq API Key** — get one free at [console.groq.com](https://console.groq.com)

### 1. Installation

```bash
git clone https://github.com/VinitHudiya19/visualization-agent.git
cd visualization-agent

# Create venv and install
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
STORAGE_TYPE=local
MAX_DATA_ROWS=1000
PORT=8003
```

### 3. Start the Server

```bash
python -m uvicorn app.main:app --reload --port 8003
```

Navigate to **http://localhost:8003/docs** for the interactive Swagger UI.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service status, model, available schemes & chart types. |
| `POST` | `/run` | **(Primary)** Unified orchestration entry point. Receives context and routes to the correct internal pipeline. |
| `POST` | `/auto-insights` | 🌟 AI picks the best 2 visualizations automatically and returns their Plotly specs. |
| `POST` | `/chart` | Generates a specific Plotly spec for a provided dataset and intentional task. |
| `POST` | `/recommend` | Rule-based chart type recommendation (Zero LLM cost). |
| `POST` | `/chart/render` | *(Utility)* Renders an existing Plotly spec into a PNG. |

### Example: `/run` Endpoint payload

The orchestrator sends tasks to the `/run` endpoint like this:

```bash
curl -X POST http://localhost:8003/run \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "viz-123",
    "agent": "viz",
    "description": "Show the trend of revenue over the last 12 months",
    "payload": {},
    "_context": {
      "data": { "columns": [...], "rows": [...] }
    }
  }'
```

### Example Response

```json
{
  "task_id": "viz-123",
  "status": "completed",
  "result": {
    "chart_type": "line",
    "spec": {
      "data": [
        {
          "x": ["2024-01", "2024-02", "2024-03"],
          "y": [12000, 15000, 9500],
          "type": "scatter",
          "mode": "lines+markers"
        }
      ],
      "layout": {
        "title": "Revenue Trend Reveals Q2 Dip",
        ...
      }
    }
  }
}
```

---

## 🏛️ Architecture

```text
 ┌──────────────────────┐
 │  Orchestrator Agent  │ (Controller)
 └──────────┬───────────┘
            │ HTTP POST /run (Task + Data)
 ┌──────────▼───────────┐
 │   FastAPI Backend    │ (Visualization Agent)
 └──────────┬───────────┘
            │
      ┌─────┴──────┐
      ▼            ▼
 ┌──────────┐ ┌──────────┐ 
 │  Pandas  │ │ Groq LLM │ 
 │Aggregator│ │  Llama 3 │ 
 └──────────┘ └──────────┘ 
      │            │
      └─────┬──────┘
            ▼            
   Sanitized Plotly JSON
     Returned to Caller
```

1. **Aggregation First**: The agent intercepts raw data, identifies the required chart type, and uses Pandas to strictly group and aggregate the dataset (e.g. by region or date).
2. **Context Enrichment**: The aggregated data, along with computed basic statistics, is formatted into a prompt.
3. **LLM Generation**: The LLM writes the layout, title, hover-templates, and color traces strictly conforming to JSON format.
4. **Sanitization Filter**: The output passes through `_sanitize_spec()` to enforce the React Plotly schema, dropping hallucinatory fields.

---

## 🧪 Running Tests

```bash
# Run the live end-to-end testing suite
python test_live.py

# Run all unit tests
python -m pytest tests/ -v
```

---

## 📄 License & Acknowledgments

- Licensed under [MIT License](LICENSE).
- Powered by **[Groq](https://groq.com/)** fast inference.
- Spec generation relies on **[Plotly Python](https://plotly.com/python/)**.

<p align="center">
  <sub>Built as a structural microservice for a sophisticated multi-agent topology.</sub>
</p>
