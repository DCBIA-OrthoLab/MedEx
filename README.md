# MedX

**MedX** is a research-oriented framework that leverages Large Language Models (LLMs) to automatically extract **temporomandibular joint (TMJ)-related comorbidities** from unstructured clinical notes. The pipeline transforms raw patient text into structured summaries and generates an **interactive visual dashboard** for cohort-level analysis.

This tool supports both **clinical research** and **decision-making** by surfacing patient-level insights and population-level statistics such as comorbidity frequencies, means, and standard deviations.

---

## 🔧 Features

- **LLM-based Comorbidity Extraction:** Supports BART and DeepSeek-based models
- **Automated Clinical Summarization:** Generates structured outputs per patient
- **Chunked Input Handling:** Efficiently processes long clinical documents
- **Interactive Dashboard:** Visualizes cohort-level statistics and trends

---

## 📁 Project Structure

- `model_run_chunks.py` – Run inference using the selected LLM model (BART or DeepSeek)
- `model_fine_tune.py` – Fine-tune your own LLM on labeled data
- `dashboard.py` – Generate an interactive Dash-based visualization for cohort summaries
- `requirements.txt` – List of required Python packages

---

## 🚀 Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run predictions:**
   ```
   python model_run_chunks.py
   ```
3. **Fine-tune a model (optional):**
   ```
   python model_fine_tune.py
   ```
4. **Launch the dashboard:**
   ```
   python dashboard.py
   ```

The dashboard visualization is built using Matplotlib
