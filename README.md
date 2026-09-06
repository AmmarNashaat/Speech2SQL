# Speech2SQL: AI-Powered Speech-to-SQL for Interactive Medical Database Exploration
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Database: PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Model: LLaMA 3.1 8B](https://img.shields.io/badge/LLM-LLaMA_3.1_8B_Instruct-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![ASR: Faster--Whisper](https://img.shields.io/badge/ASR-Faster--Whisper-black)](https://github.com/SYSTRAN/faster-whisper)
[![Frontend: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


An end-to-end, cascaded AI system that translates spoken medical inquiries into executable PostgreSQL queries, dynamic Plotly visualizations, and actionable AI business intelligence. Built as a final project for the Data Management Course (2026) at the University of Naples Federico II (UNINA).

---

## Features & Innovations

* **Cascaded Speech-to-SQL Pipeline**: Converts spoken audio into text via local ASR and generates valid PostgreSQL code using an LLM.


* **Database-Aware Preprocessing**: Integrates a lightweight normalization layer inspired by DBATI to clean ASR outputs (standardizing blood types, genders, and medical synonyms) prior to SQL generation.


* **Schema-Aware Prompting**: Passes full relational schema constraints, foreign key mappings, and table relations directly into the LLM context to drastically reduce hallucinations and invalid joins.


* **SQL Safety Guardrails**: Restricts generated queries strictly to read-only `SELECT` statements, blocking harmful operations (`DELETE`, `DROP`, `UPDATE`, `ALTER`).


* **Query Caching & Reuse**: Utilizes a PostgreSQL `query_cache` table to store verified query pairs, enabling instant retrieval for repeated questions without triggering LLM calls.


* **Automated Data Insights**: Generates smart Plotly chart visualizer selections (bar, pie, scatter, line, histogram) and provides secondary AI-driven business recommendations directly from raw SQL result sets.


* **Normalized Medical Schema**: Restructured an unorganized flat Kaggle dataset into a normalized, 7-table relational PostgreSQL schema.



---

## Architecture & Pipeline Flow

The system employs a modular, cascaded pipeline architecture designed for high interpretability and straightforward debugging:

```
┌──────────────┐     ┌─────────────────────┐     ┌───────────────────────┐
│ User Voice   │ ──> │  Faster-Whisper     │ ──> │ Text Normalization    │
│ Input        │     │  Speech Recognition │     │ & Error Correction    │
└──────────────┘     └─────────────────────┘     └───────────┬───────────┘
                                                             │
                                                             ▼
                                                 ┌───────────────────────┐
                                                 │   Query Cache Lookup  │
                                                 └───────────┬───────────┘
                                                             │
                                     ┌───────────────────────┴───────────────────────┐
                                     ▼ (Cache Hit)                                   ▼ (Cache Miss)
                         ┌───────────────────────┐                       ┌───────────────────────┐
                         │ Retrieve Cached SQL   │                       │ NVIDIA API            │
                         └───────────┬───────────┘                       │ LLaMA 3.1 8B Instruct │
                                     │                                   └───────────┬───────────┘
                                     │                                               │
                                     │                                               ▼
                                     │                                   ┌───────────────────────┐
                                     │                                   │ Save SQL to Cache     │
                                     │                                   └───────────┬───────────┘
                                     │                                               │
                                     └───────────────────────┬───────────────────────┘
                                                             │
                                                             ▼
                                                 ┌───────────────────────┐
                                                 │ PostgreSQL Execution  │
                                                 └───────────┬───────────┘
                                                             │
                                                             ▼
                                                 ┌───────────────────────┐
                                                 │ Smart Visualization   │
                                                 │ (Plotly / Tables)     │
                                                 └───────────┬───────────┘
                                                             │
                                                             ▼
                                                 ┌───────────────────────┐
                                                 │ AI Business Insights  │
                                                 │ Generation            │
                                                 └───────────────────────┘
```[cite: 1]

---

## Technology Stack

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.11[cite: 1] | System implementation[cite: 1] |
| **Speech Recognition** | Faster-Whisper (`base.en`)[cite: 1] | Quantized local ASR transcription[cite: 1] |
| **Text-to-SQL & Insights** | NVIDIA API + LLaMA 3.1 8B Instruct[cite: 1] | Schema understanding, SQL generation & BI analysis[cite: 1] |
| **Database** | PostgreSQL[cite: 1] | Relational data engine & query cache store[cite: 1] |
| **Database Driver** | psycopg2[cite: 1] | PostgreSQL connection pooling[cite: 1] |
| **User Interface** | Streamlit[cite: 1] | Interactive Web Dashboard[cite: 1] |
| **Data & Visuals** | Pandas, Plotly[cite: 1] | Data manipulation & automated dynamic charting[cite: 1] |

---

## Database Schema Summary

The normalized PostgreSQL database contains seven core relational tables designed to support complex queries involving multi-table joins, aggregations, groupings, and temporal trends[cite: 1]:

* `patients`: Primary patient demographic data (`patient_id`, `age`, `gender`, `blood_type`)[cite: 1].
* `doctors`: Physician details and specialties (`doctor_id`, `doctor_name`, `specialization`, `hospital_id`)[cite: 1].
* `hospitals`: Healthcare facility information (`hospital_id`, `hospital_name`)[cite: 1].
* `insurance_providers`: Insurance mapping (`insurance_id`, `provider_name`)[cite: 1].
* `admissions`: Core transaction/fact table tracking stays, billing, and conditions (`admission_id`, `patient_id`, `doctor_id`, `hospital_id`, `insurance_id`, `medical_condition`, `admission_type`, `date_of_admission`, `discharge_date`, `billing_amount`)[cite: 1].
* `medications`: Prescribed drugs linked to admissions (`medication_id`, `admission_id`, `medication_name`)[cite: 1].
* `medical_tests`: Diagnostic test records (`test_id`, `admission_id`, `test_name`, `test_result`)[cite: 1].

---

## Quick Start

### 1. Prerequisites
* **Python**: `v3.11` or higher[cite: 1]
* **PostgreSQL**: Installed and running locally or on a server[cite: 1]
* **NVIDIA API Key**: Required for LLaMA 3.1 8B Instruct access[cite: 1]

### 2. Installation
```bash
# Clone repository
git clone https://github.com/your-username/Speech2SQL.git
cd Speech2SQL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=medical_db
DB_USER=postgres
DB_PASSWORD=your_password
NVIDIA_API_KEY=your_nvidia_api_key

```

### 4. Database Setup

1. Execute your database initialization script to create the 7 normalized tables along with the foreign key dependencies.


2. Create the query caching table:

```sql
CREATE TABLE query_cache (
    cache_id SERIAL PRIMARY KEY,
    normalized_question TEXT UNIQUE NOT NULL,
    generated_sql TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```[cite: 1]

### 5. Launch the Dashboard
```bash
streamlit run app.py

```

---

## Credits & Acknowledgments

* **Author**: Ammar Gharaf


* **Academic Supervision**: Prof. Vincenzo Moscato & Dott. Francesco Di Serio


* **Institution**: University of Naples Federico II (UNINA) — Data Management Course (2026)
