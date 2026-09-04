import warnings
warnings.filterwarnings("ignore")
import time
import os
import re
import psycopg2
from psycopg2 import pool
import sounddevice as sd
import pandas as pd
import numpy as np
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI
from tabulate import tabulate
from functools import lru_cache
from faster_whisper import WhisperModel
load_dotenv()
# =========================================================
# LOAD FASTER WHISPER ONCE
# =========================================================
model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8" # Fast because it's quantized as quantization reduce computational & memory usage
)
# =========================================================
# NVIDIA Llama Client
# =========================================================
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)
# =========================================================
# POSTGRESQL CONNECTION POOL
# Instead of opening/closing a new database connection for every request (slow & expensive), a pool pre-creates and reuses a set of connections.
# =========================================================
db_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host="localhost",
    database="medical_voice",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)
if db_pool:
    print("✅ Connection pool created successfully")
# =========================================================
# TEXT NORMALIZATION LAYER
# =========================================================
def normalize_text(text):
    text = text.lower().strip()
    # -------------------------------------------------
    # Gender normalization
    # -------------------------------------------------
    gender_map = {
        r"\bmen\b": "male",
        r"\bman\b": "male",
        r"\bguy\b": "male",
        r"\bguys\b": "male",
        r"\bboy\b": "male",
        r"\bboys\b": "male",
        r"\bwomen\b": "female",
        r"\bwoman\b": "female",
        r"\blady\b": "female",
        r"\bladies\b": "female",
        r"\bgirl\b": "female",
        r"\bgirls\b": "female"
    }
    # -------------------------------------------------
    # Blood type normalization
    # -------------------------------------------------
    blood_map = {
        r"\ba positive\b": "A+",
        r"\ba plus\b": "A+",
        r"\ba-positive\b": "A+",
        r"\ba negative\b": "A-",
        r"\ba minus\b": "A-",
        r"\ba-negative\b": "A-",
        r"\bb positive\b": "B+",
        r"\bb plus\b": "B+",
        r"\bb-positive\b": "B+",
        r"\bb negative\b": "B-",
        r"\bb minus\b": "B-",
        r"\bb-negative\b": "B-",
        r"\bab positive\b": "AB+",
        r"\bab plus\b": "AB+",
        r"\bab-positive\b": "AB+",
        r"\bab negative\b": "AB-",
        r"\bab minus\b": "AB-",
        r"\bab-negative\b": "AB-",
        r"\bo positive\b": "O+",
        r"\bo plus\b": "O+",
        r"\bo-positive\b": "O+",
        r"\bo negative\b": "O-",
        r"\bo minus\b": "O-",
        r"\bo-negative\b": "O-"
    }
    # -------------------------------------------------
    # General synonym normalization
    # -------------------------------------------------
    synonyms = {
        r"\bphysician\b": "doctor",
        r"\bphysicians\b": "doctors",
        r"\bvisit\b": "admission",
        r"\bvisits\b": "admissions",
        r"\bpeople\b": "patients",
        r"\bperson\b": "patient",
        r"\btest\b": "medical test",
        r"\btests\b": "medical tests"
    }
    for pattern, replacement in gender_map.items():
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in blood_map.items():
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in synonyms.items():
        text = re.sub(pattern, replacement, text)
    return text
# =========================================================
# AUDIO RECORDING
# =========================================================
def record_audio(filename="input.wav", duration=6, fs=16000):
    print("\n🎤 Listening... Speak now...")
    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1
    )
    sd.wait()
    write(filename, fs, recording)
# =========================================================
# SPEECH TO TEXT
# =========================================================
def speech_to_text(audio_file):
    segments, _ = model.transcribe(
        audio_file,
        language="en",
        beam_size=1,
        vad_filter=True
    )
    text = ""
    print("🎤 ", end="", flush=True)
    for segment in segments:
        print(segment.text, end="", flush=True)
        text += segment.text
    print()
    return text.strip()
# =========================================================
# TEXT TO SQL
# =========================================================
def text_to_sql(text):
    prompt = f"""
You are an expert PostgreSQL Data Analyst.

Generate ONLY valid PostgreSQL SELECT query.

STRICT RULES:
- ONLY SELECT queries
- NEVER use UPDATE, DELETE, DROP, INSERT, ALTER, TRUNCATE
- No explanation
- No markdown
- No ```sql
- Return SQL only
- If the user question is NOT related to the provided database schema, or asks about entities/columns/tables that do not exist (example: football players, goals, sales, employees, products, schools, etc.), Return EXACTLY this: INVALID_QUERY
- NEVER guess missing tables.
- NEVER invent business domains outside the medical database.
- ONLY answer using the provided schema.
- If the request cannot be answered using the schema, return INVALID_QUERY.
- DO NOT generate fake SQL.

ALIAS RULE:
- If you use a table alias like p, a, d, h, m: you MUST define it in FROM or JOIN.
- For single-table queries: DO NOT use aliases unless necessary.


SMART RULES:
- Use LIMIT 100 unless user asks for all
- Use JOIN automatically when needed
- NEVER use JOIN ON 1=1
- NEVER use CROSS JOIN
- Use proper foreign-key JOINs only
- Use GROUP BY for aggregation queries
- Use ORDER BY DESC for ranked results

VERY IMPORTANT:
medical_condition exists ONLY in admissions table.

GENERAL SQL SAFETY RULES:
1. NEVER assume columns exist in the wrong table.
Examples:
- medical_condition exists ONLY in admissions
- gender exists ONLY in patients
- age exists ONLY in patients
- blood_type exists ONLY in patients
- doctor_name exists ONLY in doctors
- specialization exists ONLY in doctors
- hospital_name exists ONLY in hospitals
- medication_name exists ONLY in medications
- test_result exists ONLY in medical_tests

2. ALWAYS use proper JOIN paths using foreign keys.
Examples:
To access gender:
admissions -> patients
To access doctor_name:
admissions -> doctors
To access hospital_name:
admissions -> hospitals
To access medication_name:
admissions -> medications
To access test_result:
admissions -> medical_tests
NEVER skip relationship paths.

3. NEVER perform mathematical operations on TEXT columns.
Examples:
WRONG:
test_result - age
WRONG:
medical_condition + billing_amount
If test_result is text:
- use GROUP BY
- use COUNT()
- use comparison by category
- use distribution analysis
DO NOT use subtraction or AVG unless column is numeric.

4. For comparison queries:
Use grouping comparison, not mathematical subtraction.
Example:
"comparison between test results and gender"
Correct:
SELECT
    p.gender,
    m.test_result,
    COUNT(*) AS total
FROM admissions a
JOIN patients p
    ON a.patient_id = p.patient_id
JOIN medical_tests m
    ON a.admission_id = m.admission_id
GROUP BY p.gender, m.test_result
ORDER BY total DESC
LIMIT 100;

5. For comparison with age:
Use age grouping or aggregation.
Correct:
SELECT
    p.age,
    m.test_result,
    COUNT(*) AS total
FROM admissions a
JOIN patients p
    ON a.patient_id = p.patient_id
JOIN medical_tests m
    ON a.admission_id = m.admission_id
GROUP BY p.age, m.test_result
ORDER BY total DESC
LIMIT 100;
DO NOT generate invalid arithmetic operations.

NEVER use:
patients.medical_condition
ALWAYS use:
admissions.medical_condition
doctor_name exists ONLY in doctors table.
hospital_name exists ONLY in hospitals table.
medication_name exists ONLY in medications table.

HOSPITAL STAY DURATION RULE:
For hospital stay duration calculations:
Use:
(discharge_date - date_of_admission)
DO NOT use:
EXTRACT(EPOCH FROM ...)

DO NOT use:
AGE()
because both columns are DATE type and subtraction directly returns number of days.

SPECIAL COUNT RULES:
If user asks:
"How many blood types"
Use:
SELECT COUNT(DISTINCT blood_type)
FROM patients;
If user asks:
"How many specializations"
Use:
SELECT COUNT(DISTINCT specialization)
FROM doctors;
If user asks:
"How many doctors"
Use:
SELECT COUNT(DISTINCT doctor_id)
FROM doctors;
If user asks:
"How many hospitals"
Use:
SELECT COUNT(DISTINCT hospital_name)
FROM hospitals;
If user asks:
"How many medications"
Use:
SELECT COUNT(DISTINCT medication_name)
FROM medications;
If user asks:
"How many blood types, specializations and doctors"
Use EXACTLY:

SELECT
    (SELECT COUNT(DISTINCT blood_type) FROM patients) AS blood_types,
    (SELECT COUNT(DISTINCT specialization) FROM doctors) AS specializations,
    (SELECT COUNT(DISTINCT doctor_id) FROM doctors) AS doctors;
DO NOT JOIN unrelated tables for simple counts.

6. NEVER invent tables that do not exist.
Examples:
WRONG:
JOIN specializations s
WRONG:
FROM conditions
WRONG:
JOIN blood_types b
Because these are columns, not tables.
CORRECT:
Use doctors.specialization
Use admissions.medical_condition
Use patients.blood_type
Only use tables that exist in schema:
patients
doctors
hospitals
admissions
medications
medical_tests
insurance_providers

7. For time-based filtering like:
"last 30 days"
"last month"
"recent admissions"
DO NOT use CURRENT_DATE or NOW() for historical datasets.
Instead use the latest available date inside the dataset.
Correct:
WHERE date_of_admission >= (
    SELECT MAX(date_of_admission) - INTERVAL '30 day'
    FROM admissions
)
because dataset may not contain current real-time dates.

8. medical_condition exists ONLY in admissions table.
NEVER use:
patients.medical_condition
doctors.medical_condition
hospitals.medical_condition
ALWAYS use:
admissions.medical_condition
Example:
Correct:
WHERE a.medical_condition ILIKE '%asthma%'

MATCHING RULES:
- gender → exact match using LOWER()
- blood_type → exact match using UPPER()
- specialization → exact match
- names → partial match using ILIKE
- hospital → partial match using ILIKE
- medication → partial match using ILIKE
- condition → partial match using ILIKE

DATABASE SCHEMA:
patients(
    patient_id,
    patient_name,
    age,
    gender,
    blood_type
)

hospitals(
    hospital_id,
    hospital_name
)

insurance_providers(
    insurance_id,
    provider_name
)

doctors(
    doctor_id,
    doctor_name,
    specialization,
    hospital_id
)

admissions(
    admission_id,
    patient_id,
    doctor_id,
    hospital_id,
    insurance_id,
    medical_condition,
    admission_type,
    room_number,
    date_of_admission,
    discharge_date,
    billing_amount
)

medications(
    medication_id,
    admission_id,
    medication_name
)

medical_tests(
    test_id,
    admission_id,
    test_result
)

RELATIONSHIPS:
doctors.hospital_id → hospitals.hospital_id
admissions.patient_id → patients.patient_id
admissions.doctor_id → doctors.doctor_id
admissions.hospital_id → hospitals.hospital_id
admissions.insurance_id → insurance_providers.insurance_id
medications.admission_id → admissions.admission_id
medical_tests.admission_id → admissions.admission_id

GOOD EXAMPLE:
Question:
Doctors with highest number of hypertension patients
Correct SQL:
SELECT
    d.doctor_name,
    COUNT(*) AS total_patients
FROM admissions a
JOIN doctors d
    ON a.doctor_id = d.doctor_id
WHERE a.medical_condition ILIKE '%hypertension%'
GROUP BY d.doctor_name
ORDER BY total_patients DESC
LIMIT 10;

Question:
Show average hospital stay duration by medical condition
Correct SQL:
SELECT
    medical_condition,
    AVG(discharge_date - date_of_admission) AS avg_stay_days
FROM admissions
GROUP BY medical_condition
ORDER BY avg_stay_days DESC
LIMIT 100;
User Question:
{text}
SQL:
"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )
            sql = response.choices[0].message.content.strip()
            if "INVALID_QUERY" in sql.upper():
                return "INVALID_QUERY"
            # -------------------------------------------------
            # Output cleaning
            # -------------------------------------------------
            sql = sql.replace("```sql", "")
            sql = sql.replace("```", "")
            sql = sql.strip()
            if "SQL:" in sql:
                sql = sql.split("SQL:")[-1].strip()
            if "SELECT" in sql.upper():
                sql = sql[sql.upper().find("SELECT"):]
            if not sql.endswith(";"):
                sql += ";"
            return sql
        except Exception:
            print("⚠️ Rate limited, retrying...")
            time.sleep(2)
    raise Exception("LLM failed after retries")
# =========================================================
# STREAMING QUERY EXECUTION
# =========================================================
@lru_cache(maxsize=32)
def run_query_cached(sql, chunk_size=100):
    conn = None
    cur = None
    try:
        conn = db_pool.getconn()
        print("✅ Connection taken from pool")
        cur = conn.cursor()
        if "limit" not in sql.lower():
            sql = sql.rstrip(";") + f" LIMIT {chunk_size};"
        cur.execute(sql)
        rows = cur.fetchmany(chunk_size)
        if not cur.description:
            return pd.DataFrame()
        colnames = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=colnames)
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if cur:
            cur.close()
        if conn:
            db_pool.putconn(conn)
            print("✅ Connection returned to pool")
# =========================================================
def get_cached_sql(normalized_text):
    conn = db_pool.getconn()
    cur = conn.cursor()

    try:
        cur.execute(
            "SELECT sql_query FROM query_cache WHERE normalized_question = %s",
            (normalized_text,)
        )
        result = cur.fetchone()
        return result[0] if result else None

    finally:
        cur.close()
        db_pool.putconn(conn)
# =========================================================
def save_query_to_cache(normalized_text, sql):
    conn = db_pool.getconn()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO query_cache (normalized_question, sql_query)
            VALUES (%s, %s)
            ON CONFLICT (normalized_question) DO NOTHING
            """,
            (normalized_text, sql)
        )
        conn.commit()

    finally:
        cur.close()
        db_pool.putconn(conn)
# =========================================================
# Business Insight Function
# =========================================================
def generate_business_insight(user_question, sql, result_df):
    if result_df.empty:
        return "No data available for business insights."
    table_preview = result_df.head(10).to_string(index=False)
    prompt = f"""
You are a senior Healthcare Business Intelligence Analyst.
Your job is to generate accurate business insights based on:
1. User Question
2. Generated SQL Query
3. Actual Query Result
IMPORTANT RULES:
- NEVER invent assumptions
- NEVER give generic advice
- ONLY analyze based on real data shown
- Understand what the user actually asked
- Interpret the numbers correctly
- Be specific and realistic
- Keep insights concise and professional
USER QUESTION:
{user_question}
GENERATED SQL:
{sql}
QUERY RESULT:
{table_preview}
Generate:
1. Business Insight
2. Important Finding
3. Actionable Recommendation
Return clean professional text only.
"""
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Could not generate business insights."
# =========================================================
# MAIN LOOP
# =========================================================
def run_voice_assistant():
    print("\n🎤 Voice Medical Assistant Ready (say 'exit' to stop)\n")
    while True:
        record_audio(duration=6)
        text = speech_to_text("input.wav")
        print("\n🎤 You said:", text)
        if not text:
            print("⚠️ No speech detected. Please try again.")
            print("\n" + "=" * 80)
            continue
        if "exit" in text.lower() or "stop" in text.lower():
            print("👋 Exiting...")
            break
        normalized_text = normalize_text(text)
        print("\n🧹 Normalized:", normalized_text)
        # 1. Try cache first
        sql = get_cached_sql(normalized_text)

        if sql:
            print("⚡ Using cached SQL")
        else:
            # 2. Generate SQL
            sql = text_to_sql(normalized_text)

            # 3. Save it
            if sql != "INVALID_QUERY":
                save_query_to_cache(normalized_text, sql)
        print("\n🧠 SQL:", sql)
        if sql == "INVALID_QUERY":
            print("Please ask only about patients, doctors, hospitals, admissions, medications, or medical tests.")
            continue
        if "select" not in sql.lower():
            print("❌ Invalid SQL generated")
            continue
        try:
            result = run_query_cached(sql)
            print("\n📊 Result Table:\n")
            if result.empty:
                print("No matching records found.")
            else:
                print(
                    tabulate(
                        result.head(20),
                        headers="keys",
                        tablefmt="psql",
                        showindex=False
                    )
                )
                # Add AI Business Insights here
                insight = generate_business_insight(
                    user_question=text,
                    sql=sql,
                    result_df=result
                )
                print("\n🧠 AI Business Insights\n")
                print(insight)
        except Exception as e:
            print("\n❌ Query Error:", e)
        print("\n" + "=" * 80)
# =========================================================
# SAFE ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run_voice_assistant()