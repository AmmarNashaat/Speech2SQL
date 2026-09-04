import streamlit as st
import pandas as pd
import plotly.express as px
from app import get_cached_sql, save_query_to_cache
from app import (
    record_audio,
    speech_to_text,
    normalize_text,
    text_to_sql,
    run_query_cached,
    generate_business_insight,
    client
)
if "executed" not in st.session_state:
    st.session_state.executed = False

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Medical Assistant",
    layout="wide",
    page_icon="🎤"
)

# =========================================================
# CLEAN UI STYLE
# =========================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .stButton > button {
            width: 100%;
            height: 70px;
            border-radius: 50px;
            font-size: 24px;
            font-weight: bold;
            background-color: #4B8BBE;
            color: white;
            border: none;
        }

        .stButton > button:hover {
            background-color: #366a96;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>
        AI Medical Assistant
    </h1>

    <h4 style='text-align: center; color: gray;'>
        Smart Medical Database Assistant with Visualization & AI Insights
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "sql" not in st.session_state:
    st.session_state.sql = ""

if "text" not in st.session_state:
    st.session_state.text = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

# =========================================================
# HYBRID SMART CHART SELECTION ENGINE
# =========================================================

def choose_best_chart(df, sql=""):
    """
    Hybrid BI Chart Selection Engine

    Decision based on:
    1. SQL intent
    2. Column data types
    3. Cardinality
    4. Safe fallback

    Returns:
    - pie
    - bar
    - line
    - scatter
    - histogram
    - table
    """

    if df is None or df.empty:
        return "table"

    sql = sql.lower()

    # ---------------------------------
    # Detect column types
    # ---------------------------------

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    object_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    datetime_cols = df.select_dtypes(
        include=["datetime64"]
    ).columns.tolist()

    # ---------------------------------
    # KPI single-value result
    # Example:
    # SELECT COUNT(*) ...
    # ---------------------------------

    if len(df.columns) == 1 and len(df) == 1:
        return "table"

    # ---------------------------------
    # Trend Analysis
    # Example:
    # admissions over time
    # billing by admission date
    # ---------------------------------

    if "date" in sql or len(datetime_cols) >= 1:
        if len(numeric_cols) >= 1:
            return "line"

    # ---------------------------------
    # Distribution Analysis
    # Example:
    # age distribution
    # billing distribution
    # ---------------------------------

    if "age" in sql:
        return "histogram"

    if len(numeric_cols) == 1 and len(object_cols) == 0:
        return "histogram"

    # ---------------------------------
    # Comparison / Count / Group By
    # Example:
    # count by gender
    # patients by doctor
    # admissions by hospital
    # ---------------------------------

    if "count(" in sql or "group by" in sql:

        if len(object_cols) >= 1 and len(numeric_cols) >= 1:

            first_cat_col = object_cols[0]
            unique_vals = df[first_cat_col].nunique()

            # Few categories → Pie
            if unique_vals <= 6:
                return "pie"

            # Many categories → Bar
            return "bar"

    # ---------------------------------
    # Aggregation Analysis
    # Example:
    # AVG billing by condition
    # SUM revenue by hospital
    # ---------------------------------

    if "avg(" in sql or "sum(" in sql or "max(" in sql or "min(" in sql:

        if len(object_cols) >= 1 and len(numeric_cols) >= 1:
            return "bar"

    # ---------------------------------
    # Correlation Analysis
    # Example:
    # age vs billing
    # stay duration vs billing
    # ---------------------------------

    if len(numeric_cols) >= 2:
        return "scatter"

    # ---------------------------------
    # Safe fallback
    # ---------------------------------

    return "table"

# =========================================================
# AI BUSINESS INSIGHTS
# =========================================================
def generate_ai_summary(user_question, sql, df):

    if df is None or df.empty:
        return "No data available for AI analysis."

    return generate_business_insight(
        user_question=user_question,
        sql=sql,
        result_df=df
    )

# =========================================================
# MICROPHONE BUTTON (PERFECTLY CENTERED)
# =========================================================

st.markdown(
    """
    <h3 style='text-align: center; margin-top: 10px;'>
        🎙️ Tap the microphone to ask your question
    </h3>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([2, 0.8, 2])

with col2:
    run_btn = st.button("🎤 Start Voice Query")

# =========================================================
# MAIN PIPELINE
# =========================================================

if run_btn:
    st.session_state.executed = True

    # =====================================================
    # STEP 1 — VOICE INPUT ONLY
    # =====================================================

    st.info("🎤 Listening... Please speak now")

    record_audio(duration=6)
    text = speech_to_text("input.wav")

    st.session_state.text = text

    if not text.strip():
        st.warning("No voice input detected.")
        st.stop()

    st.subheader("🗣️ Recognized Speech")
    st.write(text)

    # =====================================================
    # STEP 2 — NORMALIZATION
    # =====================================================

    normalized = normalize_text(text)

    # =====================================================
    # STEP 3 — SQL GENERATION
    # =====================================================

    # 1. Try cache first
    sql = get_cached_sql(normalized)

    if sql:
        st.info("⚡ Using cached query")
    else:
        sql = text_to_sql(normalized)

        if sql != "INVALID_QUERY":
            save_query_to_cache(normalized, sql)
    st.session_state.sql = sql

    if sql == "INVALID_QUERY":
        st.error("❌ This question is outside the medical database scope.")
        st.stop()

    # =====================================================
    # STEP 4 — QUERY EXECUTION
    # =====================================================

    try:
        df = run_query_cached(sql)
        st.session_state.df = df

    except Exception as e:
        st.error(f"❌ Query Error: {str(e)}")
        st.stop()

    # =====================================================
    # TOP METRICS
    # =====================================================

    chart_type = choose_best_chart(df)

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            label="Rows Returned",
            value=len(df)
        )

    with metric2:
        st.metric(
            label="Columns",
            value=len(df.columns)
        )

    with metric3:
        st.metric(
            label="Visualization",
            value=chart_type.upper() if chart_type else "TABLE"
        )

    st.markdown("---")

    # =====================================================
    # SQL + DATA PREVIEW
    # =====================================================

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🧠 Generated SQL")
        st.code(sql, language="sql")

    with col2:
        st.subheader("📊 Data Preview")

        if df.empty:
            st.warning("No results found.")
        else:
            st.dataframe(
                df,
                use_container_width=True
            )
    # =========================================================
    # SMART VISUALIZATION BLOCK
    # =========================================================
    # Input: DataFrame + SQL string
    #     ↓
    #    1. Empty data?          → table
    #    2. Has date/datetime?   → line
    #    3. Has "age" in SQL?    → histogram
    #    4. Single numeric col?  → histogram
    #    5. Has COUNT/GROUP BY?  → pie (≤6 unique vals) or bar (>6)
    #    6. Has AVG/SUM/MAX/MIN? → bar
    #    7. Has 2+ numeric cols? → scatter
    #    8. Nothing matched?     → table (fallback)
    if df is not None and not df.empty:

        chart_type = choose_best_chart(df, sql)

        st.markdown("---")
        st.subheader("📈 Smart Visualization")

        fig = None

        try:
            # ---------------------------------
            # BAR CHART
            # ---------------------------------

            if chart_type == "bar":
                fig = px.bar(
                    df,
                    x=df.columns[0],
                    y=df.columns[1],
                    title="Bar Chart Analysis"
                )

            # ---------------------------------
            # LINE CHART
            # ---------------------------------

            elif chart_type == "line":
                fig = px.line(
                    df,
                    x=df.columns[0],
                    y=df.columns[1],
                    title="Trend Analysis"
                )

            # ---------------------------------
            # SCATTER PLOT
            # ---------------------------------

            elif chart_type == "scatter":
                fig = px.scatter(
                    df,
                    x=df.columns[0],
                    y=df.columns[1],
                    title="Correlation Analysis"
                )

            # ---------------------------------
            # PIE CHART
            # ---------------------------------

            elif chart_type == "pie":
                fig = px.pie(
                    df,
                    names=df.columns[0],
                    values=df.columns[1],
                    title="Composition Analysis"
                )

            # ---------------------------------
            # HISTOGRAM
            # ---------------------------------

            elif chart_type == "histogram":
                fig = px.histogram(
                    df,
                    x=df.columns[0],
                    title="Distribution Analysis"
                )

            # ---------------------------------
            # TABLE FALLBACK
            # ---------------------------------

            elif chart_type == "table":
                st.info("Table view is the best visualization for this dataset.")

            # ---------------------------------
            # Render Chart
            # ---------------------------------

            if fig:
                fig.update_layout(
                    height=500
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except Exception as e:
            st.warning(
                f"Visualization issue: {str(e)}"
            )

    # =====================================================
    # AI BUSINESS INSIGHTS
    # =====================================================

    st.markdown("---")
    st.subheader("🧠 AI Business Insights")

    with st.spinner("We are analyzing your data..."):
        summary = generate_ai_summary(
            user_question=text,
            sql=sql,
            df=df
        )
        st.session_state.summary = summary
    st.success(summary)