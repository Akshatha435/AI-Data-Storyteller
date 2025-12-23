# app_streamlit.py
"""
AI Data Storyteller – Interactive Dashboard

Flow:
1. Landing: 3-step explanation cards
2. Upload & preprocessing
3. Tabs:
   - Overview & EDA summary
   - Visual analytics (uni / bi / multi)
   - AI narrative & Q&A
   - Export options
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Streamlit page config ----------
st.set_page_config(page_title="AI Data Storyteller", layout="wide", page_icon="📊")
# ---------- Session state initialization ----------
if "auto_ai_story" not in st.session_state:
    st.session_state["auto_ai_story"] = None

if "report_ai_answers" not in st.session_state:
    st.session_state["report_ai_answers"] = []
if "report_visuals" not in st.session_state:
    st.session_state["report_visuals"] = []
    
# -------------------------------
# DEFAULT AI INSIGHT PROMPT
# -------------------------------
default_insight_question = """
Provide a comprehensive, end-to-end narrative explaining this dataset.

Cover:
1. What this dataset represents in real-world terms
2. Overall structure, data quality, and variable types
3. Key patterns and trends
4. Relationships and drivers influencing outcomes
5. Risks, anomalies, and warning signals
6. Opportunities for optimization and improvement
7. Strategic decisions this data can support
8. Future scope including predictive and advanced use cases

Explain everything in simple business language for non-technical stakeholders.
"""
# -----------------------
def ask_llm_about_data(question: str, eda_results: dict, df):
    try:
        from openai import OpenAI
        import streamlit as st

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        data_summary = f"""
        Dataset shape: {df.shape}
        Columns: {list(df.columns)}
        EDA summary: {eda_results}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior data analyst. "
                        "Explain insights in executive-level, business-focused language. "
                        "Avoid technical jargon."
                    )
                },
                {
                    "role": "user",
                    "content": f"{question}\n\nContext:\n{data_summary}"
                }
            ],
            max_tokens=900
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ AI Error: {e}"

# ---------- Global styling (dark navy background + visibility) ----------
st.markdown(
    """
    <style>
        /* App background – Option A (navy to dark blue/black) */
        .stApp {
            background: radial-gradient(circle at top left,
                                        #020764 0%,
                                        #030b25 40%,
                                        #000000 100%);
            color: #F5F5FF;
        }

        .block-container {
            max-width: 1150px;
            padding-top: 1.2rem;
            padding-bottom: 2.0rem;
            margin: 0 auto;
        }

        /* Headings & labels visible on dark bg */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
        }
        label[data-testid="stWidgetLabel"] {
            color: #F5F5FF !important;
            font-size: 0.86rem;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(3, 8, 18, 0.9);
            color: #E5F0FF;
            border-radius: 999px;
            padding: 4px 14px;
            font-size: 0.83rem;
            border: 1px solid rgba(2, 94, 196, 0.5);
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #043780;   /* darker blue */
    color: white;
    border-color: #0ECCED;
}


        /* Primary buttons */
        .stButton > button {
            background-color: #025EC4;
            color: white;
            border-radius: 999px;
            border: 1px solid #0ECCED;
            padding: 0.35rem 1.1rem;
            font-size: 0.86rem;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #043780;
            border-color: #0ECCED;
        }

        /* Download buttons (cleaned CSV + PDF) */
        div[data-testid="stDownloadButton"] > button {
            background-color: #025EC4 !important;
            color: #FFFFFF !important;
            border-radius: 999px !important;
            border: 1px solid #0ECCED !important;
            padding: 0.35rem 1.1rem !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background-color: #043780 !important;
        }

        /* File uploader – clearly visible */
        div[data-testid="stFileUploadDropzone"] {
            background-color: rgba(255,255,255,0.06);
            border: 1px dashed rgba(14,204,237,0.7);
        }
        div[data-testid="stFileUploadDropzone"] span {
            color: #F5F5FF !important;
        }

        /* Light info card */
        .light-card {
            background-color: rgba(2, 7, 100, 0.18);
            border-radius: 14px;
            border: 1px solid rgba(14, 204, 237, 0.35);
            padding: 14px 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Paths / imports from code/ ----------
CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

try:
    from eda import load_csv, run_eda
except Exception as e:
    st.error(
        "Could not import functions from code/eda.py. "
        "Ensure load_csv and run_eda exist. Error: " + str(e)
    )
    st.stop()

try:
    from llm_interface import get_prompt_for_eda
except Exception:
    get_prompt_for_eda = None

try:
    from report_generator import create_report
except Exception:
    create_report = None

# OpenAI client (new SDK)
try:
    from openai import OpenAI
    openai_client = OpenAI()  # uses OPENAI_API_KEY from environment
except Exception:
    openai_client = None

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(exist_ok=True)

# ---------- Session state for report selections ----------
if "report_visuals" not in st.session_state:
    st.session_state["report_visuals"] = []  # list of dict: {image,title,insight}
if "report_ai_answers" not in st.session_state:
    st.session_state["report_ai_answers"] = []  # list of dict: {question,answer}
if "combined_text_for_report" not in st.session_state:
    st.session_state["combined_text_for_report"] = ""
if "auto_ai_story" not in st.session_state:
    st.session_state["auto_ai_story"] = None

# =====================================================
#   HEADER + 3 STREAMLIT CARDS (NO HTML)
# =====================================================
st.title("🤖 AI Data Storyteller")
st.write(
    "Turn raw data into insights, visuals, and an AI-generated narrative — all in one place."
)

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.subheader("📥 Upload & Prepare")
        st.write(
            "- Upload any CSV once\n"
            "- Remove duplicates\n"
            "- Handle missing values\n"
            "- Drop mostly-empty columns\n\n"
            "Start your analysis with a clean dataset."
        )

with c2:
    with st.container(border=True):
        st.subheader("📊 Visual Analytics")
        st.write(
            "- Explore Univariate, bivariate, multivariate views\n"
            "- Uncover distributions, relationships, patterns that \n"
            " Drive data backed decisions \n"
        )

with c3:
    with st.container(border=True):
        st.subheader("🤖 AI Narrative & Export")
        st.write(
            "- Consolidate insights, visuals and AI interpretations \n"
            " into a professional report to support"
            "decision-making \n"
        )

st.markdown("---")

# =====================================================
#   STEP 1 – UPLOAD + PREPROCESSING
# =====================================================
st.subheader("Step 1 – Upload dataset")
uploaded = st.file_uploader("Upload a CSV file to start the analysis", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to unlock cleaning, visual analytics, AI narrative and export.")
    st.stop()

st.subheader("Step 2 – Preprocessing options")

col_left, col_right = st.columns(2)
with col_left:
    drop_duplicates = st.checkbox("Drop duplicate rows", value=True)
    missing_thresh = st.slider("Drop columns with more than % missing", 0, 90, 60)
    remove_outliers = st.checkbox("Remove outliers by IQR (numeric columns)", value=False)
with col_right:
    num_fill_strategy = st.selectbox(
        "Numeric missing value strategy", ["median", "mean", "zero"], index=0
    )
    cat_fill_strategy = st.selectbox(
        "Categorical missing value strategy", ["mode", "Unknown"], index=0
    )
    one_hot = st.checkbox("One-hot encode categorical columns", value=False)
    scale_numeric = st.checkbox("Standard scale numeric columns", value=False)

# Load raw data
df_raw = load_csv(uploaded)

st.markdown("#### Raw data preview")
st.write(f"Rows: {df_raw.shape[0]} — Columns: {df_raw.shape[1]}")
st.dataframe(df_raw.head(6))

# Apply preprocessing
df = df_raw.copy()

if drop_duplicates:
    before = df.shape[0]
    df = df.drop_duplicates()
    st.write(f"Removed duplicates: {before - df.shape[0]} rows")

missing_pct = df.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > missing_thresh].index.tolist()
if cols_to_drop:
    st.write(f"Dropping columns with >{missing_thresh}% missing: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

for c in num_cols:
    if df[c].isnull().any():
        if num_fill_strategy == "median":
            df[c] = df[c].fillna(df[c].median())
        elif num_fill_strategy == "mean":
            df[c] = df[c].fillna(df[c].mean())
        else:
            df[c] = df[c].fillna(0)

for c in cat_cols:
    if df[c].isnull().any():
        if cat_fill_strategy == "mode":
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown")
        else:
            df[c] = df[c].fillna("Unknown")

if one_hot and cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

if scale_numeric and num_cols:
    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    except Exception:
        st.warning("scikit-learn not installed; skipping scaling.")

if remove_outliers and num_cols:
    initial = df.shape[0]
    for c in num_cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df = df[(df[c] >= low) & (df[c] <= high)]
    st.write(f"Rows after outlier removal: {df.shape[0]} (dropped {initial - df.shape[0]})")

st.markdown("#### Cleaned data preview")
st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
st.dataframe(df.head(6))

st.download_button(
    "Download cleaned CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_data.csv",
)

# Run EDA once
eda_results = run_eda(df)

# =====================================================
#   COMMON UTILS
# =====================================================
def save_figure(fig, name_prefix: str) -> str:
    """Save fig as full-size PNG, return path."""
    path_full = OUT_DIR / f"{name_prefix}.png"
    fig.savefig(path_full, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(path_full)


def make_pdf_safe(text: str) -> str:
    """Avoid latin-1 encoding errors inside some PDF libraries."""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("latin-1", "replace").decode("latin-1")


def ask_llm_about_data(question: str, eda_results: dict, df: pd.DataFrame) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")

    if openai_client is not None and api_key:
        if get_prompt_for_eda is not None:
            eda_prompt = get_prompt_for_eda(eda_results, max_chars=2000)
        else:
            eda_prompt = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}."

        sample_csv = df.head(25).to_csv(index=False)

        system_msg = (
            "You are a senior business and data analytics consultant. "
            "Explain insights in structured, decision-oriented language."
        )

        user_msg = (
            f"Question:\n{question}\n\n"
            f"EDA Summary:\n{eda_prompt}\n\n"
            f"Sample data:\n{sample_csv}"
        )

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.35,
                max_tokens=700,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"AI error: {e}"

    return (
        "AI insights unavailable. Based on the data structure, focus on trends, "
        "outliers, relationships, and segments that influence outcomes."
    )
def to_executive_bullets(ai_text: str) -> list:
    """
    Converts AI narrative into executive-ready bullet insights
    for non-technical stakeholders.
    """
    if not ai_text or not isinstance(ai_text, str):
        return []

    lines = [l.strip() for l in ai_text.split("\n") if len(l.strip()) > 25]

    bullets = []
    for line in lines:
        clean = (
            line.replace("•", "")
                .replace("-", "")
                .replace("1.", "")
                .replace("2.", "")
                .replace("3.", "")
                .strip()
        )
        if clean and clean not in bullets:
            bullets.append(clean)

        if len(bullets) == 6:  # executive limit
            break

    return bullets


def format_as_executive_bullets(text: str):
    """
    Converts AI text into executive bullet points.
    """
    if not text:
        return []

    lines = [
        l.strip("-• ").strip()
        for l in text.split("\n")
        if len(l.strip()) > 12
    ]

    bullets = []
    for l in lines:
        if not l.endswith("."):
            l += "."
        bullets.append(f"- {l}")

    return bullets[:6]  # executive limit
# ---------- Session state initialization ----------
if "auto_ai_story" not in st.session_state:
    st.session_state["auto_ai_story"] = None

if "report_ai_answers" not in st.session_state:
    st.session_state["report_ai_answers"] = []

# =====================================================
#   STEP 3 – TABS (OVERVIEW, VISUALS, Q&A, EXPORT)
# =====================================================
st.markdown("---")
st.subheader("Step 3 – Explore, ask questions and export")

tab_overview, tab_visuals, tab_qna, tab_export = st.tabs(
    ["Overview & EDA summary", "Visual analytics", "AI narrative & Q&A", "Export options"]
)

# ---------- TAB 1: OVERVIEW ----------
with tab_overview:
    st.markdown("### Dataset overview")

    st.write(
        f"- **Shape:** {eda_results['summary']['shape'][0]} rows × "
        f"{eda_results['summary']['shape'][1]} columns."
    )

    st.markdown("**Detected column types**")
    st.write({k: eda_results["types"].get(k, [])[:10] for k in eda_results["types"]})

    st.markdown("**Missing values (top 10 columns)**")
    st.write(
        dict(
            sorted(
                eda_results["summary"]["missing_values"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        )
    )

def ai_explain_chart(chart_type: str, columns: list, df: pd.DataFrame) -> str:
    """
    AI explanation for charts – stakeholder-ready insights.
    Uses OpenAI if available, otherwise falls back to rule-based insight.
    """

    api_key = os.environ.get("OPENAI_API_KEY")

    # ---------- AI path ----------
    if openai_client and api_key:
        try:
            sample = df[columns].head(20).to_csv(index=False)

            prompt = (
                "You are a senior business data analyst.\n\n"
                f"Chart type: {chart_type}\n"
                f"Columns involved: {columns}\n\n"
                "Explain:\n"
                "1) What this chart reveals about the data\n"
                "2) Why it matters for decision-making\n"
                "3) What actions a stakeholder might consider\n\n"
                f"Sample data:\n{sample}"
            )

            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You explain charts clearly for business users."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.35,
                max_tokens=220,
            )
            return resp.choices[0].message.content.strip()

        except Exception:
            pass  # fall back safely

    # ---------- Fallback (NO AI) ----------
    return (
        f"This {chart_type.lower()} highlights patterns across {', '.join(columns)}. "
        "It helps identify dominant segments, trends, or relationships that may impact "
        "performance, planning, or prioritization decisions."
    )

# ---------- TAB 2: VISUAL ANALYTICS ----------
with tab_visuals:
    st.markdown("### Visual analytics")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ===============================
    # UNIVARIATE ANALYSIS
    # ===============================
    st.markdown("## Univariate analysis")

    u1, u2 = st.columns(2)
    with u1:
        uni_col = st.selectbox("Select column", df.columns)
    with u2:
        uni_chart = st.selectbox(
            "Chart type",
            ["Histogram", "Bar chart", "Boxplot", "Pie chart"]
        )

    fig, ax = plt.subplots()

    if uni_chart == "Histogram" and uni_col in num_cols:
        ax.hist(df[uni_col].dropna(), bins=30, color="#025EC4")
    elif uni_chart == "Bar chart":
        df[uni_col].value_counts().head(10).plot(kind="bar", ax=ax, color="#025EC4")
    elif uni_chart == "Boxplot" and uni_col in num_cols:
        sns.boxplot(x=df[uni_col], ax=ax, color="#025EC4")
    elif uni_chart == "Pie chart":
        df[uni_col].value_counts().head(6).plot(kind="pie", ax=ax, autopct="%1.1f%%")

    ax.set_title(f"{uni_chart} – {uni_col}")
    st.pyplot(fig)

    if st.button("Save univariate chart to report"):
        img_path = save_figure(fig, f"uni_{uni_col}")
        st.session_state["report_visuals"].append(
            {
                "image": img_path,
                "title": f"{uni_chart} – {uni_col}",
                "insight": ""  # intentionally empty
            }
        )
        st.success("Univariate chart saved.")

    # ===============================
    # BIVARIATE ANALYSIS
    # ===============================
    st.markdown("## Bivariate analysis")

    b1, b2, b3 = st.columns(3)
    with b1:
        x_col = st.selectbox("X axis", df.columns, key="bi_x")
    with b2:
        y_col = st.selectbox("Y axis", df.columns, key="bi_y")
    with b3:
        bi_chart = st.selectbox(
            "Chart type",
            ["Scatter plot", "Grouped bar chart", "Line chart", "Pie chart"]
        )

    fig, ax = plt.subplots()

    if bi_chart == "Scatter plot" and x_col in num_cols and y_col in num_cols:
        ax.scatter(df[x_col], df[y_col], alpha=0.6, color="#025EC4")
    elif bi_chart == "Grouped bar chart":
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
    elif bi_chart == "Line chart":
        ax.plot(df[x_col], df[y_col], color="#025EC4")
    elif bi_chart == "Pie chart":
        df.groupby(x_col)[y_col].sum().head(6).plot(kind="pie", ax=ax, autopct="%1.1f%%")

    ax.set_title(f"{bi_chart} – {x_col} vs {y_col}")
    st.pyplot(fig)

    if st.button("Save bivariate chart to report"):
        img_path = save_figure(fig, f"bi_{x_col}_{y_col}")
        st.session_state["report_visuals"].append(
            {
                "image": img_path,
                "title": f"{bi_chart} – {x_col} vs {y_col}",
                "insight": ""
            }
        )
        st.success("Bivariate chart saved.")
    # ===============================
# MULTIVARIATE ANALYSIS
# ===============================
st.markdown("## Multivariate analysis")

m1, m2 = st.columns(2)

with m1:
    multi_chart = st.selectbox(
        "Chart type",
        [
            "Correlation heatmap",
            "Stacked bar chart",
            "Boxplot by category",
            "Scatter with hue",
            "Line chart",
            "Pairplot (numeric only)",
        ],
        key="multi_chart_type",
    )

with m2:
    multi_cols = st.multiselect(
        "Select columns (2 or more)",
        df.columns,
        key="multi_cols",
    )

# ---------- VALIDATION ----------
if len(multi_cols) < 2:
    st.info("Select at least two columns to generate a multivariate chart.")

else:
    data = df[multi_cols].copy()

    # ---------- CORRELATION ----------
    if multi_chart == "Correlation heatmap":
        numeric_df = data.select_dtypes(include="number")

        if numeric_df.shape[1] < 2:
            st.warning("Select at least two numeric columns.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                numeric_df.corr(),
                cmap="Blues",
                annot=True,
                linewidths=0.5,
                ax=ax,
            )
            st.pyplot(fig)

    # ---------- STACKED BAR ----------
    elif multi_chart == "Stacked bar chart":
        fig, ax = plt.subplots(figsize=(6, 4))
        pivot = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
        pivot.plot(kind="bar", stacked=True, ax=ax)
        st.pyplot(fig)

    # ---------- BOXPLOT ----------
    elif multi_chart == "Boxplot by category":
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=df,
            x=multi_cols[0],
            y=multi_cols[1],
            ax=ax,
        )
        st.pyplot(fig)

    # ---------- SCATTER ----------
    elif multi_chart == "Scatter with hue":
        fig, ax = plt.subplots(figsize=(6, 4))

        if len(multi_cols) >= 3:
            sns.scatterplot(
                data=df,
                x=multi_cols[0],
                y=multi_cols[1],
                hue=multi_cols[2],
                ax=ax,
            )
        else:
            sns.scatterplot(
                data=df,
                x=multi_cols[0],
                y=multi_cols[1],
                ax=ax,
            )

        st.pyplot(fig)

    # ---------- LINE ----------
    elif multi_chart == "Line chart":
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df[multi_cols[0]], df[multi_cols[1]])
        st.pyplot(fig)

    # ---------- PAIRPLOT ----------
    elif multi_chart == "Pairplot (numeric only)":
        numeric_df = data.select_dtypes(include="number")

        if numeric_df.shape[1] < 2:
            st.warning("Select at least two numeric columns.")
        else:
            pair_fig = sns.pairplot(numeric_df)
            st.pyplot(pair_fig.fig)
            
# ---------- TAB 3: AI INSIGHTS ----------
with tab_qna:
    st.markdown("### AI insights")

    st.markdown(
        "<div class='light-card'>"
        "<b>What this section does</b><br>"
        "• Explains the dataset end-to-end in business terms<br>"
        "• Highlights patterns, risks, opportunities, and decisions<br>"
        "• Acts as a stakeholder-ready intelligence layer"
        "</div>",
        unsafe_allow_html=True,
    )

    if df is None or eda_results is None:
        st.info("Upload a dataset and complete EDA to generate AI insights.")
    else:
        if st.session_state["auto_ai_story"] is None:
            with st.spinner("Generating AI narrative..."):
                st.session_state["auto_ai_story"] =
        ask_llm_about_data(
                    default_insight_question,
                    eda_results,
                    df
                )

        st.markdown(
            "<div class='light-card'>"
            "<div style='font-size:0.95rem; line-height:1.7;'>"
            f"{st.session_state['auto_ai_story']}"
            "</div></div>",
            unsafe_allow_html=True,
        )

        if st.button("Save insights to report", key="save_ai_only"):
            st.session_state["report_ai_answers"].append(
                {
                    "question": "Comprehensive AI Narrative",
                    "answer": st.session_state["auto_ai_story"],
                }
            )
            st.success("Saved AI narrative to report.")
            # ---------- TAB 4: EXPORT ----------
with tab_export:
    st.markdown("### Export options")

    st.markdown(
        """
        The report will include:
        - The edited overview / cleaning notes from the **AI narrative & Q&A** tab.
        - Any visuals where you clicked **“Save this visual to report”**.
        - Any AI story or Q&A answers where you clicked **“Save this story/answer to report”**.
        """
    )

    st.write(
        f"**Saved visuals:** {len(st.session_state['report_visuals'])}  |  "
        f"**Saved AI answers:** {len(st.session_state['report_ai_answers'])}"
    )

    report_name = st.text_input(
        "Report filename (without extension)",
        value="ai_data_story"
    )

    if st.button("Create full PDF (summary + visuals + AI insights)"):
        report_path = str(REPORT_DIR / f"{report_name}.pdf")

        # Build text block
        lines = []
        lines.append("Overview & data notes:\n")
        lines.append(st.session_state["combined_text_for_report"])
        lines.append("\n\nSaved visuals (high-level interpretation):\n")

        for v in st.session_state["report_visuals"]:
            lines.append(f"- {v.get('title', 'Visual')}: {v.get('insight', '')}")

        lines.append("\n\nExecutive AI Insights:\n")

        for qa in st.session_state["report_ai_answers"]:
            lines.append(f"{qa.get('question','')}:")
            bullets = format_as_executive_bullets(qa.get("answer", ""))
            for b in bullets:
                lines.append(b)
            lines.append("")

        report_text = "\n".join(lines)
        report_text_safe = make_pdf_safe(report_text)

        # Prepare image info
        images_info = []
        for v in st.session_state["report_visuals"]:
            img_path = v.get("image")
            if img_path and Path(img_path).exists():
                images_info.append(
                    {
                        "image": img_path,
                        "title": make_pdf_safe(v.get("title", "")),
                        "interpretation": make_pdf_safe(v.get("insight", "")),
                    }
                )

        if create_report is not None:
            try:
                create_report(
                    report_path,
                    report_text_safe,
                    images_info=images_info
                )

                if Path(report_path).exists():
                    with open(report_path, "rb") as f:
                        st.success("Report generated.")
                        st.download_button(
                            "Download PDF",
                            data=f,
                            file_name=f"{report_name}.pdf",
                            mime="application/pdf",
                        )
                else:
                    st.error(
                        "create_report ran but no PDF was found at the expected path."
                    )
            except Exception as e:
                st.error(f"Failed to create report via report_generator: {e}")
        else:
            st.error(
                "report_generator.create_report is not available. "
                "Make sure code/report_generator.py exists."
            )
