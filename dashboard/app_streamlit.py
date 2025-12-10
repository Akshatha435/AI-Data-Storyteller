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
            "- Univariate, bivariate, multivariate views\n"
            "- Histograms, boxplots, bar charts\n"
            "- Scatter plots and correlation heatmaps\n"
            "- Professional dark-blue visual style"
        )

with c3:
    with st.container(border=True):
        st.subheader("🤖 AI Narrative & Export")
        st.write(
            "- Ask AI to explain what the data is saying\n"
            "- Get clear, business-ready commentary\n"
            "- Save key visuals and AI answers\n"
            "- Export a consolidated PDF report"
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

# ---------- TAB 2: VISUALS ----------
with tab_visuals:
    st.markdown("### Visual analytics")

    cat_cols_v = eda_results["types"].get("categorical", [])
    num_cols_v = eda_results["types"].get("numerical", [])

    st.markdown("**Univariate analysis (single column)**")
    c1, c2 = st.columns(2)
    with c1:
        show_bar = st.checkbox("Bar chart (categorical)", value=bool(cat_cols_v))
        show_hist = st.checkbox("Histogram (numeric)", value=bool(num_cols_v))
    with c2:
        show_box = st.checkbox("Boxplot (numeric)", value=bool(num_cols_v))

    st.markdown("**Bivariate analysis (two columns)**")
    show_scatter = st.checkbox("Scatter plot (numeric vs numeric)", value=False)

    st.markdown("**Multivariate analysis (multiple numeric columns)**")
    show_heatmap = st.checkbox(
        "Correlation heatmap", value=(len(num_cols_v) >= 2)
    )

    # Bar chart
    if show_bar and cat_cols_v:
        bar_col = st.selectbox("Bar chart – choose categorical column", cat_cols_v, key="bar_col")
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = df[bar_col].value_counts().nlargest(10)
        counts.plot(kind="bar", ax=ax, color="#025EC4")
        ax.set_title(f"Top categories – {bar_col}")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig, use_container_width=False)
        full = save_figure(fig, f"bar_{bar_col}")
        if len(counts) > 0:
            top_cat = counts.index[0]
            top_pct = counts.iloc[0] / counts.sum()
            insight_bar = (
                f"For **{bar_col}**, the top category '{top_cat}' contributes about "
                f"{top_pct:.1%} of occurrences. This helps you focus on the most dominant segments."
            )
        else:
            insight_bar = f"Bar chart ({bar_col}): no data available."
        st.caption(insight_bar)

        if st.button("Save this visual to report", key=f"save_bar_{bar_col}"):
            st.session_state["report_visuals"].append(
                {"image": full, "title": f"Bar chart – {bar_col}", "insight": insight_bar}
            )
            st.success("Saved bar chart to report.")

    # Histogram
    if show_hist and num_cols_v:
        hist_col = st.selectbox("Histogram – choose numeric column", num_cols_v, key="hist_col")
        bins = st.slider("Histogram bins", 10, 80, 30, key="hist_bins")
        fig, ax = plt.subplots(figsize=(5, 3))
        df[hist_col].dropna().plot(kind="hist", bins=bins, ax=ax, color="#043780", alpha=0.9)
        ax.set_title(f"Distribution of {hist_col}")
        st.pyplot(fig, use_container_width=False)
        full = save_figure(fig, f"hist_{hist_col}")
        s = df[hist_col].dropna()
        skew = s.skew()
        mean = s.mean()
        median = s.median()
        std = s.std()
        shape_text = (
            "right-skewed (tail on the right)" if skew > 0.3
            else "left-skewed (tail on the left)" if skew < -0.3
            else "roughly symmetric"
        )
        insight_hist = (
            f"The numeric feature **{hist_col}** has mean {mean:.2f} and median {median:.2f}, "
            f"with standard deviation {std:.2f}. The distribution is {shape_text}, "
            "which matters when interpreting averages and extremes."
        )
        st.caption(insight_hist)

        if st.button("Save this visual to report", key=f"save_hist_{hist_col}"):
            st.session_state["report_visuals"].append(
                {"image": full, "title": f"Histogram – {hist_col}", "insight": insight_hist}
            )
            st.success("Saved histogram to report.")

    # Boxplot
    if show_box and num_cols_v:
        box_col = st.selectbox("Boxplot – choose numeric column", num_cols_v, key="box_col")
        fig, ax = plt.subplots(figsize=(5, 2.2))
        sns.boxplot(x=df[box_col], ax=ax, color="#0ECCED")
        ax.set_title(f"Boxplot of {box_col}")
        st.pyplot(fig, use_container_width=False)
        full = save_figure(fig, f"box_{box_col}")
        q1 = df[box_col].quantile(0.25)
        q3 = df[box_col].quantile(0.75)
        iqr = q3 - q1
        out_low = int((df[box_col] < (q1 - 1.5 * iqr)).sum())
        out_high = int((df[box_col] > (q3 + 1.5 * iqr)).sum())
        insight_box = (
            f"The feature **{box_col}** shows around {out_low + out_high} potential outliers. "
            "These extreme values can strongly influence averages and models, so they are "
            "good candidates for review or capping."
        )
        st.caption(insight_box)

        if st.button("Save this visual to report", key=f"save_box_{box_col}"):
            st.session_state["report_visuals"].append(
                {"image": full, "title": f"Boxplot – {box_col}", "insight": insight_box}
            )
            st.success("Saved boxplot to report.")

    # Scatter
    if show_scatter and len(num_cols_v) >= 2:
        scatter_x = st.selectbox("Scatter X axis", num_cols_v, key="scatter_x")
        scatter_y = st.selectbox(
            "Scatter Y axis",
            [c for c in num_cols_v if c != scatter_x],
            key="scatter_y",
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(df[scatter_x], df[scatter_y], alpha=0.5, s=12, color="#0ECCED")
        ax.set_xlabel(scatter_x)
        ax.set_ylabel(scatter_y)
        ax.set_title(f"Scatter: {scatter_x} vs {scatter_y}")
        st.pyplot(fig, use_container_width=False)
        full = save_figure(fig, f"scatter_{scatter_x}_vs_{scatter_y}")
        corr_val = df[[scatter_x, scatter_y]].corr().iloc[0, 1]
        insight_scatter = (
            f"Between **{scatter_x}** and **{scatter_y}**, the correlation is {corr_val:.2f}. "
            "Values closer to +1 or -1 indicate a stronger linear relationship, which helps "
            "when selecting drivers for modelling or forecasting."
        )
        st.caption(insight_scatter)

        if st.button("Save this visual to report", key=f"save_scatter_{scatter_x}_{scatter_y}"):
            st.session_state["report_visuals"].append(
                {
                    "image": full,
                    "title": f"Scatter – {scatter_x} vs {scatter_y}",
                    "insight": insight_scatter,
                }
            )
            st.success("Saved scatter plot to report.")

    # Heatmap
    if show_heatmap and len(num_cols_v) >= 2:
        corr_df = pd.DataFrame(eda_results["correlations"])
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.heatmap(corr_df.astype(float), annot=True, cmap="Blues", ax=ax)
        ax.set_title("Correlation heatmap")
        st.pyplot(fig, use_container_width=False)
        full = save_figure(fig, "corr_heatmap")
        pairs = []
        for a in corr_df:
            for b, v in corr_df[a].items():
                if a != b:
                    pairs.append(((a, b), v))
        pairs_sorted = sorted(
            pairs,
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
            reverse=True,
        )
        if pairs_sorted:
            (a, b), v = pairs_sorted[0]
            insight_heat = (
                f"The strongest linear link in this dataset is between **{a}** and **{b}** "
                f"(correlation ≈ {v:.2f}). These variables move together and deserve attention "
                "when building metrics or models."
            )
        else:
            insight_heat = "No strong correlations detected between numeric features."
        st.caption(insight_heat)

        if st.button("Save this visual to report", key="save_heatmap"):
            st.session_state["report_visuals"].append(
                {"image": full, "title": "Correlation heatmap", "insight": insight_heat}
            )
            st.success("Saved heatmap to report.")

# ---------- Combined observations for summary / report ----------
obs_lines = []
obs_lines.append(
    f"- Dataset shape: {eda_results['summary']['shape'][0]} rows × "
    f"{eda_results['summary']['shape'][1]} columns."
)
missing_sorted = sorted(
    eda_results["summary"]["missing_values"].items(),
    key=lambda x: x[1],
    reverse=True,
)
missing_top = [f"{k}: {v}" for k, v in missing_sorted if v > 0][:6]
if missing_top:
    obs_lines.append("- Columns with missing values (top): " + "; ".join(missing_top))
else:
    obs_lines.append("- No missing values after preprocessing.")

nums_sample = eda_results["types"].get("numerical", [])[:6]
cats_sample = eda_results["types"].get("categorical", [])[:6]
if nums_sample:
    obs_lines.append(f"- Example numeric columns: {', '.join(nums_sample)}")
if cats_sample:
    obs_lines.append(f"- Example categorical columns: {', '.join(cats_sample)}")

combined_text_default = "\n".join(obs_lines)
if not st.session_state["combined_text_for_report"]:
    st.session_state["combined_text_for_report"] = combined_text_default


# ---------- LLM helper ----------
def ask_llm_about_data(question: str, eda_results: dict, df: pd.DataFrame) -> str:
    """
    Use OpenAI to answer questions about the dataset and suggest actions.
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    if openai_client is not None and api_key:
        if get_prompt_for_eda is not None:
            eda_prompt = get_prompt_for_eda(eda_results, max_chars=2000)
        else:
            eda_prompt = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}."

        sample_csv = df.head(30).to_csv(index=False)

        system_msg = (
            "You are a senior data analyst. You receive: (1) an EDA-style summary, "
            "and (2) a small sample of the dataset.\n"
            "Write in clear business language (no heavy statistics). "
            "Structure your answer as:\n"
            "1) A narrative explaining what this dataset is about and what the numbers are saying.\n"
            "2) 3–5 numbered, concrete actions or decisions a business could take.\n"
        )

        user_msg = (
            f"Question: {question}\n\n"
            f"EDA summary:\n{eda_prompt}\n\n"
            f"Sample data (CSV head):\n{sample_csv}"
        )

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.35,
                max_tokens=900,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"AI error while answering question: {e}"

    # Fallback text if API not available
    return (
        "AI insights could not be generated (missing API key or client).\n\n"
        "However, based on the current analysis you can already note:\n\n"
        + st.session_state["combined_text_for_report"]
        + "\n\nUse this to explain the dataset structure, key fields and any data quality points."
    )


# ---------- TAB 3: AI NARRATIVE & Q&A ----------
with tab_qna:
    st.markdown("### AI narrative & Q&A")

    st.markdown("**Overview & cleaning notes (used in the report):**")
    combined_text_editable = st.text_area(
        "",
        value=st.session_state["combined_text_for_report"],
        height=160,
    )
    st.session_state["combined_text_for_report"] = combined_text_editable

    st.markdown("---")

    # Automatic long-form dataset story – run once
    default_story_question = (
        "Give a comprehensive, narrative-style explanation of this dataset – "
        "what it contains, how the main variables behave, key patterns, risks and "
        "opportunities, and how a business should interpret it."
    )
    if st.session_state["auto_ai_story"] is None:
        with st.spinner("Generating an overall story for this dataset..."):
            st.session_state["auto_ai_story"] = ask_llm_about_data(
                default_story_question, eda_results, df
            )

    st.markdown("#### AI dataset story")
    st.markdown(
        "<div class='light-card'>"
        f"<div style='font-size:0.9rem;line-height:1.55;color:#E5F0FF;'>"
        f"{st.session_state['auto_ai_story']}"
        "</div></div>",
        unsafe_allow_html=True,
    )

    if st.button("Save this story to report"):
        st.session_state["report_ai_answers"].append(
            {
                "question": "Overall dataset story",
                "answer": st.session_state["auto_ai_story"],
            }
        )
        st.success("Saved AI dataset story to report.")

    st.markdown("---")
    st.markdown("#### Ask your own question on this dataset")

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    suggestions = []
    if numeric_cols:
        suggestions.append(f"What is the overall trend of '{numeric_cols[0]}'?")
        if len(numeric_cols) > 1:
            suggestions.append(
                f"How are '{numeric_cols[0]}' and '{numeric_cols[1]}' related?"
            )
    if categorical_cols:
        suggestions.append(
            f"Which categories in '{categorical_cols[0]}' should we pay most attention to?"
        )
    suggestions.append(
        "What are the most important insights from this dataset and what actions would you suggest?"
    )

    st.markdown("**Suggested questions:**")
    chosen_q = st.radio("", suggestions, index=len(suggestions) - 1)

    st.markdown("**Or type your own question:**")
    user_question = st.text_input(
        "",
        value="",
        placeholder="Example: Which variables appear to drive performance the most?",
    )

    if st.button("Ask AI on this dataset"):
        final_q = user_question.strip() or chosen_q
        with st.spinner("Thinking about your data..."):
            answer = ask_llm_about_data(final_q, eda_results, df)

        st.markdown(
            "<div class='light-card'>"
            "<div style='font-size:1rem;font-weight:650;color:#0ECCED;margin-bottom:6px;'>"
            "AI answer</div>"
            f"<div style='font-size:0.9rem;line-height:1.55;color:#E5F0FF;'>{answer}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button("Save this answer to report"):
            st.session_state["report_ai_answers"].append(
                {"question": final_q, "answer": answer}
            )
            st.success("Saved AI answer to report.")

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

    report_name = st.text_input("Report filename (without extension)", value="ai_data_story")

    if st.button("Create full PDF (summary + visuals + AI insights)"):
        report_path = str(REPORT_DIR / f"{report_name}.pdf")

        # Build text block
        lines = []
        lines.append("Overview & data notes:\n")
        lines.append(st.session_state["combined_text_for_report"])
        lines.append("\n\nSaved visuals (high-level interpretation):\n")
        for v in st.session_state["report_visuals"]:
            lines.append(f"- {v.get('title', 'Visual')}: {v.get('insight', '')}")
        lines.append("\n\nAI narratives & Q&A:\n")
        for qa in st.session_state["report_ai_answers"]:
            lines.append(f"Q: {qa.get('question','')}\nA: {qa.get('answer','')}\n")

        report_text = "\n".join(lines)
        report_text_safe = make_pdf_safe(report_text)

        # Prepare image info for report_generator
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
                # Try different call signatures but always with safe text
                try:
                    create_report(report_path, report_text_safe, images_info=images_info)
                except TypeError:
                    try:
                        image_paths = [
                            v["image"]
                            for v in st.session_state["report_visuals"]
                            if "image" in v and Path(v["image"]).exists()
                        ]
                        create_report(report_path, report_text_safe, image_paths=image_paths)
                    except TypeError:
                        image_paths = [
                            v["image"]
                            for v in st.session_state["report_visuals"]
                            if "image" in v and Path(v["image"]).exists()
                        ]
                        create_report(report_path, report_text_safe, image_paths)
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
                        "create_report ran but no PDF was found at the expected path. "
                        "Check report_generator.create_report implementation."
                    )
            except Exception as e:
                st.error(f"Failed to create report via report_generator: {e}")
        else:
            st.error(
                "report_generator.create_report is not available; cannot generate PDF here. "
                "Make sure code/report_generator.py exists and is imported correctly."
            )