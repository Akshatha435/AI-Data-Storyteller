# llm_interface.py
"""
Local LLM interface using Hugging Face transformers as a fallback when OpenAI is not available.
This tries, in order:
 1. Use OpenAI if OPENAI_API_KEY is set (optional)
 2. Use a small HF model loaded locally (flan-t5-small or distilbart)
 3. Return a templated plain-English summary if models are unavailable

This file exposes:
- get_prompt_for_eda(eda_results)
- generate_insights_with_llm(eda_results)
"""

import os
import textwrap

# Try import for OpenAI usage (optional)
try:
    import openai
except Exception:
    openai = None

# Try to import transformers for local model usage
_transformers_available = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    _transformers_available = True
except Exception:
    _transformers_available = False

# Choose a small, fast summarization/generation model suitable for CPU
# Options:
# - "google/flan-t5-small" (instruction-tuned, small)
# - "sshleifer/distilbart-cnn-12-6" (summarization)
HF_MODEL = os.environ.get("HF_MODEL", "google/flan-t5-small")

_local_pipeline = None

def _init_local_model():
    global _local_pipeline
    if not _transformers_available:
        return None
    if _local_pipeline is not None:
        return _local_pipeline
    try:
        # For instruction-like tasks use text2text-generation if available
        _local_pipeline = pipeline("text2text-generation", model=HF_MODEL, tokenizer=HF_MODEL, device=-1)
        return _local_pipeline
    except Exception:
        # fallback to summarization pipeline for long prompts
        try:
            _local_pipeline = pipeline("summarization", model=HF_MODEL, tokenizer=HF_MODEL, device=-1)
            return _local_pipeline
        except Exception:
            _local_pipeline = None
            return None

def get_prompt_for_eda(eda_results, max_chars=2500):
    """Build a short text prompt summarizing the EDA results for the LLM."""
    types = eda_results.get("types", {})
    summary = eda_results.get("summary", {})
    numeric_stats = eda_results.get("numeric_stats", {})
    corr = eda_results.get("correlations", {})

    prompt_lines = []
    prompt_lines.append(f"Dataset shape: {summary.get('shape')}.")
    prompt_lines.append("Columns and dtypes:")
    for k, v in summary.get("dtypes", {}).items():
        prompt_lines.append(f"- {k}: {v}")
    missing = summary.get("missing_values", {})
    missing_cols = {k: v for k, v in missing.items() if v > 0}
    if missing_cols:
        prompt_lines.append("Columns with missing values:")
        for k, v in missing_cols.items():
            prompt_lines.append(f"- {k}: {v} missing")
    if numeric_stats:
        prompt_lines.append("Numeric summary (sample):")
        for i, (col, stats) in enumerate(numeric_stats.items()):
            if i >= 6:
                break
            mean = stats.get("mean", "NA")
            std = stats.get("std", "NA")
            prompt_lines.append(f"- {col}: mean={round(mean,2) if isinstance(mean,(int,float)) else mean}, std={round(std,2) if isinstance(std,(int,float)) else std}, min={stats.get('min')}, max={stats.get('max')}")
    if corr:
        pairs = []
        for a in corr:
            for b, v in corr[a].items():
                if a != b:
                    pairs.append(((a, b), v))
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]) if isinstance(x[1], (int,float)) else 0, reverse=True)
        if pairs_sorted:
            prompt_lines.append("Top correlations (sample):")
            for (a,b),v in pairs_sorted[:5]:
                prompt_lines.append(f"- {a} vs {b}: {v}")
    full_prompt = "\n".join(prompt_lines)
    if len(full_prompt) > max_chars:
        full_prompt = full_prompt[:max_chars]
    return full_prompt

def _call_openai(prompt_text, max_tokens=300):
    """Call OpenAI if configured (optional)."""
    try:
        key = os.environ.get("OPENAI_API_KEY")
        if not key or openai is None:
            return None
        openai.api_key = key
        # Chat completion usage; if unavailable, this will raise
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content": prompt_text}],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

def _call_local_model(prompt_text, max_length=256):
    """Call a local HF pipeline if available."""
    try:
        pipe = _init_local_model()
        if pipe is None:
            return None
        # Some text2text pipelines need instruction prefix, keep it simple
        # truncate prompt if extremely long
        if len(prompt_text) > 3000:
            prompt_text = prompt_text[-3000:]
        out = pipe(prompt_text, max_length=max_length, truncation=True)
        # pipeline returns list of dicts with 'summary_text' or 'generated_text' keys
        if isinstance(out, list) and len(out) > 0:
            o = out[0]
            return o.get("generated_text") or o.get("summary_text") or str(o)
        return str(out)
    except Exception:
        return None

def generate_insights_with_llm(eda_results):
    """
    Highest-level function called by the Streamlit app.
    Tries OpenAI -> local HF model -> fallback templated insights.
    """
    prompt = get_prompt_for_eda(eda_results)
    full_task = (
        "You are a helpful assistant. Read the EDA summary below and produce "
        "6 short, plain-English insights (one line each) and then 3 suggested questions a user might ask.\n\n"
        f"{prompt}\n\nOutput format:\nInsight 1: ...\n...\nSuggested questions:\n1. ...\n2. ..."
    )

    # 1) Try OpenAI if key present
    try:
        ans = _call_openai(full_task)
        if ans:
            return ans
    except Exception:
        pass

    # 2) Try local HF model
    try:
        ans = _call_local_model(full_task, max_length=300)
        if ans:
            return ans
    except Exception:
        pass

    # 3) Fallback templated summary
    lines = []
    summary = eda_results.get("summary", {})
    types = eda_results.get("types", {})
    lines.append(f"The dataset has {summary.get('shape')[0]} rows and {summary.get('shape')[1]} columns.")
    missing = summary.get("missing_values", {})
    missing_cols = [k for k,v in missing.items() if v>0]
    if missing_cols:
        lines.append(f"Missing values in: {', '.join(missing_cols)}. Numeric filled with median, categorical with mode.")
    else:
        lines.append("No missing values detected.")
    if types.get("numerical"):
        sample_nums = ", ".join(types.get("numerical")[:5])
        lines.append(f"Numerical columns: {sample_nums}.")
    if types.get("categorical"):
        sample_cats = ", ".join(types.get("categorical")[:5])
        lines.append(f"Categorical columns: {sample_cats}.")
    # add simple correlation highlight if present
    corr = eda_results.get("correlations", {})
    if corr:
        pairs = []
        for a in corr:
            for b,v in corr[a].items():
                if a!=b:
                    pairs.append(((a,b), v))
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]) if isinstance(x[1], (int,float)) else 0, reverse=True)
        if pairs_sorted:
            (a,b),v = pairs_sorted[0]
            lines.append(f"Top correlation sample: {a} and {b} with correlation {v}.")
    lines.append("Suggested questions:\n- Which columns have missing values?\n- What are the top categories in column X?\n- Which numeric features are most correlated?")
    return "\n".join(lines)
