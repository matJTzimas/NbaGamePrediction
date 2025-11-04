from __future__ import annotations

import os
import sys
from datetime import date, datetime
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from zoneinfo import ZoneInfo  # Py ≥3.9

# --- Repo imports (clean + robust) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nba.utils.main_utils import Storage
from nba.entity.config_entity import InferenceConfig

# --- Config / Constants ---
APP_TZ = "Europe/Athens"
PAGE_TITLE = "NBA Predictions"
PAGE_ICON = "🏀"
CACHE_TTL_SEC = 60  # tweak to your refresh cadence

load_dotenv()
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- Header / Nav ---
st.title(f"{PAGE_ICON} {PAGE_TITLE} — Today’s Games")
st.caption(
    "Predicting today's games with ML. "
    "Implementation details in **Documentation**; training info in **Training Details**."
)

with st.sidebar:
    st.title("📂 Navigation")
    # Prefer page links (Streamlit multipage apps)
    st.page_link("Predictions.py", label="🏀 Predictions")
    st.page_link("pages/Training_Details.py", label="📊 Training Details")
    st.page_link("pages/Documentation.py", label="📚 Documentation")

    st.markdown("---")
    use_test_date = st.checkbox("Use test date", value=False)
    test_date_str = st.text_input("Test date (YYYY-MM-DD)", value="2024-10-30")
    st.caption("Uncheck to use current Athens date.")

# --- Inference config / storage ---
inference_config = InferenceConfig(model_name="mlp")
storage = Storage(cloud_option=inference_config.cloud_option)

# --- Data loaders ---
@st.cache_data(ttl=CACHE_TTL_SEC)
def load_data(_storage: Storage) -> Optional[pd.DataFrame]:
    """Load the full predictions table. Returns None if not available."""
    df = _storage.read_csv()
    if df is None or df.empty:
        return None
    # normalize schema
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.date.astype(str)
    # derive winner abbrev column
    if {"HOME_ABBR", "AWAY_ABBR", "WINNER PRED"}.issubset(df.columns):
        df["WINNER PRED ABBS"] = df.apply(
            lambda r: r["HOME_ABBR"] if r["WINNER PRED"] == "HOME" else r["AWAY_ABBR"],
            axis=1,
        )
    return df


def resolve_today_str() -> str:
    if use_test_date:
        try:
            # Validate test date
            _ = datetime.strptime(test_date_str, "%Y-%m-%d").date()
            return test_date_str
        except ValueError:
            st.warning("Invalid test date. Falling back to current Athens date.")
    return datetime.now(ZoneInfo(APP_TZ)).date().isoformat()


def fmt_percent(x: float | int | None) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "-"

# --- Load data ---
df = load_data(storage)

if df is None or df.empty:
    st.error("No predictions available yet. Please check back later or verify your pipeline.")
    st.stop()

# --- Today's view ---
today_str = resolve_today_str()
st.caption(f"Showing games for: **{today_str}** ({APP_TZ})")

required_cols_today = {"GAME_DATE", "HOME_ABBR", "PROB_HOME_WIN", "AWAY_ABBR", "PROB_AWAY_WIN", "WINNER PRED ABBS"}
missing_today = required_cols_today - set(df.columns)
if missing_today:
    st.error(f"Missing columns for today's view: {', '.join(sorted(missing_today))}")
else:
    today_df = df.loc[df["GAME_DATE"] == today_str, list(required_cols_today)].copy()
    if today_df.empty:
        st.info("No games scheduled for today.")
    else:
        st.caption(f"Found **{len(today_df)}** game(s) scheduled for today.")
        # Nicely formatted dataframe (percentages)
        try:
            # enforce a stable column order
            desired = [
            "GAME_DATE",
            "HOME_ABBR",
            "PROB_HOME_WIN",
            "AWAY_ABBR",
            "PROB_AWAY_WIN",
            "WINNER PRED ABBS",
            ]
            cols = [c for c in desired if c in today_df.columns]
            display_df = today_df[cols].sort_values("GAME_DATE")
            col_cfg = {}
            if "PROB_HOME_WIN" in cols:
                col_cfg["PROB_HOME_WIN"] = st.column_config.NumberColumn("PROB_HOME_WIN", format="%.2f%%")
            if "PROB_AWAY_WIN" in cols:
                col_cfg["PROB_AWAY_WIN"] = st.column_config.NumberColumn("PROB_AWAY_WIN", format="%.2f%%")
            st.dataframe(display_df, width="stretch", column_config=col_cfg)
        except Exception:
            # Fallback for older Streamlit versions
            show_df = today_df.copy()
            if "PROB_HOME_WIN" in show_df.columns:
                show_df["PROB_HOME_WIN"] = show_df["PROB_HOME_WIN"].map(fmt_percent)
            if "PROB_AWAY_WIN" in show_df.columns:
                show_df["PROB_AWAY_WIN"] = show_df["PROB_AWAY_WIN"].map(fmt_percent)
            cols = [c for c in desired if c in show_df.columns]
            st.dataframe(show_df[cols].sort_values("GAME_DATE"), width="stretch")

# --- Previous games with results ---
st.subheader("Previous Games With Results")

if "GAME_DATE" not in df.columns or "RESULT" not in df.columns:
    st.info("Historical result columns not found.")
else:
    previous_df = df.loc[df["GAME_DATE"] < today_str].copy()
    if "ACTUAL" in previous_df.columns:
        previous_df = previous_df[previous_df["ACTUAL"].notna() & (previous_df["ACTUAL"] != "-")].copy()
    elif "RESULT" in previous_df.columns:
        previous_df = previous_df[previous_df["RESULT"].notna() & (previous_df["RESULT"] != "-")].copy()

        
    if previous_df.empty:
        st.info("No previous games found.")
    else:
        # Accuracy
        try:
            accuracy = float(previous_df["RESULT"].mean()) * 100.0
            st.markdown(f"**Model accuracy on previous games:** {accuracy:.2f}%")
        except Exception:
            st.markdown("**Model accuracy on previous games:** n/a")




        # Pretty result glyphs
        previous_df["RESULT"] = previous_df["RESULT"].map({True: "✅", False: "❌"})#.fillna("-")

        # Highlight winning team/prob columns based on ACTUAL
        def highlight_actual(row):
            styles = [''] * len(row)
            if row["PROB_HOME_WIN"] > row["PROB_AWAY_WIN"]:
                for i, c in enumerate(row.index):
                    if c in ("HOME_ABBR", "PROB_HOME_WIN"):
                        styles[i] = "background-color: #6A0DAD"  # dark purple
            elif row["PROB_AWAY_WIN"] > row["PROB_HOME_WIN"]:
                for i, c in enumerate(row.index):
                        if c in ("AWAY_ABBR", "PROB_AWAY_WIN"):
                            styles[i] = "background-color: #6A0DAD"
            return styles

        # Only apply style if using pandas Styler (Streamlit >=1.32)
        prev_cols = [
            col for col in [
                "GAME_DATE", "HOME_ABBR", "PROB_HOME_WIN",
                "AWAY_ABBR", "PROB_AWAY_WIN", "RESULT"
                ] if col in previous_df.columns
        ]

        ordered = previous_df[prev_cols].sort_values("GAME_DATE", ascending=False)

        styled_prev = ordered.style.apply(highlight_actual, axis=1)
        # styled_prev = styled_prev[prev_cols]
        mask = ((previous_df["PROB_HOME_WIN"] > 0.6) | (previous_df["PROB_AWAY_WIN"] > 0.6))
        confident_df = previous_df.loc[mask]

        if len(confident_df) > 0:
            try:
                correct_preds = len(confident_df[confident_df["RESULT"] == "✅"])
                conf_accuracy = (correct_preds / len(confident_df)) * 100.0
                st.markdown(f"**Model accuracy on previous games with confident predictions (>60%):** {conf_accuracy:.2f}%")
            except Exception:
                st.markdown("**Model accuracy on previous games with confident predictions (>60%):** n/a")
        try:
            st.dataframe(
                styled_prev,
                width='stretch',
                column_config={
                    "PROB_HOME_WIN": st.column_config.NumberColumn("PROB_HOME_WIN", format="%.2f%%"),
                    "PROB_AWAY_WIN": st.column_config.NumberColumn("PROB_AWAY_WIN", format="%.2f%%"),
                },
            )
        except Exception:
            show_prev = previous_df[prev_cols].copy()
            for c in ("PROB_HOME_WIN", "PROB_AWAY_WIN"):
                if c in show_prev.columns:
                    show_prev[c] = show_prev[c].map(fmt_percent)
            st.dataframe(show_prev.sort_values("GAME_DATE", ascending=False), width='stretch')

