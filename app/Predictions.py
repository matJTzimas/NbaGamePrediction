import pandas as pd
import streamlit as st
from datetime import datetime, date
from zoneinfo import ZoneInfo   # Python 3.9+

st.set_page_config(page_title="NBA Predictions", page_icon="🏀", layout="wide")

st.title("🏀 NBA Predictions — Today's Games")
st.caption("Predicting Today's Games Using ML Model. You can find implementation details in 'Documentation' page, while training informaiton \
    are available at 'Training Details' page.")


st.sidebar.title("📂 Navigation")
if st.sidebar.button("🏀 Predictions"):
    st.switch_page("Predictions.py")
if st.sidebar.button("📊 Training Details"):
    st.switch_page("pages/Training_Details.py")
if st.sidebar.button("📚 Documentation"):
    st.switch_page("pages/Documentation.py")


CSV_PATH = "Artifacts/inference/predictions.csv"

# 1) Load data (cached for speed)
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date.astype(str)
    return df

df = load_data(CSV_PATH)
df['WINNER PRED ABBS'] = df.apply(lambda row : row['HOME_ABBR'] if row['WINNER PRED'] == 'HOME' else row['AWAY_ABBR'], axis=1)

# 2) Choose what “today” means
today_athens = datetime.now(ZoneInfo("Europe/Athens")).date()
today_str = today_athens.isoformat()
today_str = "2024-10-30"  # For testing purposes, comment out for production

st.caption(f"Showing games for: **{today_str}** (Europe/Athens)")


# 3) Filter to today's games
# today_df = df[df["GAME_DATE"] == today_str].copy()
today_df = df[df["GAME_DATE"] == today_str].copy()

if today_df.empty:
    st.markdown("<h3 style='color: red;'>No games scheduled for today.</h3>", unsafe_allow_html=True)
else: 
    st.caption(f"Found **{len(today_df)}** games scheduled for today.")
    # 5) Table
    display_cols = ['GAME_DATE', 'HOME_ABBR', 'PROB_HOME_WIN', 'AWAY_ABBR', 'PROB_AWAY_WIN', 'WINNER PRED ABBS']
    st.dataframe(
        today_df[display_cols].sort_values(by=display_cols[0]),  # simple stable sort
        use_container_width=True,
    )

st.title("Previous Games With Results")

previous_df = df[df["GAME_DATE"] < today_str].copy()
accuracy = previous_df['RESULT'].mean() * 100
st.markdown(f"<h3 style='color: white;'> Model accuracy on previous games: {accuracy:.2f}% </h3>", unsafe_allow_html=True)
# st.markdown(f"<h3 style='color: white;'>{len(previous_df)} previous games found with accu</h3>", unsafe_allow_html=True)
previous_df['RESULT'] = previous_df['RESULT'].map({True: "✅" , False: "❌" })
if previous_df.empty:
    st.markdown("<h3 style='color: red;'>No previous games found.</h3>", unsafe_allow_html=True)
else:
    display_cols = ['GAME_DATE', 'HOME_ABBR', 'PROB_HOME_WIN', 'AWAY_ABBR', 'PROB_AWAY_WIN', 'WINNER PRED ABBS','ACTUAL','RESULT']
    st.dataframe(
        previous_df[display_cols].sort_values(by=display_cols[0], ascending=False).style.set_properties(**{"text-align": "center"}),  # simple stable sort
        use_container_width=True,
    )

