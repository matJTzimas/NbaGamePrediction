import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA Predictions Documentation", layout="wide")
st.title("🏀 NBA Predictions Documentation")

st.sidebar.title("📂 Navigation")
if st.sidebar.button("🏀 Predictions"):
    st.switch_page("Predictions.py")
if st.sidebar.button("📊 Training Details"):
    st.switch_page("pages/Training_Details.py")
if st.sidebar.button("📚 Documentation"):
    st.switch_page("pages/Documentation.py")

st.markdown("""
This application predicts the outcomes of NBA games using a machine learning model.  
It was built as a project to practice modern machine learning techniques used in the industry today.

Unlike traditional approaches that rely mainly on team-level historical data, this implementation introduces two key differences:

- **Player-Level Data**:  
  Team-based models are often misleading because they don’t account for player availability.  
  For example, a team without its star player has a much lower chance of winning than its historical averages suggest.  
  To address this, the input data here includes both all-time and current-season player statistics, making the model more robust.

- **Target Variable**:  
  Most implementations use the game outcome (win/loss) as the target variable. However, this introduces significant noise, since human factors and randomness can strongly influence results.  
  Instead, this model uses **betting odds** as the target variable. Betting odds are based on advanced statistical models and provide a more probabilistic perspective.  
  This approach reduces noise in the training process and allows the model to focus on the most confident predictions.

To demonstrate that the model outputs are statistically meaningful, the overall accuracy is reported alongside the accuracy of the **top 10, 20, and 50 predictions**.
""")