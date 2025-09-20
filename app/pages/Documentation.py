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
This application predicts the outcomes of NBA games using a machine learning model. I have created this application to 
practise machine learning techniques used in the industry today.

Compared to tradition ML methods that rely on team historical data, this implementation has two key differences:
- **PLAYERS DATA**: Building a method on team data has the disadvantage of player injury status. A team without its superstar has not the 
same win percantage as normally have. Therefore for input data, all time and current season players statistics have been used.
- **TARGET VARIABLE**: The majority of implementations are using game outcome (win/loss) as target variable. However, this is not
always the best option, as there is the human factor that can affect the outcome. This irreducible ``error" can produce noice in the training
loss surface and make the model less accurate. To overcome this, I have used the betting odds as target variable, which relies on advanced statistics
and brings a more probabilistic approach to the problem. This way, we can rely also on the most confident predictions. 

To prove that that the model output indeed have a statistical significance, here is the overall accuracy along with of the top 10, 20 and 50 """)