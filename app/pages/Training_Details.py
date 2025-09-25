import os
import sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # this reads .env and puts vars into os.environ
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("https://dagshub.com/matJTzimas/NbaGamePrediction.mlflow")
import matplotlib.pyplot as plt
import pickle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from nba.utils.main_utils import Storage
from nba.entity.config_entity import InferenceConfig
inference_config = InferenceConfig(model_name="mlp")
storage = Storage(cloud_option=inference_config.cloud_option)

st.sidebar.title("📂 Navigation")
if st.sidebar.button("🏀 Predictions"):
    st.switch_page("Predictions.py")
if st.sidebar.button("📊 Training Details"):
    st.switch_page("pages/Training_Details.py")
if st.sidebar.button("📚 Documentation"):
    st.switch_page("pages/Documentation.py")



st.set_page_config(page_title="NBA Predictions — Training Details", page_icon="🏀", layout="wide")
st.title("MLP Training Details")
# Pull all runs as a DataFrame
# Tip: you can filter, e.g. "metrics.accuracy > 0.9 and tags.stage = 'prod'"
df = mlflow.search_runs(
    experiment_ids="0",
    filter_string="",
    max_results=10,               # bump if you have many runs
    output_format="pandas"
)

# get last training runs 


df = df[df['tags.mlflow.runName'].str.contains('mlp_nba', na=False)]
runname = list(df['tags.mlflow.runName'])
runname =[run_exp for run_exp in runname if run_exp != "mlp_nba_main_run"]

days_training = [int(run_exp.split('_')[-1]) for run_exp in runname]
last_day = str(max(days_training))

df = df[df['tags.mlflow.runName'].str.contains(last_day, na=False)]

st.subheader(f"Showing training runs from last training day: {last_day[6:8]}-{last_day[4:6]}-{last_day[0:4]}")


setting_df = df[["tags.mlflow.runName", "params.lr", "params.batch_size", "metrics.best_val_acc"]].copy()
setting_df = setting_df.sort_values(by="tags.mlflow.runName", ascending=False).reset_index(drop=True).dropna()

# Assume your DataFrame is called setting_df
max_idx = setting_df["metrics.best_val_acc"].idxmax()

def highlight_max_row(row):
    color = "background-color: lightgreen"
    if row.name == max_idx:
        return [color] * len(row)
    else:
        return [""] * len(row)

styled_df = setting_df.style.apply(highlight_max_row, axis=1)

st.dataframe(styled_df, use_container_width=True)

st.title("Visualization of all MLP runs")

df_new = df[df['status']=="FINISHED"].reset_index(drop=True).copy()

@st.cache_data(show_spinner=False)
def load_histories(keys, df_length, last_day_training):
    file_path = f"app/store/histories_{last_day_training}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            out = pickle.load(f)
            return out

    out = {k: [] for k in keys}
    for key in keys:
        for i in range(df_length):
            run = client.get_run(df_new.loc[i, 'run_id'])
            pts = client.get_metric_history(run.info.run_id, key)
            out[key].append((run.info.run_name, [pt.step for pt in pts], [pt.value for pt in pts]))

    with open(file_path, "wb") as f:
        pickle.dump(out, f)
    return out

client = MlflowClient()

metric_keys = ['overall', 'train_loss', 'top_10_preds', 'top_20_preds', 'top_50_preds']

histories = load_histories( metric_keys, len(df_new),last_day)

# --- Visualization (append below your existing code) ---

st.subheader("Training Curves per Metric (all runs)")

# Filter out empty metrics (no history)
available_metrics = metric_keys.copy()

if not available_metrics:
    st.warning("No per-epoch metric history found. Make sure metrics were logged with a 'step' (epoch).")
else:
    # Let user pick which metrics to show
    selected_metrics = st.multiselect(
        "Select metrics to display:",
        options=available_metrics,
        default=available_metrics
    )

    if selected_metrics:
        tabs = st.tabs(selected_metrics)
        for tab, metric in zip(tabs, selected_metrics):
            with tab:
                fig, ax = plt.subplots(figsize=(6, 4))
                series = histories.get(metric, [])
                any_plotted = False
                for run_name, steps, values in series:
                    if steps and values:
                        ax.plot(steps, values, label=run_name)
                        any_plotted = True

                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                ax.set_title(metric)
                if any_plotted:
                    ax.legend(loc="best")
                st.pyplot(fig,width=600)
                plt.close(fig)
    else:
        st.info("Select at least one metric to visualize.")
