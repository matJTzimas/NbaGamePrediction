# NBA MACHINE LEARNING PROJECT

## Streamlit APP
[NBA Game Prediction App](https://nbagameprediction.streamlit.app)

## OVERVIEW
This project focuses on predicting NBA game outcomes. It’s important to mention that the main goal was not model research but practicing with machine learning tools and pipelines. Specifically:

- **ETL Pipelines**:  
  - **EXTRACT** data from multiple sources (nba_api and Kaggle).  
  - **TRANSFORM** them into a unified data source.  
  - **LOAD** the data and feed them into an artificial neural network (ANN).  
  - The project structure is built around this methodology.  

- **MLFLOW**:  
  - Experiment tracking using a remote MLflow server (Dagshub). The training details are automatically displayed in the Streamlit app (see training details page).  
  - The best model is also registered on the remote MLflow server for use during inference.  

- **STREAMLIT**:  
  - Used to create the web application.  
  - All necessary files (.pkl, .pth, etc.) are stored either on an S3 Dagshub server or in MLflow, enabling the app to run fully on the cloud.  

- **GitHub Actions Workflows**:  
  - A GitHub workflow runs daily to update the latest game outcomes and injury information.  
  - This workflow is self-hosted to avoid IP blocking from the nba_api.  

## MLP Methodology
A small MLP network predicts the probability of a home team victory. This implementation differs from other methods in two main ways:  

1. **Input Data**: The neural network uses player stats as input instead of team history data. This allows the model to adapt dynamically to roster changes (injuries, trades, etc.), which directly impact team performance.  

2. **Target Variable**: The home team’s betting odds are used as the prediction target, with a sigmoid function as the final activation. Using only binary outcomes ([victory: 1, loss: 0]) introduces noise in the training loss function because the favorite team does not always win (i.e., discrepancies between statistics and actual results). In addition, binary targets are discrete, making training less smooth.  