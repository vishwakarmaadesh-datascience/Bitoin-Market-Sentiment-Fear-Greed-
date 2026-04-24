# Crypto Trader Performance & Behavior Analysis

## Overview
This project provides a comprehensive analytical framework and a predictive model to understand and forecast cryptocurrency trader performance and behavior. By integrating historical trade data with external market sentiment, it offers insights into how market psychology influences trading outcomes, segments traders into distinct behavioral archetypes, and predicts future profitability.

## Features
-   **Data Integration & Cleaning**: Robust loading and preprocessing of historical trade data and the Fear & Greed Index.
-   **Key Performance Metrics**: Calculation and visualization of daily PnL, win rates, average trade sizes, and long/short ratios.
-   **Sentiment-based Analysis**: Examination of how different market sentiments (Extreme Fear, Fear, Neutral, Greed, Extreme Greed) correlate with trader performance and specific behavioral patterns.
-   **Trader Segmentation**: Utilizes K-Means clustering to identify distinct trader archetypes based on their trading volume, profitability, and win rate.
-   **Drawdown Analysis**: Calculates and visualizes maximum drawdowns per account to assess risk.
-   **Predictive Modeling**: A Random Forest Classifier to predict next-day trader profitability based on previous day's activity and sentiment.
-   **Interactive Streamlit Dashboard**: A user-friendly web application to visualize all analyses, model results, and key insights.

## Technologies Used
-   **Python**: Primary programming language.
-   **Pandas & NumPy**: Data manipulation and numerical operations.
-   **Matplotlib & Seaborn**: Data visualization.
-   **Scikit-learn**: Machine learning (Random Forest, K-Means, StandardScaler).
-   **Streamlit**: For building the interactive web dashboard.
-   **Ngrok**: For exposing the local Streamlit application to the internet.

## Setup and Installation
To set up and run this project locally, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd crypto-trader-analysis
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt # (assuming you generate a requirements.txt)
    # Or manually install:
    pip install pandas numpy scikit-learn matplotlib seaborn streamlit pyngrok
    ```

4.  **Download Data**: Ensure `fear_greed_index.csv` and `historical_data.csv` are in the project root directory (or update paths in `app.py`).

5.  **Ngrok Setup (Optional, for public access)**:
    *   Sign up for Ngrok at [ngrok.com](https://ngrok.com/).
    *   Obtain your authtoken and configure it:
        ```bash
        ngrok authtoken YOUR_NGROK_AUTH_TOKEN
        ```
    *   The `UFMXNps5IOoS` cell in the Colab notebook handles Ngrok setup if running there.

## Usage

1.  **Run the Streamlit Dashboard**:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser, typically at `http://localhost:8501`.

2.  **Explore the Notebook**: The Colab notebook (`crypto_trader_analysis.ipynb` or similar) provides a step-by-step breakdown of the analysis, from data loading to model training and evaluation.

## Project Structure

```
. # Project Root
├── app.py                     # Streamlit dashboard application
├── fear_greed_index.csv       # Dataset: Fear & Greed Index
├── historical_data.csv        # Dataset: Historical trading data
├── README.md                  # This README file
└── requirements.txt (optional) # List of Python dependencies
```

## Insights & Outcomes
-   Identified periods of 'Fear' sentiment as historically most profitable in terms of overall PnL, suggesting contrarian trading strategies.
-   Revealed 'Extreme Greed' periods to have the highest win rates, indicating favorable conditions for high-probability trades.
-   Segmented traders into archetypes like 'High-Volume Profiteers' and 'Conservative Winners', enabling tailored risk management and support strategies.
-   Developed a predictive model to proactively identify traders likely to be profitable or non-profitable the next day.

---
