
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Install necessary libraries (will only run once if not already installed)
import subprocess
import sys
st.set_page_config(layout="wide", page_title="Trader Performance Analysis")
st.title("Trader Performance and Behavior Analysis")
@st.cache_data
def install_packages():
    packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'streamlit']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="Crypto Trader Analysis Dashboard")

# --- 1. Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    fear_greed_index = pd.read_csv('fear_greed_index.csv')
    historical_data = pd.read_csv('historical_data.csv', low_memory=False)

    fear_greed_index['date'] = pd.to_datetime(fear_greed_index['date'])
    historical_data['Timestamp IST'] = pd.to_datetime(historical_data['Timestamp IST'], dayfirst=True, errors='coerce')
    historical_data.dropna(subset=['Timestamp IST'], inplace=True)

    historical_data['Date'] = historical_data['Timestamp IST'].dt.normalize()
    fear_greed_index['Date'] = fear_greed_index['date'].dt.normalize()

    merged_data = pd.merge(historical_data, fear_greed_index, on='Date', how='inner')
    merged_data = merged_data.drop_duplicates()

    return fear_greed_index, historical_data, merged_data

fear_greed_index, historical_data, merged_data = load_data()

# --- Helper function for Predictive Model ---
@st.cache_data
def prepare_and_train_model(merged_data, fear_greed_index):
    daily_account_metrics = merged_data.groupby(['Date', 'Account']).agg(
        total_daily_pnl=('Closed PnL', 'sum'),
        daily_trades=('Order ID', 'size'),
        avg_daily_trade_size_tokens=('Size Tokens', 'mean'),
        avg_daily_trade_size_usd=('Size USD', 'mean'),
        num_daily_long_trades=('Side', lambda x: (x == 'BUY').sum()),
        num_daily_short_trades=('Side', lambda x: (x == 'SELL').sum())
    ).reset_index()

    daily_account_metrics['daily_long_short_ratio'] = daily_account_metrics['num_daily_long_trades'] / daily_account_metrics['num_daily_short_trades']
    daily_account_metrics['daily_long_short_ratio'] = daily_account_metrics['daily_long_short_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

    sentiment_features = fear_greed_index[['Date', 'value', 'classification']].copy()
    sentiment_features.rename(columns={'value': 'sentiment_value', 'classification': 'sentiment_class'}, inplace=True)
    sentiment_features['Date'] = pd.to_datetime(sentiment_features['Date'])
    modeling_data = pd.merge(daily_account_metrics, sentiment_features, on='Date', how='left')

    modeling_data = modeling_data.sort_values(by=['Account', 'Date'])
    modeling_data['next_day_profitable'] = modeling_data.groupby('Account')['total_daily_pnl'].shift(-1).apply(lambda x: 1 if x > 0 else 0)
    modeling_data.dropna(subset=['next_day_profitable'], inplace=True)
    modeling_data['next_day_profitable'] = modeling_data['next_day_profitable'].astype(int)

    lag_features = ['total_daily_pnl', 'daily_trades', 'avg_daily_trade_size_tokens',
                    'avg_daily_trade_size_usd', 'num_daily_long_trades', 'num_daily_short_trades',
                    'daily_long_short_ratio', 'sentiment_value']
    for feature in lag_features:
        modeling_data[f'prev_day_{feature}'] = modeling_data.groupby('Account')[feature].shift(1)

    modeling_data['sentiment_class_encoded'] = modeling_data['sentiment_class'].astype('category').cat.codes
    modeling_data['prev_day_sentiment_class_encoded'] = modeling_data.groupby('Account')['sentiment_class_encoded'].shift(1)
    modeling_data.dropna(subset=[f'prev_day_{f}' for f in lag_features] + ['prev_day_sentiment_class_encoded'], inplace=True)

    original_sentiment_categories = modeling_data['sentiment_class'].astype('category').cat.categories
    modeling_data['prev_day_sentiment_class'] = pd.Categorical.from_codes(modeling_data['prev_day_sentiment_class_encoded'].astype(int),
                                                                       categories=original_sentiment_categories)
    modeling_data = pd.get_dummies(modeling_data, columns=['prev_day_sentiment_class'], prefix='sentiment')

    X_cols = [col for col in modeling_data.columns if col.startswith('prev_day_') or (col.startswith('sentiment_') and col not in ['sentiment_class', 'sentiment_class_encoded', 'prev_day_sentiment_class_encoded'])]
    X = modeling_data[X_cols]
    y = modeling_data['next_day_profitable']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, X_cols


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Sentiment Analysis",
    "Trader Segmentation",
    "Drawdown Analysis",
    "Predictive Model"
])

# --- Page: Overview ---
if page == "Overview":
    st.title("Dashboard Overview")
    st.write("### Key Performance Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_trades = len(merged_data)
        st.metric("Total Trades", f"{total_trades:,}")

    with col2:
        winning_trades = merged_data[merged_data['Closed PnL'] > 0]
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        st.metric("Overall Win Rate", f"{win_rate:.2f}%")

    with col3:
        avg_trade_size_usd = merged_data['Size USD'].mean()
        st.metric("Average Trade Size (USD)", f"${avg_trade_size_usd:,.2f}")

    col4, col5 = st.columns(2)

    with col4:
        num_long = len(merged_data[merged_data['Side'] == 'BUY'])
        num_short = len(merged_data[merged_data['Side'] == 'SELL'])
        long_short_ratio = num_long / num_short if num_short > 0 else float('inf')
        st.metric("Long/Short Ratio", f"{long_short_ratio:.2f}")

    with col5:
        daily_pnl_per_account = merged_data.groupby(['Date', 'Account'])['Closed PnL'].sum().reset_index()
        total_pnl = daily_pnl_per_account['Closed PnL'].sum()
        st.metric("Total Closed PnL", f"${total_pnl:,.2f}")

    st.write("### Data Snapshots")
    st.subheader("Fear & Greed Index (Head)")
    st.dataframe(fear_greed_index.head())
    st.subheader("Historical Trading Data (Head)")
    st.dataframe(historical_data.head())
    st.subheader("Merged Data (Head)")
    st.dataframe(merged_data.head())

# --- Page: Sentiment Analysis ---
elif page == "Sentiment Analysis":
    st.title("Trader Behavior Analysis by Sentiment")

    st.write("#### Performance by Sentiment Classification")
    pnl_by_sentiment = merged_data.groupby('classification')['Closed PnL'].sum().reset_index()
    winning_trades_by_sentiment = merged_data[merged_data['Closed PnL'] > 0].groupby('classification').size().reset_index(name='Winning Trades')
    total_trades_by_sentiment = merged_data.groupby('classification').size().reset_index(name='Total Trades')
    win_rate_by_sentiment = pd.merge(winning_trades_by_sentiment, total_trades_by_sentiment, on='classification', how='outer').fillna(0)
    win_rate_by_sentiment['Win Rate (%)'] = (win_rate_by_sentiment['Winning Trades'] / win_rate_by_sentiment['Total Trades']) * 100
    performance_by_sentiment = pd.merge(pnl_by_sentiment, win_rate_by_sentiment[['classification', 'Win Rate (%)']], on='classification', how='left')
    st.dataframe(performance_by_sentiment.round(2))

    col1, col2 = st.columns(2)
    with col1:
        fig_pnl, ax_pnl = plt.subplots(figsize=(10, 6))
        sns.barplot(x='classification', y='Closed PnL', data=performance_by_sentiment, palette='viridis', ax=ax_pnl, hue='classification', legend=False)
        ax_pnl.set_title('Total PnL by Fear & Greed Classification')
        ax_pnl.set_xlabel('Sentiment Classification')
        ax_pnl.set_ylabel('Total Closed PnL')
        ax_pnl.tick_params(axis='x', rotation=45)
        st.pyplot(fig_pnl)

    with col2:
        fig_winrate, ax_winrate = plt.subplots(figsize=(10, 6))
        sns.barplot(x='classification', y='Win Rate (%)', data=performance_by_sentiment, palette='magma', ax=ax_winrate, hue='classification', legend=False)
        ax_winrate.set_title('Win Rate by Fear & Greed Classification')
        ax_winrate.set_xlabel('Sentiment Classification')
        ax_winrate.set_ylabel('Win Rate (%)')
        ax_winrate.tick_params(axis='x', rotation=45)
        st.pyplot(fig_winrate)

    st.write("#### Trade Frequency by Sentiment Classification")
    trade_frequency_by_sentiment = merged_data.groupby('classification').size().reset_index(name='Trade Count')
    st.dataframe(trade_frequency_by_sentiment.round(2))
    fig_freq, ax_freq = plt.subplots(figsize=(10, 6))
    sns.barplot(x='classification', y='Trade Count', data=trade_frequency_by_sentiment, palette='coolwarm', ax=ax_freq, hue='classification', legend=False)
    ax_freq.set_title('Trade Frequency by Fear & Greed Classification')
    ax_freq.set_xlabel('Sentiment Classification')
    ax_freq.set_ylabel('Number of Trades')
    ax_freq.tick_params(axis='x', rotation=45)
    st.pyplot(fig_freq)

    st.write("#### Long/Short Bias by Sentiment Classification")
    long_short_by_sentiment = merged_data.groupby(['classification', 'Side']).size().unstack(fill_value=0).reset_index()
    long_short_by_sentiment['Long/Short Ratio'] = long_short_by_sentiment['BUY'] / long_short_by_sentiment['SELL']
    st.dataframe(long_short_by_sentiment.round(2))
    fig_ls, ax_ls = plt.subplots(figsize=(10, 6))
    sns.barplot(x='classification', y='Long/Short Ratio', data=long_short_by_sentiment, palette='coolwarm', ax=ax_ls, hue='classification', legend=False)
    ax_ls.set_title('Long/Short Ratio by Fear & Greed Classification')
    ax_ls.set_xlabel('Sentiment Classification')
    ax_ls.set_ylabel('Long/Short Ratio')
    ax_ls.tick_params(axis='x', rotation=45)
    st.pyplot(fig_ls)

    st.write("#### Average Position Sizes by Sentiment Classification")
    position_size_by_sentiment = merged_data.groupby('classification')[['Size Tokens', 'Size USD']].mean().reset_index()
    st.dataframe(position_size_by_sentiment.round(2))

# --- Page: Trader Segmentation ---
elif page == "Trader Segmentation":
    st.title("Trader Segmentation into Behavioral Archetypes")

    trader_performance = merged_data.groupby('Account').agg(
        total_trades=('Order ID', 'size'),
        total_pnl=('Closed PnL', 'sum'),
        winning_trades=('Closed PnL', lambda x: (x > 0).sum())
    ).reset_index()
    trader_performance['win_rate'] = (trader_performance['winning_trades'] / trader_performance['total_trades']) * 100
    st.write("### Aggregated Trader Performance")
    st.dataframe(trader_performance.head())

    st.write("### Clustering Traders (K-Means)")
    clustering_features = trader_performance[['total_trades', 'total_pnl', 'win_rate']]
    clustering_features = clustering_features.fillna(0).replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_features)

    # Using optimal K=3 from notebook analysis
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    trader_performance['cluster'] = kmeans.fit_predict(scaled_features)

    cluster_summary = trader_performance.groupby('cluster')[['total_trades', 'total_pnl', 'win_rate']].mean().reset_index()
    st.write(f"#### Cluster Summary (K={optimal_k})")
    st.dataframe(cluster_summary.round(2))

    fig_cluster, ax_cluster = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='total_trades', y='win_rate', hue='cluster', data=trader_performance, palette='viridis', s=100, alpha=0.8, ax=ax_cluster)
    ax_cluster.set_title('Trader Clusters based on Total Trades and Win Rate')
    ax_cluster.set_xlabel('Total Trades')
    ax_cluster.set_ylabel('Win Rate (%)')
    ax_cluster.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_cluster)

    st.write("#### Trader Archetypes Identified")
    st.markdown("Based on K-Means clustering, traders are segmented into archetypes with distinct characteristics in terms of trade volume, profitability, and win rate.")

# --- Page: Drawdown Analysis ---
elif page == "Drawdown Analysis":
    st.title("Drawdown Analysis")

    st.write("### Maximum Drawdown per Account")
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    daily_pnl = merged_data.groupby(['Account', 'Date'])['Closed PnL'].sum().reset_index()
    daily_pnl = daily_pnl.sort_values(by=['Account', 'Date'])
    daily_pnl['Cumulative PnL'] = daily_pnl.groupby('Account')['Closed PnL'].cumsum()
    daily_pnl['Peak Cumulative PnL'] = daily_pnl.groupby('Account')['Cumulative PnL'].cummax()
    daily_pnl['Drawdown'] = daily_pnl['Cumulative PnL'] - daily_pnl['Peak Cumulative PnL']
    max_drawdown_per_account = daily_pnl.groupby('Account')['Drawdown'].min().reset_index()
    max_drawdown_per_account.rename(columns={'Drawdown': 'Max Drawdown'}, inplace=True)
    st.dataframe(max_drawdown_per_account.head())

    fig_drawdown, ax_drawdown = plt.subplots(figsize=(10, 6))
    sns.histplot(max_drawdown_per_account['Max Drawdown'], bins=30, kde=True, ax=ax_drawdown)
    ax_drawdown.set_title('Distribution of Maximum Drawdown per Account')
    ax_drawdown.set_xlabel('Maximum Drawdown (USD)')
    ax_drawdown.set_ylabel('Number of Accounts')
    ax_drawdown.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_drawdown)

# --- Page: Predictive Model ---
elif page == "Predictive Model":
    st.title("Predictive Model for Next-Day Trader Profitability")

    st.write("This section trains a RandomForestClassifier to predict whether a trader will be profitable the next day based on their previous day's activity and market sentiment.")

    model, X_test, y_test, y_pred, X_cols = prepare_and_train_model(merged_data, fear_greed_index)

    st.write("### Model Evaluation Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.json(report)

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    st.write("### Feature Importances")
    feature_importances = pd.Series(model.feature_importances_, index=X_cols).sort_values(ascending=False)
    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax_fi)
    ax_fi.set_title('Feature Importances for Next-Day Profitability Prediction')
    ax_fi.set_xlabel('Importance')
    ax_fi.set_ylabel('Feature')
    st.pyplot(fig_fi)
