"""
Stock Price Movement Prediction using Decision Tree Classifier
A Streamlit-based interactive application for predicting stock price direction
using a Decision Tree classifier with synthetic financial data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Function to generate synthetic stock data
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic stock market data with realistic patterns.
    Features include:
    - Open, High, Low, Close prices
    - Volume
    - Technical indicators (SMA, RSI, Volatility)
    Target: 1 if price increases next day, 0 otherwise
    """
    np.random.seed(42)
    
    # Generate base price using random walk
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate Open, High, Low, Close prices
    open_prices = prices * (1 + np.random.normal(0, 0.005, n_samples))
    close_prices = prices * (1 + np.random.normal(0.005, 0.01, n_samples))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
    
    # Generate volume (correlated with price movement)
    volume_base = np.random.normal(1e6, 2e5, n_samples)
    volume = volume_base * (1 + np.abs(returns) * 5)
    volume = np.maximum(volume, 1e5)
    
    # Calculate technical indicators
    # Simple Moving Average (5-day)
    sma_5 = pd.Series(close_prices).rolling(window=5).mean().fillna(method='bfill').values
    
    # Simple Moving Average (20-day)
    sma_20 = pd.Series(close_prices).rolling(window=20).mean().fillna(method='bfill').values
    
    # Volatility (standard deviation of returns over 10 days)
    volatility = pd.Series(returns).rolling(window=10).std().fillna(0.02).values
    
    # Relative Strength Index (simplified)
    gains = np.maximum(returns, 0)
    losses = np.abs(np.minimum(returns, 0))
    avg_gain = pd.Series(gains).rolling(window=14).mean().fillna(0.01).values
    avg_loss = pd.Series(losses).rolling(window=14).mean().fillna(0.01).values
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    
    # Price to SMA ratio
    price_sma_ratio = close_prices / (sma_20 + 1e-6)
    
    # Target: 1 if price increases tomorrow, else 0
    target = (np.roll(close_prices, -1) > close_prices).astype(int)
    target = target[:-1]  # Remove last row as no future price
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=n_samples, freq='D')[:n_samples-1],
        'Open': open_prices[:-1],
        'High': high_prices[:-1],
        'Low': low_prices[:-1],
        'Close': close_prices[:-1],
        'Volume': volume[:-1],
        'SMA_5': sma_5[:-1],
        'SMA_20': sma_20[:-1],
        'Volatility': volatility[:-1],
        'RSI': rsi[:-1],
        'Price_SMA_Ratio': price_sma_ratio[:-1],
        'Target': target
    })
    
    return df

# Function to prepare features for modeling
def prepare_features(df):
    """Select and prepare features for the model"""
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA_5', 'SMA_20', 'Volatility', 'RSI', 'Price_SMA_Ratio']
    
    X = df[feature_cols]
    y = df['Target']
    
    return X, y, feature_cols

# Function to train and evaluate model
def train_model(X_train, X_test, y_train, y_test, max_depth=None, min_samples_split=2):
    """Train Decision Tree classifier and return model with metrics"""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        criterion='entropy'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, y_pred, accuracy

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">📊 Stock Price Movement Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Using Decision Tree Classifier | Synthetic Market Data</div>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data generation parameters
        st.subheader("Data Settings")
        n_samples = st.slider("Number of Samples", 500, 5000, 1000, step=500)
        
        # Model parameters
        st.subheader("Model Parameters")
        max_depth = st.slider("Max Depth", 1, 20, 5, help="Maximum depth of the decision tree")
        min_samples_split = st.slider("Min Samples Split", 2, 20, 5, help="Minimum samples required to split a node")
        
        # Test size
        test_size = st.slider("Test Size (%)", 10, 40, 20, step=5) / 100
        
        # Generate new data button
        if st.button("🔄 Generate New Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Generate data
    with st.spinner("Generating synthetic stock data..."):
        df = generate_synthetic_data(n_samples)
    
    # Display data overview
    st.header("📋 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{len(df.columns) - 1}")
    with col3:
        up_days = (df['Target'] == 1).sum()
        st.metric("Up Days", f"{up_days} ({up_days/len(df)*100:.1f}%)")
    with col4:
        down_days = (df['Target'] == 0).sum()
        st.metric("Down Days", f"{down_days} ({down_days/len(df)*100:.1f}%)")
    
    # Display raw data
    with st.expander("📊 View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 rows of {len(df)} total rows")
    
    # Feature preparation
    X, y, feature_cols = prepare_features(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train model
    with st.spinner("Training Decision Tree Classifier..."):
        model, y_pred, accuracy = train_model(X_train, X_test, y_train, y_test, max_depth, min_samples_split)
    
    # Display metrics
    st.header("🎯 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}", 
                  delta=f"{accuracy - 0.5:.2%}" if accuracy > 0.5 else None)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax_cm.set_title('Confusion Matrix')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)
        plt.close()
    
    with col2:
        # Classification Report
        report = classification_report(y_test, y_pred, target_names=['Down', 'Up'], output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)
        
        # Feature Importance
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        bars = ax_imp.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        ax_imp.set_xlabel('Importance')
        ax_imp.set_title('Feature Importance')
        ax_imp.invert_yaxis()
        for bar in bars:
            width = bar.get_width()
            ax_imp.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                       ha='left', va='center', fontsize=9)
        st.pyplot(fig_imp)
        plt.close()
    
    # Decision Tree Visualization
    st.header("🌳 Decision Tree Visualization")
    st.info("Note: Tree visualization is simplified for readability. Larger trees may appear condensed.")
    
    # Limit tree depth for visualization if too deep
    viz_depth = min(max_depth, 4) if max_depth else 4
    fig_tree, ax_tree = plt.subplots(figsize=(20, 12))
    plot_tree(model, 
              feature_names=feature_cols,
              class_names=['Down', 'Up'],
              filled=True,
              rounded=True,
              max_depth=viz_depth,
              fontsize=10,
              ax=ax_tree)
    st.pyplot(fig_tree)
    plt.close()
    
    # Prediction Section
    st.header("🔮 Make Predictions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enter Stock Features")
        
        # Create input widgets for each feature
        input_values = {}
        for feature in feature_cols:
            if feature in ['Open', 'High', 'Low', 'Close', 'SMA_5', 'SMA_20']:
                input_values[feature] = st.number_input(feature, value=100.0, step=1.0, format="%.2f")
            elif feature == 'Volume':
                input_values[feature] = st.number_input(feature, value=1000000.0, step=100000.0, format="%.0f")
            elif feature == 'Volatility':
                input_values[feature] = st.number_input(feature, value=0.02, step=0.001, format="%.3f")
            elif feature == 'RSI':
                input_values[feature] = st.slider(feature, 0.0, 100.0, 50.0, step=1.0)
            else:  # Price_SMA_Ratio
                input_values[feature] = st.number_input(feature, value=1.0, step=0.01, format="%.2f")
    
    with col2:
        st.subheader("Prediction Result")
        
        # Create input array
        input_array = np.array([input_values[f] for f in feature_cols]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0]
        
        # Display prediction
        if prediction == 1:
            st.markdown("""
                <div style="background-color: #2ecc71; padding: 2rem; border-radius: 10px; text-align: center;">
                    <h2 style="color: white;">📈 UP ⬆️</h2>
                    <p style="color: white; font-size: 1.2rem;">Price predicted to INCREASE</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color: #e74c3c; padding: 2rem; border-radius: 10px; text-align: center;">
                    <h2 style="color: white;">📉 DOWN ⬇️</h2>
                    <p style="color: white; font-size: 1.2rem;">Price predicted to DECREASE</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Probability bars
        st.markdown("**Prediction Probabilities**")
        proba_df = pd.DataFrame({
            'Direction': ['Down', 'Up'],
            'Probability': [proba[0], proba[1]]
        })
        fig_prob, ax_prob = plt.subplots(figsize=(6, 3))
        bars = ax_prob.bar(proba_df['Direction'], proba_df['Probability'], 
                          color=['#e74c3c', '#2ecc71'])
        ax_prob.set_ylim(0, 1)
        ax_prob.set_ylabel('Probability')
        ax_prob.set_title('Class Probabilities')
        for bar in bars:
            height = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}', ha='center', va='bottom')
        st.pyplot(fig_prob)
        plt.close()
    
    # Price Chart Visualization
    st.header("📈 Stock Price Trend")
    
    fig_price, ax_price = plt.subplots(figsize=(12, 5))
    ax_price.plot(df['Date'], df['Close'], label='Close Price', color='steelblue', linewidth=1)
    ax_price.plot(df['Date'], df['SMA_20'], label='SMA 20', color='orange', linewidth=1.5, linestyle='--')
    ax_price.fill_between(df['Date'], df['Low'], df['High'], alpha=0.2, color='gray')
    ax_price.set_xlabel('Date')
    ax_price.set_ylabel('Price')
    ax_price.set_title('Stock Price with Support/Resistance Zone')
    ax_price.legend()
    ax_price.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_price)
    plt.close()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>📊 Decision Tree Classifier for Stock Price Direction Prediction | Data is synthetically generated for demonstration purposes</p>
            <p>⚠️ This is for educational purposes only. Not financial advice.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()