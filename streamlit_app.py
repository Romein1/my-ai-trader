import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gemini AI Live Trader", layout="wide")

# --- AUTHENTICATION ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    API_AVAILABLE = True
except:
    API_AVAILABLE = False

# --- CLASSES ---

class MarketData:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_live_data(self, period="5d", interval="15m"):
        try:
            # Fetch data
            data = yf.download(self.ticker, period=period, interval=interval, progress=False)
            
            if data.empty: 
                return None
            
            # --- CRITICAL FIX FOR YFINANCE UPDATES ---
            # If yfinance returns a MultiIndex (e.g., ('Close', 'AAPL')), flatten it to just 'Close'
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ensure columns are float type
            cols_to_fix = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in cols_to_fix:
                if col in data.columns:
                    data[col] = data[col].astype(float)
                    
            return data
        except Exception as e:
            st.error(f"Data Error: {e}")
            return None

    def calculate_technicals(self, df):
        df = df.copy()
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

        # Bollinger Bands
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['StdDev'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['SMA20'] + (df['StdDev'] * 2)
        df['Lower_Band'] = df['SMA20'] - (df['StdDev'] * 2)
        
        return df.dropna()

class GeminiNewsAgent:
    def analyze_dummy(self):
        return 0.0, "API Key Missing or Error - Using Neutral Sentiment"

    def analyze_sentiment(self, ticker):
        if not API_AVAILABLE:
            return self.analyze_dummy()
            
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Act as a senior financial analyst. 
        Generate 3 hypothetical but realistic breaking news headlines for {ticker} stock based on general market volatility patterns.
        Then, analyze the sentiment of these headlines.
        Return a single line: First the sentiment score (-1.0 to 1.0), then a separator "|", then the headlines summary.
        Example: 0.8 | Strong earnings expected.
        """
        
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            parts = text.split('|')
            score = float(parts[0].strip())
            summary = parts[1].strip() if len(parts) > 1 else "Analysis complete"
            return score, summary
        except:
            return 0.0, "Gemini Analysis Failed"

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def train(self, data):
        # Target: 1 if Price rises in next candle, 0 if falls
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        features = ['RSI', 'VWAP', 'Upper_Band', 'Lower_Band', 'Volume']
        
        clean_data = data.dropna()
        if len(clean_data) < 20: return 0 # Not enough data
        
        X = clean_data[features]
        y = clean_data['Target']
        
        self.model.fit(X, y)
        self.is_trained = True
        return self.model.score(X, y)

    def predict(self, latest_row):
        if not self.is_trained: return 0, 0.5
        features = ['RSI', 'VWAP', 'Upper_Band', 'Lower_Band', 'Volume']
        
        # Ensure values are simple floats (handle Series/Arrays)
        values = [float(latest_row[f]) for f in features]
        X_new = np.array(values).reshape(1, -1)
        
        prediction = self.model.predict(X_new)[0]
        prob = self.model.predict_proba(X_new)[0][1] 
        return prediction, prob

# --- MAIN APP UI ---
st.title("⚡ Gemini AI Live Trader")

if not API_AVAILABLE:
    st.warning("⚠️ Gemini API Key not found in Secrets. Sentiment analysis is disabled.")

# Sidebar
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
refresh = st.sidebar.button("Refresh Data Now")

# Main Logic
market = MarketData(ticker)
news_agent = GeminiNewsAgent()
predictor = MLPredictor()

data = market.get_live_data()

if data is not None and not data.empty:
    processed_data = market.calculate_technicals(data)
    
    # Train Model
    acc = predictor.train(processed_data)
    
    # Analyze
    latest = processed_data.iloc[-1]
    
    # --- SAFE VALUE EXTRACTION ---
    # We explicitly convert these to python floats to avoid TypeError
    current_price = float(latest['Close'])
    rsi_val = float(latest['RSI'])
    
    sentiment, news = news_agent.analyze_sentiment(ticker)
    pred, prob = predictor.predict(latest)
    
    # --- DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1: st.metric("Price", f"${current_price:.2f}")
    with col2: st.metric("RSI", f"{rsi_val:.1f}")
    with col3: st.metric("AI Sentiment", f"{sentiment:.2f}")
    
    # Signal Logic
    signal = "NEUTRAL"
    color = "gray"
    if pred == 1 and prob > 0.6 and sentiment > 0.1:
        signal = "BUY"
        color = "green"
    elif pred == 0 and
