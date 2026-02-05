import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gemini Pro Trading Terminal", layout="wide", page_icon="💹")

# --- AUTHENTICATION ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    API_AVAILABLE = True
except:
    API_AVAILABLE = False

# --- CONFIGURATION ---
WATCHLIST = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'NFLX', 'INTC']

# --- ADVANCED ANALYSIS CLASS ---
class MarketEngine:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_batch_data(self):
        """Fetches data for ALL stocks in watchlist at once"""
        try:
            data = yf.download(self.tickers, period="5d", interval="15m", group_by='ticker', progress=False)
            return data
        except Exception as e:
            st.error(f"Data Fetch Error: {e}")
            return None

    def analyze_ticker(self, ticker_data):
        """Calculates indicators for a single stock"""
        df = ticker_data.copy()
        if df.empty: return None

        # Fix MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 1. RSI (Momentum)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. VWAP (Trend)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

        # 3. Buyers vs Sellers (Money Flow Proxy)
        # If Close > Open, we assume buying pressure. If Close < Open, selling.
        df['Buy_Vol'] = np.where(df['Close'] > df['Open'], df['Volume'], 0)
        df['Sell_Vol'] = np.where(df['Close'] < df['Open'], df['Volume'], 0)
        
        # 4. Support & Resistance (Pivot Points)
        last_day = df.iloc[-20:] # Last ~5 hours
        df['Support'] = last_day['Low'].min()
        df['Resistance'] = last_day['High'].max()

        return df.dropna()

class MLBrain:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def get_prediction(self, df):
        # Prepare Data
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        features = ['RSI', 'VWAP', 'Open', 'Volume']
        
        train_df = df.dropna()
        if len(train_df) < 50: return 0, 0.5 # Not enough data

        X = train_df[features]
        y = train_df['Target']
        
        self.model.fit(X, y)
        
        latest = df.iloc[-1][features].values.reshape(1, -1)
        pred = self.model.predict(latest)[0]
        prob = self.model.predict_proba(latest)[0][1]
        
        return pred, prob

# --- MAIN APP LOGIC ---
st.title("💹 Gemini AI Pro Terminal")

# 1. SIDEBAR - CONTROL PANEL
with st.sidebar:
    st.header("📡 Market Scanner")
    refresh_btn = st.button("🔄 Scan Market Now", type="primary")
    selected_ticker = st.selectbox("Select Stock for Deep Dive", WATCHLIST)
    
    st.markdown("---")
    st.markdown("### 🤖 AI Settings")
    risk_tolerance = st.select_slider("Risk Profile", options=["Conservative", "Balanced", "Aggressive"], value="Balanced")

# 2. DATA PROCESSING
engine = MarketEngine(WATCHLIST)
ml = MLBrain()

if 'market_data' not in st.session_state or refresh_btn:
    with st.spinner("Scanning Global Markets..."):
        st.session_state['market_data'] = engine.get_batch_data()
        st.toast("Market Data Updated!")

raw_data = st.session_state['market_data']

if raw_data is not None:
    # --- DASHBOARD: MARKET OVERVIEW ---
    st.subheader("🔥 Live Intraday Scanner")
    
    scanner_results = []
    
    for t in WATCHLIST:
        try:
            # Handle YFinance weirdness with MultiIndex
            stock_df = raw_data[t].copy() if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
            
            processed = engine.analyze_ticker(stock_df)
            if processed is not None:
                latest = processed.iloc[-1]
                pred, prob = ml.get_prediction(processed)
                
                # Signal Logic
                signal = "HOLD"
                score = 0
                if pred == 1 and prob > 0.6: 
                    signal = "BUY"
                    score = 1
                elif pred == 0 and prob < 0.4: 
                    signal = "SELL"
                    score = -1
                
                scanner_results.append({
                    "Ticker": t,
                    "Price": f"${latest['Close']:.2f}",
                    "RSI": f"{latest['RSI']:.1f}",
                    "Prediction": signal,
                    "Confidence": f"{prob*100:.0f}%",
                    "Score": score # Hidden for sorting
                })
        except:
            continue

    # Create Scanner Dataframe
    scan_df = pd.DataFrame(scanner_results)
    
    # Styling signals
    def color_signal(val):
        color = 'white'
        if val == "BUY": color = '#4CAF50' # Green
        elif val == "SELL": color = '#FF5252' # Red
        return f'color: {color}; font-weight: bold'

    st.dataframe(scan_df.drop('Score', axis=1).style.map(color_signal, subset=['Prediction']), use_container_width=True)

    # --- DEEP DIVE SECTION ---
    st.markdown("---")
    st.header(f"🔎 Deep Dive: {selected_ticker}")
    
    # Get specific stock data
    stock_df = raw_data[selected_ticker].copy()
    processed = engine.analyze_ticker(stock_df)
    latest = processed.iloc[-1]
    pred, prob = ml.get_prediction(processed)
    
    # 1. KEY METRICS ROW
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${latest['Close']:.2f}", f"{(latest['Close'] - processed.iloc[-2]['Close']):.2f}")
    with col2:
        st.metric("Support (Buy Zone)", f"${latest['Support']:.2f}")
    with col3:
        st.metric("Resistance (Sell Zone)", f"${latest['Resistance']:.2f}")
    with col4:
        # Buyer vs Seller Pressure Gauge
        buy_pressure = latest['Buy_Vol']
        sell_pressure = latest['Sell_Vol']
        total = buy_pressure + sell_pressure
        if total == 0: total = 1
        buy_pct = (buy_pressure / total) 
        st.metric("Buying Pressure", f"{buy_pct*100:.0f}%")
        st.progress(buy_pct)

    # 2. ADVANCED CHARTS
    tab1, tab2 = st.tabs(["📈 Technical Chart", "📊 Order Book Analysis"])
    
    with tab1:
        fig = go.Figure()
        # Candles
        fig.add_trace(go.Candlestick(x=processed.index, open=processed['Open'], high=processed['High'],
                        low=processed['Low'], close=processed['Close'], name='Price'))
        # VWAP
        fig.add_trace(go.Scatter(x=processed.index, y=processed['VWAP'], line=dict(color='orange', width=2), name='VWAP'))
        # Support/Resistance
        fig.add_hline(y=latest['Support'], line_dash="dash", line_color="green", annotation_text="Support")
        fig.add_hline(y=latest['Resistance'], line_dash="dash", line_color="red", annotation_text="Resistance")
        
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("💡 Buyer/Seller Pressure is estimated from Intraday Volume Flow.")
        # Volume Color Chart
        colors = ['red' if row['Open'] > row['Close'] else 'green' for index, row in processed.iterrows()]
        fig_vol = go.Figure(data=[go.Bar(x=processed.index, y=processed['Volume'], marker_color=colors)])
        fig_vol.update_layout(title="Volume Flow (Green=Buying, Red=Selling)", height=300)
        st.plotly_chart(fig_vol, use_container_width=True)

    # 3. GEMINI AI ADVISOR
    st.subheader("🧠 Gemini AI Strategy")
    if API_AVAILABLE:
        if st.button("Generate AI Trade Plan"):
            with st.spinner("Gemini is analyzing market structure..."):
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                Analyze {selected_ticker} based on this data:
                - Price: {latest['Close']}
                - RSI: {latest['RSI']} (Over 70=Overbought, Under 30=Oversold)
                - VWAP: {latest['VWAP']} (Price above VWAP is bullish)
                - Support: {latest['Support']}
                - Resistance: {latest['Resistance']}
                
                Risk Profile: {risk_tolerance}.
                
                Give a strict INTRADAY trading plan in 3 bullet points:
                1. Entry Price (Where to buy/sell)
                2. Stop Loss (Where to cut losses)
                3. Target (Where to take profit)
                """
                response = model.generate_content(prompt)
                st.success("Analysis Complete")
                st.write(response.text)
    else:
        st.warning("Connect Gemini API Key in Secrets for AI Plans")

else:
    st.error("Waiting for data... Click 'Scan Market Now'")
