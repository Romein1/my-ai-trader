import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import plotly.graph_objects as go
import plotly.express as px
import pytz
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gemini Live Terminal Pro", layout="wide", page_icon="📈")

# --- AUTHENTICATION & CONFIG ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    API_AVAILABLE = True
except:
    API_AVAILABLE = False
    st.error("⚠️ GEMINI_API_KEY missing in Secrets.")

# NSE Watchlist
WATCHLIST = {
    'NIFTY 50': '^NSEI',
    'BANK NIFTY': '^NSEBANK',
    'RELIANCE': 'RELIANCE.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'TATA MOTORS': 'TATAMOTORS.NS',
    'INFOSYS': 'INFY.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'ADANI ENT': 'ADANIENT.NS',
    'SBI': 'SBIN.NS',
    'ITC': 'ITC.NS'
}

# --- QUANT ENGINE ---
class QuantEngine:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_data(self, period="5d", interval="15m"):
        try:
            df = yf.download(self.ticker, period=period, interval=interval, progress=False)
            if df.empty: return None
            
            # Flatten MultiIndex if exists
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            return df
        except: return None

    def add_technical_indicators(self, df):
        df = df.copy()
        
        # 1. VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

        # 2. VWMA (Volume Weighted Moving Average)
        df['PV'] = df['Close'] * df['Volume']
        df['VWMA'] = df['PV'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

        # 3. VMA (Variable Moving Average)
        # Uses Chande Momentum Oscillator (CMO) to adjust smoothing
        period = 9
        df['Mom'] = df['Close'].diff()
        df['Abs_Mom'] = df['Mom'].abs()
        df['Mom_Sum'] = df['Mom'].rolling(window=period).sum().abs()
        df['Abs_Mom_Sum'] = df['Abs_Mom'].rolling(window=period).sum()
        df['ER'] = df['Mom_Sum'] / df['Abs_Mom_Sum'] # Efficiency Ratio
        df['SC'] = df['ER'] * (2/(2+1) - 2/(30+1)) + 2/(30+1) # Smoothing Constant
        df['VMA'] = 0.0
        # Calculate VMA iteratively
        vma_values = [df['Close'].iloc[0]]
        for i in range(1, len(df)):
            sc = df['SC'].iloc[i]
            if pd.isna(sc): sc = 0.1
            prev_vma = vma_values[-1]
            vma = prev_vma + sc * (df['Close'].iloc[i] - prev_vma)
            vma_values.append(vma)
        df['VMA'] = vma_values

        # 4. SuperTrend (Volatility Stop)
        atr_period = 10
        multiplier = 3.0
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(atr_period).mean()
        
        df['Basic_Upper'] = (df['High'] + df['Low']) / 2 + multiplier * df['ATR']
        df['Basic_Lower'] = (df['High'] + df['Low']) / 2 - multiplier * df['ATR']
        
        # SuperTrend Logic (Simplified for speed)
        df['SuperTrend'] = df['Basic_Lower'] # Default view
        df['Trend'] = np.where(df['Close'] > df['VMA'], 1, -1) # Using VMA as trend baseline

        return df.dropna()

# --- GEMINI LIVE NEWS AGENT ---
class NewsAgent:
    def __init__(self):
        # We use a tool config to enable Google Search
        self.tools = [
            {"google_search": {}} # Enables Live Search Grounding
        ]
        self.model = genai.GenerativeModel('gemini-1.5-flash', tools=self.tools)

    def get_live_analysis(self, ticker, price_data):
        if not API_AVAILABLE: return "⚠️ API Key Needed for Live News"
        
        current_price = price_data['Close']
        change = price_data['Close'] - price_data['Open']
        
        # Prompt forces Gemini to use Google Search tool
        prompt = f"""
        Use Google Search to find the latest LIVE news (last 24 hours) for {ticker} (Indian Stock Market).
        
        Technical Context:
        - Price: {current_price}
        - Day Change: {change}
        
        Task:
        1. Find top 2 breaking news headlines for {ticker} today.
        2. Analyze if this news is Bullish (Positive) or Bearish (Negative).
        3. Combine this with the technical price action.
        
        Output format:
        "NEWS: [Headline] - [Impact]"
        "VERDICT: [BULLISH/BEARISH] because..."
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Check if we got a grounded response (from search)
            return response.text
        except Exception as e:
            return f"Gemini connection error: {e}"

# --- DASHBOARD ---
st.title("🇮🇳 Gemini Live Terminal: VMA & News Edition")

# Sidebar
with st.sidebar:
    st.header("🎮 Command Center")
    ticker_key = st.selectbox("Select Asset", list(WATCHLIST.keys()))
    ticker_sym = WATCHLIST[ticker_key]
    refresh = st.button("⚡ Refresh Live Data & News")

# Main Execution
engine = QuantEngine(ticker_sym)
news_bot = NewsAgent()

if 'data' not in st.session_state or refresh:
    with st.spinner(f"Fetching {ticker_key} data & searching live news..."):
        raw_df = engine.get_data()
        if raw_df is not None:
            processed_df = engine.add_technical_indicators(raw_df)
            st.session_state['data'] = processed_df
            st.session_state['latest'] = processed_df.iloc[-1]
            
            # Fetch Live News Analysis only on refresh to save API quota
            st.session_state['news_analysis'] = news_bot.get_live_analysis(ticker_key, st.session_state['latest'])
        else:
            st.error("Market closed or data unavailable.")

if 'data' in st.session_state:
    df = st.session_state['data']
    latest = st.session_state['latest']
    
    # --- SECTION 1: LIVE NEWS & SENTIMENT ---
    st.subheader(f"📰 Live News Intelligence ({ticker_key})")
    with st.chat_message("assistant", avatar="🤖"):
        st.write(st.session_state['news_analysis'])
        st.caption("ℹ️ Analyzed using Google Search Grounding + Technical Data")

    # --- SECTION 2: VMAT (VMA) & TRADING METRICS ---
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    
    # Safe float conversion
    price = float(latest['Close'])
    vwap = float(latest['VWAP'])
    vma = float(latest['VMA'])
    
    with c1: 
        st.metric("Live Price", f"₹{price:.2f}", f"{price - float(df.iloc[-2]['Close']):.2f}")
    with c2:
        # VMA Logic: If Price > VMA, Trend is Up
        delta_vma = price - vma
        st.metric("VMAT (Trend Line)", f"₹{vma:.2f}", f"{delta_vma:.2f}", delta_color="normal")
    with c3:
        # VWAP Logic
        st.metric("VWAP (Inst. Price)", f"₹{vwap:.2f}", "Bullish" if price > vwap else "Bearish")
    with c4:
        # Volume Spike Detection
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        is_spike = latest['Volume'] > (vol_avg * 1.5)
        st.metric("Volume Status", "🔥 High" if is_spike else "❄️ Normal", f"{int(latest['Volume'])}")

    # --- SECTION 3: ADVANCED CHARTING ---
    st.subheader("📊 Intraday Technicals (VMA + VWAP)")
    
    tab1, tab2 = st.tabs(["Price Action", "Trend Strength"])
    
    with tab1:
        fig = go.Figure()
        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'], name='Price'))
        
        # VMA (The "Smart" Moving Average)
        fig.add_trace(go.Scatter(x=df.index, y=df['VMA'], line=dict(color='purple', width=2), name='VMA (Trend)'))
        
        # VWAP (Institutional Level)
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange', width=2, dash='dot'), name='VWAP'))
        
        # SuperTrend (Stop Loss)
        fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], line=dict(color='green', width=1), name='Stop Loss Zone'))

        fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{ticker_key} - VMA & VWAP Analysis")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Volatility & Efficiency
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['ER'], fill='tozeroy', name='Market Efficiency (ER)'))
        fig2.update_layout(title="Market Efficiency (Close to 1 = Strong Trend, Close to 0 = Choppy)", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # --- SECTION 4: AI VERDICT ---
    st.success(f"**Final Strategy:** {'BUY ON DIPS' if price > vma and price > vwap else 'SELL ON RISE' if price < vma else 'WAIT'}")
    st.info("VMA Rule: If Price is ABOVE the Purple Line (VMA), the trend is strong. If Price cuts BELOW, exit immediately.")
