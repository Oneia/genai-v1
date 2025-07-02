# -*- coding: utf-8 -*-
"""
Trading LLM Analysis - Streamlit Application
A comprehensive trading analysis tool that combines Reddit sentiment with financial fundamentals.
"""

import streamlit as st
import json
import datetime as dt
from textwrap import dedent
from typing import List, Dict, Optional
import pandas as pd

# Import the existing modules
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.utils.pprint import pprint_run_response
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from reddit import RedditService

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Trading LLM Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Data models
class TickerSocialInsight(BaseModel):
    """One ticker's subreddit / social-media pulse."""
    symbol: str = Field(..., description="Ticker symbol, e.g. TSLA.")
    mentions: int = Field(..., description="Total number of posts + comments that mention the ticker in the last 24 h.")
    avg_sentiment: float = Field(..., description="Mean polarity across all mentions (-1 ... +1).")
    bullish_sentiment: float = Field(..., description="Mean polarity of mentions tagged bullish (> +0.05).")
    bearish_sentiment: float = Field(..., description="Mean polarity of mentions tagged bearish (< -0.05).")
    volume_change_pct: float = Field(..., description="Percentage change in mention volume versus the preceding 24 h window.")
    summary: str = Field(..., description="Concise natural-language takeaway of the social chatter.")

class SentimentSnapshot(BaseModel):
    """Collection of ticker-level insights captured at one point in time."""
    tickers: List[TickerSocialInsight] = Field(..., description="List of social-sentiment insights for all tickers in the scan.")

class ComponentBreakdown(BaseModel):
    """Granular factors feeding the value-risk signal (0 = worst ... 1 = best)."""
    valuation: float = Field(..., ge=0.0, le=1.0)
    growth: float = Field(..., ge=0.0, le=1.0)
    profitability: float = Field(..., ge=0.0, le=1.0)
    leverage: float = Field(..., ge=0.0, le=1.0)
    cash_flow: float = Field(..., ge=0.0, le=1.0)
    shareholder_return: float = Field(..., ge=0.0, le=1.0)
    red_flag: float = Field(..., ge=0.0, le=1.0, description="Higher = greater risk red flags")

class ValueRiskInsight(BaseModel):
    """High-level fundamental outlook from the 'value_risk' agent."""
    symbol: str = Field(..., description="Ticker symbol, e.g. TSLA.")
    sentiment: float = Field(..., ge=-1.0, le=1.0, description="Overall stance (-1 bear ... +1 bull).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0...1).")
    component_breakdown: ComponentBreakdown
    rationale: str = Field(..., description="<= 45-word natural-language justification of the score.")


def initialize_services():
    """Initialize the AI models and services."""
    try:
        # Use Gemini model (you can switch to OpenAI if needed)
        model = Gemini(id='gemini-2.5-flash-preview-04-17')
        
        # Initialize Reddit service
        reddit_service = RedditService()
        
        return model, reddit_service
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        return None, None
class FinancialAnalysisResult(BaseModel):
    """Collection of financial analysis results."""
    results: List[ValueRiskInsight] = Field(..., description="List of financial analysis insights for all analyzed tickers.")
    window: str = Field(default="last_24h", description="Time window for the analysis.")


def initialize_agents(model):
    """Initialize the analysis agents."""
    if not model:
        return None, None, None
    
    # Reddit sentiment agent
    reddit_sentiment_agent = Agent(
        model=model,
        tools=[YFinanceTools(enable_all=True)],
        instructions=dedent("""
        You are the Reddit-Analysis Agent. Follow these steps EXACTLY:

        STEP 1: INPUT ANALYSIS
        - Parse the input phrase to identify:
          * **Ticker symbols**: 1-5 letter words that look like stock symbols (NVDA, AMD, etc.)
          * **Sector/area keywords**: remaining words that describe industries/sectors

        STEP 2: TICKER RESOLUTION
        - Convert any identified ticker words to uppercase
        - If sector keywords exist, use YFinance to find 8-10 largest market cap stocks in that sector
        - Final ticker list = direct tickers + sector tickers

        STEP 3: REDDIT DATA PROCESSING
        - Filter reddit_data to last 24h
        - For each final ticker, scan all posts/comments for mentions
        - Calculate sentiment metrics for each ticker

        STEP 4: ENSURE NON-EMPTY RESULTS
        - The tickers array must NOT be empty
        - If no tickers found after steps 1-2, use default popular tickers
        - Each TickerSocialInsight must have realistic data
        """),
        use_json_mode=True,
        response_model=SentimentSnapshot,
    )
    
    # Value risk agent
    value_risk_agent = Agent(
        model=model,
        tools=[YFinanceTools(enable_all=True), DuckDuckGoTools()],
        instructions=dedent("""
        You are the **Value / Risk Analyst**, a veteran fundamental PM.

        INPUT FORMAT: <free-text phrase> (e.g. "ai sector" or "NVDA AMD")

        PRE-PARSE RULES:
        • Extract 1-to-5 letter, case-insensitive tokens → tickers (e.g. "nvda" → NVDA).
        • Remaining words → sector keyword(s).

        TICKER RESOLUTION:
        1. If at least one ticker extracted → use those.
        2. ELSE (sector only):
           – If sector ∈ predefined map below, use those tickers.
           – Otherwise, fetch up to **10** largest-cap tickers that have
             `sector == <keyword>` via yfinance screener.

        FUNDAMENTAL SCORING:
        For each resolved ticker:
        1. Retrieve price and fundamentals via yfinance.
        2. Apply the 7-pillar scoring table (valuation, growth, profitability,
           leverage, cash-flow, shareholder return, red-flag).
        3. sentiment_raw = Σ(weighted pillar scores), clamp –1…+1.
        4. confidence = 0.1 + 0.9*abs(sentiment).
        5. rationale = one sentence (≤45 words) citing strongest + & – pillars.
     """),  # Add this line
        use_json_mode=True,
        response_model=FinancialAnalysisResult,
    )
    
    # Summarize agent
    summarize_agent = Agent(
        model=model,
        tools=[],
        instructions=dedent("""
        You are the **Synthesizer / Portfolio Manager**.

        INPUT: A single text string formatted exactly as:
        reddit sentiments: <REDDIT_JSON>, finance data: <FINANCE_JSON>

        TASKS:
        1. **Parse** the string to obtain two Python/JSON objects:
           `reddit = {...}` , `finance = {...}`.

        2. **For each ticker** appearing in either object:
           • sentiment = reddit.entities[sym].avg_sentiment  (null → 0)  
           • valuation  = finance[sym].get("pe_ttm")         (null → "n/a")  
           • build a bullet-point **Bull case** if sentiment>0.2 or pe_ttm<25.  
           • build a **Bear case** if sentiment<-0.2 or pe_ttm>35.  
           • Add a risk note if `reddit.entities[sym].volume_change_pct` > 50% 
             (crowd-hype risk) or if pe_ttm is "n/a".

        3. **Verdict & Size**
           • net = sentiment  
           • long if net>0.15 ; short if net<-0.15 ; else flat.  
           • size_pct = round(net * 50) capped ±30.

        4. **Compose a human-readable Markdown block** per ticker with all the above information.
        """),
        markdown=True
    )
    
    return reddit_sentiment_agent, value_risk_agent, summarize_agent


def get_reddit_data(reddit_service, subreddits):
    """Fetch Reddit data from specified subreddits."""
    if not reddit_service:
        return []
    
    reddit_data = []
    with st.spinner("Fetching Reddit data..."):
        for subreddit in subreddits:
            try:
                posts = reddit_service.get_posts_with_top_comments(subreddit, 10)
                reddit_data.extend(posts)
            except Exception as e:
                st.warning(f"Error fetching data from r/{subreddit}: {str(e)}")
    
    return reddit_data

# Main application
def main():
    st.markdown('<h1 class="main-header">📈 Trading LLM Analysis</h1>', unsafe_allow_html=True)
    
    # Initialize services
    model, reddit_service = initialize_services()
    if not model or not reddit_service:
        st.error("Failed to initialize services. Please check your environment variables.")
        return
    
    # Initialize agents
    reddit_sentiment_agent, value_risk_agent, summarize_agent = initialize_agents(model)
    if not all([reddit_sentiment_agent, value_risk_agent, summarize_agent]):
        st.error("Failed to initialize analysis agents.")
        return
    
    # Sidebar configuration
    st.sidebar.title("🔧 Configuration")
    
    # Subreddits selection
    default_subreddits = [
        "wallstreetbets", "stocks", "investing", "StockMarket",
        "personalfinance", "financialindependence", "ValueInvesting",
        "pennystocks", "options", "SecurityAnalysis", "dividends",
        "Bogleheads", "ETFs"
    ]
    
    selected_subreddits = st.sidebar.multiselect(
        "Select Subreddits",
        default_subreddits,
        default=default_subreddits[:5],
        help="Choose which subreddits to analyze for sentiment"
    )
    
    # Main content area - Stock List Analysis only
    st.subheader("📋 Stock Analysis")
    stock_input = st.text_area(
        "Enter stock symbols (one per line or comma-separated)",
        placeholder="NVDA\nAMD\nTSLA\nAAPL\nMSFT",
        height=150
    )
    
    if st.button("🚀 Analyze Stocks", type="primary"):
        if stock_input.strip():
            # Convert to comma-separated format
            stocks = [s.strip() for s in stock_input.replace('\n', ',').split(',') if s.strip()]
            user_input = ' '.join(stocks)
            analyze_request(user_input, selected_subreddits, reddit_service,
                          reddit_sentiment_agent, value_risk_agent, summarize_agent)
        else:
            st.warning("Please enter at least one stock symbol.")
    
    # Display usage instructions
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        ### 📊 Trading LLM Analysis App
        
        This app combines **social sentiment analysis** from Reddit with **fundamental financial analysis** 
        to provide comprehensive trading insights.
        
        #### 📈 Analysis Components:
        - **Reddit Sentiment**: Social media sentiment from selected subreddits
        - **Financial Fundamentals**: Valuation, growth, profitability metrics
        - **Combined Analysis**: Synthesized trading recommendations
        
        #### 🔧 Configuration:
        - Select relevant subreddits for sentiment analysis
        - Enter stock symbols to analyze
        - Get comprehensive trading insights with position sizing
        
        #### 💡 Example Stock Inputs:
        - `NVDA AMD INTC` (semiconductor stocks)
        - `TSLA AAPL MSFT GOOGL` (tech giants)
        - `JPM BAC WFC` (banking sector)
        """)

def analyze_request(user_input, selected_subreddits, reddit_service, 
                   reddit_sentiment_agent, value_risk_agent, summarize_agent):
    """Perform the complete analysis pipeline."""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Get Reddit data
        status_text.text("Step 1/4: Fetching Reddit data...")
        progress_bar.progress(25)
        
        reddit_data = get_reddit_data(reddit_service, selected_subreddits)
        
        if not reddit_data:
            st.warning("No Reddit data available. Proceeding with financial analysis only.")
        
        # Step 2: Reddit sentiment analysis
        status_text.text("Step 2/4: Analyzing Reddit sentiment...")
        progress_bar.progress(50)
        
        # Update agent instructions with current data
        current_time = dt.datetime.utcnow().isoformat() + "Z"
        
        try:
            reddit_response = reddit_sentiment_agent.run(user_input)
            reddit_sentiment_data = reddit_response.content
        except Exception as e:
            st.error(f"Error in Reddit sentiment analysis: {str(e)}")
            reddit_sentiment_data = None
        
        # Step 3: Financial analysis
        status_text.text("Step 3/4: Performing financial analysis...")
        progress_bar.progress(75)
        
        try:
            finance_response = value_risk_agent.run(user_input)
            finance_data = finance_response.content
        except Exception as e:
            st.error(f"Error in financial analysis: {str(e)}")
            finance_data = None
        
        # Step 4: Combined analysis
        status_text.text("Step 4/4: Synthesizing results...")
        progress_bar.progress(100)
        
        if reddit_sentiment_data and finance_data:
            combined_request = f"reddit sentiments: {reddit_sentiment_data}, finance data: {finance_data}"
            
            try:
                summary_response = summarize_agent.run(combined_request)
                display_results(summary_response.content, reddit_sentiment_data, finance_data)
            except Exception as e:
                st.error(f"Error in combined analysis: {str(e)}")
                # Display individual results
                display_individual_results(reddit_sentiment_data, finance_data)
        else:
            # Display individual results if combined analysis fails
            display_individual_results(reddit_sentiment_data, finance_data)
        
        status_text.text("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        status_text.text("❌ Analysis failed!")
    
    finally:
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

def display_results(summary_content, reddit_data, finance_data):
    """Display the combined analysis results."""
    st.subheader("📊 Analysis Results")
    
    # Display the summary content
    st.markdown(summary_content)
    
    # Create tabs for detailed views
    tab1, tab2, tab3 = st.tabs(["📈 Combined Analysis", "💬 Reddit Sentiment", "💰 Financial Data"])
    
    with tab1:
        st.markdown("### Combined Trading Analysis")
        st.markdown(summary_content)
    
    with tab2:
        st.markdown("### Reddit Sentiment Analysis")
        if reddit_data:
            try:
                # Parse and display reddit data
                if isinstance(reddit_data, str):
                    # Try to parse JSON if it's a string
                    reddit_json = json.loads(reddit_data)
                else:
                    reddit_json = reddit_data
                
                if 'tickers' in reddit_json:
                    display_reddit_sentiment_table(reddit_json['tickers'])
                else:
                    st.json(reddit_json)
            except Exception as e:
                st.error(f"Error parsing Reddit data: {str(e)}")
                st.text(reddit_data)
        else:
            st.warning("No Reddit sentiment data available.")
    
    with tab3:
        st.markdown("### Financial Analysis")
        if finance_data:
            try:
                # Parse and display finance data
                if isinstance(finance_data, str):
                    # Try to parse JSON if it's a string
                    finance_json = json.loads(finance_data)
                else:
                    finance_json = finance_data
                
                if 'results' in finance_json:
                    display_financial_table(finance_json['results'])
                else:
                    st.json(finance_json)
            except Exception as e:
                st.error(f"Error parsing financial data: {str(e)}")
                st.text(finance_data)
        else:
            st.warning("No financial data available.")

def display_individual_results(reddit_data, finance_data):
    """Display individual analysis results when combined analysis fails."""
    st.subheader("📊 Individual Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💬 Reddit Sentiment")
        if reddit_data:
            try:
                if isinstance(reddit_data, str):
                    reddit_json = json.loads(reddit_data)
                else:
                    reddit_json = reddit_data
                
                if 'tickers' in reddit_json:
                    display_reddit_sentiment_table(reddit_json['tickers'])
                else:
                    st.json(reddit_json)
            except Exception as e:
                st.error(f"Error parsing Reddit data: {str(e)}")
                st.text(reddit_data)
        else:
            st.warning("No Reddit sentiment data available.")
    
    with col2:
        st.markdown("### 💰 Financial Analysis")
        if finance_data:
            try:
                if isinstance(finance_data, str):
                    finance_json = json.loads(finance_data)
                else:
                    finance_json = finance_data
                
                if 'results' in finance_json:
                    display_financial_table(finance_json['results'])
                else:
                    st.json(finance_json)
            except Exception as e:
                st.error(f"Error parsing financial data: {str(e)}")
                st.text(finance_data)
        else:
            st.warning("No financial data available.")

def display_reddit_sentiment_table(tickers_data):
    """Display Reddit sentiment data in a table format."""
    if not tickers_data:
        st.warning("No ticker sentiment data available.")
        return
    
    # Convert to DataFrame for better display
    df_data = []
    for ticker in tickers_data:
        df_data.append({
            'Symbol': ticker.get('symbol', 'N/A'),
            'Mentions': ticker.get('mentions', 0),
            'Avg Sentiment': f"{ticker.get('avg_sentiment', 0):.3f}",
            'Bullish Sentiment': f"{ticker.get('bullish_sentiment', 0):.3f}",
            'Bearish Sentiment': f"{ticker.get('bearish_sentiment', 0):.3f}",
            'Volume Change %': f"{ticker.get('volume_change_pct', 0):.1f}%",
            'Summary': ticker.get('summary', 'N/A')[:100] + '...' if len(ticker.get('summary', '')) > 100 else ticker.get('summary', 'N/A')
        })
    
    df = pd.DataFrame(df_data)
    
    # Display with styling
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Avg Sentiment": st.column_config.NumberColumn(
                "Avg Sentiment",
                help="Average sentiment score (-1 to +1)",
                format="%.3f"
            ),
            "Bullish Sentiment": st.column_config.NumberColumn(
                "Bullish Sentiment",
                help="Bullish sentiment score",
                format="%.3f"
            ),
            "Bearish Sentiment": st.column_config.NumberColumn(
                "Bearish Sentiment",
                help="Bearish sentiment score",
                format="%.3f"
            ),
            "Volume Change %": st.column_config.NumberColumn(
                "Volume Change %",
                help="Percentage change in mention volume",
                format="%.1f%%"
            )
        }
    )

def display_financial_table(results_data):
    """Display financial analysis data in a table format."""
    if not results_data:
        st.warning("No financial analysis data available.")
        return
    
    # Convert to DataFrame for better display
    df_data = []
    for result in results_data:
        breakdown = result.get('component_breakdown', {})
        df_data.append({
            'Symbol': result.get('symbol', 'N/A'),
            'Sentiment': f"{result.get('sentiment', 0):.3f}",
            'Confidence': f"{result.get('confidence', 0):.3f}",
            'Valuation': f"{breakdown.get('valuation', 0):.3f}",
            'Growth': f"{breakdown.get('growth', 0):.3f}",
            'Profitability': f"{breakdown.get('profitability', 0):.3f}",
            'Leverage': f"{breakdown.get('leverage', 0):.3f}",
            'Cash Flow': f"{breakdown.get('cash_flow', 0):.3f}",
            'Shareholder Return': f"{breakdown.get('shareholder_return', 0):.3f}",
            'Red Flag': f"{breakdown.get('red_flag', 0):.3f}",
            'Rationale': result.get('rationale', 'N/A')[:100] + '...' if len(result.get('rationale', '')) > 100 else result.get('rationale', 'N/A')
        })
    
    df = pd.DataFrame(df_data)
    
    # Display with styling
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Sentiment": st.column_config.NumberColumn(
                "Sentiment",
                help="Overall sentiment score (-1 to +1)",
                format="%.3f"
            ),
            "Confidence": st.column_config.NumberColumn(
                "Confidence",
                help="Prediction confidence (0 to 1)",
                format="%.3f"
            )
        }
    )

if __name__ == "__main__":
    main() 