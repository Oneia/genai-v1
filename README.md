# 📈 Trading LLM Analysis App

A comprehensive Streamlit application that combines **social sentiment analysis** from Reddit with **fundamental financial analysis** to provide trading insights and recommendations.

## 🚀 Features

- **Multi-Input Methods**: Text input, stock list, or sector analysis
- **Reddit Sentiment Analysis**: Real-time sentiment from popular investing subreddits
- **Financial Fundamentals**: Valuation, growth, profitability, and risk metrics
- **Combined Analysis**: AI-powered synthesis of social and financial data
- **Interactive UI**: Beautiful, responsive Streamlit interface
- **Position Sizing**: Automated trading position recommendations

## 🛠️ Installation

1. **Clone or download the project files**

   ```bash
   # Make sure you have the following files in your directory:
   # - streamlit_app.py
   # - reddit.py
   # - requirements.txt
   # - .env (create this)
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in your project directory with:

   ```env
   # Reddit API credentials (required)
   REDDIT_APP_ID=your_reddit_app_id
   REDDIT_APP_KEY=your_reddit_app_secret

   # Optional: OpenAI API key (if you want to use GPT models)
   OPENAI_API_KEY=your_openai_api_key

   # Optional: Google API key (if you want to use Gemini models)
   GOOGLE_API_KEY=your_google_api_key
   ```

4. **Get Reddit API credentials**
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Choose "script" as the app type
   - Note down the client ID and client secret

## 🎯 Usage

### Running the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Input Methods

#### 1. **Text Input** (Free-form)

Enter any combination of stocks and sectors:

- `NVDA AMD semiconductors`
- `ai sector`
- `cars in europe sector`
- `TSLA AAPL MSFT`

#### 2. **Stock List**

Enter stock symbols directly:

```
NVDA
AMD
TSLA
AAPL
MSFT
```

#### 3. **Sector Analysis**

Focus on specific industries:

- `semiconductors`
- `healthcare`
- `banking`
- `ai`

### Configuration Options

- **Subreddits**: Select which Reddit communities to analyze for sentiment
- **Input Method**: Choose your preferred way to input analysis requests
- **Analysis Depth**: The app automatically determines the scope based on your input

## 📊 Analysis Components

### Reddit Sentiment Analysis

- **Mentions**: Number of posts/comments mentioning each ticker
- **Average Sentiment**: Overall sentiment score (-1 to +1)
- **Bullish/Bearish Sentiment**: Separate scores for positive/negative mentions
- **Volume Change**: Percentage change in mention volume
- **Summary**: AI-generated summary of social chatter

### Financial Analysis

- **Valuation**: PE ratios, price-to-book, etc.
- **Growth**: Revenue and earnings growth metrics
- **Profitability**: Margins, ROE, ROA
- **Leverage**: Debt ratios and financial health
- **Cash Flow**: Operating and free cash flow analysis
- **Shareholder Return**: Dividends and buybacks
- **Red Flags**: Risk indicators and warnings

### Combined Analysis

- **Trading Verdict**: Long/Short/Flat recommendations
- **Position Sizing**: Percentage allocation suggestions
- **Bull/Bear Cases**: Key arguments for and against
- **Risk Assessment**: Potential risks and concerns

## 🔧 Technical Details

### Architecture

- **Frontend**: Streamlit web interface
- **AI Models**: Google Gemini (default) or OpenAI GPT
- **Data Sources**:
  - Reddit API for social sentiment
  - Yahoo Finance for financial data
  - DuckDuckGo for additional research
- **Analysis Pipeline**: Multi-agent system with specialized roles

### Agents

1. **Reddit Analysis Agent**: Processes social media sentiment
2. **Value/Risk Agent**: Analyzes financial fundamentals
3. **Synthesizer Agent**: Combines insights into trading recommendations

### Caching

- Reddit data cached for 1 hour
- Agent initialization cached for session
- Results cached to improve performance

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Progress Indicators**: Real-time analysis progress
- **Tabbed Results**: Organized view of different analysis types
- **Interactive Tables**: Sortable and filterable data
- **Color-coded Sentiment**: Visual indicators for positive/negative sentiment
- **Expandable Help**: Built-in usage instructions

## 💡 Example Use Cases

### 1. **Sector Rotation Analysis**

```
Input: "technology sector"
Result: Analysis of major tech stocks with sentiment and fundamentals
```

### 2. **Individual Stock Research**

```
Input: "NVDA AMD"
Result: Detailed comparison of semiconductor competitors
```

### 3. **Market Sentiment Check**

```
Input: "meme stocks"
Result: Sentiment analysis of popular retail trading favorites
```

### 4. **Geographic Focus**

```
Input: "european banks"
Result: Analysis of European banking sector stocks
```

## ⚠️ Important Notes

### API Limits

- **Reddit API**: Rate limited, app handles gracefully
- **Yahoo Finance**: Free tier, may have delays
- **AI Models**: Subject to API rate limits and costs

### Data Accuracy

- Social sentiment is real-time but may not reflect fundamental value
- Financial data is from public sources and may have delays
- Always do your own research before making investment decisions

### Risk Disclaimer

This app is for educational and research purposes only. It does not constitute financial advice. Always consult with qualified financial professionals before making investment decisions.

## 🐛 Troubleshooting

### Common Issues

1. **Reddit API Errors**

   - Check your Reddit API credentials in `.env`
   - Ensure your Reddit app is configured as "script" type
   - Verify your Reddit account has sufficient karma

2. **Missing Financial Data**

   - Some stocks may not have complete financial data
   - International stocks may have limited data availability
   - Try different stock symbols or sectors

3. **AI Model Errors**

   - Check your API keys in `.env`
   - Ensure you have sufficient API credits
   - Try switching between different AI models

4. **Streamlit Issues**
   - Update Streamlit: `pip install --upgrade streamlit`
   - Clear cache: `streamlit cache clear`
   - Check Python version compatibility

### Getting Help

1. Check the error messages in the app
2. Verify your environment variables
3. Test with simple inputs first
4. Check the browser console for JavaScript errors

## 🔄 Updates and Maintenance

### Regular Updates

- Keep dependencies updated: `pip install -r requirements.txt --upgrade`
- Monitor API rate limits and costs
- Check for new Reddit API changes

### Customization

- Modify subreddit list in the app
- Adjust sentiment thresholds
- Add new financial metrics
- Customize AI agent instructions

## 📈 Future Enhancements

- **Real-time Alerts**: Price and sentiment alerts
- **Portfolio Tracking**: Track your positions and performance
- **Backtesting**: Historical analysis of recommendations
- **More Data Sources**: Twitter, news, earnings calls
- **Advanced Analytics**: Technical indicators, correlation analysis
- **Export Features**: PDF reports, CSV data export

---

**Happy Trading! 📈💰**
