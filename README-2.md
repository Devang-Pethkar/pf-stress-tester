# Portfolio Stress Testing Tool

An interactive Streamlit app that stress tests a stock portfolio against six historical crisis scenarios using Monte Carlo simulation.

**Live app:** _add your Streamlit link here after deploying_

## What it does

- Pulls price history with yfinance for any portfolio (up to 25 tickers, equal or custom weights)
- Computes annualized return, volatility, Sharpe ratio, maximum drawdown, and beta for each stock
- Applies six scenarios (2008 Financial Crisis, COVID Crash, Dot-Com Bust, Fed Rate Shock, Stagflation, Black Swan) with beta-adjusted shocks and stressed volatility
- Simulates thousands of price paths per scenario and rolls them up into portfolio-level outcomes (mean loss, median, 5th percentile, average max drawdown)
- Shows historical performance, drawdown, and a correlation heatmap
- Optional correlated shocks, so stocks crash together the way they do in real crises

## Method in short

For each stock: `shock = market shock x beta`, `stressed vol = normal vol x multiplier`, then daily returns are `drift + vol x random normal`, compounded into price paths (discrete geometric Brownian motion). Daily returns are floored at -100% so prices never go negative. See the Methodology tab in the app for assumptions and limitations.

## Run locally (optional)

```
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

Educational project, not investment advice.
