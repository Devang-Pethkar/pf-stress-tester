# Risk Analysis Tool 

A Streamlit app that produces a full risk report for any US-listed ticker, with direct Excel export.

**Live app:** _add your Streamlit link here after deploying_

## Tabs

Overview, Market Risk and VaR (parametric, historical, and Monte Carlo), Drawdown, Stress Testing, Correlations, Factor Analysis, Risk Dashboard, Portfolio Mode, Options and Greeks (Black-Scholes), Earnings and Events, and News Feed.

The Excel export builds a multi-sheet workbook for committee use (openpyxl).

## Run locally (optional)

```
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

Educational project, not investment advice. Market data comes from Yahoo Finance via yfinance.
