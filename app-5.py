"""
Portfolio Stress Testing Tool
=============================
Interactive Streamlit version of the PF_Stresstester notebook.

What it does
------------
1. Downloads price history for a portfolio of tickers (yfinance).
2. Computes per-stock statistics (return, volatility, Sharpe, drawdown, beta).
3. Applies six historical crisis scenarios using a beta-adjusted shock plus a
   stressed volatility, and simulates price paths with Monte Carlo.
4. Rolls the stock paths up into portfolio-level outcomes and charts them.

Changes versus the original notebook (all intentional)
------------------------------------------------------
* The notebook's hard-coded tickers and weights are now sidebar inputs.
* Duplicate setup cells are gone (the setup runs once).
* The number of simulations is a slider, and the comment/value mismatch
  (the old comment said 100 while the code used 1,000) no longer exists.
* The history chart title and the dates shown now match the data actually used,
  instead of saying "Trailing 1 Year" while plotting five years.
* Simulated daily returns are floored at -100%, so a price can never go below
  zero. A note shows how many simulated daily moves had to be floored.
* A fixed random seed makes results repeatable (Streamlit re-runs the script on
  every click, so an unseeded simulation would change numbers constantly).
* Prices now come from Tiingo first (free API key stored in Streamlit Secrets) and
  fall back to Yahoo Finance. Yahoo blocks many requests from shared cloud servers,
  which made a Yahoo-only version fail after deployment.
* NEW, optional: "Correlated shocks" draws all stocks from one joint distribution
  that matches their historical correlations. It is OFF by default so the
  results match the original notebook. See the Methodology tab for why it matters.

Everything is in this one file so it can be pasted straight into a GitHub repo.
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

TRADING_DAYS = 252          # trading days per year, used to annualize
MAX_TICKERS = 25            # keeps the simulation fast on free hosting
MIN_OBSERVATIONS = 60       # minimum days of overlapping history we require
DEFAULT_TICKERS = "ANET, TSEM, V, LLY, GE, JPM, META, MSFT, AMZN, NVDA"

# Lookback choices shown in the sidebar (label -> number of years of history)
LOOKBACKS = {"1 year": 1, "2 years": 2, "3 years": 3, "5 years": 5}

# Colors (same dark palette as the notebook)
RED = "#FF4560"
ORANGE = "#FF8C00"
YELLOW = "#FEB019"
GREEN = "#00E396"
MUTED = "#64748b"
GRID = "#1e293b"
PAPER = "#0a0e17"
PLOT_BG = "#111827"
TEXT = "#e2e8f0"
STOCK_COLORS = [
    "#FF4560", "#FF8C00", "#FEB019", "#008FFB", "#775DD0",
    "#00E396", "#ef5350", "#42A5F5", "#AB47BC", "#26a69a",
]

# The six scenarios, identical to the notebook.
#   equity_shock : total market move over the scenario (before beta scaling)
#   vol_mult     : multiplier applied to each stock's normal volatility
#   days         : length of the scenario in trading days
SCENARIOS = {
    "2008 Financial Crisis": {"equity_shock": -0.55, "vol_mult": 3.5, "days": 60},
    "COVID Crash (2020)": {"equity_shock": -0.34, "vol_mult": 4.0, "days": 25},
    "Dot-Com Bust (2000)": {"equity_shock": -0.49, "vol_mult": 2.0, "days": 90},
    "Fed Rate Shock +300bps": {"equity_shock": -0.20, "vol_mult": 2.0, "days": 45},
    "Stagflation": {"equity_shock": -0.25, "vol_mult": 1.8, "days": 60},
    "Black Swan (-3 sigma)": {"equity_shock": -0.40, "vol_mult": 5.0, "days": 10},
}

# Plain-English notes shown in the Methodology tab
SCENARIO_NOTES = {
    "2008 Financial Crisis": "S&P fell about 55%, volatility spiked, played out over roughly 3 months",
    "COVID Crash (2020)": "Sharp 34% drop with extreme volatility (VIX hit 82), over about 1 month",
    "Dot-Com Bust (2000)": "49% decline with a moderate volatility increase, a slow bleed over about 4.5 months",
    "Fed Rate Shock +300bps": "20% equity decline from aggressive Fed tightening, over about 2 months",
    "Stagflation": "25% decline from an inflation plus stagnation combination, over about 3 months",
    "Black Swan (-3 sigma)": "Sudden 40% crash with extreme volatility, over about 2 weeks",
}


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def parse_tickers(raw):
    """
    Turn the text box contents into a clean list of tickers.
    Accepts commas, spaces, or semicolons as separators, upper-cases everything,
    removes duplicates (keeping order), and reports anything that does not look
    like a ticker so we can warn the user instead of silently dropping it.
    """
    tokens = re.split(r"[,\s;]+", raw.upper().strip())
    valid, invalid, seen = [], [], set()
    for token in tokens:
        if not token:
            continue
        if re.fullmatch(r"[A-Z0-9.\-\^=]{1,12}", token):
            if token not in seen:
                seen.add(token)
                valid.append(token)
        else:
            invalid.append(token)
    return valid, invalid


def get_tiingo_key():
    """
    Read the Tiingo API key from Streamlit Secrets (never from the code, because
    this repo is public). Returns None if no key has been set up.
    """
    try:
        return st.secrets["TIINGO_API_KEY"]
    except Exception:
        return None


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def fetch_tiingo_series(ticker, years, _api_key):
    """
    Download one ticker's adjusted daily closes from Tiingo. Cached for 12 hours
    per ticker, so the free plan's request limits are not used up by repeat visits.
    (The leading underscore on _api_key tells Streamlit not to include it in the cache key.)

    Returns None if Tiingo does not know the ticker. RAISES on rate limits or
    other errors, because Streamlit does not cache exceptions, so a temporary
    problem is retried on the next click instead of being remembered.
    """
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    resp = requests.get(
        f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices",
        params={"startDate": start, "token": _api_key},
        headers={"Content-Type": "application/json"},
        timeout=20,
    )
    if resp.status_code == 404:
        return None                                   # unknown ticker
    if resp.status_code == 401:
        raise RuntimeError("Tiingo rejected the API key (check the Secrets setting).")
    if resp.status_code == 429:
        raise RuntimeError("Tiingo request limit reached (free plan). Try again later.")
    resp.raise_for_status()

    rows = resp.json()
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Dates arrive like "2026-10-06T00:00:00.000Z"; keep just the calendar date.
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"], utc=True)).tz_localize(None).normalize()
    return pd.Series(df["adjClose"].to_numpy(dtype=float), index=dates, name=ticker)


def load_prices_tiingo(tickers, years, api_key):
    """Build a price table (one column per ticker) from Tiingo, ticker by ticker."""
    series = []
    for ticker in tickers:
        s = fetch_tiingo_series(ticker, years, api_key)
        if s is not None:
            series.append(s)
    if not series:
        raise ValueError("Tiingo returned no data for these tickers.")
    return pd.concat(series, axis=1)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_prices_yahoo(tickers, years):
    """
    Backup source: download adjusted closing prices from Yahoo Finance.
    Cached for 6 hours. Like the Tiingo loader, it RAISES when nothing comes back
    so that temporary failures are not cached.
    """
    raw = yf.download(list(tickers), period=f"{years}y", auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise ValueError("No price data was returned by Yahoo Finance.")

    close = raw["Close"]
    # Older yfinance versions return a Series when only one ticker is requested.
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close = close.dropna(axis=1, how="all")          # drop tickers with no data at all
    if close.shape[1] == 0:
        raise ValueError("None of the tickers returned price data.")
    return close


def load_prices(tickers, years):
    """
    Try Tiingo first (if a key is configured), then Yahoo Finance.
    Returns (price table, name of the source that worked).
    """
    problems = []
    key = get_tiingo_key()
    if key:
        try:
            return load_prices_tiingo(tickers, years, key), "Tiingo"
        except Exception as exc:
            problems.append(f"Tiingo: {exc}")
    else:
        problems.append("Tiingo: no API key configured")

    try:
        return load_prices_yahoo(tickers, years), "Yahoo Finance"
    except Exception as exc:
        problems.append(f"Yahoo Finance: {exc}")

    raise ValueError(" | ".join(problems))


# ═══════════════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ═══════════════════════════════════════════════════════════════════

def compute_stats(prices, returns):
    """
    Per-stock summary table (same definitions as the notebook).

    Beta here is measured against the equal-weighted average of the stocks in
    the portfolio, not against the S&P 500. With a single stock the beta is 1.
    """
    basket = returns.mean(axis=1)                       # equal-weighted proxy for "the market"
    ann_return = returns.mean() * TRADING_DAYS          # average daily return x 252
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)     # daily std x sqrt(252)

    stats = pd.DataFrame({
        "Last Price": prices.iloc[-1],
        "Ann. Return": ann_return,
        "Ann. Vol": ann_vol,
        "Sharpe": ann_return / ann_vol,                 # risk-free rate assumed to be 0
        "Max Drawdown": (prices / prices.cummax() - 1).min(),
        "Beta": returns.apply(lambda s: s.cov(basket)) / basket.var(),
    })
    return stats.replace([np.inf, -np.inf], np.nan)


def safe_cholesky(corr):
    """
    Cholesky factor of a correlation matrix. If the matrix is numerically not
    positive definite, add a tiny amount to the diagonal and retry. Falls back
    to independent draws (identity matrix) only as a last resort.
    """
    n = corr.shape[0]
    jitter = 0.0
    for _ in range(10):
        try:
            return np.linalg.cholesky(corr + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0 else jitter * 10
    return np.eye(n)


@st.cache_data(show_spinner=False)
def run_stress(stats, corr, weights, n_sims, seed, correlated):
    """
    Monte Carlo stress test for every scenario. Cached, so changing a chart
    selector does not re-run the simulation.

    For each scenario and stock:
        shock          = equity_shock x beta
        stressed_vol   = normal annual vol x vol_mult
        daily drift    = shock / days             (total shock spread evenly)
        daily vol      = stressed_vol / sqrt(252)
        daily return   = drift + daily vol x z    (z ~ standard normal)
        price path     = price x cumulative product of (1 + daily return)

    Fix: daily returns are floored at -100% so a price can fall to zero but
    never below it (the original code could create negative prices at very high
    stressed volatility).

    Returns a dict keyed by scenario name. Only compact results are stored
    (percentile bands, 50 sample paths), not every simulated path.
    """
    rng = np.random.default_rng(seed)               # fixed seed = repeatable results
    tickers = list(stats.index)
    n = len(tickers)
    w = np.asarray(weights, dtype=float)

    beta = stats["Beta"].to_numpy(dtype=float)
    vol = stats["Ann. Vol"].to_numpy(dtype=float)
    price = stats["Last Price"].to_numpy(dtype=float)

    chol = safe_cholesky(np.asarray(corr, dtype=float)) if correlated else None

    out = {}
    for name, p in SCENARIOS.items():
        days = p["days"]

        shock = p["equity_shock"] * beta                    # beta-adjusted total shock, one per stock
        stressed_vol = vol * p["vol_mult"]                  # inflated annual vol, one per stock
        daily_drift = shock / days
        daily_vol = stressed_vol / np.sqrt(TRADING_DAYS)

        # Random draws, shape (simulations, days, stocks)
        z = rng.standard_normal((n_sims, days, n))
        if correlated:
            # Multiplying by the Cholesky factor turns independent draws into
            # draws whose correlations match the historical correlation matrix.
            z = z @ chol.T

        daily_ret = daily_drift + daily_vol * z

        # Fix: floor at -100% (a total loss). Count how often that happens.
        clipped = int((daily_ret < -1.0).sum())
        daily_ret = np.maximum(daily_ret, -1.0)

        # Growth factor of each stock relative to its starting price (day 0 = 1.0)
        growth = np.cumprod(1.0 + daily_ret, axis=1)
        growth = np.concatenate([np.ones((n_sims, 1, n)), growth], axis=1)   # (sims, days+1, stocks)

        # Portfolio value indexed to 1.0 (weights applied to relative growth,
        # so a $200 stock and a $50 stock contribute by weight, not by price)
        port = growth @ w                                   # (sims, days+1)

        final_price = price * growth[:, -1, :]
        port_final = port[:, -1]
        running_max = np.maximum.accumulate(port, axis=1)
        drawdowns = port / running_max - 1.0

        out[name] = {
            "days": days,
            "shock": shock,
            "stressed_vol": stressed_vol,
            "median_final": np.median(final_price, axis=0),
            "worst_5pct": np.percentile(final_price, 5, axis=0),
            "stock_median_path": np.median(growth, axis=0) * 100,      # (days+1, stocks), start = 100
            "port_p5": np.percentile(port, 5, axis=0) * 100,
            "port_median": np.median(port, axis=0) * 100,
            "port_p95": np.percentile(port, 95, axis=0) * 100,
            "sample_paths": port[:50] * 100,
            "portfolio_loss": float(np.mean(port_final) - 1.0),        # average ending value minus 1
            "portfolio_median": float(np.median(port_final) - 1.0),
            "portfolio_p5": float(np.percentile(port_final, 5) - 1.0),
            "portfolio_max_dd": float(np.mean(np.min(drawdowns, axis=1))),
            "clipped": clipped,
            "total_draws": int(daily_ret.size),
        }
    return out


def build_summary_table(results):
    """Scenario summary table (strings, formatted for display)."""
    rows = {}
    for name, r in results.items():
        rows[name] = {
            "Days": r["days"],
            "Portfolio Loss (mean)": f"{r['portfolio_loss']:.1%}",
            "Portfolio Loss (median)": f"{r['portfolio_median']:.1%}",
            "Worst 5% Outcome": f"{r['portfolio_p5']:.1%}",
            "Avg. Max Drawdown": f"{r['portfolio_max_dd']:.1%}",
        }
    return pd.DataFrame(rows).T


def build_detail_table(results, stats):
    """One row per scenario and stock."""
    rows = []
    for name, r in results.items():
        for i, ticker in enumerate(stats.index):
            last = stats.loc[ticker, "Last Price"]
            rows.append({
                "Scenario": name,
                "Ticker": ticker,
                "Beta-Adj Shock": f"{r['shock'][i]:.1%}",
                "Stressed Vol": f"{r['stressed_vol'][i]:.1%}",
                "Median Final": f"${r['median_final'][i]:.2f}",
                "Median Return": f"{r['median_final'][i] / last - 1:.1%}",
                "Worst 5%": f"${r['worst_5pct'][i]:.2f}",
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════

def style_fig(fig, height=420, title=None):
    """Apply the shared dark theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT, size=12),
        height=height,
        margin=dict(l=10, r=10, t=60 if title else 30, b=10),
        title=dict(text=title, font=dict(size=16)) if title else None,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def spaghetti_xy(paths):
    """
    Pack many paths into ONE line trace by putting a gap (NaN) between paths.
    One trace with 50 paths is far faster to draw than 50 separate traces.
    """
    k, m = paths.shape
    x = np.tile(np.append(np.arange(m, dtype=float), np.nan), k)
    y = np.hstack([paths, np.full((k, 1), np.nan)]).ravel()
    return x, y


def fig_scenario_impact(results):
    """Horizontal bars: expected portfolio change in each scenario."""
    names = list(results.keys())
    losses = [results[n]["portfolio_loss"] * 100 for n in names]
    # Red for severe (worse than -30%), orange for bad (worse than -20%), yellow otherwise
    colors = [RED if v < -30 else ORANGE if v < -20 else YELLOW for v in losses]

    fig = go.Figure(go.Bar(
        x=losses, y=names, orientation="h", marker_color=colors,
        text=[f"{v:.1f}%" for v in losses],
        textposition="inside", insidetextanchor="end",
        textfont=dict(color="white", size=13),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Portfolio change (%)")
    return style_fig(fig, height=380, title="Portfolio Impact by Stress Scenario")


def fig_path_grid(results):
    """2 x 3 grid: Monte Carlo portfolio paths for every scenario."""
    names = list(results.keys())
    fig = make_subplots(rows=2, cols=3, subplot_titles=names,
                        horizontal_spacing=0.06, vertical_spacing=0.14)

    for idx, name in enumerate(names):
        r = results[name]
        row, col = idx // 3 + 1, idx % 3 + 1
        days_axis = np.arange(r["days"] + 1)
        first = idx == 0    # show the legend only once

        # Faint sample paths (the "spaghetti")
        x, y = spaghetti_xy(r["sample_paths"])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                 line=dict(color="rgba(255,69,96,0.14)", width=1),
                                 showlegend=False, hoverinfo="skip"), row=row, col=col)

        # 5th to 95th percentile band (upper edge first, lower edge fills up to it)
        fig.add_trace(go.Scatter(x=days_axis, y=r["port_p95"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"),
                      row=row, col=col)
        fig.add_trace(go.Scatter(x=days_axis, y=r["port_p5"], mode="lines",
                                 line=dict(color=ORANGE, width=1, dash="dash"),
                                 fill="tonexty", fillcolor="rgba(255,69,96,0.15)",
                                 name="5th percentile", legendgroup="p5", showlegend=first),
                      row=row, col=col)

        # Median path
        fig.add_trace(go.Scatter(x=days_axis, y=r["port_median"], mode="lines",
                                 line=dict(color=RED, width=2),
                                 name="Median", legendgroup="median", showlegend=first),
                      row=row, col=col)

        fig.add_hline(y=100, line_dash="dot", line_color=MUTED, line_width=1, row=row, col=col)

    fig.update_xaxes(title_text="Days", row=2)
    fig.update_yaxes(title_text="Portfolio value (start = 100)", col=1)
    fig.update_layout(legend=dict(orientation="h", y=-0.12))
    return style_fig(fig, height=640, title="Monte Carlo Portfolio Paths Under Stress")


def fig_drilldown(results, scenario, tickers):
    """Left: beta-adjusted shock per stock. Right: median simulated price path per stock."""
    r = results[scenario]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.1,
                        subplot_titles=(f"Beta-adjusted shock: {scenario}",
                                        f"Median simulated price paths: {scenario}"))

    colors = [STOCK_COLORS[i % len(STOCK_COLORS)] for i in range(len(tickers))]
    shocks = [s * 100 for s in r["shock"]]
    fig.add_trace(go.Bar(x=shocks, y=tickers, orientation="h", marker_color=colors,
                         text=[f"{s:.1f}%" for s in shocks], textposition="inside",
                         insidetextanchor="end", textfont=dict(color="white"),
                         showlegend=False), row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="Assumed loss (%)", row=1, col=1)

    days_axis = np.arange(r["days"] + 1)
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(x=days_axis, y=r["stock_median_path"][:, i], mode="lines",
                                 line=dict(color=colors[i], width=2), name=ticker),
                      row=1, col=2)
    fig.add_hline(y=100, line_dash="dot", line_color=MUTED, line_width=1, row=1, col=2)
    fig.update_xaxes(title_text="Days", row=1, col=2)
    fig.update_yaxes(title_text="Indexed price (start = 100)", row=1, col=2)
    return style_fig(fig, height=460)


def fig_history(weighted_returns):
    """Top: growth of 100 invested (green above 100, red below). Bottom: drawdown."""
    cumulative = (1 + weighted_returns).cumprod() * 100
    drawdown = (cumulative / cumulative.cummax() - 1) * 100
    baseline = pd.Series(100.0, index=cumulative.index)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)

    # Shaded areas: draw a flat baseline at 100, then fill up to the clipped series.
    fig.add_trace(go.Scatter(x=cumulative.index, y=baseline, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.clip(lower=100), mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor="rgba(0,227,150,0.2)",
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cumulative.index, y=baseline, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.clip(upper=100), mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor="rgba(255,69,96,0.2)",
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative, mode="lines",
                             line=dict(color=GREEN, width=2), name="Portfolio"), row=1, col=1)

    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines",
                             line=dict(color=RED, width=1), fill="tozeroy",
                             fillcolor="rgba(255,69,96,0.3)", name="Drawdown"), row=2, col=1)

    fig.update_yaxes(title_text="Value (start = 100)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_layout(showlegend=False)
    return style_fig(fig, height=560), cumulative, drawdown


def fig_correlation(corr_df):
    """Heatmap of daily return correlations with the value printed in each cell."""
    values = corr_df.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=values, x=list(corr_df.columns), y=list(corr_df.index),
        zmin=-1, zmax=1, colorscale="RdYlGn",
        text=np.round(values, 2), texttemplate="%{text}",
        textfont=dict(color="black", size=12),
        colorbar=dict(title="Corr"),
    ))
    fig.update_yaxes(autorange="reversed")
    height = max(380, 60 * len(corr_df) + 120)
    return style_fig(fig, height=height, title="Return Correlation Matrix")


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def sidebar_inputs():
    """Draw the sidebar and return everything the user chose."""
    with st.sidebar:
        st.header("Portfolio")
        raw = st.text_input(
            "Tickers",
            value=DEFAULT_TICKERS,
            help=f"Separate with commas or spaces. Up to {MAX_TICKERS} tickers.",
        )
        tickers, invalid = parse_tickers(raw)
        if invalid:
            st.warning("Ignored (not valid tickers): " + ", ".join(invalid))
        if len(tickers) > MAX_TICKERS:
            st.warning(f"Using the first {MAX_TICKERS} tickers.")
            tickers = tickers[:MAX_TICKERS]

        lookback_label = st.selectbox("History used for statistics", list(LOOKBACKS.keys()), index=3)

        weight_mode = st.radio("Weights", ["Equal weight", "Custom"], horizontal=True)
        weights_pct = None
        if tickers:
            if weight_mode == "Equal weight":
                weights_pct = pd.Series(100.0 / len(tickers), index=tickers)
            else:
                base = pd.DataFrame({
                    "Ticker": tickers,
                    "Weight (%)": [round(100.0 / len(tickers), 2)] * len(tickers),
                })
                # The key includes the tickers so the table resets when they change.
                edited = st.data_editor(
                    base,
                    key="weights_" + "_".join(tickers),
                    hide_index=True,
                    disabled=["Ticker"],
                    column_config={"Weight (%)": st.column_config.NumberColumn(
                        min_value=0.0, max_value=100.0, step=0.5, format="%.2f")},
                )
                weights_pct = pd.Series(edited["Weight (%)"].to_numpy(dtype=float), index=tickers)
                st.caption(f"Weights total {weights_pct.sum():.1f}% (they are rescaled to 100% automatically).")

        st.header("Simulation")
        n_sims = st.slider("Simulations per scenario", 500, 5000, 1000, step=500)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1,
                               help="Same seed gives the same results every time.")
        correlated = st.checkbox(
            "Correlated shocks",
            value=False,
            help="Off: each stock is simulated independently (matches the original notebook). "
                 "On: stocks move together according to their historical correlations, which "
                 "gives a more realistic, usually larger, portfolio loss. See the Methodology tab.",
        )

        st.divider()
        st.caption("Built by Devang Pethkar. Educational tool, not investment advice.")

    return tickers, lookback_label, weights_pct, int(n_sims), int(seed), bool(correlated)


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Portfolio Stress Tester", page_icon="📉", layout="wide")

    st.title("Portfolio Stress Testing Tool")
    st.caption("Monte Carlo stress tests across six historical crisis scenarios, "
               "with beta-adjusted shocks and stressed volatility.")

    tickers, lookback_label, weights_pct, n_sims, seed, correlated = sidebar_inputs()

    if not tickers:
        st.info("Enter at least one ticker in the sidebar to begin.")
        st.stop()

    # ── 1. Load prices ──
    try:
        with st.spinner("Downloading price history..."):
            prices, source = load_prices(tuple(tickers), LOOKBACKS[lookback_label])
    except Exception as exc:
        st.error("Could not load price data. The data providers may be temporarily limiting "
                 "requests, or a ticker may be mistyped. Please check the tickers and try again in a few minutes.")
        st.caption(f"Details: {exc}")
        st.stop()

    # Tickers that returned nothing are dropped, and the weights are re-scaled.
    st.caption(f"Price data source: {source}")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning("No data found for: " + ", ".join(missing) + ". They were removed from the portfolio.")
    kept = [t for t in tickers if t in prices.columns]

    # Keep only days where every remaining ticker has a price. A recently listed
    # stock therefore shortens the window for the whole portfolio.
    prices = prices[kept].dropna()
    returns = prices.pct_change().dropna()    # (today's price - yesterday's price) / yesterday's price
    if len(returns) < MIN_OBSERVATIONS:
        st.error(f"Only {len(returns)} days of overlapping history were found. "
                 f"At least {MIN_OBSERVATIONS} are needed. Try fewer tickers or a longer history window.")
        st.stop()

    # ── 2. Weights ──
    raw_w = weights_pct.reindex(kept).to_numpy(dtype=float)
    if raw_w.sum() <= 0:
        st.error("All weights are zero. Enter at least one positive weight.")
        st.stop()
    weights = raw_w / raw_w.sum()

    # ── 3. Statistics and simulation ──
    stats = compute_stats(prices, returns)
    if stats["Beta"].isna().any() or stats["Ann. Vol"].isna().any():
        st.error("Could not compute statistics for one of the tickers (not enough price variation). "
                 "Try removing it.")
        st.stop()

    corr_df = returns.corr()
    with st.spinner("Running Monte Carlo simulations..."):
        results = run_stress(stats, corr_df.to_numpy(), tuple(weights), n_sims, seed, correlated)

    # ── 4. Headline numbers ──
    worst_name = min(results, key=lambda n: results[n]["portfolio_loss"])
    avg_loss = float(np.mean([r["portfolio_loss"] for r in results.values()]))
    start, end = returns.index[0].date(), returns.index[-1].date()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Worst scenario", worst_name, f"{results[worst_name]['portfolio_loss']:.1%}",
              delta_color="off")
    k2.metric("Average across scenarios", f"{avg_loss:.1%}")
    k3.metric("Stocks in portfolio", len(kept))
    k4.metric("History used", f"{len(returns)} days", f"{start} to {end}", delta_color="off")

    total_clipped = sum(r["clipped"] for r in results.values())
    if total_clipped:
        total_draws = sum(r["total_draws"] for r in results.values())
        st.info(f"{total_clipped:,} of {total_draws:,} simulated daily moves "
                f"would have taken a price below zero and were floored at a 100% loss.")

    # ── 5. Tabs ──
    tab_over, tab_paths, tab_drill, tab_hist, tab_corr, tab_detail, tab_method = st.tabs([
        "Overview", "Monte Carlo Paths", "Scenario Drill-Down",
        "Historical Performance", "Correlations", "Detail Table", "Methodology",
    ])

    with tab_over:
        st.plotly_chart(fig_scenario_impact(results))
        st.subheader("Scenario summary")
        st.dataframe(build_summary_table(results))
        st.caption("Portfolio Loss (mean) is the average ending portfolio value across all simulations minus 1. "
                   "Worst 5% Outcome is the 5th percentile ending value, a VaR-style tail measure.")

        st.subheader("Stock summary")
        shown = stats.copy()
        shown["Last Price"] = shown["Last Price"].map("${:,.2f}".format)
        for col in ["Ann. Return", "Ann. Vol", "Max Drawdown"]:
            shown[col] = shown[col].map("{:.1%}".format)
        shown["Sharpe"] = shown["Sharpe"].map("{:.2f}".format)
        shown["Beta"] = shown["Beta"].map("{:.2f}".format)
        st.dataframe(shown)
        st.caption("Sharpe assumes a 0% risk-free rate. Beta is measured against the equal-weighted "
                   "average of the stocks in this portfolio, not the S&P 500.")

    with tab_paths:
        st.plotly_chart(fig_path_grid(results))
        st.caption("Faint lines are 50 sample paths. The solid line is the median and the shaded band "
                   "spans the 5th to 95th percentile of all simulations.")

    with tab_drill:
        scenario = st.selectbox("Scenario", list(results.keys()))
        st.plotly_chart(fig_drilldown(results, scenario, kept))
        st.caption("The left chart shows the assumed beta-adjusted shock (an input to the simulation). "
                   "The right chart shows the median simulated outcome, which also reflects the added volatility.")

    with tab_hist:
        weighted_returns = (returns * weights).sum(axis=1)     # daily portfolio return
        hist_fig, cumulative, drawdown = fig_history(weighted_returns)
        total_return = cumulative.iloc[-1] / 100 - 1
        st.subheader(f"Portfolio performance, {start} to {end}")
        h1, h2 = st.columns(2)
        h1.metric("Total return over the window", f"{total_return:+.1%}")
        h2.metric("Worst drawdown", f"{drawdown.min():.1f}%")
        st.plotly_chart(hist_fig)
        st.caption("Assumes the portfolio is rebalanced back to the target weights every day. "
                   "Past performance does not predict future results.")

    with tab_corr:
        st.plotly_chart(fig_correlation(corr_df))
        st.caption("Values near +1 mean two stocks tend to move together. High correlations mean less "
                   "diversification, which matters most in a crisis.")

    with tab_detail:
        detail = build_detail_table(results, stats)
        choice = st.selectbox("Filter by scenario", ["All scenarios"] + list(results.keys()), key="detail_filter")
        if choice != "All scenarios":
            detail = detail[detail["Scenario"] == choice]
        st.dataframe(detail, hide_index=True)

    with tab_method:
        render_methodology(correlated)


def render_methodology(correlated):
    """Plain-English explanation of the model and its limits."""
    st.markdown("""
### How the stress test works

1. **Statistics.** For every stock, the app measures annualized return, volatility, Sharpe ratio,
   maximum drawdown, and beta from the selected history window.
2. **Scenario shock.** Each scenario has a market-wide move (for example -55% for 2008). A stock's
   shock is that move multiplied by its beta, so a stock with beta 1.2 falls 1.2 times as much.
3. **Stressed volatility.** Each stock's normal volatility is multiplied by a scenario factor
   (for example 3.5x for 2008) to reflect how violent crises are.
4. **Simulation.** The total shock is spread evenly across the scenario's days as a daily drift.
   Each day adds a random move scaled by the stressed daily volatility (a discrete version of
   geometric Brownian motion). Thousands of paths are simulated per scenario.
5. **Portfolio roll-up.** Each stock's path is indexed to 1.0 and combined using the portfolio weights.

### Scenario assumptions
""")
    table = pd.DataFrame([
        {
            "Scenario": name,
            "Market shock": f"{p['equity_shock']:.0%}",
            "Volatility multiplier": f"{p['vol_mult']}x",
            "Days": p["days"],
            "Context": SCENARIO_NOTES[name],
        }
        for name, p in SCENARIOS.items()
    ])
    st.dataframe(table, hide_index=True)

    st.markdown("""
### Known limitations (worth knowing before reading the numbers)

* **Independent vs. correlated shocks.** By default each stock gets its own independent random
  draws, as in the original notebook. Real stocks move together, and even more so in a crisis,
  so independent draws spread the risk out and tend to **understate** the portfolio's loss spread.
  Turn on **Correlated shocks** in the sidebar to use the historical correlation matrix instead.
* **Beta proxy.** Beta is measured against the equal-weighted average of your own stocks, not the
  S&P 500, so it describes each stock relative to this portfolio.
* **Simplified scenarios.** The shock sizes, volatility multipliers, and durations are rounded
  approximations of each historical episode, not a replay of actual prices.
* **Normal returns.** Daily shocks are normally distributed, which understates extreme moves.
* **Weights.** The simulation holds the starting weights (no rebalancing). The historical
  performance chart assumes daily rebalancing to the target weights.
* **Price floor.** Daily returns are floored at -100%, so a price can reach zero but never go negative.
* **Data.** Adjusted daily closing prices come from Tiingo (with Yahoo Finance as a backup) and can be delayed or occasionally missing.

This is an educational tool and not investment advice.
""")
    if correlated:
        st.success("Correlated shocks are currently ON.")
    else:
        st.info("Correlated shocks are currently OFF (matches the original notebook).")


# Streamlit runs this file as a script, so __name__ is "__main__" when deployed.
if __name__ == "__main__":
    main()
