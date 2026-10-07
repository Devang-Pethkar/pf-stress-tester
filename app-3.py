"""
Risk Analysis Tool: 360 Huntington Fund
Search any ticker for a full risk report with direct Excel export
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Risk Analysis | 360 Huntington Fund",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #CC0000; }
    h3 { color: #1a1a2e; border-bottom: 2px solid #CC0000; padding-bottom: 6px; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #f5f5f5;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #CC0000 !important;
        color: white !important;
    }
    div[data-testid="metric-container"] {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #CC0000;
        padding: 12px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════
BENCHMARK  = "SPY"
SECTOR_ETF = "IGV"
RF_RATE    = 0.045
NU_RED     = "CC0000"
NU_NAVY    = "002147"

QUICK_PICKS = {
    "Tech":     ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Software": ["NOW", "CRM", "WDAY", "SNOW", "ADSK"],
    "Finance":  ["JPM", "GS", "MS", "BAC", "BLK"],
    "Health":   ["LLY", "UNH", "JNJ", "ABBV", "PFE"],
    "Industrials": ["HEICO", "TTEK", "GE", "RTX", "HON"],
}

STRESS_SCENARIOS = {
    "COVID Crash (Feb-Mar 2020)":     ("2020-02-19", "2020-03-23"),
    "2022 Rate Shock (Jan-Oct 2022)": ("2022-01-03", "2022-10-13"),
    "2018 Q4 Selloff":                ("2018-10-03", "2018-12-24"),
    "2015-16 China Slowdown":         ("2015-07-20", "2016-02-11"),
    "2023 SVB/Banking Crisis":        ("2023-03-08", "2023-03-24"),
}

RISK_LIMITS = {
    "VaR 95% (% of position)":  {"green": 3.0,  "yellow": 5.0,  "unit": "%",  "higher_is_worse": True},
    "CVaR 95% (% of position)": {"green": 5.0,  "yellow": 8.0,  "unit": "%",  "higher_is_worse": True},
    "Annualized Volatility":     {"green": 35.0, "yellow": 50.0, "unit": "%",  "higher_is_worse": True},
    "Max Drawdown":              {"green": 25.0, "yellow": 40.0, "unit": "%",  "higher_is_worse": True},
    "Beta vs SPY":               {"green": 1.5,  "yellow": 2.0,  "unit": "x",  "higher_is_worse": True},
    "Sharpe Ratio":              {"green": 0.8,  "yellow": 0.5,  "unit": "",   "higher_is_worse": False},
    "Sortino Ratio":             {"green": 1.0,  "yellow": 0.7,  "unit": "",   "higher_is_worse": False},
    "Calmar Ratio":              {"green": 0.5,  "yellow": 0.3,  "unit": "",   "higher_is_worse": False},
}


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR: TICKER SEARCH
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Ticker Search")

    ticker_input = st.text_input(
        "Enter any ticker symbol",
        value=st.session_state.get("ticker", "NOW"),
        placeholder="e.g. AAPL, MSFT, NOW, LLY",
        help="Type any US-listed stock ticker and press Enter"
    ).upper().strip()

    if st.button("▶ Run Analysis", type="primary", use_container_width=True):
        st.session_state["ticker"] = ticker_input

    st.markdown("---")
    st.markdown("**Quick Pick by Sector**")
    for sector, tickers in QUICK_PICKS.items():
        with st.expander(sector):
            cols = st.columns(len(tickers))
            for i, t in enumerate(tickers):
                if cols[i].button(t, key=f"qp_{t}", use_container_width=True):
                    st.session_state["ticker"] = t
                    st.rerun()

    st.markdown("---")
    st.markdown("**Parameters**")
    lookback_map = {"1 Year": 365, "2 Years": 730, "3 Years": 1095, "5 Years": 1825}
    lookback     = st.selectbox("Lookback Period", list(lookback_map.keys()), index=2)
    end_date     = datetime.today()
    start_date   = end_date - timedelta(days=lookback_map[lookback])

    position_value  = st.number_input("Position Value ($)", value=75_000, step=5_000, format="%d")
    confidence_lvls = st.multiselect(
        "VaR Confidence Levels", [0.90, 0.95, 0.99], default=[0.95, 0.99],
        format_func=lambda x: f"{int(x*100)}%"
    )
    mc_sims    = st.selectbox("Monte Carlo Simulations", [1_000, 5_000, 10_000], index=1)
    mc_horizon = st.selectbox("MC Horizon (Days)", [1, 5, 10, 21], index=0)

    st.markdown("---")
    st.markdown("**Peer Comparison**")
    peers = st.multiselect(
        "Peers", ["CRM", "WDAY", "SAP", "MSFT", "ADSK", "ORCL", "SNOW", "VEEV",
                  "AAPL", "GOOGL", "AMZN", "NVDA", "META"],
        default=["CRM", "WDAY", "MSFT"]
    )

    st.markdown("---")
    st.caption("360 Huntington Fund | Northeastern University")

TICKER = st.session_state.get("ticker", ticker_input)


# ═══════════════════════════════════════════════════════════════════
# VALIDATE TICKER & LOAD DATA
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers: list, start: datetime, end: datetime) -> pd.DataFrame:
    raw    = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return prices.dropna(how="all")

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

all_tickers = [TICKER, BENCHMARK, SECTOR_ETF] + peers

with st.spinner(f"Loading data for {TICKER}..."):
    data = load_market_data(all_tickers, start_date, end_date)
    info = get_company_info(TICKER)

if TICKER not in data.columns or data[TICKER].dropna().empty:
    st.error(f"❌ Could not load data for **{TICKER}**. Please check the ticker symbol and try again.")
    st.info("💡 Examples: AAPL, MSFT, NVDA, NOW, LLY, JPM, GS")
    st.stop()

company_name     = info.get("longName", TICKER)
company_sector   = info.get("sector", "N/A")
company_industry = info.get("industry", "N/A")
market_cap       = info.get("marketCap", None)
mc_str           = f"${market_cap/1e9:.1f}B" if market_cap else "N/A"

now_px  = data[TICKER].dropna()
spy_px  = data[BENCHMARK].dropna()
igv_px  = data[SECTOR_ETF].dropna() if SECTOR_ETF in data.columns else None

now_ret = now_px.pct_change().dropna()
spy_ret = spy_px.pct_change().dropna()

common = now_ret.index.intersection(spy_ret.index)
nr = now_ret[common]
sr = spy_ret[common]


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def var_parametric(ret, cl, pv):
    z = stats.norm.ppf(1 - cl)
    return -(ret.mean() + z * ret.std()) * pv

def var_historical(ret, cl, pv):
    return -np.percentile(ret, (1 - cl) * 100) * pv

def var_montecarlo(ret, cl, pv, n, h=1):
    mu, sig = ret.mean() * h, ret.std() * np.sqrt(h)
    sims = np.random.normal(mu, sig, n)
    return -np.percentile(sims, (1 - cl) * 100) * pv

def cvar(ret, cl, pv):
    thresh = np.percentile(ret, (1 - cl) * 100)
    return -ret[ret <= thresh].mean() * pv

def drawdown_series(px):
    return (px - px.expanding().max()) / px.expanding().max() * 100

def beta(s_ret, b_ret):
    return np.cov(s_ret, b_ret)[0, 1] / np.var(b_ret)

def sharpe(ret, rf=RF_RATE):
    return (ret.mean() * 252 - rf) / (ret.std() * np.sqrt(252))

def sortino(ret, rf=RF_RATE):
    dd = ret[ret < 0].std() * np.sqrt(252)
    return (ret.mean() * 252 - rf) / dd if dd else np.nan

def calmar(ret, px):
    ann  = (1 + ret.mean()) ** 252 - 1
    mdd  = abs(drawdown_series(px).min())
    return ann / mdd * 100 if mdd else np.nan

def rolling_beta(s_ret, b_ret, w=63):
    return s_ret.rolling(w).cov(b_ret) / b_ret.rolling(w).var()

def rag(val, green, yellow, higher_is_worse=True):
    if higher_is_worse:
        if val <= green:  return "🟢 GREEN"
        if val <= yellow: return "🟡 YELLOW"
        return "🔴 RED"
    else:
        if val >= green:  return "🟢 GREEN"
        if val >= yellow: return "🟡 YELLOW"
        return "🔴 RED"

_beta      = beta(nr.values, sr.values)
_sharpe    = sharpe(nr)
_sortino   = sortino(nr)
_calmar    = calmar(nr, now_px)
_ann_vol   = nr.std() * np.sqrt(252) * 100
_dd_series = drawdown_series(now_px)
_max_dd    = abs(_dd_series.min())
_ann_ret   = ((1 + nr.mean()) ** 252 - 1) * 100
_ytd_ret   = (now_px.iloc[-1] / now_px[now_px.index.year == datetime.today().year].iloc[0] - 1) * 100 \
             if len(now_px[now_px.index.year == datetime.today().year]) else np.nan
_ir        = (nr - sr).mean() * 252 / ((nr - sr).std() * np.sqrt(252))

slope, intercept, r_val, p_val, _ = stats.linregress(sr, nr)
_alpha_ann = intercept * 252 * 100


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"# 📊 {company_name} ({TICKER}): Risk Analysis Report")
h1, h2, h3, h4, h5 = st.columns(5)
h1.markdown(f"**Fund:** 360 Huntington Fund")
h2.markdown(f"**Sector:** {company_sector}")
h3.markdown(f"**Market Cap:** {mc_str}")
h4.markdown(f"**Position:** ${position_value:,.0f}")
h5.markdown(f"**As of:** {datetime.today().strftime('%B %d, %Y')}")
st.divider()


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 Overview",
    "⚠️ Market Risk & VaR",
    "📉 Drawdown",
    "🔥 Stress Testing",
    "🔗 Correlations",
    "📐 Factor Analysis",
    "🚦 Risk Dashboard",
    "💼 Portfolio Mode",
    "📊 Options & Greeks",
    "📅 Earnings & Events",
    "📰 News Feed",
    "📥 Export to Excel",
])


# ───────────────────────────────────────────────────────────────────
# TAB 0: OVERVIEW
# ───────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Company Snapshot")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Price",       f"${now_px.iloc[-1]:,.2f}")
    m2.metric("YTD Return",  f"{_ytd_ret:.1f}%", delta=f"{_ytd_ret:.1f}%")
    m3.metric("Ann. Return", f"{_ann_ret:.1f}%")
    m4.metric("Ann. Vol",    f"{_ann_vol:.1f}%")
    m5.metric("Beta",        f"{_beta:.2f}")
    m6.metric("Sharpe",      f"{_sharpe:.2f}")

    st.markdown("#### Indexed Performance (Base = 100)")
    fig = go.Figure()
    for name, px_, color, dash in [
        (TICKER, now_px, "#CC0000", "solid"),
        ("SPY",  spy_px, "#002147", "dash"),
    ]:
        norm = px_ / px_.iloc[0] * 100
        fig.add_trace(go.Scatter(x=norm.index, y=norm, name=name,
                                  line=dict(color=color, width=2, dash=dash)))
    if igv_px is not None:
        norm = igv_px / igv_px.iloc[0] * 100
        fig.add_trace(go.Scatter(x=norm.index, y=norm, name="IGV (Software)",
                                  line=dict(color="#888", width=1.5, dash="dot")))
    fig.update_layout(yaxis_title="Index Value", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08), height=380,
                      margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Daily Return Distribution")
        x = np.linspace(nr.min(), nr.max(), 200)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=nr * 100, nbinsx=60, histnorm="probability density",
                                         marker_color="#CC0000", opacity=0.65, name="Empirical"))
        fig_dist.add_trace(go.Scatter(x=x * 100, y=stats.norm.pdf(x, nr.mean(), nr.std()),
                                       name="Normal Fit", line=dict(color="#002147", width=2)))
        fig_dist.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Density",
                                legend=dict(orientation="h", y=1.1), height=300,
                                margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.markdown("#### Descriptive Statistics")
        desc = pd.DataFrame({
            "Metric": ["Mean Daily Return", "Daily Std Dev", "Annualized Return",
                        "Annualized Volatility", "Skewness", "Excess Kurtosis",
                        "Best Day", "Worst Day", "Sharpe", "Sortino"],
            TICKER: [f"{nr.mean()*100:.3f}%", f"{nr.std()*100:.3f}%",
                      f"{_ann_ret:.1f}%", f"{_ann_vol:.1f}%",
                      f"{stats.skew(nr):.3f}", f"{stats.kurtosis(nr):.3f}",
                      f"{nr.max()*100:.2f}%", f"{nr.min()*100:.2f}%",
                      f"{_sharpe:.2f}", f"{_sortino:.2f}"],
            "SPY":   [f"{sr.mean()*100:.3f}%", f"{sr.std()*100:.3f}%",
                      f"{((1+sr.mean())**252-1)*100:.1f}%", f"{sr.std()*np.sqrt(252)*100:.1f}%",
                      f"{stats.skew(sr):.3f}", f"{stats.kurtosis(sr):.3f}",
                      f"{sr.max()*100:.2f}%", f"{sr.min()*100:.2f}%",
                      f"{sharpe(sr):.2f}", f"{sortino(sr):.2f}"],
        })
        st.dataframe(desc, hide_index=True, use_container_width=True, height=330)


# ───────────────────────────────────────────────────────────────────
# TAB 1: MARKET RISK & VaR
# ───────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Value at Risk & Risk-Adjusted Metrics")

    var_rows = []
    for cl in confidence_lvls:
        vp  = var_parametric(nr, cl, position_value)
        vh  = var_historical(nr, cl, position_value)
        vmc = var_montecarlo(nr, cl, position_value, mc_sims, mc_horizon)
        cv  = cvar(nr, cl, position_value)
        var_rows.append({
            "Confidence":              f"{int(cl*100)}%",
            "Horizon":                 f"{mc_horizon}d",
            "VaR (Parametric)":        f"${vp:,.0f}",
            "VaR (Historical)":        f"${vh:,.0f}",
            "VaR (Monte Carlo)":       f"${vmc:,.0f}",
            "CVaR / Exp. Shortfall":   f"${cv:,.0f}",
            "VaR as % of Position":    f"{vp/position_value*100:.2f}%",
        })

    st.markdown(f"#### Daily VaR Summary (Position: ${position_value:,.0f} | MC: {mc_sims:,} sims)")
    st.dataframe(pd.DataFrame(var_rows), hide_index=True, use_container_width=True)
    st.caption("CVaR = average loss conditional on exceeding VaR, the tail loss on a bad day.")

    st.markdown("#### Risk-Adjusted Return Metrics")
    rm1, rm2, rm3, rm4, rm5 = st.columns(5)
    rm1.metric("Sharpe Ratio",      f"{_sharpe:.2f}")
    rm2.metric("Sortino Ratio",     f"{_sortino:.2f}")
    rm3.metric("Calmar Ratio",      f"{_calmar:.2f}")
    rm4.metric("Info. Ratio (SPY)", f"{_ir:.2f}")
    rm5.metric("Ann. Alpha (CAPM)", f"{_alpha_ann:.2f}%")

    st.markdown("#### Rolling Annualized Volatility")
    fig_vol = go.Figure()
    for window, name, color, dash in [(21,"21-Day","#CC0000","solid"),(63,"63-Day","#002147","dash"),(126,"126-Day","#888","dot")]:
        rv = nr.rolling(window).std() * np.sqrt(252) * 100
        fig_vol.add_trace(go.Scatter(x=rv.index, y=rv, name=name,
                                      line=dict(color=color, width=1.5, dash=dash)))
    fig_vol.add_hline(y=_ann_vol, line_dash="longdash", line_color="black",
                       annotation_text=f"Full-Period: {_ann_vol:.1f}%")
    fig_vol.update_layout(yaxis_title="Annualized Volatility (%)", hovermode="x unified",
                           height=350, legend=dict(orientation="h", y=1.08),
                           margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown(f"#### Monte Carlo P&L Distribution ({mc_horizon}-Day Horizon)")
    np.random.seed(42)
    mc_raw = np.random.normal(nr.mean()*mc_horizon, nr.std()*np.sqrt(mc_horizon), mc_sims) * position_value
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(x=mc_raw, nbinsx=80, marker_color="#CC0000", opacity=0.7))
    for cl in confidence_lvls:
        cut = np.percentile(mc_raw, (1-cl)*100)
        fig_mc.add_vline(x=cut, line_dash="dash", line_color="#002147",
                          annotation_text=f"VaR {int(cl*100)}%: ${abs(cut):,.0f}",
                          annotation_position="top right")
    fig_mc.update_layout(xaxis_title="P&L ($)", yaxis_title="Frequency",
                          height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_mc, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 2: DRAWDOWN
# ───────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Drawdown Analysis")

    dd_now      = drawdown_series(now_px)
    dd_spy      = drawdown_series(spy_px)
    max_dd_date = dd_now.idxmin()
    peak_date   = now_px[:max_dd_date].idxmax()
    post        = dd_now[max_dd_date:]
    rec_dates   = post[post >= 0].index
    rec_date    = rec_dates[0] if len(rec_dates) > 0 else None

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Max Drawdown", f"{dd_now.min():.1f}%")
    d2.metric("Peak Date",    peak_date.strftime("%b %d, %Y"))
    d3.metric("Trough Date",  max_dd_date.strftime("%b %d, %Y"))
    d4.metric("Recovery",     rec_date.strftime("%b %d, %Y") if rec_date else "Not yet")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_now.index, y=dd_now, fill="tozeroy", name=f"{TICKER} Drawdown",
                                  line=dict(color="#CC0000", width=1.5), fillcolor="rgba(204,0,0,0.2)"))
    fig_dd.add_trace(go.Scatter(x=dd_spy.index, y=dd_spy, name="SPY Drawdown",
                                  line=dict(color="#002147", width=1.5, dash="dash")))
    fig_dd.add_annotation(x=max_dd_date, y=dd_now.min(),
                           text=f"Max DD: {dd_now.min():.1f}%", showarrow=True,
                           arrowhead=2, arrowcolor="#CC0000", bgcolor="white", bordercolor="#CC0000")
    fig_dd.update_layout(title="Underwater Curve: Drawdown from Peak (%)",
                          yaxis_title="Drawdown (%)", hovermode="x unified",
                          legend=dict(orientation="h", y=1.08), height=400,
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("#### Monthly Return Heatmap")
    monthly    = now_ret.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
    monthly_df = monthly.to_frame("Return")
    monthly_df["Year"]  = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.strftime("%b")
    pivot = monthly_df.pivot_table(index="Year", columns="Month", values="Return")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0,"#8B0000"],[0.3,"#CC0000"],[0.5,"#ffffff"],[0.7,"#1a6632"],[1,"#0d3319"]],
        zmid=0, text=np.round(pivot.values,1), texttemplate="%{text}%",
        colorbar=dict(title="Return %"),
    ))
    fig_heat.update_layout(title="Monthly Returns (%)", height=300, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_heat, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 3: STRESS TESTING
# ───────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Stress Testing & Scenario Analysis")
    st.markdown("#### Historical Stress Scenarios")

    scen_rows = []
    for name, (s, e) in STRESS_SCENARIOS.items():
        mask_n = (now_px.index >= s) & (now_px.index <= e)
        mask_s = (spy_px.index >= s) & (spy_px.index <= e)
        if mask_n.sum() < 3: continue
        n_ret = (now_px[mask_n].iloc[-1] / now_px[mask_n].iloc[0] - 1) * 100
        s_ret = (spy_px[mask_s].iloc[-1] / spy_px[mask_s].iloc[0] - 1) * 100 if mask_s.sum() >= 3 else np.nan
        scen_rows.append({
            "Scenario": name, "Start": s, "End": e,
            f"{TICKER} (%)": round(n_ret, 1),
            "SPY (%)": round(s_ret, 1),
            "Est. P&L ($)": round(n_ret/100*position_value, 0),
            "Active Return (%)": round(n_ret - s_ret, 1),
        })

    if scen_rows:
        sdf = pd.DataFrame(scen_rows)
        display_sdf = sdf.copy()
        display_sdf[f"{TICKER} (%)"]     = display_sdf[f"{TICKER} (%)"].apply(lambda x: f"{x:.1f}%")
        display_sdf["SPY (%)"]           = display_sdf["SPY (%)"].apply(lambda x: f"{x:.1f}%")
        display_sdf["Est. P&L ($)"]      = display_sdf["Est. P&L ($)"].apply(lambda x: f"${x:,.0f}")
        display_sdf["Active Return (%)"] = display_sdf["Active Return (%)"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_sdf[["Scenario","Start","End",f"{TICKER} (%)","SPY (%)","Est. P&L ($)","Active Return (%)"]],
                     hide_index=True, use_container_width=True)

        fig_sc = go.Figure()
        ticker_vals = sdf[f"{TICKER} (%)"].tolist()
        colors = ["#CC0000" if x < 0 else "#1a6632" for x in ticker_vals]
        fig_sc.add_trace(go.Bar(x=sdf["Scenario"], y=ticker_vals, marker_color=colors,
                                 text=[f"{v:.1f}%" for v in ticker_vals], textposition="outside", name=TICKER))
        fig_sc.add_trace(go.Bar(x=sdf["Scenario"], y=sdf["SPY (%)"], marker_color="#002147", opacity=0.5,
                                 text=[f"{v:.1f}%" for v in sdf["SPY (%)"]], textposition="outside", name="SPY"))
        fig_sc.update_layout(barmode="group", yaxis_title="Return (%)", height=380,
                              legend=dict(orientation="h", y=1.08), margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("#### Custom Shock Simulator")
    col_s1, col_s2, col_s3 = st.columns(3)
    shock_now = col_s1.slider(f"{TICKER} Price Shock (%)", -70, 50, -20)
    shock_mkt = col_s2.slider("Market (SPY) Shock (%)", -50, 30, -10)
    shock_vol = col_s3.slider("Vol Spike (x)", 1.0, 4.0, 1.5)

    direct_pnl  = shock_now / 100 * position_value
    beta_pnl    = _beta * shock_mkt / 100 * position_value
    shocked_var = var_historical(nr * shock_vol, 0.95, position_value)

    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Direct P&L",        f"${direct_pnl:,.0f}")
    cs2.metric("Beta-Adj Mkt P&L",  f"${beta_pnl:,.0f}")
    cs3.metric("Total Est. P&L",    f"${direct_pnl+beta_pnl:,.0f}")
    cs4.metric("VaR (Shocked Vol)", f"${shocked_var:,.0f}")


# ───────────────────────────────────────────────────────────────────
# TAB 4: CORRELATIONS
# ───────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### Correlation Analysis")

    corr_cols = [c for c in [TICKER, BENCHMARK, SECTOR_ETF] + peers if c in data.columns]
    corr_mat  = data[corr_cols].pct_change().dropna().corr()

    fig_c = go.Figure(data=go.Heatmap(
        z=corr_mat.values, x=corr_mat.columns, y=corr_mat.columns,
        colorscale=[[0,"#003366"],[0.5,"#ffffff"],[1,"#CC0000"]],
        zmin=-1, zmax=1, text=np.round(corr_mat.values, 2), texttemplate="%{text}",
        colorbar=dict(title="Corr"),
    ))
    fig_c.update_layout(title="Return Correlation Matrix", height=420, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_c, use_container_width=True)

    st.markdown(f"#### Rolling 63-Day Correlation: {TICKER} vs SPY")
    roll_c_spy = nr.rolling(63).corr(sr)
    fig_rc = go.Figure()
    fig_rc.add_trace(go.Scatter(x=roll_c_spy.index, y=roll_c_spy, name=f"{TICKER}/SPY",
                                  fill="tozeroy", fillcolor="rgba(204,0,0,0.1)", line=dict(color="#CC0000")))
    if igv_px is not None:
        igv_ret = igv_px.pct_change().dropna()
        common2 = nr.index.intersection(igv_ret.index)
        roll_c_igv = nr[common2].rolling(63).corr(igv_ret[common2])
        fig_rc.add_trace(go.Scatter(x=roll_c_igv.index, y=roll_c_igv, name=f"{TICKER}/IGV",
                                      line=dict(color="#002147", dash="dash")))
    fig_rc.add_hline(y=roll_c_spy.mean(), line_dash="dot", line_color="gray",
                      annotation_text=f"SPY avg: {roll_c_spy.mean():.2f}")
    fig_rc.update_layout(yaxis=dict(range=[-1,1]), yaxis_title="Correlation",
                          hovermode="x unified", height=330,
                          legend=dict(orientation="h", y=1.08), margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig_rc, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 5: FACTOR ANALYSIS
# ───────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### Factor Analysis: CAPM Regression")

    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Beta",           f"{slope:.3f}")
    f2.metric("Daily Alpha",    f"{intercept*100:.4f}%")
    f3.metric("Ann. Alpha",     f"{_alpha_ann:.2f}%")
    f4.metric("R²",             f"{r_val**2:.3f}")
    f5.metric("p-value (beta)", f"{p_val:.4f}")

    col_l, col_r = st.columns(2)
    with col_l:
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=sr*100, y=nr*100, mode="markers",
                                      marker=dict(color="#CC0000", opacity=0.25, size=4), name="Daily Returns"))
        xr = np.linspace(sr.min()*100, sr.max()*100, 100)
        fig_reg.add_trace(go.Scatter(x=xr, y=slope*xr+intercept*100,
                                      name=f"Fit b={slope:.2f}", line=dict(color="#002147", width=2)))
        fig_reg.update_layout(title=f"CAPM Scatter: {TICKER} vs SPY",
                               xaxis_title="SPY (%)", yaxis_title=f"{TICKER} (%)",
                               height=370, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_r:
        rb = rolling_beta(nr, sr, 63)
        fig_rb = go.Figure()
        fig_rb.add_trace(go.Scatter(x=rb.index, y=rb, name="63-Day Rolling Beta", line=dict(color="#CC0000")))
        fig_rb.add_hline(y=_beta, line_dash="dash", line_color="#002147",
                          annotation_text=f"Full-Period b={_beta:.2f}")
        fig_rb.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="b=1.0")
        fig_rb.update_layout(title="Rolling Beta (63-Day)", yaxis_title="Beta",
                              height=370, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_rb, use_container_width=True)

    st.markdown("#### Rolling Annualized Alpha (%)")
    roll_alpha = (nr.rolling(63).mean() - slope * sr.rolling(63).mean()) * 252 * 100
    fig_ra = go.Figure()
    fig_ra.add_trace(go.Scatter(x=roll_alpha.index, y=roll_alpha, name="Rolling Alpha",
                                 fill="tozeroy", fillcolor="rgba(204,0,0,0.1)", line=dict(color="#CC0000")))
    fig_ra.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    fig_ra.add_hline(y=_alpha_ann, line_dash="dash", line_color="#002147",
                     annotation_text=f"Full-Period: {_alpha_ann:.1f}%")
    fig_ra.update_layout(yaxis_title="Alpha (%)", hovermode="x unified",
                          height=320, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig_ra, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 6: RISK DASHBOARD
# ───────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("### Risk Limits Dashboard")
    st.caption("360 Huntington Fund indicative thresholds")

    var95_pct  = var_historical(nr, 0.95, position_value) / position_value * 100
    cvar95_pct = cvar(nr, 0.95, position_value) / position_value * 100

    current_vals = {
        "VaR 95% (% of position)":  var95_pct,
        "CVaR 95% (% of position)": cvar95_pct,
        "Annualized Volatility":     _ann_vol,
        "Max Drawdown":              _max_dd,
        "Beta vs SPY":               _beta,
        "Sharpe Ratio":              _sharpe,
        "Sortino Ratio":             _sortino,
        "Calmar Ratio":              _calmar,
    }

    dash_rows = []
    for metric, limits in RISK_LIMITS.items():
        val  = current_vals[metric]
        g, y = limits["green"], limits["yellow"]
        u    = limits["unit"]
        dash_rows.append({
            "Metric":       metric,
            "Current":      f"{val:.2f}{u}",
            "Green Limit":  f"{g}{u}",
            "Yellow Limit": f"{y}{u}",
            "Status":       rag(val, g, y, limits["higher_is_worse"]),
        })

    st.dataframe(pd.DataFrame(dash_rows), hide_index=True, use_container_width=True, height=330)

    reds    = sum(1 for r in dash_rows if "RED"    in r["Status"])
    yellows = sum(1 for r in dash_rows if "YELLOW" in r["Status"])
    greens  = sum(1 for r in dash_rows if "GREEN"  in r["Status"])

    g1, g2, g3 = st.columns(3)
    g1.metric("🟢 Within Limits", str(greens))
    g2.metric("🟡 Watch",         str(yellows))
    g3.metric("🔴 Breach",        str(reds))

    overall = ("🔴 CAUTION: Limit Breach(es) Detected" if reds > 0
               else "🟡 WATCH: Monitor These Metrics" if yellows >= 2
               else "🟢 HEALTHY: All Limits Satisfied")
    st.info(f"**Overall Risk Status: {overall}**")

    st.markdown("#### Risk Profile Radar")
    radar_metrics    = ["Ann. Volatility", "Beta", "Max Drawdown", "VaR 95%", "CVaR 95%"]
    raw_vals         = [_ann_vol/60, _beta/2.5, _max_dd/50, var95_pct/8, cvar95_pct/12]
    raw_vals_clipped = [min(v, 1.0) for v in raw_vals]
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=raw_vals_clipped + [raw_vals_clipped[0]],
        theta=radar_metrics + [radar_metrics[0]],
        fill="toself", fillcolor="rgba(204,0,0,0.2)",
        line=dict(color="#CC0000", width=2)
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1],
                                    tickvals=[0.25,0.5,0.75], ticktext=["Low","Mid","High"])),
        height=380, showlegend=False, margin=dict(l=40,r=40,t=40,b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 7: PORTFOLIO MODE
# ───────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("### Portfolio Risk Analysis")
    st.caption("Compose a multi-asset portfolio, set weights, and view consolidated risk metrics.")

    port_raw = st.text_input(
        "Portfolio Tickers (comma-separated, max 10)",
        value=f"{TICKER}, MSFT, GOOGL, JPM",
        help="Enter any US-listed tickers separated by commas.",
        key="port_tickers_input",
    )
    port_tickers_raw = [t.strip().upper() for t in port_raw.split(",") if t.strip()][:10]

    if len(port_tickers_raw) < 2:
        st.warning("Enter at least 2 tickers to run portfolio analysis.")
    else:
        st.markdown("**Weights**: enter decimals (e.g. 0.25). Will be normalized automatically.")
        default_w = round(1 / len(port_tickers_raw), 2)
        w_cols = st.columns(min(len(port_tickers_raw), 5))
        raw_weights = []
        for i, t in enumerate(port_tickers_raw):
            raw_weights.append(
                w_cols[i % 5].number_input(
                    t, min_value=0.0, max_value=1.0,
                    value=default_w, step=0.05, key=f"pw_{t}_{i}"
                )
            )

        w_total = sum(raw_weights)
        if w_total == 0:
            st.error("All weights are zero.")
            st.stop()
        norm_weights = np.array(raw_weights) / w_total
        if abs(w_total - 1.0) > 0.01:
            st.caption(f"Weights summed to {w_total:.2f}, normalized to 1.00.")

        with st.spinner("Loading portfolio data..."):
            try:
                port_prices = load_market_data(port_tickers_raw, start_date, end_date)
            except Exception as e:
                st.error(f"Data load error: {e}")
                st.stop()

        valid_tickers_port = [
            t for t in port_tickers_raw
            if t in port_prices.columns and not port_prices[t].dropna().empty
        ]
        invalid_tickers = [t for t in port_tickers_raw if t not in valid_tickers_port]
        if invalid_tickers:
            st.warning(f"Could not load data for: {', '.join(invalid_tickers)}, excluded.")
        if len(valid_tickers_port) < 2:
            st.error("Need at least 2 valid tickers.")
            st.stop()

        valid_idx_map = [port_tickers_raw.index(t) for t in valid_tickers_port]
        w_arr = norm_weights[valid_idx_map]
        w_arr = w_arr / w_arr.sum()

        port_ret_df  = port_prices[valid_tickers_port].pct_change().dropna()
        port_ret_ser = (port_ret_df * w_arr).sum(axis=1)

        cov_matrix     = port_ret_df.cov()
        port_daily_vol = np.sqrt(w_arr @ cov_matrix.values @ w_arr)
        port_ann_vol   = port_daily_vol * np.sqrt(252) * 100
        port_ann_ret   = ((1 + port_ret_ser.mean()) ** 252 - 1) * 100
        port_sharpe_r  = (port_ret_ser.mean() * 252 - RF_RATE) / (port_daily_vol * np.sqrt(252))
        port_var95     = -np.percentile(port_ret_ser, 5) * position_value

        port_cum = (1 + port_ret_ser).cumprod()
        port_dd  = (port_cum / port_cum.expanding().max() - 1) * 100
        port_mdd = abs(port_dd.min())

        indiv_vols = np.array([port_ret_df[t].std() * np.sqrt(252) * 100 for t in valid_tickers_port])
        div_ratio  = (w_arr @ indiv_vols) / port_ann_vol

        pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
        pm1.metric("Ann. Return",     f"{port_ann_ret:.1f}%")
        pm2.metric("Ann. Volatility", f"{port_ann_vol:.1f}%")
        pm3.metric("Sharpe Ratio",    f"{port_sharpe_r:.2f}")
        pm4.metric("VaR 95%",         f"${port_var95:,.0f}")
        pm5.metric("Max Drawdown",    f"-{port_mdd:.1f}%")
        pm6.metric("Div. Ratio",      f"{div_ratio:.2f}",
                   help="Weighted-avg individual vol / portfolio vol. >1 = diversification benefit.")

        st.markdown("#### Indexed Performance (Base = 100)")
        port_idx_ser = port_cum / port_cum.iloc[0] * 100
        spy_aligned  = spy_px[spy_px.index.isin(port_idx_ser.index)]
        spy_norm_p   = spy_aligned / spy_aligned.iloc[0] * 100

        COLORS_PORT = ["#FF6B6B","#4ECDC4","#45B7D1","#FFA07A","#DDA0DD",
                       "#98FB98","#FFB347","#B0C4DE","#F0E68C","#E6E6FA"]

        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(x=port_idx_ser.index, y=port_idx_ser, name="Portfolio",
                                       line=dict(color="#CC0000", width=2.5)))
        fig_port.add_trace(go.Scatter(x=spy_norm_p.index, y=spy_norm_p, name="SPY",
                                       line=dict(color="#002147", width=1.8, dash="dash")))
        for i, t in enumerate(valid_tickers_port):
            px_t   = port_prices[t].dropna()
            norm_t = px_t / px_t.iloc[0] * 100
            fig_port.add_trace(go.Scatter(x=norm_t.index, y=norm_t, name=t,
                                           line=dict(width=1, dash="dot",
                                                     color=COLORS_PORT[i % len(COLORS_PORT)]),
                                           opacity=0.65))
        fig_port.update_layout(yaxis_title="Index Value", hovermode="x unified", height=390,
                                legend=dict(orientation="h", y=1.1),
                                margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_port, use_container_width=True)

        col_pie_p, col_corr_p = st.columns(2)
        with col_pie_p:
            st.markdown("#### Weight Allocation")
            fig_pie_p = go.Figure(data=go.Pie(
                labels=valid_tickers_port, values=(w_arr * 100).round(1),
                marker_colors=COLORS_PORT[:len(valid_tickers_port)],
                textinfo="label+percent", hole=0.35
            ))
            fig_pie_p.update_layout(height=320, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
            st.plotly_chart(fig_pie_p, use_container_width=True)

        with col_corr_p:
            st.markdown("#### Constituent Correlation")
            corr_port = port_ret_df.corr()
            fig_corr_p = go.Figure(data=go.Heatmap(
                z=corr_port.values, x=list(corr_port.columns), y=list(corr_port.columns),
                colorscale=[[0,"#003366"],[0.5,"#ffffff"],[1,"#CC0000"]],
                zmin=-1, zmax=1, text=np.round(corr_port.values, 2),
                texttemplate="%{text}", colorbar=dict(title="Corr")
            ))
            fig_corr_p.update_layout(height=320, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_corr_p, use_container_width=True)

        st.markdown("#### Portfolio Drawdown vs SPY")
        spy_dd_port = (spy_aligned / spy_aligned.expanding().max() - 1) * 100
        fig_pdd = go.Figure()
        fig_pdd.add_trace(go.Scatter(x=port_dd.index, y=port_dd, name="Portfolio",
                                      fill="tozeroy", fillcolor="rgba(204,0,0,0.15)",
                                      line=dict(color="#CC0000", width=1.5)))
        fig_pdd.add_trace(go.Scatter(x=spy_dd_port.index, y=spy_dd_port, name="SPY",
                                      line=dict(color="#002147", width=1.5, dash="dash")))
        fig_pdd.update_layout(yaxis_title="Drawdown (%)", hovermode="x unified", height=300,
                               legend=dict(orientation="h", y=1.08),
                               margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_pdd, use_container_width=True)

        st.markdown("#### Constituent Risk Summary")
        ind_rows_port = []
        for i, t in enumerate(valid_tickers_port):
            t_ret  = port_ret_df[t]
            t_spy  = sr[sr.index.isin(t_ret.index)].reindex(t_ret.index).dropna()
            t_ret_a = t_ret[t_ret.index.isin(t_spy.index)]
            t_beta = beta(t_ret_a.values, t_spy.values) if len(t_ret_a) > 10 else np.nan
            ind_rows_port.append({
                "Ticker":           t,
                "Weight":           f"{w_arr[i]*100:.1f}%",
                "Ann. Return":      f"{((1+t_ret.mean())**252-1)*100:.1f}%",
                "Ann. Vol":         f"{t_ret.std()*np.sqrt(252)*100:.1f}%",
                "Beta vs SPY":      f"{t_beta:.2f}" if not np.isnan(t_beta) else "N/A",
                "Sharpe":           f"{sharpe(t_ret):.2f}",
                "Contrib. Vol":     f"{(w_arr[i]*t_ret.std()*np.sqrt(252)*100):.1f}%",
                "Marginal VaR":     f"${-np.percentile(t_ret,5)*position_value*w_arr[i]:,.0f}",
            })
        st.dataframe(pd.DataFrame(ind_rows_port), hide_index=True, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 8: OPTIONS & GREEKS
# ───────────────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown("### Options Chain & Greeks")
    st.caption("Live options chain via yfinance. Greeks computed via Black-Scholes.")

    @st.cache_data(ttl=1800, show_spinner=False)
    def get_expirations(ticker: str):
        try:
            return list(yf.Ticker(ticker).options)
        except Exception:
            return []

    @st.cache_data(ttl=1800, show_spinner=False)
    def get_option_chain(ticker: str, exp: str):
        try:
            chain = yf.Ticker(ticker).option_chain(exp)
            return chain.calls, chain.puts
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    def bs_greeks(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return {k: np.nan for k in ["delta","gamma","theta","vega","rho"]}
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_delta = stats.norm.cdf(d1)
        gamma  = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta  = (-(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                   - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
        vega   = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
        rho    = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        return {
            "delta": round(call_delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega":  round(vega, 4),
            "rho":   round(rho, 4),
        }

    expirations = get_expirations(TICKER)

    if not expirations:
        st.warning(f"No options data available for **{TICKER}**.")
    else:
        opt_c1, opt_c2, opt_c3 = st.columns(3)
        selected_exp = opt_c1.selectbox("Expiration Date", expirations[:15])
        opt_type     = opt_c2.selectbox("Option Type", ["Calls", "Puts"])
        strike_range = opt_c3.slider("Strike Range (% around spot)", 5, 50, 20, 5)

        spot = now_px.iloc[-1]
        T    = max((pd.to_datetime(selected_exp) - pd.Timestamp.now()).days, 0) / 365

        calls_df, puts_df = get_option_chain(TICKER, selected_exp)
        chain_df = calls_df if opt_type == "Calls" else puts_df

        if chain_df.empty:
            st.warning("No chain data returned for this expiration.")
        else:
            lo = spot * (1 - strike_range / 100)
            hi = spot * (1 + strike_range / 100)
            chain_df = chain_df[(chain_df["strike"] >= lo) & (chain_df["strike"] <= hi)].copy()

            greeks_rows = []
            for _, row in chain_df.iterrows():
                iv = row.get("impliedVolatility", np.nan)
                if pd.isna(iv) or iv <= 0:
                    continue
                g = bs_greeks(spot, row["strike"], T, RF_RATE, iv)
                put_delta = g["delta"] - 1
                greeks_rows.append({
                    "Strike":        row["strike"],
                    "Last Price":    round(row.get("lastPrice", np.nan), 2),
                    "Bid":           round(row.get("bid", np.nan), 2),
                    "Ask":           round(row.get("ask", np.nan), 2),
                    "IV (%)":        round(iv * 100, 1),
                    "Volume":        int(row.get("volume", 0) or 0),
                    "Open Interest": int(row.get("openInterest", 0) or 0),
                    "Delta":         g["delta"] if opt_type == "Calls" else round(put_delta, 4),
                    "Gamma":         g["gamma"],
                    "Theta":         g["theta"],
                    "Vega":          g["vega"],
                })

            if not greeks_rows:
                st.info("No valid IV data: chain may have stale quotes.")
            else:
                gdf = pd.DataFrame(greeks_rows)
                atm_strike = gdf.iloc[(gdf["Strike"] - spot).abs().argsort()[:1]]["Strike"].values[0]

                st.markdown(
                    f"#### {TICKER} {opt_type} | Exp: **{selected_exp}** | "
                    f"Spot: **${spot:.2f}** | T: **{T*365:.0f}d** | ATM: **{atm_strike}**"
                )
                st.dataframe(gdf, hide_index=True, use_container_width=True, height=300)

                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.markdown("#### IV Smile")
                    fig_iv = go.Figure()
                    fig_iv.add_trace(go.Scatter(x=gdf["Strike"], y=gdf["IV (%)"],
                                                mode="lines+markers",
                                                line=dict(color="#CC0000", width=2),
                                                marker=dict(size=5)))
                    fig_iv.add_vline(x=spot, line_dash="dash", line_color="#002147",
                                     annotation_text=f"Spot ${spot:.0f}")
                    fig_iv.update_layout(xaxis_title="Strike", yaxis_title="IV (%)",
                                         height=310, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_iv, use_container_width=True)

                with g_col2:
                    st.markdown("#### Delta by Strike")
                    delta_colors = ["#CC0000" if abs(d) >= 0.5 else "#002147" for d in gdf["Delta"]]
                    fig_delta = go.Figure()
                    fig_delta.add_trace(go.Bar(x=gdf["Strike"], y=gdf["Delta"],
                                               marker_color=delta_colors))
                    fig_delta.add_vline(x=spot, line_dash="dash", line_color="gray")
                    fig_delta.update_layout(xaxis_title="Strike", yaxis_title="Delta",
                                            height=310, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_delta, use_container_width=True)

                st.markdown("#### Open Interest by Strike")
                oi_colors = ["#CC0000" if k <= spot else "#002147" for k in gdf["Strike"]]
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(x=gdf["Strike"], y=gdf["Open Interest"],
                                        marker_color=oi_colors,
                                        text=gdf["Open Interest"].apply(lambda x: f"{x:,}"),
                                        textposition="outside"))
                fig_oi.add_vline(x=spot, line_dash="dash", line_color="#888",
                                 annotation_text=f"Spot ${spot:.0f}")
                fig_oi.update_layout(xaxis_title="Strike", yaxis_title="Open Interest",
                                     height=320, margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig_oi, use_container_width=True)

                st.markdown("#### Payoff Diagram at Expiry")
                strike_options = sorted(gdf["Strike"].tolist())
                atm_default    = [min(strike_options, key=lambda x: abs(x - spot))]
                strikes_pay    = st.multiselect("Strike(s)", strike_options,
                                                default=atm_default, key="payoff_strikes")
                if strikes_pay:
                    S_range = np.linspace(spot * 0.65, spot * 1.35, 300)
                    fig_pay = go.Figure()
                    for k_pay in strikes_pay:
                        row_k   = gdf[gdf["Strike"] == k_pay].iloc[0]
                        premium = row_k["Last Price"]
                        if opt_type == "Calls":
                            payoff = np.maximum(S_range - k_pay, 0) - premium
                        else:
                            payoff = np.maximum(k_pay - S_range, 0) - premium
                        fig_pay.add_trace(go.Scatter(x=S_range, y=payoff,
                                                      name=f"K={k_pay} (prem=${premium:.2f})"))
                    fig_pay.add_hline(y=0, line_color="black", line_width=1)
                    fig_pay.add_vline(x=spot, line_dash="dash", line_color="#002147",
                                      annotation_text="Spot")
                    fig_pay.update_layout(xaxis_title="Price at Expiry ($)",
                                          yaxis_title="P&L per contract ($)",
                                          hovermode="x unified", height=360,
                                          legend=dict(orientation="h", y=1.08),
                                          margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_pay, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 9: EARNINGS & EVENTS
# ───────────────────────────────────────────────────────────────────
with tabs[9]:
    st.markdown("### Earnings & Event Risk")
    st.caption("Earnings dates, historical earnings-day returns, and upcoming event countdown.")

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_earnings_data(ticker: str):
        t = yf.Ticker(ticker)
        try:
            cal = t.calendar
        except Exception:
            cal = None
        try:
            earn_dates = t.earnings_dates
        except Exception:
            earn_dates = None
        return cal, earn_dates

    calendar_data, earnings_hist = get_earnings_data(TICKER)

    st.markdown("#### Upcoming Earnings")
    upcoming_date = None

    if calendar_data is not None:
        try:
            if isinstance(calendar_data, pd.DataFrame):
                if "Earnings Date" in calendar_data.columns:
                    upcoming_date = pd.to_datetime(calendar_data["Earnings Date"].iloc[0])
                elif "Earnings Date" in calendar_data.index:
                    upcoming_date = pd.to_datetime(calendar_data.loc["Earnings Date"].iloc[0])
            elif isinstance(calendar_data, dict):
                earn_list = calendar_data.get("Earnings Date", [])
                if earn_list:
                    upcoming_date = pd.to_datetime(earn_list[0])
        except Exception:
            upcoming_date = None

    if upcoming_date is None and earnings_hist is not None and not earnings_hist.empty:
        try:
            now_tz = pd.Timestamp.now(tz=earnings_hist.index.tz)
            future = earnings_hist[earnings_hist.index > now_tz]
        except Exception:
            future = pd.DataFrame()
        if not future.empty:
            upcoming_date = future.index[-1]

    if upcoming_date:
        # Normalize to tz-naive so comparisons with tz-naive indices don't error
        try:
            upcoming_date = upcoming_date.tz_convert(None)
        except TypeError:
            pass
        try:
            upcoming_date = upcoming_date.tz_localize(None)
        except TypeError:
            pass
        days_to_earn  = (upcoming_date - pd.Timestamp.now()).days
        expected_move = nr.std() * 100 * 1.5
        e_col1, e_col2, e_col3 = st.columns(3)
        e_col1.metric("Next Earnings Date",    upcoming_date.strftime("%b %d, %Y"))
        e_col2.metric("Days Until Earnings",   str(max(days_to_earn, 0)))
        e_col3.metric("Expected Move (±1σ×1.5)", f"±{expected_move:.1f}%",
                      help="1.5x daily std dev as a rough earnings-day volatility proxy.")
    else:
        st.info("No upcoming earnings date found via yfinance.")

    st.markdown("#### Price History with Earnings Dates Overlaid")
    fig_earn_px = go.Figure()
    fig_earn_px.add_trace(go.Scatter(x=now_px.index, y=now_px, name=TICKER,
                                      line=dict(color="#CC0000", width=2)))

    earn_dates_plotted = []
    if earnings_hist is not None and not earnings_hist.empty:
        try:
            now_tz = pd.Timestamp.now(tz=earnings_hist.index.tz)
            past_earnings = earnings_hist[earnings_hist.index <= now_tz].index
        except Exception:
            past_earnings = earnings_hist.index
        for ed in past_earnings:
            if now_px.index.min() <= ed <= now_px.index.max():
                idx_pos = now_px.index.searchsorted(ed)
                if idx_pos >= len(now_px.index):
                    continue
                nearest = now_px.index[idx_pos]
                px_e    = now_px.iloc[idx_pos]
                ret_e   = now_ret.get(nearest, np.nan) * 100 if nearest in now_ret.index else np.nan
                color_e = "#1a6632" if (not np.isnan(ret_e) and ret_e > 0) else "#CC0000"
                fig_earn_px.add_trace(go.Scatter(
                    x=[nearest], y=[px_e], mode="markers",
                    marker=dict(symbol="diamond", size=10, color=color_e,
                                line=dict(width=1, color="white")),
                    name="Earnings" if not earn_dates_plotted else "",
                    showlegend=len(earn_dates_plotted) == 0,
                    hovertemplate=(
                        f"{ed.strftime('%b %d, %Y')}<br>Price: ${px_e:.2f}<br>"
                        f"Day Return: {ret_e:.1f}%<extra></extra>"
                        if not np.isnan(ret_e)
                        else f"{ed.strftime('%b %d, %Y')}<br>Price: ${px_e:.2f}<extra></extra>"
                    )
                ))
                earn_dates_plotted.append(ed)

    if upcoming_date and now_px.index.min() < upcoming_date:
        fig_earn_px.add_vline(
            x=upcoming_date.timestamp() * 1000,
            line_dash="dash", line_color="#002147",
            annotation_text=f"Next: {upcoming_date.strftime('%b %d')}",
            annotation_position="top right"
        )

    fig_earn_px.update_layout(yaxis_title="Price ($)", hovermode="x unified", height=390,
                               legend=dict(orientation="h", y=1.08),
                               margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig_earn_px, use_container_width=True)

    st.markdown("#### Historical Earnings-Day Returns")
    if earn_dates_plotted:
        earn_ret_rows = []
        for ed in earn_dates_plotted:
            idx_pos = now_px.index.searchsorted(ed)
            if idx_pos >= len(now_px.index):
                continue
            nearest = now_px.index[idx_pos]
            ret_e   = now_ret.get(nearest, np.nan) * 100 if nearest in now_ret.index else np.nan
            if not np.isnan(ret_e):
                earn_ret_rows.append({
                    "Date":              nearest.strftime("%Y-%m-%d"),
                    "1-Day Return (%)":  round(ret_e, 2),
                    "Direction":         "📈 Up" if ret_e > 0 else "📉 Down",
                })

        if earn_ret_rows:
            edf      = pd.DataFrame(earn_ret_rows)
            avg_abs  = edf["1-Day Return (%)"].abs().mean()
            pct_up   = (edf["1-Day Return (%)"] > 0).mean() * 100
            max_up   = edf["1-Day Return (%)"].max()
            max_down = edf["1-Day Return (%)"].min()

            ea1, ea2, ea3, ea4 = st.columns(4)
            ea1.metric("Avg Absolute Move",    f"{avg_abs:.1f}%")
            ea2.metric("% Positive Reactions", f"{pct_up:.0f}%")
            ea3.metric("Best Earnings Day",    f"+{max_up:.1f}%")
            ea4.metric("Worst Earnings Day",   f"{max_down:.1f}%")

            fig_ebar = go.Figure()
            fig_ebar.add_trace(go.Bar(
                x=edf["Date"], y=edf["1-Day Return (%)"],
                marker_color=["#1a6632" if r > 0 else "#CC0000" for r in edf["1-Day Return (%)"]],
                text=[f"{r:+.1f}%" for r in edf["1-Day Return (%)"]],
                textposition="outside"
            ))
            fig_ebar.add_hline(y=0, line_color="black", line_width=1)
            fig_ebar.add_hline(y=avg_abs, line_dash="dot", line_color="#1a6632",
                               annotation_text=f"+{avg_abs:.1f}% avg")
            fig_ebar.add_hline(y=-avg_abs, line_dash="dot", line_color="#CC0000",
                               annotation_text=f"-{avg_abs:.1f}% avg")
            fig_ebar.update_layout(xaxis_title="Earnings Date", yaxis_title="1-Day Return (%)",
                                    height=340, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_ebar, use_container_width=True)
            st.dataframe(edf, hide_index=True, use_container_width=True)

            st.markdown("#### Pre vs Post-Earnings Volatility")
            vol_spikes = []
            for ed in earn_dates_plotted:
                idx_pos = now_px.index.searchsorted(ed)
                pre  = now_ret.iloc[max(0, idx_pos-5):idx_pos]
                post = now_ret.iloc[idx_pos:min(len(now_ret), idx_pos+5)]
                if len(pre) > 0 and len(post) > 0:
                    vol_spikes.append({
                        "Date":                  ed.strftime("%Y-%m-%d"),
                        "Pre-Earn Vol (ann.)":   round(pre.std()  * np.sqrt(252) * 100, 1),
                        "Post-Earn Vol (ann.)":  round(post.std() * np.sqrt(252) * 100, 1),
                    })
            if vol_spikes:
                vsdf = pd.DataFrame(vol_spikes)
                fig_vs = go.Figure()
                fig_vs.add_trace(go.Bar(x=vsdf["Date"], y=vsdf["Pre-Earn Vol (ann.)"],
                                        name="Pre-Earnings", marker_color="#002147"))
                fig_vs.add_trace(go.Bar(x=vsdf["Date"], y=vsdf["Post-Earn Vol (ann.)"],
                                        name="Post-Earnings", marker_color="#CC0000"))
                fig_vs.update_layout(barmode="group", yaxis_title="Ann. Volatility (%)",
                                      height=300, legend=dict(orientation="h", y=1.08),
                                      margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig_vs, use_container_width=True)
        else:
            st.info("Earnings dates found but no return data available in the selected window.")
    else:
        st.info(f"No historical earnings dates found for **{TICKER}** in the selected lookback window.")


# ───────────────────────────────────────────────────────────────────
# TAB 10: NEWS FEED
# ───────────────────────────────────────────────────────────────────
with tabs[10]:
    st.markdown("### Recent News")
    st.caption("Latest headlines via yfinance. Tone classification is keyword-based and indicative only.")

    @st.cache_data(ttl=1800, show_spinner=False)
    def get_ticker_news(ticker: str):
        try:
            return yf.Ticker(ticker).news or []
        except Exception:
            return []

    POSITIVE_WORDS = [
        "beat","exceed","growth","surge","record","upgrade","profit","gain",
        "strong","bullish","outperform","raise","raised","rally","buy",
        "positive","expand","expansion","wins","win","awarded","partnership",
        "deal","acquisition","breakthrough","innovative",
    ]
    NEGATIVE_WORDS = [
        "miss","missed","decline","loss","downgrade","risk","fall","drop",
        "weak","bearish","underperform","cut","concern","probe","investigation",
        "lawsuit","penalty","fine","warning","layoff","layoffs","recall",
        "delay","disappoints","disappointing","sell",
    ]

    def classify_tone(text: str):
        tl  = text.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in tl)
        neg = sum(1 for w in NEGATIVE_WORDS if w in tl)
        if pos > neg: return "🟢 Positive", "#d4edda"
        if neg > pos: return "🔴 Negative", "#f8d7da"
        return "⚪ Neutral",  "#f5f5f5"

    news_items = get_ticker_news(TICKER)

    col_news_left, col_news_right = st.columns([3, 1])

    with col_news_left:
        st.markdown(f"#### {company_name} ({TICKER}): {len(news_items)} Articles")

        if not news_items:
            st.info(f"No recent news found for {TICKER}.")
        else:
            tones = [classify_tone(item.get("title",""))[0] for item in news_items]
            n_pos = sum(1 for t in tones if "Positive" in t)
            n_neg = sum(1 for t in tones if "Negative" in t)
            n_neu = sum(1 for t in tones if "Neutral"  in t)

            nc1, nc2, nc3 = st.columns(3)
            nc1.metric("🟢 Positive", str(n_pos))
            nc2.metric("🔴 Negative", str(n_neg))
            nc3.metric("⚪ Neutral",  str(n_neu))

            net_score = n_pos - n_neg
            overall_sentiment = (
                "🟢 Net Positive Tone"  if net_score >  1 else
                "🔴 Net Negative Tone"  if net_score < -1 else
                "⚪ Mixed / Neutral Tone"
            )
            st.info(f"**Sentiment Summary:** {overall_sentiment}  (+{n_pos} / {n_neu} / -{n_neg})")

            for item in news_items:
                title     = item.get("title", "No title")
                link      = item.get("link") or item.get("url", "#")
                publisher = item.get("publisher", "Unknown Source")
                pub_ts    = item.get("providerPublishTime") or item.get("published", None)
                try:
                    date_str = pd.Timestamp(pub_ts, unit="s").strftime("%b %d, %Y  %H:%M")
                except Exception:
                    date_str = ""
                tone_label, _ = classify_tone(title)
                with st.expander(f"{tone_label}  |  {title}"):
                    st.markdown(
                        f"**Source:** {publisher}  |  **Published:** {date_str}  \n"
                        f"[Read full article →]({link})"
                    )

    with col_news_right:
        st.markdown("#### News Pulse: Peers")
        pulse_tickers = [TICKER] + peers[:4]
        pulse_rows    = []
        for pt in pulse_tickers:
            pt_news  = get_ticker_news(pt)
            pt_tones = [classify_tone(i.get("title",""))[0] for i in pt_news]
            p_pos    = sum(1 for t in pt_tones if "Positive" in t)
            p_neg    = sum(1 for t in pt_tones if "Negative" in t)
            p_neu    = sum(1 for t in pt_tones if "Neutral"  in t)
            pulse_rows.append({
                "Ticker":   pt,
                "Articles": len(pt_news),
                "🟢": p_pos, "🔴": p_neg, "⚪": p_neu,
                "Net":      p_pos - p_neg,
            })

        pulse_df = pd.DataFrame(pulse_rows)
        st.dataframe(pulse_df, hide_index=True, use_container_width=True, height=220)

        fig_pulse = go.Figure()
        fig_pulse.add_trace(go.Bar(
            x=pulse_df["Ticker"], y=pulse_df["Net"],
            marker_color=["#1a6632" if n >= 0 else "#CC0000" for n in pulse_df["Net"]],
            text=pulse_df["Net"].apply(lambda x: f"{x:+d}"),
            textposition="outside"
        ))
        fig_pulse.add_hline(y=0, line_color="black", line_width=1)
        fig_pulse.update_layout(title="Net Sentiment Score",
                                 yaxis_title="Positive − Negative",
                                 height=280, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_pulse, use_container_width=True)
        st.caption("Keyword-based: always read the full article before acting on any signal.")


# ───────────────────────────────────────────────────────────────────
# TAB 11: EXCEL EXPORT
# ───────────────────────────────────────────────────────────────────
with tabs[11]:
    st.markdown("### Export Risk Report to Excel")
    st.markdown("""
    Generates a fully formatted **.xlsx** with Northeastern red styling across 7 sheets:
    Summary, VaR/CVaR, Risk Dashboard, Stress Scenarios, Daily Returns, Correlation Matrix, Factor Regression.
    """)

    def build_excel() -> bytes:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        RED     = PatternFill("solid", fgColor=NU_RED)
        NAVY    = PatternFill("solid", fgColor=NU_NAVY)
        LGRAY   = PatternFill("solid", fgColor="F2F2F2")
        LRED    = PatternFill("solid", fgColor="FFD7D7")
        LYELLOW = PatternFill("solid", fgColor="FFF3CD")
        LGREEN  = PatternFill("solid", fgColor="D4EDDA")

        HDR_FONT   = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
        BOLD_FONT  = Font(name="Calibri", bold=True, size=10)
        STD_FONT   = Font(name="Calibri", size=10)
        TITLE_FONT = Font(name="Calibri", bold=True, size=14, color=NU_RED)
        CENTER     = Alignment(horizontal="center", vertical="center", wrap_text=True)
        LEFT       = Alignment(horizontal="left",   vertical="center")
        thin       = Side(border_style="thin", color="CCCCCC")
        BORDER     = Border(left=thin, right=thin, top=thin, bottom=thin)

        def hdr(ws, row, values, fill=RED):
            for col, val in enumerate(values, 1):
                c = ws.cell(row=row, column=col, value=val)
                c.font = HDR_FONT; c.fill = fill; c.alignment = CENTER; c.border = BORDER

        def row_data(ws, row, values, fill=None):
            for col, val in enumerate(values, 1):
                c = ws.cell(row=row, column=col, value=val)
                c.font = STD_FONT; c.alignment = LEFT; c.border = BORDER
                if fill: c.fill = fill

        def auto_width(ws):
            for col in ws.columns:
                best = max((len(str(c.value)) for c in col if c.value), default=12)
                ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(best+3,12),40)

        def title_block(ws, title, sub=""):
            ws.merge_cells("A1:G1")
            ws["A1"].value = title; ws["A1"].font = TITLE_FONT; ws["A1"].alignment = LEFT
            ws.row_dimensions[1].height = 28
            if sub:
                ws.merge_cells("A2:G2")
                ws["A2"].value = sub
                ws["A2"].font = Font(name="Calibri", italic=True, size=10, color="666666")
                ws["A2"].alignment = LEFT
            return 4 if sub else 3

        # Sheet 1: Summary
        ws1 = wb.create_sheet("Summary")
        r   = title_block(ws1, f"{company_name} ({TICKER}): Risk Analysis Report",
                          f"360 Huntington Fund | {datetime.today().strftime('%B %d, %Y')} | Position: ${position_value:,.0f}")
        hdr(ws1, r, ["Metric", "Value", "Benchmark"], fill=NAVY)
        for i, rv in enumerate([
            ("Ticker", TICKER, "SPY"), ("Company", company_name, ""),
            ("Sector", company_sector, ""), ("Market Cap", mc_str, ""),
            ("Report Date", datetime.today().strftime("%Y-%m-%d"), ""),
            ("Position Value", f"${position_value:,.0f}", ""),
            ("Current Price", f"${now_px.iloc[-1]:,.2f}", f"${spy_px.iloc[-1]:,.2f}"),
            ("YTD Return", f"{_ytd_ret:.1f}%", ""),
            ("Annualized Return", f"{_ann_ret:.1f}%", f"{((1+sr.mean())**252-1)*100:.1f}%"),
            ("Annualized Volatility", f"{_ann_vol:.1f}%", f"{sr.std()*np.sqrt(252)*100:.1f}%"),
            ("Max Drawdown", f"{_dd_series.min():.1f}%", f"{drawdown_series(spy_px).min():.1f}%"),
            ("Beta vs SPY", f"{_beta:.3f}", "1.000"),
            ("Alpha (Ann.)", f"{_alpha_ann:.2f}%", "0.00%"),
            ("R²", f"{r_val**2:.3f}", ""),
            ("Sharpe Ratio", f"{_sharpe:.3f}", f"{sharpe(sr):.3f}"),
            ("Sortino Ratio", f"{_sortino:.3f}", f"{sortino(sr):.3f}"),
            ("Calmar Ratio", f"{_calmar:.3f}", ""),
            ("Skewness", f"{stats.skew(nr):.3f}", f"{stats.skew(sr):.3f}"),
            ("Excess Kurtosis", f"{stats.kurtosis(nr):.3f}", f"{stats.kurtosis(sr):.3f}"),
        ]):
            row_data(ws1, r+1+i, list(rv), fill=LGRAY if i%2==0 else None)
        auto_width(ws1)

        # Sheet 2: VaR
        ws2 = wb.create_sheet("VaR_CVaR")
        r2  = title_block(ws2, "Value at Risk & Expected Shortfall",
                          f"Position: ${position_value:,.0f} | MC: {mc_sims:,} sims | Horizon: {mc_horizon}d")
        hdr(ws2, r2, ["Confidence","Horizon","VaR Parametric ($)","VaR Historical ($)",
                       "VaR Monte Carlo ($)","CVaR / ES ($)","VaR % of Position"])
        for i, rv in enumerate(var_rows):
            row_data(ws2, r2+1+i, [rv["Confidence"],rv["Horizon"],rv["VaR (Parametric)"],
                                    rv["VaR (Historical)"],rv["VaR (Monte Carlo)"],
                                    rv["CVaR / Exp. Shortfall"],rv["VaR as % of Position"]],
                     fill=LGRAY if i%2==0 else None)
        auto_width(ws2)

        # Sheet 3: Risk Dashboard
        ws3 = wb.create_sheet("Risk_Dashboard")
        r3  = title_block(ws3, "Risk Limits Dashboard")
        hdr(ws3, r3, ["Metric","Current Value","Green Threshold","Yellow Threshold","Status"])
        for i, rv in enumerate(dash_rows):
            s = rv["Status"]
            fill = LGREEN if "GREEN" in s else (LYELLOW if "YELLOW" in s else LRED)
            row_data(ws3, r3+1+i, [rv["Metric"],rv["Current"],rv["Green Limit"],rv["Yellow Limit"],s], fill=fill)
        auto_width(ws3)

        # Sheet 4: Stress Scenarios
        if scen_rows:
            ws4 = wb.create_sheet("Stress_Scenarios")
            r4  = title_block(ws4, "Historical Stress Scenarios")
            hdr(ws4, r4, ["Scenario","Start","End",f"{TICKER} Return (%)","SPY Return (%)","Est. P&L ($)","Active Return (%)"])
            for i, rv in enumerate(scen_rows):
                fill = LRED if rv[f"{TICKER} (%)"] < -15 else (LYELLOW if rv[f"{TICKER} (%)"] < 0 else LGREEN)
                row_data(ws4, r4+1+i,
                         [rv["Scenario"],rv["Start"],rv["End"],
                          f"{rv[f'{TICKER} (%)']:.1f}%", f"{rv['SPY (%)']:.1f}%",
                          f"${rv['Est. P&L ($)']:,.0f}", f"{rv['Active Return (%)']:.1f}%"], fill=fill)
            auto_width(ws4)

        # Sheet 5: Daily Returns
        ws5 = wb.create_sheet("Daily_Returns")
        r5  = title_block(ws5, "Daily Return Series")
        hdr(ws5, r5, ["Date",f"{TICKER} Daily Return","SPY Daily Return",f"{TICKER} Cumulative","SPY Cumulative"])
        ret_df = pd.DataFrame({TICKER: nr, "SPY": sr}).dropna()
        ret_df[f"{TICKER}_CUM"] = (1+ret_df[TICKER]).cumprod()-1
        ret_df["SPY_CUM"]       = (1+ret_df["SPY"]).cumprod()-1
        for i, (date, rv) in enumerate(ret_df.iterrows()):
            row_data(ws5, r5+1+i,
                     [date.strftime("%Y-%m-%d"), f"{rv[TICKER]*100:.4f}%", f"{rv['SPY']*100:.4f}%",
                      f"{rv[f'{TICKER}_CUM']*100:.2f}%", f"{rv['SPY_CUM']*100:.2f}%"],
                     fill=LGRAY if i%2==0 else None)
        auto_width(ws5)

        # Sheet 6: Correlation
        ws6 = wb.create_sheet("Correlation_Matrix")
        r6  = title_block(ws6, f"Correlation Matrix: {TICKER} & Peers")
        cols_list = list(corr_mat.columns)
        hdr(ws6, r6, [""]+cols_list)
        for i, idx in enumerate(corr_mat.index):
            label = ws6.cell(row=r6+1+i, column=1, value=idx)
            label.font = BOLD_FONT; label.fill = LGRAY; label.alignment = CENTER; label.border = BORDER
            for j, col_ in enumerate(corr_mat.columns):
                val = corr_mat.loc[idx, col_]
                c   = ws6.cell(row=r6+1+i, column=j+2, value=round(val,4))
                c.font = STD_FONT; c.alignment = CENTER; c.border = BORDER
                if idx == col_: c.fill = NAVY; c.font = Font(name="Calibri",bold=True,color="FFFFFF",size=10)
                elif val > 0.7: c.fill = LRED
                elif val < 0.3: c.fill = LGREEN
        auto_width(ws6)

        # Sheet 7: Factor Regression
        ws7 = wb.create_sheet("Factor_Regression")
        r7  = title_block(ws7, f"CAPM Regression: {TICKER} vs SPY")
        hdr(ws7, r7, ["Parameter","Value","Interpretation"])
        for i, rv in enumerate([
            ("Beta", f"{slope:.4f}", f"Every 1% SPY move = {slope:.2f}% {TICKER} move"),
            ("Alpha (Daily)", f"{intercept*100:.4f}%", "Excess daily return vs CAPM"),
            ("Alpha (Ann.)", f"{_alpha_ann:.2f}%", "Annualized excess return"),
            ("R²", f"{r_val**2:.4f}", f"{r_val**2*100:.1f}% of variance explained by SPY"),
            ("p-value (Beta)", f"{p_val:.6f}", "Statistical significance"),
            ("Observations", str(len(nr)), f"Trading days ({lookback})"),
            ("Risk-Free Rate", f"{RF_RATE*100:.1f}%", "10-yr proxy"),
        ]):
            row_data(ws7, r7+1+i, list(rv), fill=LGRAY if i%2==0 else None)
        auto_width(ws7)

        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔨 Build Excel Report", type="primary", use_container_width=True):
            with st.spinner("Building workbook..."):
                xlsx_bytes = build_excel()
            fname = f"{TICKER}_Risk_Report_{datetime.today().strftime('%Y%m%d')}.xlsx"
            st.download_button(
                label=f"⬇️ Download {fname}",
                data=xlsx_bytes, file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.success(f"Ready: {fname}")
    with col2:
        st.markdown("""
        | Sheet | Contents |
        |---|---|
        | Summary | Key metrics + company info |
        | VaR_CVaR | All VaR methods + CVaR |
        | Risk_Dashboard | Traffic-light table |
        | Stress_Scenarios | 5 historical crises |
        | Daily_Returns | Full return series |
        | Correlation_Matrix | Peer correlations |
        | Factor_Regression | CAPM outputs |
        """)
