"""
Algorithmic Trading Backtester — Streamlit Dashboard
Portfolio-grade backtesting engine with three systematic strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "JPM"]
BENCHMARK = "SPY"
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"
RISK_FREE_RATE = 0.045
TX_COST = 0.001  # 0.1 % per trade (entry or exit)
TRADING_DAYS = 252

STRESS_PERIODS: Dict[str, Tuple[str, str]] = {
    "COVID Crash": ("2020-02-19", "2020-03-23"),
    "2022 Rate Hike Cycle": ("2022-01-01", "2022-12-31"),
    "SVB Crisis": ("2023-03-08", "2023-03-31"),
}

STRATEGY_NAMES = {
    "ma_crossover": "Moving Average Crossover",
    "momentum": "Momentum (12-1 Month)",
    "rsi_mean_reversion": "RSI Mean Reversion",
}

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def download_price_data() -> Dict[str, pd.Series]:
    """Download adjusted close prices for all tickers + benchmark."""
    all_tickers = list(dict.fromkeys(TICKERS + [BENCHMARK]))
    raw = yf.download(
        all_tickers,
        start=START_DATE,
        end="2024-01-01",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    prices = prices.dropna(how="all").sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.loc[START_DATE:END_DATE]

    result: Dict[str, pd.Series] = {}
    for ticker in all_tickers:
        if ticker in prices.columns:
            result[ticker] = prices[ticker].dropna()
    return result


# ---------------------------------------------------------------------------
# Indicators & signals
# ---------------------------------------------------------------------------


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ma_crossover_signals(prices: pd.Series) -> pd.Series:
    sma50 = prices.rolling(50).mean()
    sma200 = prices.rolling(200).mean()
    return (sma50 > sma200).astype(int)


def momentum_signals(prices: pd.Series) -> pd.Series:
    """Long when 12-minus-1 month return is positive."""
    mom = prices.shift(21) / prices.shift(252) - 1
    return (mom > 0).astype(int).fillna(0)


def rsi_mean_reversion_signals(prices: pd.Series) -> pd.Series:
    rsi = compute_rsi(prices, 14)
    position = pd.Series(0, index=prices.index, dtype=int)
    in_trade = False
    for i, dt in enumerate(prices.index):
        if not in_trade and rsi.iloc[i] < 30:
            in_trade = True
        elif in_trade and rsi.iloc[i] > 70:
            in_trade = False
        position.iloc[i] = 1 if in_trade else 0
    return position


SIGNAL_GENERATORS = {
    "ma_crossover": ma_crossover_signals,
    "momentum": momentum_signals,
    "rsi_mean_reversion": rsi_mean_reversion_signals,
}


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float


@dataclass
class BacktestResult:
    strategy_key: str
    ticker: str
    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]


def run_backtest(
    prices: pd.Series,
    strategy_key: str,
    tx_cost: float = TX_COST,
) -> BacktestResult:
    signals = SIGNAL_GENERATORS[strategy_key](prices)
    aligned = pd.DataFrame({"price": prices, "position": signals}).dropna()
    aligned["position"] = aligned["position"].astype(int)

    if len(aligned) < 2:
        empty = pd.Series(dtype=float)
        return BacktestResult(
            strategy_key=strategy_key,
            ticker=prices.name or "ASSET",
            equity_curve=empty,
            daily_returns=empty,
            trades=[],
            metrics=empty_metrics(),
        )

    daily_rets: List[float] = []
    trades: List[Trade] = []
    entry_date: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    prev_pos = 0

    for i, (dt, row) in enumerate(aligned.iterrows()):
        price = float(row["price"])
        pos = int(row["position"])
        day_ret = 0.0

        if i > 0:
            prev_price = float(aligned.iloc[i - 1]["price"])
            if prev_pos == 1:
                day_ret = price / prev_price - 1

        if prev_pos == 0 and pos == 1:
            day_ret -= tx_cost
            entry_date = dt
            entry_price = price
        elif prev_pos == 1 and pos == 0:
            day_ret -= tx_cost
            if entry_date is not None and entry_price is not None:
                gross = price / entry_price - 1
                net = (1 + gross) * (1 - tx_cost) ** 2 - 1
                trades.append(
                    Trade(
                        entry_date=entry_date,
                        exit_date=dt,
                        entry_price=entry_price,
                        exit_price=price,
                        return_pct=net * 100,
                    )
                )
            entry_date = None
            entry_price = None

        daily_rets.append(day_ret)
        prev_pos = pos

    if prev_pos == 1 and entry_date is not None and entry_price is not None:
        last_dt = aligned.index[-1]
        last_price = float(aligned.iloc[-1]["price"])
        gross = last_price / entry_price - 1
        net = (1 + gross) * (1 - tx_cost) - 1
        trades.append(
            Trade(
                entry_date=entry_date,
                exit_date=last_dt,
                entry_price=entry_price,
                exit_price=last_price,
                return_pct=net * 100,
            )
        )

    ret_series = pd.Series(daily_rets, index=aligned.index, name="returns")
    eq_series = (1 + ret_series).cumprod()
    eq_series.name = "equity"

    metrics = compute_metrics(eq_series, ret_series, trades)
    return BacktestResult(
        strategy_key=strategy_key,
        ticker=prices.name or "ASSET",
        equity_curve=eq_series,
        daily_returns=ret_series,
        trades=trades,
        metrics=metrics,
    )


def buy_and_hold(prices: pd.Series) -> BacktestResult:
    aligned = prices.dropna()
    if len(aligned) < 2:
        empty = pd.Series(dtype=float)
        return BacktestResult(
            strategy_key="buy_hold",
            ticker=prices.name or BENCHMARK,
            equity_curve=empty,
            daily_returns=empty,
            trades=[],
            metrics=empty_metrics(),
        )

    daily_ret = aligned.pct_change().fillna(0)
    equity = (1 + daily_ret).cumprod()
    trades: List[Trade] = []
    if len(aligned) >= 2:
        total_ret = aligned.iloc[-1] / aligned.iloc[0] - 1
        trades.append(
            Trade(
                entry_date=aligned.index[0],
                exit_date=aligned.index[-1],
                entry_price=float(aligned.iloc[0]),
                exit_price=float(aligned.iloc[-1]),
                return_pct=total_ret * 100,
            )
        )
    metrics = compute_metrics(equity, daily_ret, trades)
    return BacktestResult(
        strategy_key="buy_hold",
        ticker=prices.name or BENCHMARK,
        equity_curve=equity,
        daily_returns=daily_ret,
        trades=trades,
        metrics=metrics,
    )


def compute_metrics(
    equity: pd.Series,
    daily_returns: pd.Series,
    trades: List[Trade],
) -> Dict[str, float]:
    if len(equity) < 2:
        return empty_metrics(trades)

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    rf_daily = RISK_FREE_RATE / TRADING_DAYS
    excess = daily_returns - rf_daily
    std = daily_returns.std()
    down_std = daily_returns[daily_returns < 0].std()

    # Sharpe = annualized excess return / annualized volatility
    sharpe = (excess.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    sortino = (excess.mean() / down_std * np.sqrt(TRADING_DAYS)) if down_std > 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    closed = [t for t in trades if t.entry_date != t.exit_date or len(trades) == 1]
    wins = sum(1 for t in closed if t.return_pct > 0)
    win_rate = (wins / len(closed) * 100) if closed else 0.0
    n_trades = len(closed)

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "num_trades": n_trades,
    }


def period_return(equity: pd.Series, start: str, end: str) -> float:
    mask = (equity.index >= start) & (equity.index <= end)
    sub = equity.loc[mask]
    if len(sub) < 2:
        return 0.0
    return (sub.iloc[-1] / sub.iloc[0] - 1) * 100


def has_min_rows(series: pd.Series, min_rows: int = 2) -> bool:
    return series is not None and len(series) >= min_rows


def empty_metrics(trades: Optional[List[Trade]] = None) -> Dict[str, float]:
    closed = trades or []
    wins = sum(1 for t in closed if t.return_pct > 0)
    return {
        "total_return": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "win_rate": (wins / len(closed) * 100) if closed else 0.0,
        "num_trades": len(closed),
    }


# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------


def build_empty_figure(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        height=480,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="#9ca3af"),
            )
        ],
    )
    return fig


def build_equity_chart(
    strategy_eq: pd.Series,
    benchmark_eq: pd.Series,
    strategy_name: str,
    ticker: str,
) -> go.Figure:
    if not has_min_rows(strategy_eq) or not has_min_rows(benchmark_eq):
        return build_empty_figure("No data available")

    strat_pct = (strategy_eq / strategy_eq.iloc[0] - 1) * 100
    bench_pct = (benchmark_eq / benchmark_eq.iloc[0] - 1) * 100

    fig = go.Figure()

    stress_colors = {
        "COVID Crash": "rgba(239, 68, 68, 0.15)",
        "2022 Rate Hike Cycle": "rgba(234, 179, 8, 0.12)",
        "SVB Crisis": "rgba(168, 85, 247, 0.15)",
    }
    for label, (s, e) in STRESS_PERIODS.items():
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor=stress_colors.get(label, "rgba(128,128,128,0.1)"),
            layer="below",
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="#9ca3af",
        )

    fig.add_trace(
        go.Scatter(
            x=strat_pct.index,
            y=strat_pct.values,
            name=f"{strategy_name} ({ticker})",
            line=dict(color="#3b82f6", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bench_pct.index,
            y=bench_pct.values,
            name=f"{BENCHMARK} Buy & Hold",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        height=480,
        margin=dict(l=40, r=20, t=50, b=40),
        title=dict(
            text=f"Cumulative Return — {strategy_name} vs {BENCHMARK} Benchmark",
            font=dict(size=16),
        ),
        xaxis=dict(title="Date", gridcolor="#30363d"),
        yaxis=dict(title="Cumulative Return (%)", gridcolor="#30363d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; }
        .section-spacer { margin: 2.25rem 0; }
        .hero-subtitle {
            color: #8b949e;
            font-size: 1rem;
            margin: -0.5rem 0 2rem 0;
            letter-spacing: 0.02em;
        }
        .metric-card {
            background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
            border: 2.5px solid #484f58;
            border-radius: 12px;
            padding: 20px 22px;
            margin-bottom: 8px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
            min-height: 148px;
        }
        .metric-card .card-title {
            color: #9ca3af;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
            font-weight: 600;
        }
        .metric-card .strategy-value {
            color: #f0f6fc;
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 14px;
        }
        .metric-card .bench-row,
        .metric-card .diff-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
            font-size: 0.88rem;
        }
        .metric-card .bench-row { color: #8b949e; }
        .metric-card .diff-row {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #30363d;
            font-weight: 600;
        }
        .metric-card .diff-label { color: #9ca3af; }
        .metric-card .val-pos { color: #3fb950; }
        .metric-card .val-neg { color: #f85149; }
        .metric-card .val-neutral { color: #f0f6fc; }
        .stress-card {
            background: #161b22;
            border: 2px solid #484f58;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 12px;
        }
        .stress-card h4 { color: #f0f6fc; margin: 0 0 12px 0; font-size: 1rem; }
        .stress-card .metric-row { display: flex; justify-content: space-between; margin: 8px 0; }
        .stress-card .label { color: #9ca3af; font-size: 0.85rem; }
        .stress-card .value-pos { color: #3fb950; font-weight: 600; }
        .stress-card .value-neg { color: #f85149; font-weight: 600; }
        .stress-card .value-neutral { color: #f0f6fc; font-weight: 600; }
        .footer-note {
            text-align: center;
            color: #6e7681;
            font-size: 0.78rem;
            padding: 28px 0 12px 0;
            border-top: 1px solid #30363d;
            margin-top: 3rem;
        }
        h1 { color: #f0f6fc !important; margin-bottom: 0.25rem !important; }
        h2, h3 { color: #c9d1d9 !important; margin-top: 1.5rem !important; }
        [data-testid="stHorizontalBlock"] { gap: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_pct(val: float) -> str:
    return f"{val:+.2f}%"


def fmt_num(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}"


def value_class(diff: float, higher_is_better: bool = True) -> str:
    if abs(diff) < 1e-9:
        return "val-neutral"
    beats = diff > 0 if higher_is_better else diff < 0
    return "val-pos" if beats else "val-neg"


def render_comparison_card(
    title: str,
    strategy_val: str,
    bench_val: str,
    diff: float,
    diff_label: str,
    higher_is_better: bool = True,
) -> None:
    diff_cls = value_class(diff, higher_is_better)
    diff_fmt = f"{diff:+.2f}" if "%" not in diff_label else f"{diff:+.2f}%"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="card-title">{title}</div>
            <div class="strategy-value">{strategy_val}</div>
            <div class="bench-row">
                <span>{BENCHMARK} Benchmark</span>
                <span>{bench_val}</span>
            </div>
            <div class="diff-row">
                <span class="diff-label">{diff_label}</span>
                <span class="{diff_cls}">{diff_fmt}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_break() -> None:
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)


def render_stress_card(
    title: str,
    strat_ret: float,
    bench_ret: float,
) -> None:
    def cls(v: float) -> str:
        if v > 0:
            return "value-pos"
        if v < 0:
            return "value-neg"
        return "value-neutral"

    st.markdown(
        f"""
        <div class="stress-card">
            <h4>{title}</h4>
            <div class="metric-row">
                <span class="label">Strategy Return</span>
                <span class="{cls(strat_ret)}">{fmt_pct(strat_ret)}</span>
            </div>
            <div class="metric-row">
                <span class="label">{BENCHMARK} Benchmark</span>
                <span class="{cls(bench_ret)}">{fmt_pct(bench_ret)}</span>
            </div>
            <div class="metric-row">
                <span class="label">Alpha</span>
                <span class="{cls(strat_ret - bench_ret)}">{fmt_pct(strat_ret - bench_ret)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Algo Trading Backtester",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Sidebar
    with st.sidebar:
        st.title("📈 Algo Backtester")
        st.markdown("---")
        st.markdown(
            """
            ### Project Overview
            A professional-grade **algorithmic trading backtester** built for
            quantitative finance portfolio demonstrations. Evaluates systematic
            equity strategies against a **SPY buy-and-hold benchmark** using
            real historical data from Yahoo Finance.

            ### Methodology
            - **Universe:** SPY, QQQ, AAPL, MSFT, JPM
            - **Period:** January 2019 – December 2023
            - **Transaction costs:** 0.1% per trade (slippage + commissions)
            - **Risk-free rate:** 4.5% (Sharpe & Sortino calculations)
            - **Benchmark:** SPY buy-and-hold (always)

            ### Strategies
            1. **MA Crossover** — 50/200-day SMA golden/death cross
            2. **Momentum** — 12-minus-1 month return signal
            3. **RSI Mean Reversion** — Buy RSI < 30, sell RSI > 70

            ### Stress Tests
            Performance isolated during:
            - COVID crash (Feb–Mar 2020)
            - 2022 rate hike cycle
            - SVB banking crisis (Mar 2023)
            """
        )
        st.markdown("---")
        st.caption(f"Last updated: {date.today().isoformat()}")

    st.title("Algorithmic Trading Backtester")
    st.markdown(
        '<p class="hero-subtitle">Built by Matyas Szabo · MBA Business Analytics · '
        "Florida International University</p>",
        unsafe_allow_html=True,
    )
    st.caption("Systematic strategy evaluation · Real market data · Institutional-grade metrics")

    with st.spinner("Downloading historical price data from Yahoo Finance…"):
        try:
            price_data = download_price_data()
        except Exception as exc:
            st.error(f"Failed to download data: {exc}")
            st.stop()

    benchmark_prices = price_data[BENCHMARK]
    benchmark_prices.name = BENCHMARK
    benchmark_result = buy_and_hold(benchmark_prices)
    bench_metrics = benchmark_result.metrics

    # Pre-compute all strategy results (cached in session)
    if "all_results_v4" not in st.session_state:
        all_results: Dict[str, Dict[str, BacktestResult]] = {}
        for strat_key in SIGNAL_GENERATORS:
            all_results[strat_key] = {}
            for ticker in TICKERS:
                if ticker not in price_data:
                    continue
                series = price_data[ticker].copy()
                series.name = ticker
                all_results[strat_key][ticker] = run_backtest(series, strat_key)
        st.session_state["all_results_v4"] = all_results

    all_results = st.session_state["all_results_v4"]

    # Top row — selectors
    col_s, col_t, _ = st.columns([2, 2, 4])
    with col_s:
        selected_strategy = st.selectbox(
            "Strategy",
            options=list(STRATEGY_NAMES.keys()),
            format_func=lambda k: STRATEGY_NAMES[k],
        )
    with col_t:
        selected_ticker = st.selectbox("Ticker", options=TICKERS)

    result = all_results[selected_strategy][selected_ticker]
    m = result.metrics
    bm = bench_metrics

    st.markdown("---")
    section_break()

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_comparison_card(
            "Total Return",
            fmt_pct(m["total_return"]),
            fmt_pct(bm["total_return"]),
            m["total_return"] - bm["total_return"],
            "Difference",
        )
    with c2:
        render_comparison_card(
            "Sharpe Ratio",
            fmt_num(m["sharpe"]),
            fmt_num(bm["sharpe"]),
            m["sharpe"] - bm["sharpe"],
            "Difference",
            higher_is_better=True,
        )
    with c3:
        render_comparison_card(
            "Max Drawdown",
            fmt_pct(m["max_drawdown"]),
            fmt_pct(bm["max_drawdown"]),
            m["max_drawdown"] - bm["max_drawdown"],
            "Difference",
            higher_is_better=True,
        )
    with c4:
        render_comparison_card(
            "Sortino Ratio",
            fmt_num(m["sortino"]),
            fmt_num(bm["sortino"]),
            m["sortino"] - bm["sortino"],
            "Difference",
        )
    with c5:
        render_comparison_card(
            "Win Rate",
            fmt_pct(m["win_rate"]),
            fmt_pct(bm["win_rate"]),
            m["win_rate"] - bm["win_rate"],
            "Difference",
        )

    section_break()

    # Main chart
    fig = build_equity_chart(
        result.equity_curve,
        benchmark_result.equity_curve,
        STRATEGY_NAMES[selected_strategy],
        selected_ticker,
    )
    st.plotly_chart(fig, use_container_width=True)

    section_break()

    # Strategy comparison table
    st.subheader("Strategy Comparison")
    comparison_rows = []
    for strat_key, strat_label in STRATEGY_NAMES.items():
        r = all_results[strat_key][selected_ticker]
        rm = r.metrics
        comparison_rows.append(
            {
                "Strategy": strat_label,
                "Total Return (%)": round(rm["total_return"], 2),
                "Sharpe": round(rm["sharpe"], 2),
                "Sortino": round(rm["sortino"], 2),
                "Max Drawdown (%)": round(rm["max_drawdown"], 2),
                "Win Rate (%)": round(rm["win_rate"], 2),
                "Trades": int(rm["num_trades"]),
                f"{BENCHMARK} Return (%)": round(bm["total_return"], 2),
            }
        )
    st.dataframe(
        pd.DataFrame(comparison_rows).set_index("Strategy"),
        use_container_width=True,
    )

    section_break()

    # Stress tests
    st.subheader("Stress Test Performance")
    stress_cols = st.columns(3)
    for i, (label, (s, e)) in enumerate(STRESS_PERIODS.items()):
        strat_ret = period_return(result.equity_curve, s, e)
        bench_ret = period_return(benchmark_result.equity_curve, s, e)
        with stress_cols[i]:
            render_stress_card(label, strat_ret, bench_ret)

    section_break()

    # Trade log
    st.subheader("Trade Log")
    if result.trades:
        trade_df = pd.DataFrame(
            [
                {
                    "Entry Date": t.entry_date.strftime("%Y-%m-%d"),
                    "Exit Date": t.exit_date.strftime("%Y-%m-%d"),
                    "Entry Price": round(t.entry_price, 2),
                    "Exit Price": round(t.exit_price, 2),
                    "Return (%)": round(t.return_pct, 2),
                }
                for t in result.trades
            ]
        )
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
        st.caption(f"Total trades: {len(result.trades)} · Transaction cost: {TX_COST * 100:.1f}% per side")
    else:
        st.info("No completed trades for this strategy/ticker combination.")

    # Footer
    st.markdown(
        '<div class="footer-note">Data sourced from Yahoo Finance via <strong>yfinance</strong>. '
        "For educational and portfolio demonstration purposes only. Not investment advice.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
