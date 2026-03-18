"""
app.py – Streamlit dashboard for Pulsecast.

Features
--------
- Route selector (TLC zone / PULocationID)
- Fan chart (p10/p50/p90 over the forecast horizon)
- Ablation panel (model comparison table)
- Calibration chart (coverage probability vs. nominal quantile)
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

_API_URL = os.getenv("PULSECAST_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def fetch_forecast(route_id: int, horizon: int) -> dict | None:
    try:
        resp = requests.post(
            f"{_API_URL}/forecast",
            json={"route_id": route_id, "horizon": horizon},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _fan_chart(data: dict, horizon: int) -> None:
    """Render a fan chart from p10/p50/p90 forecast data."""
    import plotly.graph_objects as go

    n = horizon * 24
    hours = list(range(1, n + 1))
    p10 = data["p10"][:n]
    p50 = data["p50"][:n]
    p90 = data["p90"][:n]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hours + hours[::-1],
            y=p90 + p10[::-1],
            fill="toself",
            fillcolor="rgba(99,110,250,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% interval",
        )
    )
    fig.add_trace(
        go.Scatter(x=hours, y=p50, mode="lines", name="Median (p50)",
                   line=dict(color="rgb(99,110,250)", width=2))
    )
    fig.update_layout(
        title=f"Demand Forecast – Route {data['route_id']}",
        xaxis_title="Hours ahead",
        yaxis_title="Pickup volume",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def _ablation_panel() -> None:
    """Static ablation results table."""
    df = pd.DataFrame(
        {
            "Model": ["MSTL", "LightGBM", "LightGBM + delay", "TFT + delay"],
            "MAE": ["—", "—", "—", "—"],
            "RMSE": ["—", "—", "—", "—"],
            "Pinball p10": ["—", "—", "—", "—"],
            "Pinball p90": ["—", "—", "—", "—"],
        }
    )
    st.subheader("Ablation Study")
    st.dataframe(df, use_container_width=True)


def _calibration_chart(data: dict) -> None:
    """Plot coverage probability vs. nominal quantile."""
    import plotly.graph_objects as go

    # Placeholder calibration curve (diagonal = perfect calibration).
    nominal = [0.1, 0.5, 0.9]
    observed = [0.1, 0.5, 0.9]  # Replace with real empirical coverage

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Perfect calibration",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=nominal, y=observed,
            mode="markers+lines",
            name="Model calibration",
            marker=dict(size=10),
        )
    )
    fig.update_layout(
        title="Calibration Chart",
        xaxis_title="Nominal quantile",
        yaxis_title="Observed coverage",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Pulsecast", layout="wide")
    st.title("🚕 Pulsecast – Demand Forecast Dashboard")

    with st.sidebar:
        st.header("Parameters")
        route_id = st.number_input("Route / Zone ID", min_value=1, max_value=263, value=132)
        horizon = st.slider("Horizon (days)", min_value=1, max_value=7, value=3)
        run_btn = st.button("Fetch Forecast")

    if run_btn:
        with st.spinner("Fetching forecast …"):
            data = fetch_forecast(int(route_id), int(horizon))

        if data:
            col1, col2, col3 = st.columns(3)
            col1.metric("p10 (next hour)", f"{data['p10'][0]:.1f}")
            col2.metric("p50 (next hour)", f"{data['p50'][0]:.1f}")
            col3.metric("p90 (next hour)", f"{data['p90'][0]:.1f}")

            _fan_chart(data, int(horizon))
            _calibration_chart(data)

    _ablation_panel()


if __name__ == "__main__":
    main()
