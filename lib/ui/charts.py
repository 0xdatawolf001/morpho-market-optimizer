"""Altair chart builders for optimization and market detail."""

from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

from lib.config import MAX_LINE_PLOT_POINTS_PER_STRATEGY, MAX_SCATTER_PLOT_POINTS, STRATEGY_COLORS, STRATEGY_NAMES


def build_efficiency_frontier_chart(df_scatter: pd.DataFrame, highlights: pd.DataFrame):
    if df_scatter.empty:
        return None
    if len(df_scatter) > MAX_SCATTER_PLOT_POINTS:
        df_scatter = df_scatter.sample(MAX_SCATTER_PLOT_POINTS, random_state=42)
    base = alt.Chart(df_scatter).mark_circle(opacity=0.3, color="#80DEEA").encode(
        x=alt.X("Diversity Score", title="Diversity (1 - HHI)"),
        y=alt.Y("Blended APY", title="APY", axis=alt.Axis(format="%")),
        tooltip=["Diversity Score", "Blended APY", "Annual Yield ($)"],
    )
    points = alt.Chart(highlights).mark_circle(size=200, opacity=1.0).encode(
        x="Diversity Score",
        y="Blended APY",
        color=alt.Color(
            "Type",
            scale=alt.Scale(domain=STRATEGY_NAMES, range=list(STRATEGY_COLORS.values())),
        ),
        tooltip=["Type", "Blended APY", "Diversity Score"],
    )
    return base + points


def build_convergence_chart(traces: dict):
    d1 = pd.DataFrame({"Iteration": range(len(traces["yield"])), "Value": traces["yield"], "Strategy": "Best Yield"})
    d2 = pd.DataFrame({"Iteration": range(len(traces["frontier"])), "Value": traces["frontier"], "Strategy": "Frontier"})
    d4 = pd.DataFrame({"Iteration": range(len(traces["liquid"])), "Value": traces["liquid"], "Strategy": "Liquid-Yield"})
    d5 = pd.DataFrame({"Iteration": range(len(traces["whale"])), "Value": traces["whale"], "Strategy": "Whale Shield"})

    def _downsample(df, seed):
        if len(df) > MAX_LINE_PLOT_POINTS_PER_STRATEGY:
            return df.sample(MAX_LINE_PLOT_POINTS_PER_STRATEGY, random_state=seed).sort_values("Iteration")
        return df.sort_values("Iteration")

    df_hist_long = pd.concat([_downsample(d1, 42), _downsample(d2, 43), _downsample(d4, 45), _downsample(d5, 46)])
    return alt.Chart(df_hist_long).mark_line().encode(
        x="Iteration",
        y=alt.Y("Value", title="Objective Yield ($)"),
        color=alt.Color(
            "Strategy",
            scale=alt.Scale(domain=STRATEGY_NAMES, range=list(STRATEGY_COLORS.values())),
        ),
    )


def build_allocation_chart(bar_data: list[dict]):
    if not bar_data:
        return None
    df_bar = pd.DataFrame(bar_data)
    return (
        alt.Chart(df_bar)
        .mark_bar()
        .encode(
            y=alt.Y("Market:N", title=None, sort="ascending"),
            x=alt.X("Alloc ($):Q", title="Allocation (USD)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Market:N", legend=None),
            row=alt.Row(
                "Strategy:N",
                title="Portfolio Strategy",
                sort=STRATEGY_NAMES,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=14),
            ),
            tooltip=["Strategy", "Market", alt.Tooltip("Alloc ($)", format="$,.2f")],
        )
        .properties(height=alt.Step(25), width="container")
        .resolve_scale(y="independent")
        .configure_view(stroke=None)
    )


def historical_series_to_df(series: list[dict] | None, value_name: str) -> pd.DataFrame:
    if not series:
        return pd.DataFrame(columns=["date", value_name])
    rows = []
    for pt in series:
        ts = pt.get("x")
        val = pt.get("y")
        if ts is None:
            continue
        rows.append({"date": datetime.fromtimestamp(int(ts), tz=timezone.utc), value_name: float(val or 0)})
    return pd.DataFrame(rows)


def render_efficiency_frontier(df_scatter: pd.DataFrame, highlights: pd.DataFrame):
    chart = build_efficiency_frontier_chart(df_scatter, highlights)
    if chart is None:
        st.info("Not enough data to generate optimization charts. Please add capital or select more markets.")
    else:
        st.altair_chart(chart, use_container_width=True)


def render_convergence_chart(traces: dict):
    chart = build_convergence_chart(traces)
    st.altair_chart(chart, use_container_width=True)


def render_allocation_bars(bar_data: list[dict]):
    chart = build_allocation_chart(bar_data)
    if chart is None:
        st.info("No significant allocations.")
    else:
        st.altair_chart(chart, use_container_width=True)


def build_historical_line_chart(df: pd.DataFrame, y_field: str, title: str, fmt: str = ".2%"):
    if df.empty:
        return None
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{y_field}:Q", title=title, axis=alt.Axis(format=fmt)),
            tooltip=["date:T", alt.Tooltip(f"{y_field}:Q", format=fmt)],
        )
        .properties(height=220)
    )
