import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from typing import Dict, Any, List
import pandas as pd


def display_portfolio_summary(portfolio_data: Dict[str, Any]):
    """
    Display comprehensive portfolio summary including allocations, performance, and risk

    Args:
        portfolio_data: Dictionary containing portfolio information

    Usage:
        from ui.components.portfolio_summary import display_portfolio_summary
        display_portfolio_summary(portfolio_data)
    """
    if not portfolio_data:
        st.warning("No portfolio data available")
        return

    st.header("💼 Portfolio Summary")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    total_value = portfolio_data.get('total_value', 0)
    daily_change = portfolio_data.get('daily_change', 0)
    daily_change_pct = portfolio_data.get('daily_change_percent', 0)
    cash_balance = portfolio_data.get('cash_balance', 0)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${total_value:,.2f}",
            delta=f"${daily_change:,.2f} ({daily_change_pct:+.2f}%)"
        )

    with col2:
        positions_count = len(portfolio_data.get('positions', {}))
        st.metric("Active Positions", positions_count)

    with col3:
        st.metric("Cash Balance", f"${cash_balance:,.2f}")

    with col4:
        buying_power = portfolio_data.get('buying_power', 0)
        st.metric("Buying Power", f"${buying_power:,.2f}")

    # Asset allocation section
    allocations = portfolio_data.get('allocations', {})
    if allocations:
        st.subheader("📊 Asset Allocation")

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(allocations.keys()),
                values=list(allocations.values()),
                hole=0.3
            )])

            fig_pie.update_layout(
                title="Portfolio Allocation",
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Allocation table
            allocation_df = pd.DataFrame([
                {"Asset": asset, "Allocation %": f"{weight * 100:.1f}%", "Value": f"${total_value * weight:,.2f}"}
                for asset, weight in allocations.items()
            ])
            st.dataframe(allocation_df, use_container_width=True)

    # Performance metrics
    performance = portfolio_data.get('performance', {})
    if performance:
        st.subheader("📈 Performance Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Return", f"{performance.get('total_return', 0) * 100:.2f}%")
            st.metric("1M Return", f"{performance.get('1m_return', 0) * 100:.2f}%")

        with col2:
            st.metric("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}")
            st.metric("Max Drawdown", f"{performance.get('max_drawdown', 0) * 100:.2f}%")

        with col3:
            st.metric("Volatility", f"{performance.get('volatility', 0) * 100:.2f}%")
            st.metric("Beta", f"{performance.get('beta', 0):.2f}")

    # Holdings breakdown
    positions = portfolio_data.get('positions', {})
    if positions:
        st.subheader("📋 Current Holdings")

        holdings_data = []
        for symbol, position in positions.items():
            holdings_data.append({
                "Symbol": symbol,
                "Shares": position.get('quantity', 0),
                "Price": f"${position.get('current_price', 0):.2f}",
                "Market Value": f"${position.get('market_value', 0):,.2f}",
                "P&L": f"${position.get('unrealized_pl', 0):,.2f}",
                "P&L %": f"{position.get('unrealized_pl_percent', 0):+.2f}%",
                "Weight": f"{position.get('weight', 0) * 100:.1f}%"
            })

        holdings_df = pd.DataFrame(holdings_data)
        st.dataframe(holdings_df, use_container_width=True)

    # Risk metrics
    risk = portfolio_data.get('risk', {})
    if risk:
        st.subheader("⚠️ Risk Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("VaR (95%)", f"${risk.get('var_95', 0):,.2f}")
            st.metric("Portfolio Beta", f"{risk.get('portfolio_beta', 0):.2f}")

        with col2:
            st.metric("Correlation to SPY", f"{risk.get('spy_correlation', 0):.2f}")
            st.metric("Diversification Ratio", f"{risk.get('diversification_ratio', 0):.2f}")


def display_portfolio_performance_chart(performance_history: List[Dict[str, Any]],
                                        benchmark_data: List[Dict[str, Any]] = None):
    """
    Display portfolio performance chart with optional benchmark comparison

    Args:
        performance_history: List of portfolio performance data over time
        benchmark_data: Optional benchmark data for comparison
    """
    if not performance_history:
        st.warning("No performance history available")
        return

    st.subheader("📊 Portfolio Performance Chart")

    df = pd.DataFrame(performance_history)
    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure()

    # Portfolio performance line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_return'],
        mode='lines',
        name='Portfolio',
        line=dict(color='#1f77b4', width=3)
    ))

    # Benchmark comparison if provided
    if benchmark_data:
        benchmark_df = pd.DataFrame(benchmark_data)
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])

        fig.add_trace(go.Scatter(
            x=benchmark_df['date'],
            y=benchmark_df['cumulative_return'],
            mode='lines',
            name='Benchmark (SPY)',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))

    fig.update_layout(
        title="Portfolio vs Benchmark Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template="plotly_dark",
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def display_sector_allocation(sector_data: Dict[str, float]):
    """
    Display sector allocation breakdown

    Args:
        sector_data: Dictionary of sector -> allocation percentage
    """
    if not sector_data:
        st.warning("No sector allocation data available")
        return

    st.subheader("🏭 Sector Allocation")

    # Horizontal bar chart
    fig = go.Figure(go.Bar(
        x=list(sector_data.values()),
        y=list(sector_data.keys()),
        orientation='h',
        marker_color='lightblue'
    ))

    fig.update_layout(
        title="Portfolio Sector Breakdown",
        xaxis_title="Allocation (%)",
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def display_risk_exposure(risk_exposures: Dict[str, Any]):
    """
    Display risk exposure metrics and visualizations

    Args:
        risk_exposures: Dictionary containing various risk metrics
    """
    if not risk_exposures:
        st.warning("No risk exposure data available")
        return

    st.subheader("⚡ Risk Exposure Analysis")

    # Risk gauge chart
    risk_score = risk_exposures.get('overall_risk_score', 50)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)
