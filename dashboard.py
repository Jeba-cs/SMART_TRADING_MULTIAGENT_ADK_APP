import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta


def app():
    st.title("🏠 Dashboard")

    # Check if user is authenticated
    if not st.session_state.get('authenticated', False):
        st.warning("Please login first to view your dashboard.")
        return

    # Get user data and portfolio from session state
    user_data = st.session_state.get('user_data', {})
    portfolio = st.session_state.get('portfolio', {})

    # Display welcome message
    st.header(f"Welcome, {st.session_state.get('username', 'User')}!")

    # Portfolio overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Portfolio Value",
            value=f"${portfolio.get('value', 0):,.2f}",
            delta=f"${portfolio.get('daily_change', 0):,.2f}"
        )

    with col2:
        st.metric(
            label="💵 Cash Balance",
            value=f"${portfolio.get('cash', 0):,.2f}"
        )

    with col3:
        positions = portfolio.get('positions', [])
        st.metric(
            label="📊 Positions",
            value=len(positions)
        )

    with col4:
        st.metric(
            label="📈 Daily P&L",
            value=f"${portfolio.get('daily_change', 0):,.2f}",
            delta=f"{portfolio.get('daily_change_pct', 0):.2f}%"
        )

    # Portfolio allocation chart
    st.subheader("Portfolio Allocation")

    if positions:
        allocation_data = {}
        for position in positions:
            allocation_data[position.get('sym', 'Unknown')] = position.get('mv', 0)

        allocation_data['Cash'] = portfolio.get('cash', 0)

        fig = px.pie(
            values=list(allocation_data.values()),
            names=list(allocation_data.keys()),
            title="Asset Allocation",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No positions in portfolio. Your allocation is 100% cash.")

    # Quick actions
    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Market Analysis", use_container_width=True):
            st.switch_page("pages/market_analysis.py")

    with col2:
        if st.button("💼 View Portfolio", use_container_width=True):
            st.switch_page("pages/my_portfolio.py")

    with col3:
        if st.button("⚡ Start Trading", use_container_width=True):
            st.switch_page("pages/trading_hub.py")


if __name__ == "__main__":
    app()
