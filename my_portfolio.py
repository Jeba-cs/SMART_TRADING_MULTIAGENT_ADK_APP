import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass


def app():
    st.title("💼 My Portfolio")

    # Check if user is authenticated
    if not st.session_state.get('authenticated', False):
        st.warning("Please login first to view your portfolio.")
        return

    # Get user data from session state
    user_data = st.session_state.get('user_data', {})

    # Refresh portfolio button
    if st.button("🔄 Refresh Portfolio"):
        with st.spinner("Fetching latest portfolio data..."):
            refresh_portfolio(user_data)

    # Get portfolio from session state
    portfolio = st.session_state.get('portfolio', {})

    # Portfolio summary
    st.header("Portfolio Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Value",
            value=f"${portfolio.get('value', 0):,.2f}",
            delta=f"${portfolio.get('daily_change', 0):,.2f}"
        )

    with col2:
        st.metric(
            label="Cash Balance",
            value=f"${portfolio.get('cash', 0):,.2f}"
        )

    with col3:
        invested = portfolio.get('value', 0) - portfolio.get('cash', 0)
        st.metric(
            label="Invested Amount",
            value=f"${invested:,.2f}"
        )

    with col4:
        daily_change_pct = portfolio.get('daily_change_pct', 0)
        st.metric(
            label="Daily Change",
            value=f"{daily_change_pct:.2f}%",
            delta=f"${portfolio.get('daily_change', 0):,.2f}"
        )

    # Portfolio positions
    st.header("Positions")

    positions = portfolio.get('positions', [])

    if positions:
        # Convert positions to DataFrame for display
        positions_data = []
        for position in positions:
            positions_data.append({
                "Symbol": position.get('sym', ''),
                "Quantity": position.get('qty', 0),
                "Avg Price": f"${position.get('avg', 0):.2f}",
                "Current Price": f"${position.get('current_price', 0):.2f}",
                "Market Value": f"${position.get('mv', 0):,.2f}",
                "Unrealized P&L": f"${position.get('unrealized_pl', 0):,.2f}",
                "Unrealized P&L %": f"{position.get('unrealized_plpc', 0):.2f}%"
            })

        positions_df = pd.DataFrame(positions_data)
        st.dataframe(positions_df, use_container_width=True)

        # Portfolio performance chart
        st.header("Portfolio Performance")

        # Create a sample performance chart (in a real app, this would use historical data)
        dates = pd.date_range(end=datetime.now(), periods=30).tolist()
        values = [portfolio.get('value', 10000) * (1 + (i * 0.002)) for i in range(30)]

        performance_df = pd.DataFrame({
            'Date': dates,
            'Value': values
        })

        fig = px.line(
            performance_df,
            x='Date',
            y='Value',
            title="Portfolio Value (Last 30 Days)",
            labels={'Value': 'Portfolio Value ($)', 'Date': ''}
        )

        fig.update_layout(
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                tickprefix='$'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Asset allocation
        st.header("Asset Allocation")

        # Create allocation data
        allocation_data = {}
        for position in positions:
            allocation_data[position.get('sym', 'Unknown')] = position.get('mv', 0)

        # Add cash to allocation
        allocation_data['Cash'] = portfolio.get('cash', 0)

        # Create pie chart
        fig = px.pie(
            values=list(allocation_data.values()),
            names=list(allocation_data.keys()),
            title="Asset Allocation",
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No positions in your portfolio. Start trading to build your portfolio!")

    # Order history
    st.header("Order History")

    orders = portfolio.get('orders', [])

    if orders:
        # Convert orders to DataFrame for display
        orders_data = []
        for order in orders:
            orders_data.append({
                "Symbol": order.get('symbol', ''),
                "Side": order.get('side', ''),
                "Quantity": order.get('qty', 0),
                "Type": order.get('type', ''),
                "Status": order.get('status', ''),
                "Filled At": order.get('filled_at', 'N/A'),
                "Filled Price": f"${order.get('filled_avg_price', 0):.2f}" if order.get('filled_avg_price') else 'N/A'
            })

        orders_df = pd.DataFrame(orders_data)
        st.dataframe(orders_df, use_container_width=True)
    else:
        st.info("No order history available.")


def refresh_portfolio(user_data):
    """Refresh portfolio data from Alpaca API"""
    try:
        # Initialize Alpaca API with the new SDK
        trading_client = TradingClient(
            api_key=user_data.get('alpaca_key', ''),
            secret_key=user_data.get('alpaca_secret', ''),
            paper=True  # Use paper trading
        )

        # Get account information
        account = trading_client.get_account()

        # Get positions
        positions = trading_client.get_all_positions()

        # Get orders
        orders = trading_client.get_orders(limit=20)

        # Process positions
        processed_positions = []
        for position in positions:
            # Calculate unrealized P&L percentage
            unrealized_plpc = 0
            if float(position.avg_entry_price) > 0:
                unrealized_plpc = (float(position.current_price) - float(position.avg_entry_price)) / float(
                    position.avg_entry_price) * 100

            processed_positions.append({
                'sym': position.symbol,
                'qty': int(float(position.qty)),
                'avg': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'mv': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': unrealized_plpc
            })

        # Process orders
        processed_orders = []
        for order in orders:
            processed_orders.append({
                'symbol': order.symbol,
                'side': order.side.value,
                'qty': int(float(order.qty)),
                'type': order.type.value,
                'status': order.status.value,
                'filled_at': order.filled_at,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            })

        # Calculate daily change
        daily_change = float(account.equity) - float(account.last_equity)
        daily_change_pct = (daily_change / float(account.last_equity)) * 100 if float(account.last_equity) > 0 else 0

        # Update portfolio in session state
        st.session_state['portfolio'] = {
            'cash': float(account.cash),
            'value': float(account.equity),
            'buying_power': float(account.buying_power),
            'daily_change': daily_change,
            'daily_change_pct': daily_change_pct,
            'positions': processed_positions,
            'orders': processed_orders
        }

        st.success("Portfolio refreshed successfully!")

    except Exception as e:
        st.error(f"Error refreshing portfolio: {str(e)}")


if __name__ == "__main__":
    app()
