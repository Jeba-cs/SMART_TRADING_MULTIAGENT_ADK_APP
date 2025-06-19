import streamlit as st
import plotly.graph_objs as go
from typing import List, Dict, Optional
import pandas as pd


def display_order_book(order_book_data: List[Dict], symbol: str = ""):
    """
    Display order book data as an interactive table with depth visualization

    Args:
        order_book_data: List of orders with keys: price, quantity, side
        symbol: Stock symbol for display

    Usage:
        from ui.components.order_book import display_order_book
        display_order_book(order_book, "AAPL")
    """
    if not order_book_data:
        st.warning("No order book data available")
        return

    # Separate bids and asks
    bids = [order for order in order_book_data if order.get('side') == 'buy']
    asks = [order for order in order_book_data if order.get('side') == 'sell']

    # Sort orders appropriately
    bids.sort(key=lambda x: x.get('price', 0), reverse=True)  # Highest bid first
    asks.sort(key=lambda x: x.get('price', 0))  # Lowest ask first

    st.subheader(f"📊 Order Book {f'for {symbol}' if symbol else ''}")

    # Display current spread if we have both bids and asks
    if bids and asks:
        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100 if best_bid > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Bid", f"${best_bid:.2f}")
        with col2:
            st.metric("Best Ask", f"${best_ask:.2f}")
        with col3:
            st.metric("Spread", f"${spread:.2f} ({spread_pct:.2f}%)")

    # Main order book display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟢 Bids")
        if bids:
            # Create DataFrame for better display
            bids_df = pd.DataFrame(bids[:10])  # Show top 10 bids
            bids_df['Price'] = bids_df['price'].apply(lambda x: f"${x:.2f}")
            bids_df['Quantity'] = bids_df['quantity']
            bids_df['Total'] = (bids_df['price'] * bids_df['quantity']).apply(lambda x: f"${x:,.2f}")

            # Display with color coding
            st.dataframe(
                bids_df[['Price', 'Quantity', 'Total']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No bid orders available")

    with col2:
        st.markdown("### 🔴 Asks")
        if asks:
            # Create DataFrame for better display
            asks_df = pd.DataFrame(asks[:10])  # Show top 10 asks
            asks_df['Price'] = asks_df['price'].apply(lambda x: f"${x:.2f}")
            asks_df['Quantity'] = asks_df['quantity']
            asks_df['Total'] = (asks_df['price'] * asks_df['quantity']).apply(lambda x: f"${x:,.2f}")

            st.dataframe(
                asks_df[['Price', 'Quantity', 'Total']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No ask orders available")


def display_order_book_depth_chart(order_book_data: List[Dict], symbol: str = ""):
    """
    Display order book depth chart

    Args:
        order_book_data: List of orders with price, quantity, side
        symbol: Stock symbol
    """
    if not order_book_data:
        st.warning("No order book data available for depth chart")
        return

    st.subheader(f"📈 Order Book Depth {f'for {symbol}' if symbol else ''}")

    # Separate and sort orders
    bids = [order for order in order_book_data if order.get('side') == 'buy']
    asks = [order for order in order_book_data if order.get('side') == 'sell']

    bids.sort(key=lambda x: x.get('price', 0), reverse=True)
    asks.sort(key=lambda x: x.get('price', 0))

    fig = go.Figure()

    # Calculate cumulative quantities for bids
    if bids:
        bid_prices = [order['price'] for order in bids]
        bid_quantities = [order['quantity'] for order in bids]
        cumulative_bid_qty = []
        running_total = 0
        for qty in bid_quantities:
            running_total += qty
            cumulative_bid_qty.append(running_total)

        fig.add_trace(go.Scatter(
            x=bid_prices,
            y=cumulative_bid_qty,
            fill='tozeroy',
            mode='lines',
            name='Bids',
            line=dict(color='green', width=2),
            fillcolor='rgba(0, 255, 0, 0.3)'
        ))

    # Calculate cumulative quantities for asks
    if asks:
        ask_prices = [order['price'] for order in asks]
        ask_quantities = [order['quantity'] for order in asks]
        cumulative_ask_qty = []
        running_total = 0
        for qty in ask_quantities:
            running_total += qty
            cumulative_ask_qty.append(running_total)

        fig.add_trace(go.Scatter(
            x=ask_prices,
            y=cumulative_ask_qty,
            fill='tozeroy',
            mode='lines',
            name='Asks',
            line=dict(color='red', width=2),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))

    fig.update_layout(
        title=f"Order Book Depth Chart {f'for {symbol}' if symbol else ''}",
        xaxis_title="Price ($)",
        yaxis_title="Cumulative Quantity",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def display_market_depth_summary(order_book_data: List[Dict]):
    """
    Display market depth summary statistics

    Args:
        order_book_data: List of orders
    """
    if not order_book_data:
        return

    bids = [order for order in order_book_data if order.get('side') == 'buy']
    asks = [order for order in order_book_data if order.get('side') == 'sell']

    st.subheader("📊 Market Depth Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_bid_volume = sum(order['quantity'] for order in bids)
        st.metric("Total Bid Volume", f"{total_bid_volume:,}")

    with col2:
        total_ask_volume = sum(order['quantity'] for order in asks)
        st.metric("Total Ask Volume", f"{total_ask_volume:,}")

    with col3:
        bid_ask_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 0
        st.metric("Bid/Ask Ratio", f"{bid_ask_ratio:.2f}")

    with col4:
        total_orders = len(order_book_data)
        st.metric("Total Orders", total_orders)


def display_live_trades(recent_trades: List[Dict], symbol: str = ""):
    """
    Display recent trades/executions

    Args:
        recent_trades: List of recent trade data
        symbol: Stock symbol
    """
    if not recent_trades:
        st.info("No recent trades available")
        return

    st.subheader(f"🔄 Recent Trades {f'for {symbol}' if symbol else ''}")

    # Convert to DataFrame for better display
    trades_df = pd.DataFrame(recent_trades)

    if 'timestamp' in trades_df.columns:
        trades_df['Time'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%H:%M:%S')

    if 'price' in trades_df.columns:
        trades_df['Price'] = trades_df['price'].apply(lambda x: f"${x:.2f}")

    if 'quantity' in trades_df.columns:
        trades_df['Quantity'] = trades_df['quantity']

    if 'side' in trades_df.columns:
        trades_df['Side'] = trades_df['side'].apply(lambda x: "🟢 Buy" if x == 'buy' else "🔴 Sell")

    # Display recent trades table
    display_columns = ['Time', 'Price', 'Quantity', 'Side']
    available_columns = [col for col in display_columns if col in trades_df.columns]

    if available_columns:
        st.dataframe(
            trades_df[available_columns].head(20),
            use_container_width=True,
            hide_index=True
        )
