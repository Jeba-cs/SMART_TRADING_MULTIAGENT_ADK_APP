import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType


def app():
    st.title("⚡ Trading Hub")

    # Check if user is authenticated
    if not st.session_state.get('authenticated', False):
        st.warning("Please login first to access trading features.")
        return

    # Get user data from session state
    user_data = st.session_state.get('user_data', {})
    portfolio = st.session_state.get('portfolio', {})

    # Trading interface
    st.header("Place Order")

    # Create tabs for different order types
    tab1, tab2, tab3 = st.tabs(["Market Order", "Limit Order", "Stop Order"])

    with tab1:
        create_market_order_form(user_data, portfolio)

    with tab2:
        create_limit_order_form(user_data, portfolio)

    with tab3:
        create_stop_order_form(user_data, portfolio)

    # Order status
    st.header("Recent Orders")
    display_recent_orders(portfolio)

    # Watchlist
    st.header("Watchlist")
    create_watchlist()


def create_market_order_form(user_data, portfolio):
    """Create market order form"""
    with st.form("market_order_form"):
        st.subheader("Market Order")

        col1, col2 = st.columns(2)

        with col1:
            symbol = st.text_input("Symbol", value="AAPL").upper()
            side = st.selectbox("Side", ["buy", "sell"])

        with col2:
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            time_in_force = st.selectbox("Time In Force", ["day", "gtc", "ioc", "opg"])

        # Calculate estimated cost/proceeds
        if symbol and side and quantity > 0:
            estimated_value = calculate_estimated_value(symbol, side, quantity)
            st.write(f"Estimated {'Cost' if side == 'buy' else 'Proceeds'}: ${estimated_value:.2f}")

        # Submit button
        submitted = st.form_submit_button("Place Market Order")

        if submitted:
            place_order(
                user_data=user_data,
                symbol=symbol,
                side=side,
                qty=quantity,
                order_type="market",
                time_in_force=time_in_force
            )


def create_limit_order_form(user_data, portfolio):
    """Create limit order form"""
    with st.form("limit_order_form"):
        st.subheader("Limit Order")

        col1, col2 = st.columns(2)

        with col1:
            symbol = st.text_input("Symbol", value="AAPL", key="limit_symbol").upper()
            side = st.selectbox("Side", ["buy", "sell"], key="limit_side")

        with col2:
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="limit_qty")
            time_in_force = st.selectbox("Time In Force", ["day", "gtc", "ioc", "opg"], key="limit_tif")

        # Limit price
        current_price = get_current_price(symbol)
        limit_price = st.number_input(
            "Limit Price",
            min_value=0.01,
            value=float(current_price) if current_price else 100.00,
            step=0.01,
            format="%.2f"
        )

        # Calculate estimated cost/proceeds
        if symbol and side and quantity > 0 and limit_price > 0:
            estimated_value = quantity * limit_price
            st.write(f"Estimated {'Cost' if side == 'buy' else 'Proceeds'}: ${estimated_value:.2f}")

        # Submit button
        submitted = st.form_submit_button("Place Limit Order")

        if submitted:
            place_order(
                user_data=user_data,
                symbol=symbol,
                side=side,
                qty=quantity,
                order_type="limit",
                time_in_force=time_in_force,
                limit_price=limit_price
            )


def create_stop_order_form(user_data, portfolio):
    """Create stop order form"""
    with st.form("stop_order_form"):
        st.subheader("Stop Order")

        col1, col2 = st.columns(2)

        with col1:
            symbol = st.text_input("Symbol", value="AAPL", key="stop_symbol").upper()
            side = st.selectbox("Side", ["buy", "sell"], key="stop_side")

        with col2:
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="stop_qty")
            time_in_force = st.selectbox("Time In Force", ["day", "gtc", "ioc", "opg"], key="stop_tif")

        # Stop price
        current_price = get_current_price(symbol)
        stop_price = st.number_input(
            "Stop Price",
            min_value=0.01,
            value=float(current_price) if current_price else 100.00,
            step=0.01,
            format="%.2f"
        )

        # Calculate estimated cost/proceeds
        if symbol and side and quantity > 0 and stop_price > 0:
            estimated_value = quantity * stop_price
            st.write(f"Estimated {'Cost' if side == 'buy' else 'Proceeds'}: ${estimated_value:.2f}")

        # Submit button
        submitted = st.form_submit_button("Place Stop Order")

        if submitted:
            place_order(
                user_data=user_data,
                symbol=symbol,
                side=side,
                qty=quantity,
                order_type="stop",
                time_in_force=time_in_force,
                stop_price=stop_price
            )


def display_recent_orders(portfolio):
    """Display recent orders"""
    orders = portfolio.get('orders', [])

    if orders:
        # Convert orders to DataFrame for display
        orders_data = []
        for order in orders[:5]:  # Show only the 5 most recent orders
            orders_data.append({
                "Symbol": order.get('symbol', ''),
                "Side": order.get('side', '').capitalize(),
                "Quantity": order.get('qty', 0),
                "Type": order.get('type', '').capitalize(),
                "Status": order.get('status', '').capitalize(),
                "Filled At": order.get('filled_at', 'N/A'),
                "Filled Price": f"${order.get('filled_avg_price', 0):.2f}" if order.get('filled_avg_price') else 'N/A'
            })

        orders_df = pd.DataFrame(orders_data)
        st.dataframe(orders_df, use_container_width=True)
    else:
        st.info("No recent orders.")


def create_watchlist():
    """Create and display watchlist"""
    # Get watchlist from session state or initialize
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    watchlist = st.session_state['watchlist']

    # Add to watchlist
    col1, col2 = st.columns([3, 1])

    with col1:
        new_symbol = st.text_input("Add Symbol to Watchlist").upper()

    with col2:
        if st.button("Add") and new_symbol and new_symbol not in watchlist:
            watchlist.append(new_symbol)
            st.session_state['watchlist'] = watchlist
            st.success(f"Added {new_symbol} to watchlist!")
            st.experimental_rerun()

    # Display watchlist
    if watchlist:
        watchlist_data = []
        for symbol in watchlist:
            price = get_current_price(symbol)
            watchlist_data.append({
                "Symbol": symbol,
                "Price": f"${price:.2f}" if price else "N/A",
                "Actions": "Remove"
            })

        watchlist_df = pd.DataFrame(watchlist_data)
        st.dataframe(watchlist_df, use_container_width=True)

        # Remove from watchlist
        symbol_to_remove = st.selectbox("Select Symbol to Remove", watchlist)
        if st.button("Remove from Watchlist"):
            watchlist.remove(symbol_to_remove)
            st.session_state['watchlist'] = watchlist
            st.success(f"Removed {symbol_to_remove} from watchlist!")
            st.experimental_rerun()
    else:
        st.info("Your watchlist is empty. Add symbols to track them.")


def place_order(user_data, symbol, side, qty, order_type, time_in_force, limit_price=None, stop_price=None):
    """Place an order using Alpaca API"""
    try:
        # Initialize Alpaca API with the new SDK
        trading_client = TradingClient(
            api_key=user_data.get('alpaca_key', ''),
            secret_key=user_data.get('alpaca_secret', ''),
            paper=True  # Use paper trading
        )

        # Convert side string to OrderSide enum
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        # Convert time_in_force string to TimeInForce enum
        tif_map = {
            'day': TimeInForce.DAY,
            'gtc': TimeInForce.GTC,
            'ioc': TimeInForce.IOC,
            'opg': TimeInForce.OPG
        }
        time_in_force_enum = tif_map.get(time_in_force.lower(), TimeInForce.DAY)

        # Create appropriate order request based on order type
        if order_type.lower() == 'market':
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=time_in_force_enum
            )
        elif order_type.lower() == 'limit' and limit_price:
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=time_in_force_enum,
                limit_price=limit_price
            )
        elif order_type.lower() == 'stop' and stop_price:
            order_request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=time_in_force_enum,
                stop_price=stop_price
            )
        else:
            st.error(f"Invalid order type or missing price parameters")
            return

        # Submit order
        order = trading_client.submit_order(order_data=order_request)

        # Show success message
        st.success(f"Order placed successfully! Order ID: {order.id}")

        # Refresh portfolio data
        refresh_portfolio(user_data)

    except Exception as e:
        st.error(f"Error placing order: {str(e)}")


def get_current_price(symbol):
    """Get current price for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        todays_data = ticker.history(period='1d')
        return todays_data['Close'][0]
    except:
        return None


def calculate_estimated_value(symbol, side, quantity):
    """Calculate estimated cost or proceeds for an order"""
    price = get_current_price(symbol)
    if price:
        return price * quantity
    return 0


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

    except Exception as e:
        st.error(f"Error refreshing portfolio: {str(e)}")


if __name__ == "__main__":
    app()
