import streamlit as st
import json
import pandas as pd
import time
from datetime import datetime
from alpaca.trading.client import TradingClient


def app():
    st.title("⚙️ Account Settings")

    # Check if user is authenticated
    if not st.session_state.get('authenticated', False):
        st.warning("Please login first to access account settings.")
        return

    # Get user data from session state
    user_data = st.session_state.get('user_data', {})

    # Account information
    st.header("Account Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Username:** {st.session_state.get('username', 'N/A')}")
        st.write(f"**Account Created:** {format_date(user_data.get('created_at', 'N/A'))}")

    with col2:
        st.write(f"**Last Login:** {format_date(user_data.get('last_login', 'N/A'))}")
        st.write(f"**Login Count:** {user_data.get('login_count', 0)}")

    # API Keys
    st.header("API Keys")

    with st.expander("View API Keys"):
        st.warning("⚠️ Keep your API keys secure and never share them with anyone.")

        col1, col2 = st.columns(2)

        with col1:
            st.text_input(
                "Alpaca API Key",
                value=user_data.get('alpaca_key', ''),
                type="password",
                disabled=True
            )

        with col2:
            st.text_input(
                "Alpaca Secret Key",
                value=user_data.get('alpaca_secret', ''),
                type="password",
                disabled=True
            )

    # Portfolio Settings
    st.header("Portfolio Settings")

    # Export portfolio data
    portfolio = st.session_state.get('portfolio', {})

    if st.button("📊 Export Portfolio Data"):
        portfolio_json = json.dumps(portfolio, indent=2)
        st.download_button(
            label="Download Portfolio JSON",
            data=portfolio_json,
            file_name=f"portfolio_{st.session_state.get('username', 'user')}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

    # Reset portfolio
    st.write("**Reset Portfolio**")
    st.warning("⚠️ This will reset your portfolio to default values. This action cannot be undone.")

    if st.button("🔄 Reset Portfolio"):
        reset_confirmed = st.checkbox("I understand this will reset my portfolio data")

        if reset_confirmed and st.button("Confirm Reset", key="confirm_reset"):
            reset_portfolio(user_data)
            st.success("Portfolio reset successfully!")
            st.experimental_rerun()

    # Appearance Settings
    st.header("Appearance Settings")

    # Theme selection
    theme = st.selectbox(
        "Theme",
        ["Light", "Dark", "System Default"],
        index=1
    )

    # Date format
    date_format = st.selectbox(
        "Date Format",
        ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"],
        index=2
    )

    # Save appearance settings
    if st.button("Save Appearance Settings"):
        st.session_state['theme'] = theme
        st.session_state['date_format'] = date_format
        st.success("Appearance settings saved!")

    # Account Security
    st.header("Account Security")

    # Change password
    with st.form("change_password_form"):
        st.subheader("Change Password")

        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        submitted = st.form_submit_button("Change Password")

        if submitted:
            if not current_password or not new_password or not confirm_password:
                st.error("Please fill in all password fields.")
            elif new_password != confirm_password:
                st.error("New passwords do not match.")
            else:
                # In a real app, you would verify the current password and update it
                st.success("Password changed successfully!")

    # Session Management
    st.header("Session Management")

    # Display current session info
    st.write(f"**Current Session Started:** {format_time_ago(st.session_state.get('last_activity', time.time()))}")

    # Logout button
    if st.button("🚪 Logout"):
        logout()
        st.experimental_rerun()

    # Delete Account
    st.header("Delete Account")

    st.warning("⚠️ Deleting your account will permanently remove all your data. This action cannot be undone.")

    with st.form("delete_account_form"):
        st.write("To delete your account, please type 'DELETE' to confirm.")

        confirmation = st.text_input("Confirmation")
        password = st.text_input("Password", type="password")

        submitted = st.form_submit_button("Delete Account")

        if submitted:
            if confirmation != "DELETE":
                st.error("Please type 'DELETE' to confirm account deletion.")
            elif not password:
                st.error("Please enter your password.")
            else:
                # In a real app, you would verify the password and delete the account
                st.success("Account deletion request submitted. Your account will be deleted shortly.")


def format_date(date_str):
    """Format date string for display"""
    if date_str == 'N/A':
        return date_str

    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return date_str


def format_time_ago(timestamp):
    """Format timestamp as time ago"""
    now = time.time()
    diff = now - timestamp

    if diff < 60:
        return "Just now"
    elif diff < 3600:
        minutes = int(diff / 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif diff < 86400:
        hours = int(diff / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        days = int(diff / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"


def reset_portfolio(user_data):
    """Reset portfolio to default values"""
    try:
        # Initialize Alpaca API with the new SDK
        trading_client = TradingClient(
            api_key=user_data.get('alpaca_key', ''),
            secret_key=user_data.get('alpaca_secret', ''),
            paper=True  # Use paper trading
        )

        # Get account information
        account = trading_client.get_account()

        # Update portfolio in session state
        st.session_state['portfolio'] = {
            'cash': float(account.cash),
            'value': float(account.equity),
            'buying_power': float(account.buying_power),
            'daily_change': 0,
            'daily_change_pct': 0,
            'positions': [],
            'orders': []
        }

    except Exception as e:
        st.error(f"Error resetting portfolio: {str(e)}")


def logout():
    """Log out the current user"""
    # Clear session state
    for key in ['authenticated', 'username', 'user_data', 'portfolio']:
        if key in st.session_state:
            del st.session_state[key]

    st.success("Logged out successfully!")


if __name__ == "__main__":
    app()
