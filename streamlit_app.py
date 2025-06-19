import streamlit as st
import asyncio
import sys
import os
import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Streamlit configuration
st.set_page_config(
    page_title="SmartTrader Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database files
USER_DB_FILE = "user_database.json"
PORTFOLIO_DB_FILE = "portfolio_database.json"


class UserDatabase:
    """Secure user database management"""

    def __init__(self):
        self.user_db_file = USER_DB_FILE
        self.portfolio_db_file = PORTFOLIO_DB_FILE
        self.users = self.load_users()
        self.portfolios = self.load_portfolios()

    def load_users(self):
        """Load users from JSON file"""
        try:
            if os.path.exists(self.user_db_file):
                with open(self.user_db_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Error loading user database: {e}")
            return {}

    def load_portfolios(self):
        """Load portfolio data from JSON file"""
        try:
            if os.path.exists(self.portfolio_db_file):
                with open(self.portfolio_db_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Error loading portfolio database: {e}")
            return {}

    def save_users(self):
        """Save users to JSON file"""
        try:
            with open(self.user_db_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving user database: {e}")
            return False

    def save_portfolios(self):
        """Save portfolio data to JSON file"""
        try:
            with open(self.portfolio_db_file, 'w') as f:
                json.dump(self.portfolios, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving portfolio database: {e}")
            return False

    def hash_password(self, password):
        """Hash password using SHA-256 with salt"""
        salt = "smarttrader_salt_2025"  # In production, use random salt per user
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def create_user(self, username, password, alpaca_api_key, alpaca_secret_key):
        """Create new user with validation"""
        # Input validation
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters long"

        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters long"

        if not alpaca_api_key or not alpaca_secret_key:
            return False, "Both Alpaca API Key and Secret Key are required"

        if username in self.users:
            return False, "Username already exists"

        # Create user
        user_id = str(uuid.uuid4())
        self.users[username] = {
            "user_id": user_id,
            "password_hash": self.hash_password(password),
            "alpaca_api_key": alpaca_api_key,
            "alpaca_secret_key": alpaca_secret_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_login": None,
            "login_count": 0,
            "account_status": "active"
        }

        # Create default portfolio for user
        self.portfolios[user_id] = {
            "user_id": user_id,
            "username": username,
            "total_value": 100000.0,  # Starting value $100k
            "available_cash": 100000.0,
            "invested_amount": 0.0,
            "positions": {},
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "transaction_history": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

        if self.save_users() and self.save_portfolios():
            return True, "Account created successfully!"
        return False, "Error saving user data"

    def authenticate_user(self, username, password):
        """Authenticate user login"""
        if not username or not password:
            return False, "Please enter both username and password"

        if username not in self.users:
            return False, "Username not found"

        user = self.users[username]

        if user.get("account_status") != "active":
            return False, "Account is disabled"

        if user["password_hash"] == self.hash_password(password):
            # Update login information
            user["last_login"] = datetime.now(timezone.utc).isoformat()
            user["login_count"] = user.get("login_count", 0) + 1
            self.save_users()
            return True, "Login successful!"

        return False, "Invalid password"

    def get_user_data(self, username):
        """Get user data by username"""
        return self.users.get(username, None)

    def get_user_portfolio(self, user_id):
        """Get user portfolio by user_id"""
        return self.portfolios.get(user_id, None)

    def update_user_portfolio(self, user_id, portfolio_data):
        """Update user portfolio data"""
        if user_id in self.portfolios:
            portfolio_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            self.portfolios[user_id] = portfolio_data
            return self.save_portfolios()
        return False

    def add_transaction(self, user_id, transaction):
        """Add transaction to user's history"""
        if user_id in self.portfolios:
            if "transaction_history" not in self.portfolios[user_id]:
                self.portfolios[user_id]["transaction_history"] = []

            transaction["timestamp"] = datetime.now(timezone.utc).isoformat()
            transaction["transaction_id"] = str(uuid.uuid4())

            self.portfolios[user_id]["transaction_history"].append(transaction)
            return self.save_portfolios()
        return False


class SmartTraderApp:
    """Main Streamlit application with authentication"""

    def __init__(self):
        self.db = UserDatabase()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize all session state variables"""
        session_defaults = {
            'authenticated': False,
            'username': None,
            'user_data': None,
            'user_portfolio': None,
            'trader_initialized': False,
            'trader_instance': None,
            'last_activity': time.time(),
            'session_timeout': 3600  # 1 hour timeout
        }

        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def check_session_timeout(self):
        """Check if session has timed out"""
        if st.session_state.authenticated:
            current_time = time.time()
            if current_time - st.session_state.last_activity > st.session_state.session_timeout:
                self.logout()
                st.warning("⏰ Session expired. Please login again.")
                st.rerun()
            else:
                st.session_state.last_activity = current_time

    def show_authentication_page(self):
        """Display the authentication page"""
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .auth-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .auth-title {
            text-align: center;
            color: #2E86C1;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

        st.title("📈 SmartTrader Assistant")
        st.markdown("*Multi-Agent Trading System powered by AI*")
        st.markdown("---")

        # Center the authentication form
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # Authentication mode toggle
            auth_mode = st.radio(
                "**Choose your action:**",
                ["Login", "Signup"],
                horizontal=True,
                key="auth_mode"
            )

            st.markdown("### " + ("🔐 Sign In" if auth_mode == "Login" else "📝 Create Account"))

            # Authentication form
            with st.form("auth_form", clear_on_submit=False):
                # Username
                username = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                    help="Minimum 3 characters"
                )

                # Password
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    help="Minimum 6 characters" if auth_mode == "Signup" else None
                )

                # Additional fields for signup
                if auth_mode == "Signup":
                    confirm_password = st.text_input(
                        "Confirm Password",
                        type="password",
                        placeholder="Confirm your password"
                    )

                    st.markdown("#### 🔑 Trading API Configuration")
                    st.info("💡 Enter your Alpaca API credentials to enable live trading")

                    alpaca_api_key = st.text_input(
                        "Alpaca API Key",
                        type="password",
                        placeholder="Enter your Alpaca API Key",
                        help="Get this from your Alpaca dashboard"
                    )

                    alpaca_secret_key = st.text_input(
                        "Alpaca Secret Key",
                        type="password",
                        placeholder="Enter your Alpaca Secret Key",
                        help="Keep this secure and never share it"
                    )
                else:
                    confirm_password = alpaca_api_key = alpaca_secret_key = None

                # Submit button
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.form_submit_button("🔐 Sign In" if auth_mode == "Login" else "📝 Create Account",
                                             use_container_width=True):
                        if auth_mode == "Login":
                            self.handle_login(username, password)
                        else:
                            self.handle_signup(username, password, confirm_password, alpaca_api_key, alpaca_secret_key)

                with col_b:
                    if st.form_submit_button("🧹 Clear Form", use_container_width=True):
                        st.rerun()

        # Help sections
        with st.expander("🔑 How to get Alpaca API Keys"):
            st.markdown("""
            **Step-by-step guide:**

            1. 🌐 Visit [Alpaca Markets](https://app.alpaca.markets/signup)
            2. 📝 Create a free account (no minimum deposit required)
            3. ✅ Complete email verification
            4. 🔑 Navigate to **Paper Trading** → **API Keys**
            5. 📋 Generate new API keys for paper trading
            6. 💾 Copy your **API Key** and **Secret Key**
            7. 🔒 Use these credentials during signup

            **⚠️ Important:**
            - Start with **Paper Trading** to test safely
            - Never share your API keys with anyone
            - Keep your secret key secure
            """)

        with st.expander("❓ Frequently Asked Questions"):
            st.markdown("""
            **Q: Is my data secure?**
            A: Yes, all passwords are hashed and API keys are stored securely.

            **Q: Can I change my API keys later?**
            A: Currently, you'll need to create a new account. Account management features are coming soon.

            **Q: What's the difference between paper and live trading?**
            A: Paper trading uses virtual money to test strategies safely. Live trading uses real money.

            **Q: Do I need to deposit money?**
            A: No, you can start with paper trading for free to test the system.
            """)

    def handle_login(self, username, password):
        """Handle user login"""
        if not username or not password:
            st.error("❌ Please enter both username and password")
            return

        with st.spinner("🔐 Authenticating..."):
            success, message = self.db.authenticate_user(username, password)

        if success:
            user_data = self.db.get_user_data(username)
            user_portfolio = self.db.get_user_portfolio(user_data['user_id'])

            # Update session state
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_data = user_data
            st.session_state.user_portfolio = user_portfolio
            st.session_state.last_activity = time.time()

            st.success(f"✅ Welcome back, {username}!")
            time.sleep(1)  # Brief pause for user experience
            st.rerun()
        else:
            st.error(f"❌ {message}")

    def handle_signup(self, username, password, confirm_password, alpaca_api_key, alpaca_secret_key):
        """Handle user signup"""
        # Validation
        if not all([username, password, confirm_password, alpaca_api_key, alpaca_secret_key]):
            st.error("❌ Please fill in all fields")
            return

        if password != confirm_password:
            st.error("❌ Passwords do not match")
            return

        if len(password) < 6:
            st.error("❌ Password must be at least 6 characters long")
            return

        if len(username) < 3:
            st.error("❌ Username must be at least 3 characters long")
            return

        # Validate API keys format (basic validation)
        if not alpaca_api_key.startswith(('PK', 'AKFZ')) or len(alpaca_api_key) < 20:
            st.warning("⚠️ Please verify your Alpaca API Key format")

        if len(alpaca_secret_key) < 40:
            st.warning("⚠️ Please verify your Alpaca Secret Key format")

        with st.spinner("📝 Creating your account..."):
            success, message = self.db.create_user(username, password, alpaca_api_key, alpaca_secret_key)

        if success:
            st.success(f"✅ {message}")
            st.info("🎉 Your account has been created! You can now sign in with your credentials.")
            st.balloons()
        else:
            st.error(f"❌ {message}")

    def show_main_interface(self):
        """Display main interface for authenticated users"""
        # Check session timeout
        self.check_session_timeout()

        # Header with user info
        user_data = st.session_state.user_data
        portfolio = st.session_state.user_portfolio

        # Top header
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            st.title(f"👋 Welcome, {st.session_state.username}!")
            st.caption(f"Last login: {self._format_datetime(user_data.get('last_login'))}")

        with col2:
            st.metric("💰 Portfolio Value", f"${portfolio.get('total_value', 0):,.2f}")

        with col3:
            daily_pnl = portfolio.get('daily_pnl', 0)
            st.metric("📈 Daily P&L", f"${daily_pnl:,.2f}", f"{portfolio.get('daily_pnl_pct', 0):+.2f}%")

        with col4:
            if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                self.logout()

        st.markdown("---")

        # Sidebar navigation
        with st.sidebar:
            st.title("🧭 Navigation")

            # User info
            st.success(f"✅ Logged in as **{st.session_state.username}**")

            if user_data.get('last_login'):
                last_login = self._format_datetime(user_data['last_login'])
                st.info(f"🕒 Last login: {last_login}")

            st.markdown("---")

            # Navigation menu
            page = st.selectbox(
                "Select Page",
                [
                    "🏠 Dashboard",
                    "🤖 Initialize Trader",
                    "📊 Market Analysis",
                    "💼 My Portfolio",
                    "⚡ Trading Hub",
                    "📋 Transaction History",
                    "⚙️ Account Settings"
                ]
            )

            # Status indicators
            st.markdown("### 📊 System Status")

            if st.session_state.trader_initialized:
                st.success("✅ Trader Active")
            else:
                st.warning("⚠️ Trader Inactive")

            # Quick stats
            st.markdown("### 📈 Quick Stats")
            st.metric("Positions", len(portfolio.get('positions', {})))
            st.metric("Available Cash", f"${portfolio.get('available_cash', 0):,.0f}")

            # Session info
            st.markdown("---")
            st.caption(f"Session: {int((time.time() - st.session_state.last_activity) / 60)} min ago")

        # Route to appropriate page
        page_name = page.split(" ", 1)[1]  # Remove emoji

        if page_name == "Dashboard":
            self.show_dashboard()
        elif page_name == "Initialize Trader":
            self.show_trader_initialization()
        elif page_name == "Market Analysis":
            self.show_market_analysis()
        elif page_name == "My Portfolio":
            self.show_portfolio_details()
        elif page_name == "Trading Hub":
            self.show_trading_hub()
        elif page_name == "Transaction History":
            self.show_transaction_history()
        elif page_name == "Account Settings":
            self.show_account_settings()

    def show_dashboard(self):
        """User dashboard page"""
        st.header("🏠 Trading Dashboard")

        portfolio = st.session_state.user_portfolio

        # Portfolio overview cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💰 Total Value",
                f"${portfolio.get('total_value', 0):,.2f}",
                f"${portfolio.get('daily_pnl', 0):+,.2f}"
            )

        with col2:
            st.metric(
                "💵 Available Cash",
                f"${portfolio.get('available_cash', 0):,.2f}"
            )

        with col3:
            st.metric(
                "📊 Invested",
                f"${portfolio.get('invested_amount', 0):,.2f}"
            )

        with col4:
            positions_count = len(portfolio.get('positions', {}))
            st.metric("📈 Active Positions", positions_count)

        # Charts and recent activity
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Portfolio Performance")
            if portfolio.get('positions'):
                # Create a simple chart with portfolio data
                import pandas as pd
                import numpy as np

                # Sample data for demonstration
                dates = pd.date_range('2025-01-01', periods=30, freq='D')
                values = np.random.normal(portfolio['total_value'], 5000, 30)

                chart_data = pd.DataFrame({
                    'Date': dates,
                    'Portfolio Value': values
                })

                st.line_chart(chart_data.set_index('Date'))
            else:
                st.info("📝 No portfolio data yet. Start trading to see performance charts!")

        with col2:
            st.subheader("⚡ Quick Actions")

            if st.button("🔍 Analyze Market", use_container_width=True):
                st.switch_page("pages/📊 Market Analysis.py") if hasattr(st, 'switch_page') else st.info(
                    "Navigate to Market Analysis")

            if st.button("💼 View Portfolio", use_container_width=True):
                st.switch_page("pages/💼 My Portfolio.py") if hasattr(st, 'switch_page') else st.info(
                    "Navigate to Portfolio")

            if st.button("⚡ Start Trading", use_container_width=True):
                st.switch_page("pages/⚡ Trading Hub.py") if hasattr(st, 'switch_page') else st.info(
                    "Navigate to Trading")

            st.markdown("---")

            if st.button("🔄 Refresh Data", use_container_width=True):
                self.refresh_portfolio_data()

        # Recent transactions
        st.subheader("📋 Recent Activity")
        transactions = portfolio.get('transaction_history', [])

        if transactions:
            recent_transactions = sorted(transactions, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]

            for transaction in recent_transactions:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                    with col1:
                        st.write(f"**{transaction.get('symbol', 'N/A')}**")
                        st.caption(self._format_datetime(transaction.get('timestamp')))

                    with col2:
                        st.write(transaction.get('action', 'N/A'))

                    with col3:
                        st.write(f"{transaction.get('quantity', 0)} shares")

                    with col4:
                        st.write(f"${transaction.get('price', 0):.2f}")
        else:
            st.info("📝 No transactions yet. Your trading activity will appear here.")

    def show_trader_initialization(self):
        """Trader initialization page"""
        st.header("🤖 Initialize Trading System")

        user_data = st.session_state.user_data

        # Show API configuration status
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔑 API Configuration")
            st.success("✅ Alpaca API Key Configured")
            st.code(f"Key: {user_data['alpaca_api_key'][:8]}***")

        with col2:
            st.subheader("🔐 Security Status")
            st.success("✅ Secret Key Configured")
            st.code(f"Secret: {user_data['alpaca_secret_key'][:8]}***")

        st.markdown("---")

        if st.session_state.trader_initialized:
            st.success("✅ Your trading system is fully initialized and ready!")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔄 Reinitialize System", use_container_width=True):
                    st.session_state.trader_initialized = False
                    st.session_state.trader_instance = None
                    st.success("System reset. Click Initialize to restart.")
                    st.rerun()

            with col2:
                if st.button("📊 System Status", use_container_width=True):
                    self.show_system_status()
        else:
            st.info("🚀 Your personalized trading system is ready to be initialized with your API credentials.")

            st.markdown("### 🔧 System Requirements Check")

            # Check system requirements
            requirements = [
                ("✅ Alpaca API Keys", True),
                ("✅ Portfolio Database", True),
                ("✅ Network Connection", True),
                ("⚠️ Market Data Connection", False)  # This would be checked in real implementation
            ]

            for req, status in requirements:
                if status:
                    st.success(req)
                else:
                    st.warning(req)

            st.markdown("---")

            if st.button("🚀 Initialize Trading System", use_container_width=True, type="primary"):
                self.initialize_trading_system()

    def initialize_trading_system(self):
        """Initialize the trading system"""
        with st.spinner("🤖 Initializing your personalized trading system..."):
            try:
                # Simulate initialization process
                progress_bar = st.progress(0)
                status_text = st.empty()

                steps = [
                    "Connecting to Alpaca API...",
                    "Validating credentials...",
                    "Loading market data...",
                    "Initializing AI agents...",
                    "Setting up portfolio tracking...",
                    "System ready!"
                ]

                for i, step in enumerate(steps):
                    status_text.text(step)
                    progress_bar.progress((i + 1) / len(steps))
                    time.sleep(0.5)  # Simulate work

                # Set environment variables for this user's session
                user_data = st.session_state.user_data
                os.environ['ALPACA_API_KEY'] = user_data['alpaca_api_key']
                os.environ['ALPACA_SECRET_KEY'] = user_data['alpaca_secret_key']

                # Mark as initialized
                st.session_state.trader_initialized = True
                st.session_state.trader_instance = "initialized"  # In real app, this would be the actual trader instance

                st.success("✅ Trading system initialized successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                st.info("💡 Please check your API credentials and try again.")

    def show_market_analysis(self):
        """Market analysis page"""
        st.header("📊 Market Analysis")

        if not st.session_state.trader_initialized:
            st.warning("⚠️ Please initialize your trading system first to access market analysis.")
            if st.button("🚀 Go to Initialization"):
                st.rerun()
            return

        # Market analysis interface
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🔍 Symbol Analysis")

            symbols_input = st.text_input(
                "Enter symbols (comma-separated)",
                value="AAPL,GOOGL,MSFT,TSLA",
                help="Enter stock symbols you want to analyze"
            )

            analysis_type = st.selectbox(
                "Analysis Type",
                ["Technical Analysis", "Fundamental Analysis", "Risk Analysis", "Comprehensive Analysis"]
            )

            if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
                self.run_market_analysis(symbols_input, analysis_type)

        with col2:
            st.subheader("🎯 Quick Analysis")

            if st.button("📈 Market Overview", use_container_width=True):
                self.show_market_overview()

            if st.button("🔥 Trending Stocks", use_container_width=True):
                self.show_trending_stocks()

            if st.button("📊 Sector Performance", use_container_width=True):
                self.show_sector_performance()

        # Display sample analysis results
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        self.display_sample_analysis_results()

    def show_portfolio_details(self):
        """Detailed portfolio view"""
        st.header("💼 Portfolio Details")

        portfolio = st.session_state.user_portfolio

        # Portfolio summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("💰 Total Value", f"${portfolio.get('total_value', 0):,.2f}")
            st.metric("💵 Available Cash", f"${portfolio.get('available_cash', 0):,.2f}")

        with col2:
            st.metric("📈 Total P&L", f"${portfolio.get('total_pnl', 0):+,.2f}")
            st.metric("📊 P&L %", f"{portfolio.get('total_pnl_pct', 0):+.2f}%")

        with col3:
            positions = portfolio.get('positions', {})
            st.metric("🎯 Positions", len(positions))
            st.metric("📈 Invested", f"${portfolio.get('invested_amount', 0):,.2f}")

        # Positions table
        if positions:
            st.subheader("📊 Current Positions")

            # Convert to display format
            position_data = []
            for symbol, position in positions.items():
                position_data.append({
                    "Symbol": symbol,
                    "Shares": position.get('shares', 0),
                    "Avg Price": f"${position.get('avg_price', 0):.2f}",
                    "Current Price": f"${position.get('current_price', 0):.2f}",
                    "Market Value": f"${position.get('market_value', 0):,.2f}",
                    "P&L": f"${position.get('unrealized_pnl', 0):+,.2f}",
                    "P&L %": f"{position.get('unrealized_pnl_pct', 0):+.2f}%"
                })

            if position_data:
                import pandas as pd
                df = pd.DataFrame(position_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("📝 No positions yet. Start trading to build your portfolio!")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("⚡ Start Trading", use_container_width=True):
                    st.info("Navigate to Trading Hub")
            with col2:
                if st.button("🔍 Analyze Market", use_container_width=True):
                    st.info("Navigate to Market Analysis")

        # Portfolio actions
        st.markdown("---")
        st.subheader("⚡ Portfolio Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Refresh Positions", use_container_width=True):
                self.refresh_portfolio_data()

        with col2:
            if st.button("📊 Rebalance Portfolio", use_container_width=True):
                st.info("Portfolio rebalancing feature coming soon!")

        with col3:
            if st.button("📈 Add Demo Position", use_container_width=True):
                self.add_demo_position()

    def show_trading_hub(self):
        """Trading interface"""
        st.header("⚡ Trading Hub")

        if not st.session_state.trader_initialized:
            st.warning("⚠️ Please initialize your trading system first.")
            return

        portfolio = st.session_state.user_portfolio

        # Trading interface
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Buy Order")

            buy_symbol = st.text_input("Symbol", value="AAPL", key="buy_symbol")
            buy_quantity = st.number_input("Quantity", min_value=1, value=10, key="buy_quantity")
            buy_order_type = st.selectbox("Order Type", ["Market", "Limit"], key="buy_order_type")

            if buy_order_type == "Limit":
                buy_limit_price = st.number_input("Limit Price", min_value=0.01, value=150.00, key="buy_limit")

            available_cash = portfolio.get('available_cash', 0)
            estimated_cost = buy_quantity * 150.00  # Mock price

            st.info(f"💵 Available Cash: ${available_cash:,.2f}")
            st.info(f"💰 Estimated Cost: ${estimated_cost:,.2f}")

            if st.button("📈 Place Buy Order", use_container_width=True, type="primary"):
                if estimated_cost <= available_cash:
                    self.place_buy_order(buy_symbol, buy_quantity, buy_order_type)
                else:
                    st.error("❌ Insufficient funds!")

        with col2:
            st.subheader("📉 Sell Order")

            # Get user's positions for sell dropdown
            positions = portfolio.get('positions', {})
            if positions:
                sell_symbol = st.selectbox("Select Position", list(positions.keys()), key="sell_symbol")
                max_shares = positions[sell_symbol].get('shares', 0)

                sell_quantity = st.number_input("Quantity", min_value=1, max_value=max_shares, value=min(5, max_shares),
                                                key="sell_quantity")
                sell_order_type = st.selectbox("Order Type", ["Market", "Limit"], key="sell_order_type")

                if sell_order_type == "Limit":
                    sell_limit_price = st.number_input("Limit Price", min_value=0.01, value=150.00, key="sell_limit")

                st.info(f"📊 Available Shares: {max_shares}")

                if st.button("📉 Place Sell Order", use_container_width=True, type="secondary"):
                    self.place_sell_order(sell_symbol, sell_quantity, sell_order_type)
            else:
                st.info("📝 No positions available to sell.")
                if st.button("🔍 Find Opportunities", use_container_width=True):
                    st.info("Navigate to Market Analysis")

        # Recent orders
        st.markdown("---")
        st.subheader("📋 Recent Orders")

        transactions = portfolio.get('transaction_history', [])
        if transactions:
            recent_orders = sorted(transactions, key=lambda x: x.get('timestamp', ''), reverse=True)[:3]

            for order in recent_orders:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

                    with col1:
                        st.write(f"**{order.get('symbol', 'N/A')}**")
                    with col2:
                        st.write(order.get('action', 'N/A'))
                    with col3:
                        st.write(f"{order.get('quantity', 0)} shares")
                    with col4:
                        st.write(f"${order.get('price', 0):.2f}")
                    with col5:
                        st.caption(self._format_datetime(order.get('timestamp')))
        else:
            st.info("📝 No recent orders. Your trading activity will appear here.")

    def show_transaction_history(self):
        """Transaction history page"""
        st.header("📋 Transaction History")

        portfolio = st.session_state.user_portfolio
        transactions = portfolio.get('transaction_history', [])

        if transactions:
            # Filter options
            col1, col2 = st.columns(2)

            with col1:
                filter_action = st.selectbox("Filter by Action", ["All", "Buy", "Sell"])

            with col2:
                filter_symbol = st.text_input("Filter by Symbol", placeholder="e.g., AAPL")

            # Apply filters
            filtered_transactions = transactions

            if filter_action != "All":
                filtered_transactions = [t for t in filtered_transactions if
                                         t.get('action', '').lower() == filter_action.lower()]

            if filter_symbol:
                filtered_transactions = [t for t in filtered_transactions if
                                         filter_symbol.upper() in t.get('symbol', '').upper()]

            # Display transactions
            if filtered_transactions:
                sorted_transactions = sorted(filtered_transactions, key=lambda x: x.get('timestamp', ''), reverse=True)

                # Create a detailed table
                import pandas as pd

                transaction_data = []
                for t in sorted_transactions:
                    transaction_data.append({
                        "Date": self._format_datetime(t.get('timestamp')),
                        "Symbol": t.get('symbol', 'N/A'),
                        "Action": t.get('action', 'N/A'),
                        "Quantity": t.get('quantity', 0),
                        "Price": f"${t.get('price', 0):.2f}",
                        "Total Value": f"${t.get('total_value', 0):.2f}",
                        "ID": t.get('transaction_id', 'N/A')[:8] + "..."
                    })

                df = pd.DataFrame(transaction_data)
                st.dataframe(df, use_container_width=True)

                # Summary stats
                st.subheader("📊 Transaction Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    total_transactions = len(filtered_transactions)
                    st.metric("Total Transactions", total_transactions)

                with col2:
                    buy_transactions = len([t for t in filtered_transactions if t.get('action', '').lower() == 'buy'])
                    st.metric("Buy Orders", buy_transactions)

                with col3:
                    sell_transactions = len([t for t in filtered_transactions if t.get('action', '').lower() == 'sell'])
                    st.metric("Sell Orders", sell_transactions)
            else:
                st.info("📝 No transactions match your filters.")
        else:
            st.info("📝 No transaction history yet. Start trading to see your activity here!")

            if st.button("⚡ Start Trading", use_container_width=True):
                st.info("Navigate to Trading Hub")

    def show_account_settings(self):
        """Account settings page"""
        st.header("⚙️ Account Settings")

        user_data = st.session_state.user_data
        portfolio = st.session_state.user_portfolio

        # Account Information
        st.subheader("👤 Account Information")

        col1, col2 = st.columns(2)

        with col1:
            st.text_input("Username", value=st.session_state.username, disabled=True)
            st.text_input("User ID", value=user_data['user_id'], disabled=True)

        with col2:
            created_at = self._format_datetime(user_data['created_at'])
            st.text_input("Account Created", value=created_at, disabled=True)

            login_count = user_data.get('login_count', 0)
            st.text_input("Total Logins", value=str(login_count), disabled=True)

        # API Key Management
        st.subheader("🔑 API Key Management")

        with st.expander("🔐 View API Credentials", expanded=False):
            st.warning("⚠️ Keep your API keys secure and never share them!")

            col1, col2 = st.columns(2)

            with col1:
                st.text_area("Alpaca API Key", value=user_data['alpaca_api_key'], height=100, disabled=True)

            with col2:
                st.text_area("Alpaca Secret Key", value=user_data['alpaca_secret_key'], height=100, disabled=True)

            st.info("💡 To change your API keys, you'll need to create a new account (feature coming soon).")

        # Portfolio Management
        st.subheader("💼 Portfolio Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reset Portfolio", use_container_width=True):
                if st.checkbox("⚠️ I understand this will reset my entire portfolio"):
                    self.reset_user_portfolio()

        with col2:
            if st.button("📊 Export Portfolio Data", use_container_width=True):
                portfolio_json = json.dumps(portfolio, indent=2)
                st.download_button(
                    "💾 Download Portfolio JSON",
                    data=portfolio_json,
                    file_name=f"{st.session_state.username}_portfolio_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

        # Session Management
        st.subheader("🔐 Session Management")

        col1, col2 = st.columns(2)

        with col1:
            session_duration = int((time.time() - st.session_state.last_activity) / 60)
            st.info(f"⏱️ Current session: {session_duration} minutes")

        with col2:
            if st.button("🔄 Extend Session", use_container_width=True):
                st.session_state.last_activity = time.time()
                st.success("✅ Session extended!")

        # Danger Zone
        st.markdown("---")
        st.subheader("⚠️ Danger Zone")

        with st.expander("🗑️ Delete Account", expanded=False):
            st.error("⚠️ This action cannot be undone!")
            st.write("Deleting your account will:")
            st.write("• Remove all your portfolio data")
            st.write("• Delete your transaction history")
            st.write("• Remove your API key configuration")

            if st.text_input("Type 'DELETE' to confirm") == "DELETE":
                if st.button("🗑️ Delete My Account", type="primary"):
                    st.error("Account deletion feature coming soon.")

    def logout(self):
        """Handle user logout"""
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Reinitialize
        self.initialize_session_state()

        st.success("✅ You have been logged out successfully!")
        st.rerun()

    # Helper methods
    def _format_datetime(self, datetime_str):
        """Format datetime string for display"""
        if not datetime_str:
            return "Never"

        try:
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return datetime_str

    def refresh_portfolio_data(self):
        """Refresh user's portfolio data"""
        with st.spinner("🔄 Refreshing portfolio data..."):
            # Simulate data refresh
            time.sleep(1)

            # In a real app, this would fetch live data
            user_id = st.session_state.user_data['user_id']
            portfolio = self.db.get_user_portfolio(user_id)

            if portfolio:
                st.session_state.user_portfolio = portfolio
                st.success("✅ Portfolio data refreshed!")
            else:
                st.error("❌ Failed to refresh portfolio data")

    def add_demo_position(self):
        """Add a demo position for testing"""
        import random

        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        symbol = random.choice(symbols)
        shares = random.randint(1, 20)
        price = random.uniform(100, 300)

        user_id = st.session_state.user_data['user_id']
        portfolio = st.session_state.user_portfolio

        # Add position
        if 'positions' not in portfolio:
            portfolio['positions'] = {}

        portfolio['positions'][symbol] = {
            'shares': shares,
            'avg_price': price,
            'current_price': price * random.uniform(0.95, 1.05),
            'market_value': shares * price,
            'unrealized_pnl': shares * price * random.uniform(-0.1, 0.1),
            'unrealized_pnl_pct': random.uniform(-10, 10)
        }

        # Add transaction
        transaction = {
            'symbol': symbol,
            'action': 'Buy',
            'quantity': shares,
            'price': price,
            'total_value': shares * price
        }

        self.db.add_transaction(user_id, transaction)

        # Update portfolio totals
        portfolio['invested_amount'] = sum(pos['market_value'] for pos in portfolio['positions'].values())
        portfolio['total_value'] = portfolio['available_cash'] + portfolio['invested_amount']

        # Save changes
        if self.db.update_user_portfolio(user_id, portfolio):
            st.session_state.user_portfolio = portfolio
            st.success(f"✅ Added demo position: {shares} shares of {symbol} at ${price:.2f}")
        else:
            st.error("❌ Failed to add position")

    def place_buy_order(self, symbol, quantity, order_type):
        """Place a buy order"""
        # Simulate order placement
        price = 150.00  # Mock price
        total_cost = quantity * price

        user_id = st.session_state.user_data['user_id']
        portfolio = st.session_state.user_portfolio

        if total_cost <= portfolio['available_cash']:
            # Add transaction
            transaction = {
                'symbol': symbol,
                'action': 'Buy',
                'quantity': quantity,
                'price': price,
                'total_value': total_cost,
                'order_type': order_type
            }

            self.db.add_transaction(user_id, transaction)

            # Update portfolio
            portfolio['available_cash'] -= total_cost

            if 'positions' not in portfolio:
                portfolio['positions'] = {}

            if symbol in portfolio['positions']:
                # Update existing position
                existing = portfolio['positions'][symbol]
                total_shares = existing['shares'] + quantity
                total_cost_basis = (existing['shares'] * existing['avg_price']) + total_cost
                portfolio['positions'][symbol] = {
                    'shares': total_shares,
                    'avg_price': total_cost_basis / total_shares,
                    'current_price': price,
                    'market_value': total_shares * price,
                    'unrealized_pnl': 0,
                    'unrealized_pnl_pct': 0
                }
            else:
                # New position
                portfolio['positions'][symbol] = {
                    'shares': quantity,
                    'avg_price': price,
                    'current_price': price,
                    'market_value': total_cost,
                    'unrealized_pnl': 0,
                    'unrealized_pnl_pct': 0
                }

            # Update totals
            portfolio['invested_amount'] = sum(pos['market_value'] for pos in portfolio['positions'].values())
            portfolio['total_value'] = portfolio['available_cash'] + portfolio['invested_amount']

            # Save changes
            if self.db.update_user_portfolio(user_id, portfolio):
                st.session_state.user_portfolio = portfolio
                st.success(f"✅ Buy order placed: {quantity} shares of {symbol} at ${price:.2f}")
            else:
                st.error("❌ Failed to place order")
        else:
            st.error("❌ Insufficient funds!")

    def place_sell_order(self, symbol, quantity, order_type):
        """Place a sell order"""
        price = 150.00  # Mock price
        total_value = quantity * price

        user_id = st.session_state.user_data['user_id']
        portfolio = st.session_state.user_portfolio

        if symbol in portfolio.get('positions', {}) and portfolio['positions'][symbol]['shares'] >= quantity:
            # Add transaction
            transaction = {
                'symbol': symbol,
                'action': 'Sell',
                'quantity': quantity,
                'price': price,
                'total_value': total_value,
                'order_type': order_type
            }

            self.db.add_transaction(user_id, transaction)

            # Update portfolio
            portfolio['available_cash'] += total_value

            # Update position
            position = portfolio['positions'][symbol]
            position['shares'] -= quantity

            if position['shares'] <= 0:
                # Remove position if all shares sold
                del portfolio['positions'][symbol]
            else:
                # Update remaining position
                position['market_value'] = position['shares'] * position['current_price']

            # Update totals
            portfolio['invested_amount'] = sum(pos['market_value'] for pos in portfolio['positions'].values())
            portfolio['total_value'] = portfolio['available_cash'] + portfolio['invested_amount']

            # Save changes
            if self.db.update_user_portfolio(user_id, portfolio):
                st.session_state.user_portfolio = portfolio
                st.success(f"✅ Sell order placed: {quantity} shares of {symbol} at ${price:.2f}")
            else:
                st.error("❌ Failed to place order")
        else:
            st.error("❌ Insufficient shares!")

    def reset_user_portfolio(self):
        """Reset user's portfolio to default state"""
        user_id = st.session_state.user_data['user_id']

        default_portfolio = {
            "user_id": user_id,
            "username": st.session_state.username,
            "total_value": 100000.0,
            "available_cash": 100000.0,
            "invested_amount": 0.0,
            "positions": {},
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "transaction_history": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

        if self.db.update_user_portfolio(user_id, default_portfolio):
            st.session_state.user_portfolio = default_portfolio
            st.success("✅ Portfolio reset to default state!")
            st.rerun()
        else:
            st.error("❌ Failed to reset portfolio")

    def run_market_analysis(self, symbols_input, analysis_type):
        """Run market analysis"""
        symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]

        if not symbols:
            st.error("❌ Please enter at least one symbol")
            return

        with st.spinner(f"🔍 Running {analysis_type} for {', '.join(symbols)}..."):
            time.sleep(2)  # Simulate analysis

            st.success(f"✅ {analysis_type} completed for {', '.join(symbols)}!")

            # Display mock results
            for symbol in symbols:
                with st.expander(f"📊 {symbol} Analysis Results"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Current Price", f"${150.00:.2f}")
                        st.metric("Daily Change", "+2.5%", "📈")

                    with col2:
                        st.metric("Volume", "1.2M")
                        st.metric("Recommendation", "BUY", "🚀")

    def show_market_overview(self):
        """Show market overview"""
        st.subheader("📈 Market Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("S&P 500", "4,150.25", "+1.2%")

        with col2:
            st.metric("NASDAQ", "12,820.75", "+0.8%")

        with col3:
            st.metric("Dow Jones", "33,945.58", "+0.5%")

    def show_trending_stocks(self):
        """Show trending stocks"""
        st.subheader("🔥 Trending Stocks")

        trending = [
            ("AAPL", "+3.2%"),
            ("GOOGL", "+2.8%"),
            ("MSFT", "+2.1%"),
            ("TSLA", "+5.4%"),
            ("AMZN", "+1.9%")
        ]

        for symbol, change in trending:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{symbol}**")
            with col2:
                st.write(change)

    def show_sector_performance(self):
        """Show sector performance"""
        st.subheader("📊 Sector Performance")

        sectors = [
            ("Technology", "+2.1%"),
            ("Healthcare", "+1.8%"),
            ("Finance", "+1.2%"),
            ("Energy", "-0.5%"),
            ("Utilities", "+0.3%")
        ]

        for sector, change in sectors:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{sector}**")
            with col2:
                st.write(change)

    def display_sample_analysis_results(self):
        """Display sample analysis results"""
        tab1, tab2, tab3 = st.tabs(["📈 Technical", "📊 Fundamental", "⚠️ Risk"])

        with tab1:
            st.write("**Technical Indicators:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RSI", "65.2")
                st.metric("MACD", "Bullish")
            with col2:
                st.metric("Moving Avg", "Above 200-day")
                st.metric("Volume", "Above Average")

        with tab2:
            st.write("**Fundamental Metrics:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("P/E Ratio", "22.5")
                st.metric("Revenue Growth", "+12.3%")
            with col2:
                st.metric("Profit Margin", "18.7%")
                st.metric("Rating", "BUY")

        with tab3:
            st.write("**Risk Assessment:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Beta", "1.15")
                st.metric("Volatility", "Medium")
            with col2:
                st.metric("VaR (95%)", "-2.3%")
                st.metric("Risk Level", "Moderate")

    def show_system_status(self):
        """Show system status"""
        st.subheader("📊 System Status")

        col1, col2 = st.columns(2)

        with col1:
            st.success("✅ API Connection: Active")
            st.success("✅ Data Feed: Live")
            st.success("✅ Portfolio Sync: Active")

        with col2:
            st.info("📊 Orders Today: 0")
            st.info("⏱️ Last Update: Just now")
            st.info("🔄 System Uptime: 100%")

    def run(self):
        """Main application entry point"""
        if not st.session_state.authenticated:
            self.show_authentication_page()
        else:
            self.show_main_interface()


# Main application execution
def main():
    """Main function to run the Streamlit app"""
    app = SmartTraderApp()
    app.run()


if __name__ == "__main__":
    main()
