import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta


def app():
    st.title("📊 Market Analysis")

    # Check if user is authenticated
    if not st.session_state.get('authenticated', False):
        st.warning("Please login first to access market analysis.")
        return

    # Sidebar for analysis options
    st.sidebar.header("Analysis Options")

    # Symbol input
    symbol_input = st.sidebar.text_input(
        "Enter Symbol(s)",
        value="AAPL,MSFT,GOOGL",
        help="Enter stock symbols separated by commas"
    )

    # Parse symbols
    symbols = [s.strip().upper() for s in symbol_input.split(',') if s.strip()]

    # Time period selection
    time_period = st.sidebar.selectbox(
        "Time Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
        index=5
    )

    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Price Chart", "Technical Indicators", "Comparison", "Fundamentals"],
        index=0
    )

    # Run analysis button
    if st.sidebar.button("🔍 Run Analysis", use_container_width=True):
        with st.spinner(f"Analyzing {', '.join(symbols)}..."):
            if analysis_type == "Price Chart":
                display_price_chart(symbols, time_period)
            elif analysis_type == "Technical Indicators":
                display_technical_indicators(symbols[0] if symbols else "AAPL", time_period)
            elif analysis_type == "Comparison":
                display_comparison(symbols, time_period)
            elif analysis_type == "Fundamentals":
                display_fundamentals(symbols[0] if symbols else "AAPL")
    else:
        # Default view
        st.info("Select your analysis options and click 'Run Analysis' to begin.")

        # Market overview
        st.subheader("Market Overview")
        display_market_overview()


def display_price_chart(symbols, time_period):
    """Display price chart for selected symbols"""
    if not symbols:
        st.warning("Please enter at least one symbol")
        return

    for symbol in symbols:
        st.subheader(f"{symbol} - Price Chart")

        try:
            # Get data from yfinance
            data = yf.download(symbol, period=time_period)

            if data.empty:
                st.warning(f"No data available for {symbol}")
                continue

            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol
            )])

            fig.update_layout(
                title=f"{symbol} Price Movement",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display summary statistics
            st.write("**Summary Statistics:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                current_price = data['Close'].iloc[-1]
                previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                price_change = current_price - previous_price
                price_change_pct = (price_change / previous_price) * 100 if previous_price > 0 else 0

                st.metric(
                    label="Current Price",
                    value=f"${current_price:.2f}",
                    delta=f"{price_change_pct:.2f}%"
                )

            with col2:
                st.metric(
                    label="Volume",
                    value=f"{data['Volume'].iloc[-1]:,.0f}"
                )

            with col3:
                st.metric(
                    label="52-Week Range",
                    value=f"${data['Low'].min():.2f} - ${data['High'].max():.2f}"
                )

        except Exception as e:
            st.error(f"Error analyzing {symbol}: {str(e)}")


def display_technical_indicators(symbol, time_period):
    """Display technical indicators for a symbol"""
    if not symbol:
        st.warning("Please enter a symbol")
        return

    st.subheader(f"{symbol} - Technical Indicators")

    try:
        # Get data from yfinance
        data = yf.download(symbol, period=time_period)

        if data.empty:
            st.warning(f"No data available for {symbol}")
            return

        # Calculate technical indicators
        # Moving Averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()

        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Create price chart with moving averages
        fig1 = go.Figure()

        fig1.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price'
        ))

        fig1.add_trace(go.Scatter(
            x=data.index,
            y=data['MA20'],
            mode='lines',
            name='20-Day MA',
            line=dict(color='orange')
        ))

        fig1.add_trace(go.Scatter(
            x=data.index,
            y=data['MA50'],
            mode='lines',
            name='50-Day MA',
            line=dict(color='green')
        ))

        fig1.add_trace(go.Scatter(
            x=data.index,
            y=data['MA200'],
            mode='lines',
            name='200-Day MA',
            line=dict(color='red')
        ))

        fig1.update_layout(
            title=f"{symbol} Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Create RSI chart
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI'
        ))

        # Add overbought/oversold lines
        fig2.add_shape(
            type="line",
            x0=data.index[0],
            y0=70,
            x1=data.index[-1],
            y1=70,
            line=dict(color="red", width=2, dash="dash")
        )

        fig2.add_shape(
            type="line",
            x0=data.index[0],
            y0=30,
            x1=data.index[-1],
            y1=30,
            line=dict(color="green", width=2, dash="dash")
        )

        fig2.update_layout(
            title=f"{symbol} RSI (Relative Strength Index)",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Technical analysis summary
        st.subheader("Technical Analysis Summary")

        # Current values
        current_price = data['Close'].iloc[-1]
        current_ma20 = data['MA20'].iloc[-1]
        current_ma50 = data['MA50'].iloc[-1]
        current_ma200 = data['MA200'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]

        # Generate signals
        ma_signal = "Bullish" if current_price > current_ma50 else "Bearish"
        ma_trend = "Bullish" if current_ma20 > current_ma50 else "Bearish"
        rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MA Signal", ma_signal)

        with col2:
            st.metric("MA Trend", ma_trend)

        with col3:
            st.metric("RSI Signal", rsi_signal, f"{current_rsi:.1f}")

    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")


def display_comparison(symbols, time_period):
    """Display comparison chart for multiple symbols"""
    if not symbols or len(symbols) < 2:
        st.warning("Please enter at least two symbols for comparison")
        return

    st.subheader(f"Comparison: {', '.join(symbols)}")

    try:
        # Get data from yfinance
        data = yf.download(symbols, period=time_period)['Close']

        if data.empty:
            st.warning("No data available for the selected symbols")
            return

        # Normalize data to 100 for better comparison
        normalized_data = data.copy()
        for symbol in symbols:
            normalized_data[symbol] = normalized_data[symbol] / normalized_data[symbol].iloc[0] * 100

        # Create comparison chart
        fig = go.Figure()

        for symbol in symbols:
            fig.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[symbol],
                mode='lines',
                name=symbol
            ))

        fig.update_layout(
            title="Normalized Price Comparison (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Performance comparison table
        st.subheader("Performance Comparison")

        performance_data = []
        for symbol in symbols:
            symbol_data = data[symbol]
            current_price = symbol_data.iloc[-1]
            start_price = symbol_data.iloc[0]
            change_pct = ((current_price - start_price) / start_price) * 100

            performance_data.append({
                "Symbol": symbol,
                "Start Price": f"${start_price:.2f}",
                "Current Price": f"${current_price:.2f}",
                "Change %": f"{change_pct:.2f}%"
            })

        st.table(pd.DataFrame(performance_data))

    except Exception as e:
        st.error(f"Error comparing symbols: {str(e)}")


def display_fundamentals(symbol):
    """Display fundamental data for a symbol"""
    if not symbol:
        st.warning("Please enter a symbol")
        return

    st.subheader(f"{symbol} - Fundamental Analysis")

    try:
        # Get data from yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Company info
        st.write("**Company Information**")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Name:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Country:** {info.get('country', 'N/A')}")

        with col2:
            st.write(f"**Market Cap:** ${info.get('marketCap', 0):,.0f}")
            st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            st.write(f"**Dividend Yield:** {info.get('dividendYield', 0) * 100:.2f}%")
            st.write(
                f"**52 Week Range:** ${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}")

        # Financial metrics
        st.write("**Financial Metrics**")

        metrics = [
            {"name": "Revenue (TTM)", "value": info.get('totalRevenue', 0)},
            {"name": "Profit Margin", "value": info.get('profitMargins', 0)},
            {"name": "Operating Margin", "value": info.get('operatingMargins', 0)},
            {"name": "Return on Equity", "value": info.get('returnOnEquity', 0)},
            {"name": "Return on Assets", "value": info.get('returnOnAssets', 0)},
            {"name": "Debt to Equity", "value": info.get('debtToEquity', 0)}
        ]

        metrics_df = pd.DataFrame(metrics)
        metrics_df["value"] = metrics_df["value"].apply(
            lambda x: f"${x:,.0f}" if "Revenue" in metrics_df.loc[metrics_df["value"] == x, "name"].values[
                0] else f"{x:.2%}")

        st.table(metrics_df)

        # Business summary
        st.write("**Business Summary**")
        st.write(info.get('longBusinessSummary', 'No business summary available.'))

    except Exception as e:
        st.error(f"Error fetching fundamental data for {symbol}: {str(e)}")


def display_market_overview():
    """Display market overview with major indices"""
    try:
        # Major indices
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
        index_names = ['S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000']

        # Get data
        data = yf.download(indices, period="1d")['Close']
        prev_data = yf.download(indices, period="2d")['Close'].iloc[0]

        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]

        for i, (index, name) in enumerate(zip(indices, index_names)):
            current = data[index].iloc[-1]
            previous = prev_data[index]
            change = current - previous
            change_pct = (change / previous) * 100

            with cols[i]:
                st.metric(
                    label=name,
                    value=f"{current:.2f}",
                    delta=f"{change_pct:.2f}%"
                )

        # Market heatmap (simplified)
        st.subheader("Sector Performance")

        # Simulate sector data
        sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer Cyclical', 'Industrials', 'Energy']
        performances = [2.1, 1.5, -0.5, 0.8, -0.2, 1.2]  # Simulated performance data

        sector_df = pd.DataFrame({
            'Sector': sectors,
            'Performance': performances
        })

        fig = px.bar(
            sector_df,
            x='Sector',
            y='Performance',
            title="Sector Performance (%)",
            color='Performance',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[-3, 3]
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying market overview: {str(e)}")


if __name__ == "__main__":
    app()
