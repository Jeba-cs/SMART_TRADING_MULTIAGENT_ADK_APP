import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime


def plot_price_chart(symbol: str, price_data: List[Dict], title: str = None, chart_type: str = "candlestick"):
    """
    Plot interactive price chart for a given symbol using Plotly

    Args:
        symbol: Stock symbol
        price_data: List of dicts with keys: timestamp, open, high, low, close, volume
        title: Optional chart title
        chart_type: Type of chart ('candlestick', 'line', 'ohlc')

    Usage:
        from ui.components.price_chart import plot_price_chart
        plot_price_chart("AAPL", price_data, chart_type="candlestick")
    """
    if not price_data:
        st.warning(f"No price data available for {symbol}")
        return

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(price_data)

    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create the main price chart
    fig = go.Figure()

    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ))
    elif chart_type == "line":
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['close'],
            mode='lines',
            name=f"{symbol} Close Price",
            line=dict(color='#1f77b4', width=2)
        ))
    elif chart_type == "ohlc":
        fig.add_trace(go.Ohlc(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        ))

    # Add volume subplot
    if 'volume' in df.columns:
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.6)',
            marker_line_color='rgba(8,48,107,1.0)',
            marker_line_width=1
        ))

        fig_volume.update_layout(
            title=f"{symbol} Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_dark",
            height=200
        )

    # Update layout
    fig.update_layout(
        title=title or f"{symbol} Price Chart ({chart_type.title()})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )

    # Display charts
    st.plotly_chart(fig, use_container_width=True)

    if 'volume' in df.columns:
        st.plotly_chart(fig_volume, use_container_width=True)


def plot_technical_indicators(symbol: str, price_data: List[Dict], indicators: Dict[str, List[float]]):
    """
    Plot technical indicators alongside price data

    Args:
        symbol: Stock symbol
        price_data: Price data
        indicators: Dict of indicator names and values
    """
    if not price_data or not indicators:
        st.warning("No data available for technical indicators")
        return

    df = pd.DataFrame(price_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    fig = go.Figure()

    # Add price line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='white', width=2)
    ))

    # Add indicators
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
    for i, (name, values) in enumerate(indicators.items()):
        if len(values) == len(df):
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=values,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=1.5)
            ))

    fig.update_layout(
        title=f"{symbol} Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_comparison_chart(symbols_data: Dict[str, List[Dict]], normalize: bool = True):
    """
    Plot comparison chart for multiple symbols

    Args:
        symbols_data: Dict of symbol -> price_data
        normalize: Whether to normalize prices to 100 base
    """
    if not symbols_data:
        st.warning("No data available for comparison")
        return

    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for i, (symbol, price_data) in enumerate(symbols_data.items()):
        if not price_data:
            continue

        df = pd.DataFrame(price_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        prices = df['close']
        if normalize:
            prices = (prices / prices.iloc[0]) * 100

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=prices,
            mode='lines',
            name=symbol,
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    y_title = "Normalized Price (Base 100)" if normalize else "Price (USD)"

    fig.update_layout(
        title="Symbol Comparison Chart",
        xaxis_title="Date",
        yaxis_title=y_title,
        template="plotly_dark",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)
