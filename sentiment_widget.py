import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from typing import Dict, Any, List
import pandas as pd


def display_sentiment_summary(sentiment_data: Dict[str, Any], symbol: str):
    """
    Display a comprehensive sentiment analysis summary for a given symbol

    Args:
        sentiment_data: Dictionary containing sentiment scores and metadata
        symbol: Stock symbol

    Usage:
        from ui.components.sentiment_widget import display_sentiment_summary
        display_sentiment_summary(sentiment_data, "TSLA")
    """
    if not sentiment_data:
        st.warning("No sentiment data available")
        return

    symbol_sentiment = sentiment_data.get('symbol_sentiment', {}).get(symbol, {})
    news_sentiment = sentiment_data.get('sentiment_by_symbol', {}).get(symbol, {})

    st.subheader(f"📊 Sentiment Analysis for {symbol}")

    # Main sentiment metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📰 News Sentiment")
        news_score = news_sentiment.get('weighted_sentiment', 0)
        news_confidence = news_sentiment.get('confidence', 0)

        # Color-code sentiment
        color = "normal"
        if news_score > 0.1:
            color = "normal"
        elif news_score < -0.1:
            color = "inverse"

        st.metric(
            "Sentiment Score",
            f"{news_score:.2f}",
            delta=f"Confidence: {news_confidence:.1%}",
            delta_color=color
        )

        # News article count
        article_count = news_sentiment.get('article_count', 0)
        st.metric("Articles Analyzed", article_count)

    with col2:
        st.markdown("### 💬 Social Media Sentiment")
        social_score = symbol_sentiment.get('sentiment_score', 0)
        social_confidence = symbol_sentiment.get('confidence', 0)

        color = "normal"
        if social_score > 0.1:
            color = "normal"
        elif social_score < -0.1:
            color = "inverse"

        st.metric(
            "Sentiment Score",
            f"{social_score:.2f}",
            delta=f"Confidence: {social_confidence:.1%}",
            delta_color=color
        )

        # Post count
        post_count = symbol_sentiment.get('post_count', 0)
        st.metric("Posts Analyzed", post_count)

    with col3:
        st.markdown("### 🎯 Overall Sentiment")
        overall_score = (news_score + social_score) / 2
        overall_confidence = (news_confidence + social_confidence) / 2

        # Determine sentiment label
        if overall_score > 0.2:
            sentiment_label = "🟢 Bullish"
        elif overall_score > 0.05:
            sentiment_label = "🟡 Slightly Bullish"
        elif overall_score > -0.05:
            sentiment_label = "⚪ Neutral"
        elif overall_score > -0.2:
            sentiment_label = "🟡 Slightly Bearish"
        else:
            sentiment_label = "🔴 Bearish"

        st.metric("Overall Sentiment", sentiment_label)
        st.metric("Combined Score", f"{overall_score:.2f}")

    # Sentiment breakdown visualization
    breakdown = symbol_sentiment.get('sentiment_breakdown', {})
    if breakdown:
        st.markdown("#### 📈 Sentiment Breakdown")

        # Create pie chart for sentiment distribution
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(breakdown.keys()),
            values=list(breakdown.values()),
            hole=0.3,
            marker_colors=['#ff6b6b', '#ffd93d', '#6bcf7f']
        )])

        fig_pie.update_layout(
            title="Social Media Sentiment Distribution",
            template="plotly_dark",
            height=300
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Show raw numbers
            st.markdown("**Sentiment Counts:**")
            for sentiment, count in breakdown.items():
                st.write(f"• {sentiment.title()}: {count}")


def display_sentiment_trend(sentiment_history: List[Dict[str, Any]], symbol: str):
    """
    Display sentiment trend over time

    Args:
        sentiment_history: List of sentiment data over time
        symbol: Stock symbol
    """
    if not sentiment_history:
        st.warning("No sentiment trend data available")
        return

    st.subheader(f"📈 Sentiment Trend for {symbol}")

    df = pd.DataFrame(sentiment_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    fig = go.Figure()

    # Add news sentiment trend
    if 'news_sentiment' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['news_sentiment'],
            mode='lines+markers',
            name='News Sentiment',
            line=dict(color='#1f77b4', width=2)
        ))

    # Add social sentiment trend
    if 'social_sentiment' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['social_sentiment'],
            mode='lines+markers',
            name='Social Media Sentiment',
            line=dict(color='#ff7f0e', width=2)
        ))

    # Add neutral line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"Sentiment Trend for {symbol}",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def display_sentiment_heatmap(multi_symbol_sentiment: Dict[str, Dict[str, float]]):
    """
    Display sentiment heatmap for multiple symbols

    Args:
        multi_symbol_sentiment: Dict of symbol -> sentiment scores
    """
    if not multi_symbol_sentiment:
        st.warning("No multi-symbol sentiment data available")
        return

    st.subheader("🗺️ Sentiment Heatmap")

    # Prepare data for heatmap
    symbols = list(multi_symbol_sentiment.keys())
    sentiment_types = ['news_sentiment', 'social_sentiment', 'overall_sentiment']

    heatmap_data = []
    for sentiment_type in sentiment_types:
        row = []
        for symbol in symbols:
            score = multi_symbol_sentiment.get(symbol, {}).get(sentiment_type, 0)
            row.append(score)
        heatmap_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=symbols,
        y=['News', 'Social Media', 'Overall'],
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title="Sentiment Score")
    ))

    fig.update_layout(
        title="Sentiment Heatmap Across Symbols",
        template="plotly_dark",
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)
