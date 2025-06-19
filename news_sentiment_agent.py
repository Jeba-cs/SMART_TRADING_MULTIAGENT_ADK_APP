import asyncio
import logging
import os  # Added missing import
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import aiohttp
import json
from textblob import TextBlob
import re

# Import with proper error handling
try:
    from google.adk.agents import Agent
    from google.adk.tools import Tool
except ImportError:
    # Fallback implementation
    class Tool:
        def __init__(self, name=None, description=None):
            self.name = name
            self.description = description

        async def call(self, *args, **kwargs):
            raise NotImplementedError("Subclasses must implement call method")


    class Agent:
        def __init__(self, model=None, name=None, description=None, instructions=None, tools=None):
            self.model = model
            self.name = name
            self.description = description
            self.instructions = instructions
            self.tools = tools or []

logger = logging.getLogger(__name__)


class NewsSentimentTool(Tool):
    """Tool for collecting and analyzing news sentiment"""

    def __init__(self):
        super().__init__(
            name="news_sentiment_analyzer",
            description="Collect financial news and analyze sentiment for market impact"
        )

        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')

    async def call(self, symbols: List[str], lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Collect and analyze news sentiment for given symbols
        Args:
            symbols: List of stock symbols to analyze
            lookback_hours: How many hours back to look for news
        Returns:
            News sentiment analysis results
        """
        try:
            logger.info(f"Collecting news sentiment for symbols: {symbols}")
            all_news = []
            sentiment_results = {}

            async with aiohttp.ClientSession() as session:
                # Collect news from multiple sources
                tasks = []
                for symbol in symbols:
                    # NewsAPI
                    if self.news_api_key:
                        tasks.append(self._fetch_newsapi_data(session, symbol, lookback_hours))
                    # Finnhub
                    if self.finnhub_api极_key:
                        tasks.append(self._fetch_finnhub_news(session, symbol, lookback_hours))

                # Wait for all news collection tasks
                news_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for result in news_results:
                    if isinstance(result, Exception):
                        logger.warning(f"News collection error: {str(result)}")
                        continue
                    if isinstance(result, list):
                        all_news.extend(result)

            # Analyze sentiment for each symbol
            for symbol in symbols:
                symbol_news = [news for news in all_news if symbol.upper() in news.get('title', '').upper()
                               or symbol.upper() in news.get('description', '').upper()]
                sentiment_analysis = self._analyze_sentiment(symbol_news)
                sentiment_results[symbol] = {
                    "news_count": len(symbol_news),
                    "sentiment_score": sentiment_analysis['overall_sentiment'],
                    "sentiment_label": sentiment_analysis['sentiment_label'],
                    "confidence": sentiment_analysis['confidence'],
                    "recent_headlines": [news['title'] for news in symbol_news[:5]],
                    "sentiment_breakdown": sentiment_analysis['breakdown'],
                    "market_impact_score": self._calculate_market_impact(sentiment_analysis, len(symbol_news))
                }

            # Overall market sentiment
            overall_sentiment = self._calculate_overall_market_sentiment(all_news)

            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "lookback_hours": lookback_hours,
                "total_news_articles": len(all_news),
                "symbol_sentiment": sentiment_results,
                "overall_market_sentiment": overall_sentiment,
                "news_sources": self._get_news_sources_summary(all_news)
            }

        except Exception as e:
            error_msg = f"Error in news sentiment analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _fetch_newsapi_data(self, session: aiohttp.ClientSession, symbol: str, lookback_hours: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        try:
            from_date = (datetime.utcnow() - timedelta(hours=lookback_hours)).isoformat()
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{symbol} OR "{symbol}" stock OR trading',
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 50,
                'apiKey': self.news_api_key
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    # Process and clean articles
                    processed_articles = []
                    for article in articles:
                        processed_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'source': article.get('source', {}).get('name', 'NewsAPI'),
                            'symbol': symbol
                        })
                    return processed_articles
                else:
                    logger.warning(f"NewsAPI request failed with status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching NewsAPI data for {symbol}: {str(e)}")
            return []

    async def _fetch_finnhub_news(self, session: aiohttp.ClientSession, symbol: str, lookback_hours: int) -> List[Dict]:
        """Fetch news from Finnhub"""
        try:
            from_date = int((datetime.utcnow() - timedelta(hours=lookback_hours)).timestamp())
            to_date = int(datetime.utcnow().timestamp())
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': datetime.fromtimestamp(from_date).strftime('%Y-%m-%d'),
                'to': datetime.fromtimestamp(to_date).strftime('%Y-%m-%d'),
                'token': self.finnhub_api_key
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    articles = await response.json()
                    processed_articles = []
                    for article in articles:
                        processed_articles.append({
                            'title': article.get('headline', ''),
                            'description': article.get('summary', ''),
                            'content': article.get('summary', ''),
                            'url': article.get('url', ''),
                            'published_at': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                            'source': 'Finnhub',
                            'symbol': symbol,
                            'related_symbols': article.get('related', '').split(',') if article.get('related') else []
                        })
                    return processed_articles
                else:
                    logger.warning(f"Finnhub request failed with status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching Finnhub data for {symbol}: {str(e)}")
            return []

    def _analyze_sentiment(self, news_articles: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment of news articles"""
        if not news_articles:
            return {
                "overall_sentiment": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "breakdown": {"positive": 0, "negative": 0, "neutral": 0}
            }

        sentiments = []
        breakdown = {"positive": 0, "negative": 0, "neutral": 0}

        for article in news_articles:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if not text.strip():
                continue

            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
            sentiments.append(polarity)

            # Categorize sentiment
            if polarity > 0.1:
                breakdown["positive"] += 1
            elif polarity < -0.1:
                breakdown["negative"] += 1
            else:
                breakdown["neutral"] += 1

        if not sentiments:
            return {
                "overall_sentiment": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "breakdown": breakdown
            }

        # Calculate overall sentiment
        overall_sentiment = sum(sentiments) / len(sentiments)

        # Determine sentiment label
        if overall_sentiment > 0.1:
            sentiment_label = "positive"
        elif overall_sentiment < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        # Calculate confidence (inverse of variance)
        variance = sum((s - overall_sentiment) ** 2 for s in sentiments) / len(sentiments)
        confidence = max(0.0, min(1.0, 1.0 - variance))

        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_label": sentiment_label,
            "confidence": confidence,
            "breakdown": breakdown
        }

    def _calculate_market_impact(self, sentiment_analysis: Dict[str, Any], news_count: int) -> float:
        """Calculate potential market impact score"""
        sentiment_score = abs(sentiment_analysis['overall_sentiment'])
        confidence = sentiment_analysis['confidence']
        volume_factor = min(1.0, news_count / 10.0)  # Normalize news volume

        # Market impact score (0-100)
        impact_score = (sentiment_score * confidence * volume_factor) * 100
        return round(impact_score, 2)

    def _calculate_overall_market_sentiment(self, all_news: List[Dict]) -> Dict[str, Any]:
        """Calculate overall market sentiment from all news"""
        if not all_news:
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0
            }

        # Analyze sentiment for all news
        market_sentiment = self._analyze_sentiment(all_news)

        # Add market-specific metrics
        market_sentiment["news_volume"] = len(all_news)
        market_sentiment["market_stress_indicator"] = self._calculate_market_stress(all_news)
        return market_sentiment

    def _calculate_market_stress(self, all_news: List[Dict]) -> str:
        """Calculate market stress level based on news patterns"""
        stress_keywords = ['crash', 'panic', 'volatility', 'uncertainty', 'fear', 'selloff', 'plunge']
        stress_mentions = 0

        for article in all_news:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            stress_mentions += sum(1 for keyword in stress_keywords if keyword in text)

        if stress_mentions > len(all_news) * 0.3:
            return "high"
        elif stress_mentions > len(all_news) * 0.1:
            return "moderate"
        else:
            return "low"

    def _get_news_sources_summary(self, all_news: List[Dict]) -> Dict[str, int]:
        """Get summary of news sources"""
        sources = {}
        for article in all_news:
            source = article.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        return sources


class NewsSentimentAgent(Agent):
    """Specialized agent for news collection and sentiment analysis"""

    def __init__(self, gcp_services, config):
        super().__init__(
            model="gemini-2.0-flash-exp",
            name="news_sentiment_analyzer",
            description="Expert news sentiment analyst specializing in financial market impact analysis",
            instructions="""You are a specialized news sentiment analysis agent with expertise in:
1. Multi-source financial news collection (NewsAPI, Finnhub, etc.)
2. Advanced sentiment analysis and market impact assessment
3. Real-time news monitoring and trend identification
4. Market stress indicator calculation
5. Cross-symbol sentiment correlation analysis

Your primary responsibilities:
- Collect relevant financial news from multiple sources
- Perform accurate sentiment analysis using NLP techniques
- Assess potential market impact of news sentiment
- Identify emerging market trends and sentiment shifts
- Provide actionable insights for trading decisions

Focus on providing high-quality sentiment analysis that considers:
- News source credibility and impact
- Sentiment confidence levels
- Market stress indicators
- Cross-asset sentiment correlations""",
            tools=[NewsSentimentTool()]
        )
        self.gcp_services = gcp_services
        self.config = config
        logger.info("NewsSentimentAgent initialized")

    async def analyze_news_sentiment(self, symbols: List[str], lookback_hours: int = 24) -> Dict[str, Any]:
        """Main method to analyze news sentiment for given symbols"""
        try:
            logger.info(f"Starting news sentiment analysis for {len(symbols)} symbols")

            # Use the news sentiment tool
            sentiment_data = await self.tools[0].call(
                symbols=symbols,
                lookback_hours=lookback_hours
            )

            # Store results in Firestore
            await self.gcp_services.store_news_sentiment(sentiment_data)

            # Store in BigQuery for analytics
            await self.gcp_services.store_bigquery_sentiment_data(sentiment_data)

            # Add agent metadata
            sentiment_data["analysis_metadata"] = {
                "agent": self.name,
                "analysis_time": datetime.utcnow().isoformat(),
                "symbols_analyzed": symbols,
                "lookback_hours": lookback_hours,
                "sentiment_quality_score": self._calculate_sentiment_quality_score(sentiment_data)
            }

            logger.info(
                f"News sentiment analysis completed. Quality score: {sentiment_data['analysis_metadata']['sentiment_quality_score']}")
            return sentiment_data

        except Exception as e:
            error_msg = f"Error in news sentiment analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.name
            }

    def _calculate_sentiment_quality_score(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate quality score for sentiment analysis"""
        if sentiment_data.get('status') != 'success':
            return 0.0

        # Base score from news article count
        total_articles = sentiment_data.get('total_news_articles', 0)
        base_score = min(50.0, (total_articles / 20.0) * 50.0)  # Max 50 points for article count

        # Quality score from sentiment confidence
        symbol_sentiment = sentiment_data.get('symbol_sentiment', {})
        if symbol_sentiment:
            avg_confidence = sum(data.get('confidence', 0) for data in symbol_sentiment.values()) / len(
                symbol_sentiment)
            confidence_score = avg_confidence * 30.0  # Max 30 points for confidence
        else:
            confidence_score = 0.0

        # Bonus for multiple sources
        news_sources = sentiment_data.get('news_sources', {})
        source_diversity_score = min(20.0, len(news_sources) * 5.0)  # Max 20 points for source diversity

        total_score = base_score + confidence_score + source_diversity_score
        return round(min(100.0, total_score), 2)

    async def get_breaking_news_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Get sentiment for very recent breaking news (last 2 hours)"""
        return await self.analyze_news_sentiment(symbols, lookback_hours=2)

    async def get_daily_sentiment_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive daily sentiment summary"""
        return await self.analyze_news_sentiment(symbols, lookback_hours=24)

    async def monitor_sentiment_changes(self, symbols: List[str], comparison_hours: int = 6) -> Dict[str, Any]:
        """Monitor sentiment changes over time"""
        try:
            # Get current sentiment
            current_sentiment = await self.analyze_news_sentiment(symbols, lookback_hours=comparison_hours)

            # Get historical sentiment for comparison
            historical_sentiment = await self.analyze_news_sentiment(symbols, lookback_hours=comparison_hours * 2)

            # Calculate sentiment changes
            sentiment_changes = {}
            for symbol in symbols:
                current_data = current_sentiment.get('symbol_sentiment', {}).get(symbol, {})
                historical_data = historical_sentiment.get('symbol_sentiment', {}).get(symbol, {})
                current_score = current_data.get('sentiment_score', 0)
                historical_score = historical_data.get('sentiment_score', 0)
                change = current_score - historical_score

                sentiment_changes[symbol] = {
                    "current_sentiment": current_score,
                    "historical_sentiment": historical_score,
                    "sentiment_change": change,
                    "change_significance": self._assess_change_significance(change),
                    "trend_direction": "positive" if change > 0.05 else "negative" if change < -0.05 else "stable"
                }

            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "comparison_hours": comparison_hours,
                "sentiment_changes": sentiment_changes,
                "agent": self.name
            }

        except Exception as e:
            error_msg = f"Error monitoring sentiment changes: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.name
            }

    def _assess_change_significance(self, change: float) -> str:
        """Assess the significance of sentiment change"""
        abs_change = abs(change)
        if abs_change > 0.3:
            return "high"
        elif abs_change > 0.1:
            return "moderate"
        else:
            return "low"
