import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List
import aiohttp
import asyncio
from textblob import TextBlob
import re

# Python version compatibility check
if sys.version_info < (3, 7):
    raise RuntimeError("This application requires Python 3.7 or higher")

# Import with proper error handling
try:
    from google.adk.agents import Agent
    from google.adk.tools import Tool
except ImportError:
    # Fallback implementations
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


class SocialMediaTool(Tool):
    """Tool for collecting and analyzing social media sentiment"""

    def __init__(self):
        # Use explicit parent class reference for Python 2.7 compatibility
        Tool.__init__(
            self,
            name="social_media_sentiment_analyzer",
            description="Collect and analyze social media sentiment from Twitter and Reddit"
        )
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')

    async def call(self, symbols: List[str], lookback_hours: int = 24, max_posts: int = 100) -> Dict[str, Any]:
        """
        Collect and analyze social media sentiment for given symbols
        Args:
            symbols: List of stock symbols to analyze
            lookback_hours: How many hours back to look
            max_posts: Maximum posts to analyze per symbol
        Returns:
            Social media sentiment analysis results
        """
        try:
            logger.info(f"Collecting social media sentiment for symbols: {symbols}")
            all_posts = []
            sentiment_results = {}

            async with aiohttp.ClientSession() as session:
                tasks = []
                for symbol in symbols:
                    # Twitter data collection
                    if self.twitter_bearer_token:
                        tasks.append(self._fetch_twitter_data(session, symbol, lookback_hours, max_posts))
                    # Reddit data collection
                    if self.reddit_client_id and self.reddit_client_secret:
                        tasks.append(self._fetch_reddit_data(session, symbol, lookback_hours, max_posts))

                # Wait for all social media collection tasks
                social_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for result in social_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Social media collection error: {str(result)}")
                        continue
                    if isinstance(result, list):
                        all_posts.extend(result)

            # Analyze sentiment for each symbol
            for symbol in symbols:
                symbol_posts = [post for post in all_posts if symbol.upper() in post.get('text', '').upper()]
                sentiment_analysis = self._analyze_social_sentiment(symbol_posts)
                sentiment_results[symbol] = {
                    "post_count": len(symbol_posts),
                    "sentiment_score": sentiment_analysis['overall_sentiment'],
                    "sentiment_label": sentiment_analysis['sentiment_label'],
                    "confidence": sentiment_analysis['confidence'],
                    "engagement_score": self._calculate_engagement_score(symbol_posts),
                    "trending_score": self._calculate_trending_score(symbol_posts, lookback_hours),
                    "sentiment_breakdown": sentiment_analysis['breakdown'],
                    "top_posts": self._get_top_posts(symbol_posts)[:3],
                    "influencer_mentions": self._identify_influencer_mentions(symbol_posts)
                }

            # Overall social sentiment
            overall_sentiment = self._calculate_overall_social_sentiment(all_posts)

            return {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "lookback_hours": lookback_hours,
                "total_posts": len(all_posts),
                "symbol_sentiment": sentiment_results,
                "overall_social_sentiment": overall_sentiment,
                "platform_breakdown": self._get_platform_breakdown(all_posts),
                "viral_indicators": self._detect_viral_content(all_posts)
            }

        except Exception as e:
            error_msg = f"Error in social media sentiment analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _fetch_twitter_data(self, session: aiohttp.ClientSession, symbol: str,
                                  lookback_hours: int, max_posts: int) -> List[Dict]:
        """Fetch Twitter data using Twitter API v2"""
        try:
            # Calculate time window for lookback
            cutoff_time = datetime.now(timezone.utc).timestamp() - (lookback_hours * 3600)

            # Twitter API v2 recent search endpoint
            url = "https://api.twitter.com/2/tweets/search/recent"
            # Search query for stock symbol
            query = f"({symbol} OR ${symbol} OR #{symbol}) (stock OR stocks OR trading OR invest OR market) -is:retweet lang:en"

            params = {
                'query': query,
                'max_results': min(max_posts, 100),  # API limit
                'tweet.fields': 'created_at,public_metrics,context_annotations,author_id',
                'user.fields': 'public_metrics,verified',
                'expansions': 'author_id'
            }

            headers = {
                'Authorization': f'Bearer {self.twitter_bearer_token}',
                'Content-Type': 'application/json'
            }

            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    tweets = data.get('data', [])
                    users = {user['id']: user for user in data.get('includes', {}).get('users', [])}

                    processed_posts = []
                    for tweet in tweets:
                        # Filter by time window using lookback_hours
                        tweet_time = datetime.fromisoformat(
                            tweet.get('created_at', '').replace('Z', '+00:00')).timestamp()
                        if tweet_time < cutoff_time:
                            continue

                        user = users.get(tweet.get('author_id', ''), {})
                        processed_posts.append({
                            'text': tweet.get('text', ''),
                            'created_at': tweet.get('created_at', ''),
                            'platform': 'twitter',
                            'author_id': tweet.get('author_id', ''),
                            'author_verified': user.get('verified', False),
                            'author_followers': user.get('public_metrics', {}).get('followers_count', 0),
                            'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                            'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                            'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
                            'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0),
                            'symbol': symbol,
                            'engagement_total': sum([
                                tweet.get('public_metrics', {}).get('retweet_count', 0),
                                tweet.get('public_metrics', {}).get('like_count', 0),
                                tweet.get('public_metrics', {}).get('reply_count', 0),
                                tweet.get('public_metrics', {}).get('quote_count', 0)
                            ])
                        })

                    return processed_posts
                else:
                    logger.warning(f"Twitter API request failed with status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching Twitter data for {symbol}: {str(e)}")
            return []

    async def _fetch_reddit_data(self, session: aiohttp.ClientSession, symbol: str,
                                 lookback_hours: int, max_posts: int) -> List[Dict]:
        """Fetch Reddit data from relevant subreddits"""
        try:
            # Calculate time window for lookback
            cutoff_time = datetime.now(timezone.utc).timestamp() - (lookback_hours * 3600)

            # Get Reddit access token
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                'grant_type': 'client_credentials'
            }
            auth_headers = {
                'User-Agent': 'SmartTrader/1.0'
            }

            async with session.post(auth_url, data=auth_data, headers=auth_headers,
                                    auth=aiohttp.BasicAuth(self.reddit_client_id,
                                                           self.reddit_client_secret)) as auth_response:
                if auth_response.status == 200:
                    auth_result = await auth_response.json()
                    access_token = auth_result.get('access_token')
                else:
                    logger.warning(f"Reddit auth failed with status {auth_response.status}")
                    return []

            # Search relevant subreddits
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'StockMarket', 'wallstreetbets']
            all_reddit_posts = []

            headers = {
                'Authorization': f'Bearer {access_token}',
                'User-Agent': 'SmartTrader/1.0'
            }

            for subreddit in subreddits:
                search_url = f"https://oauth.reddit.com/r/{subreddit}/search"
                params = {
                    'q': symbol,
                    'sort': 'new',
                    'limit': max_posts // len(subreddits),
                    'restrict_sr': 'true',
                    't': 'day' if lookback_hours <= 24 else 'week'
                }

                async with session.get(search_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get('data', {}).get('children', [])

                        for post in posts:
                            post_data = post.get('data', {})
                            # Filter by time window using lookback_hours
                            post_time = post_data.get('created_utc', 0)
                            if post_time < cutoff_time:
                                continue

                            all_reddit_posts.append({
                                'text': f"{post_data.get('title', '')} {post_data.get('selftext', '')}",
                                'created_at': datetime.fromtimestamp(post_time).replace(
                                    tzinfo=timezone.utc).isoformat(),
                                'platform': 'reddit',
                                'subreddit': subreddit,
                                'author': post_data.get('author', ''),
                                'score': post_data.get('score', 0),
                                'upvote_ratio': post_data.get('upvote_ratio', 0),
                                'num_comments': post_data.get('num_comments', 0),
                                'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                'symbol': symbol,
                                'engagement_total': post_data.get('score', 0) + post_data.get('num_comments', 0)
                            })

            return all_reddit_posts

        except Exception as e:
            logger.error(f"Error fetching Reddit data for {symbol}: {str(e)}")
            return []

    def _analyze_social_sentiment(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment of social media posts"""
        if not posts:
            return {
                "overall_sentiment": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "breakdown": {"positive": 0, "negative": 0, "neutral": 0}
            }

        sentiments = []
        weights = []
        breakdown = {"positive": 0, "negative": 0, "neutral": 0}

        for post in posts:
            text = post.get('text', '').strip()
            if not text:
                continue

            # Clean text for sentiment analysis
            cleaned_text = self._clean_social_text(text)

            # Use TextBlob for sentiment analysis
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity

            # Calculate weight based on engagement
            engagement = post.get('engagement_total', 1)
            weight = min(10.0, 1.0 + (engagement / 100.0))  # Cap weight at 10x

            sentiments.append(polarity)
            weights.append(weight)

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

        # Calculate weighted sentiment
        weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)

        # Determine sentiment label
        if weighted_sentiment > 0.1:
            sentiment_label = "positive"
        elif weighted_sentiment < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        # Calculate confidence
        variance = sum(w * (s - weighted_sentiment) ** 2 for s, w in zip(sentiments, weights)) / sum(weights)
        confidence = max(0.0, min(1.0, 1.0 - variance))

        return {
            "overall_sentiment": weighted_sentiment,
            "sentiment_label": sentiment_label,
            "confidence": confidence,
            "breakdown": breakdown
        }

    def _clean_social_text(self, text: str) -> str:
        """Clean social media text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove mentions and hashtags for cleaner sentiment
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    @staticmethod
    def _calculate_engagement_score(posts: List[Dict]) -> float:
        """Calculate engagement score for posts"""
        if not posts:
            return 0.0

        total_engagement = sum(post.get('engagement_total', 0) for post in posts)
        avg_engagement = total_engagement / len(posts)

        # Normalize to 0-100 scale
        return min(100.0, avg_engagement / 10.0)

    @staticmethod
    def _calculate_trending_score(posts: List[Dict], lookback_hours: int) -> float:
        """Calculate how trending a symbol is based on post frequency"""
        if not posts or lookback_hours == 0:
            return 0.0

        posts_per_hour = len(posts) / lookback_hours

        # Normalize to 0-100 scale (assuming 10 posts/hour = 100% trending)
        return min(100.0, (posts_per_hour / 10.0) * 100.0)

    def _get_top_posts(self, posts: List[Dict]) -> List[Dict]:
        """Get top posts by engagement"""
        sorted_posts = sorted(posts, key=lambda x: x.get('engagement_total', 0), reverse=True)
        return [{
            'text': post.get('text', '')[:200] + '...' if len(post.get('text', '')) > 200 else post.get('text', ''),
            'platform': post.get('platform', ''),
            'engagement': post.get('engagement_total', 0),
            'created_at': post.get('created_at', '')
        } for post in sorted_posts[:5]]

    def _identify_influencer_mentions(self, posts: List[Dict]) -> List[Dict]:
        """Identify posts from influential accounts"""
        influencer_posts = []

        for post in posts:
            # Twitter influencer criteria
            if post.get('platform') == 'twitter':
                followers = post.get('author_followers', 0)
                verified = post.get('author_verified', False)
                if verified or followers > 10000:
                    influencer_posts.append({
                        'author_id': post.get('author_id', ''),
                        'followers': followers,
                        'verified': verified,
                        'text': post.get('text', '')[:150] + '...',
                        'engagement': post.get('engagement_total', 0)
                    })

            # Reddit influencer criteria (high karma posts)
            elif post.get('platform') == 'reddit':
                score = post.get('score', 0)
                if score > 100:
                    influencer_posts.append({
                        'author': post.get('author', ''),
                        'subreddit': post.get('subreddit', ''),
                        'score': score,
                        'text': post.get('text', '')[:150] + '...',
                        'comments': post.get('num_comments', 0)
                    })

        return sorted(influencer_posts, key=lambda x: x.get('followers', x.get('score', 0)), reverse=True)[:5]

    def _calculate_overall_social_sentiment(self, all_posts: List[Dict]) -> Dict[str, Any]:
        """Calculate overall social media sentiment"""
        overall_analysis = self._analyze_social_sentiment(all_posts)

        # Add social-specific metrics
        overall_analysis.update({
            "total_posts": len(all_posts),
            "viral_threshold_posts": len([p for p in all_posts if p.get('engagement_total', 0) > 1000]),
            "platform_dominance": self._get_platform_dominance(all_posts)
        })

        return overall_analysis

    def _get_platform_breakdown(self, all_posts: List[Dict]) -> Dict[str, int]:
        """Get breakdown of posts by platform"""
        platforms = {}
        for post in all_posts:
            platform = post.get('platform', 'unknown')
            platforms[platform] = platforms.get(platform, 0) + 1
        return platforms

    def _get_platform_dominance(self, all_posts: List[Dict]) -> str:
        """Determine which platform has the most posts"""
        platform_counts = self._get_platform_breakdown(all_posts)
        if not platform_counts:
            return "none"
        return max(platform_counts, key=platform_counts.get)

    def _detect_viral_content(self, all_posts: List[Dict]) -> Dict[str, Any]:
        """Detect viral content indicators"""
        viral_posts = [p for p in all_posts if p.get('engagement_total', 0) > 1000]

        return {
            "viral_post_count": len(viral_posts),
            "viral_percentage": (len(viral_posts) / len(all_posts) * 100) if all_posts else 0,
            "highest_engagement": max([p.get('engagement_total', 0) for p in all_posts]) if all_posts else 0,
            "viral_risk_level": "high" if len(viral_posts) > 5 else "moderate" if len(viral_posts) > 2 else "low"
        }


class SocialMediaAgent(Agent):
    """Specialized agent for social media sentiment analysis"""

    def __init__(self, gcp_services, config):
        # Use explicit parent class reference for Python 2.7 compatibility
        Agent.__init__(
            self,
            model="gemini-2.0-flash-exp",
            name="social_media_analyst",
            description="Expert social media sentiment analyst for financial markets",
            instructions="""You are a specialized social media sentiment analysis agent with expertise in:
1. Multi-platform social media data collection (Twitter, Reddit)
2. Engagement-weighted sentiment analysis
3. Viral content detection and trend identification
4. Influencer impact assessment
5. Real-time social sentiment monitoring

Your primary responsibilities:
- Collect relevant financial discussions from social platforms
- Perform weighted sentiment analysis based on engagement
- Identify trending topics and viral content
- Assess influencer impact on market sentiment
- Provide early warning signals for sentiment shifts

Focus on providing actionable insights that consider:
- Platform-specific engagement patterns
- Influencer verification and reach
- Viral content propagation risks
- Cross-platform sentiment correlation""",
            tools=[SocialMediaTool()]
        )

        self.gcp_services = gcp_services
        self.config = config
        logger.info("SocialMediaAgent initialized")

    async def analyze_social_sentiment(self, symbols: List[str], lookback_hours: int = 24) -> Dict[str, Any]:
        """Main method to analyze social media sentiment"""
        try:
            logger.info(f"Starting social media sentiment analysis for {len(symbols)} symbols")

            sentiment_data = await self.tools[0].call(
                symbols=symbols,
                lookback_hours=lookback_hours,
                max_posts=200
            )

            # Store results
            if self.gcp_services:
                if hasattr(self.gcp_services, 'store_social_sentiment'):
                    await self.gcp_services.store_social_sentiment(sentiment_data)
                if hasattr(self.gcp_services, 'store_bigquery_social_data'):
                    await self.gcp_services.store_bigquery_social_data(sentiment_data)

            # Add metadata
            sentiment_data["analysis_metadata"] = {
                "agent": self.name,
                "analysis_time": datetime.now(timezone.utc).isoformat(),
                "symbols_analyzed": symbols,
                "lookback_hours": lookback_hours,
                "quality_score": self._calculate_social_quality_score(sentiment_data)
            }

            logger.info("Social media sentiment analysis completed")
            return sentiment_data

        except Exception as e:
            error_msg = f"Error in social media sentiment analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": self.name
            }

    @staticmethod
    def _calculate_social_quality_score(sentiment_data: Dict[str, Any]) -> float:
        """Calculate quality score for social sentiment analysis"""
        if sentiment_data.get('status') != 'success':
            return 0.0

        # Base score from post count
        total_posts = sentiment_data.get('total_posts', 0)
        base_score = min(40.0, (total_posts / 100.0) * 40.0)

        # Platform diversity score
        platform_breakdown = sentiment_data.get('platform_breakdown', {})
        diversity_score = min(20.0, len(platform_breakdown) * 10.0)

        # Engagement quality score
        symbol_sentiment = sentiment_data.get('symbol_sentiment', {})
        if symbol_sentiment:
            avg_engagement = sum(data.get('engagement_score', 0) for data in symbol_sentiment.values()) / len(
                symbol_sentiment)
            engagement_score = min(25.0, avg_engagement / 4.0)
        else:
            engagement_score = 0.0

        # Confidence score
        if symbol_sentiment:
            avg_confidence = sum(data.get('confidence', 0) for data in symbol_sentiment.values()) / len(
                symbol_sentiment)
            confidence_score = avg_confidence * 15.0
        else:
            confidence_score = 0.0

        total_score = base_score + diversity_score + engagement_score + confidence_score
        return round(min(100.0, total_score), 2)
