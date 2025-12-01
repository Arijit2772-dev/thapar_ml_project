"""
Machine Learning Models for WhatsApp Chat Analysis
Includes: Sentiment Analysis, Topic Modeling, Message Clustering, Activity Prediction
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Get the directory of this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP_WORDS_PATH = os.path.join(BASE_DIR, 'stop_words', 'stop_hinglish.txt')

# For sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


def sentiment_analysis(df, selected_user='Overall'):
    """
    Perform sentiment analysis on messages using VADER (better for informal text)
    or TextBlob as fallback
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user].copy()
    else:
        df = df.copy()

    # Filter out group notifications and media
    df = df[df['user'] != 'group_notification']
    df = df[df['message'] != '<Media omitted>\n']

    if df.empty or len(df) < 5:
        return None

    sentiments = []
    polarities = []

    if VADER_AVAILABLE:
        # VADER is better for social media/informal text
        analyzer = SentimentIntensityAnalyzer()

        for message in df['message']:
            scores = analyzer.polarity_scores(message)
            compound = scores['compound']
            polarities.append(compound)

            # Classify based on compound score
            if compound >= 0.05:
                sentiments.append('Positive')
            elif compound <= -0.05:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')

    elif TEXTBLOB_AVAILABLE:
        # TextBlob fallback
        for message in df['message']:
            analysis = TextBlob(message)
            polarity = analysis.sentiment.polarity
            polarities.append(polarity)

            if polarity > 0.1:
                sentiments.append('Positive')
            elif polarity < -0.1:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')
    else:
        # Simple rule-based fallback
        positive_words = ['good', 'great', 'awesome', 'excellent', 'happy', 'love', 'best', 'nice', 'thanks', 'thank']
        negative_words = ['bad', 'worst', 'hate', 'terrible', 'sad', 'angry', 'disappointed', 'wrong', 'issue', 'problem']

        for message in df['message']:
            message_lower = message.lower()
            pos_count = sum(1 for word in positive_words if word in message_lower)
            neg_count = sum(1 for word in negative_words if word in message_lower)

            polarity = (pos_count - neg_count) / max(len(message.split()), 1)
            polarities.append(polarity)

            if pos_count > neg_count:
                sentiments.append('Positive')
            elif neg_count > pos_count:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')

    df['sentiment'] = sentiments
    df['polarity'] = polarities

    # Sentiment distribution
    sentiment_dist = df['sentiment'].value_counts()

    # Sentiment over time
    sentiment_timeline = df.groupby(['only_date', 'sentiment']).size().unstack(fill_value=0)

    # Average polarity by user (if Overall)
    if selected_user == 'Overall':
        user_sentiment = df.groupby('user')['polarity'].mean().sort_values(ascending=False).head(10)
    else:
        user_sentiment = None

    return {
        'df': df,
        'distribution': sentiment_dist,
        'timeline': sentiment_timeline,
        'user_sentiment': user_sentiment,
        'avg_polarity': df['polarity'].mean()
    }


def topic_modeling(df, selected_user='Overall', n_topics=5, method='lda'):
    """
    Discover topics in conversations using LDA or NMF
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user].copy()
    else:
        df = df.copy()

    # Filter out group notifications and media
    df = df[df['user'] != 'group_notification']
    df = df[df['message'] != '<Media omitted>\n']

    # Minimum messages needed for topic modeling
    min_messages = max(10, n_topics * 2)
    if df.empty or len(df) < min_messages:
        return None

    # Load stop words
    try:
        with open(STOP_WORDS_PATH, 'r') as f:
            stop_words = f.read().split('\n')
    except:
        stop_words = []

    # Vectorize messages - use min_df=1 for smaller datasets
    min_df = 1 if len(df) < 100 else 2

    try:
        if method == 'lda':
            vectorizer = CountVectorizer(max_features=1000, stop_words=stop_words,
                                        min_df=min_df, max_df=0.95)
            doc_term_matrix = vectorizer.fit_transform(df['message'])

            # Check if we have enough features
            if doc_term_matrix.shape[1] < n_topics:
                return None

            # LDA model
            lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
            lda_model.fit(doc_term_matrix)

            # Get top words for each topic
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx + 1,
                    'top_words': top_words
                })

            return {
                'topics': topics,
                'model': lda_model,
                'vectorizer': vectorizer,
                'method': 'LDA'
            }

        else:  # NMF
            vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words,
                                        min_df=min_df, max_df=0.95)
            tfidf_matrix = vectorizer.fit_transform(df['message'])

            # Check if we have enough features
            if tfidf_matrix.shape[1] < n_topics:
                return None

            # NMF model
            nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=200)
            nmf_model.fit(tfidf_matrix)

            # Get top words for each topic
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx + 1,
                    'top_words': top_words
                })

            return {
                'topics': topics,
                'model': nmf_model,
                'vectorizer': vectorizer,
                'method': 'NMF'
            }

    except Exception as e:
        print(f"Error in topic modeling: {e}")
        return None


def message_clustering(df, selected_user='Overall', n_clusters=5):
    """
    Cluster similar messages together using K-Means
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user].copy()
    else:
        df = df.copy()

    # Filter out group notifications and media
    df = df[df['user'] != 'group_notification']
    df = df[df['message'] != '<Media omitted>\n']

    # Need at least 2x clusters worth of messages
    min_messages = max(10, n_clusters * 2)
    if df.empty or len(df) < min_messages:
        return None

    # Load stop words
    try:
        with open(STOP_WORDS_PATH, 'r') as f:
            stop_words = f.read().split('\n')
    except:
        stop_words = []

    try:
        # TF-IDF vectorization - use min_df=1 for smaller datasets
        min_df = 1 if len(df) < 100 else 2
        vectorizer = TfidfVectorizer(max_features=500, stop_words=stop_words,
                                     min_df=min_df, max_df=0.95)
        tfidf_matrix = vectorizer.fit_transform(df['message'])

        # Check if we have enough features
        if tfidf_matrix.shape[1] < 3:
            return None

        # K-Means clustering
        actual_clusters = min(n_clusters, len(df))
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)

        df['cluster'] = clusters

        # Get representative messages from each cluster
        cluster_info = []
        for cluster_id in range(actual_clusters):
            cluster_messages = df[df['cluster'] == cluster_id]
            if len(cluster_messages) > 0:
                # Get messages closest to centroid
                cluster_indices = np.where(clusters == cluster_id)[0]
                distances = kmeans.transform(tfidf_matrix[cluster_indices])[:, cluster_id]
                closest_idx = cluster_indices[distances.argmin()]

                cluster_info.append({
                    'cluster_id': cluster_id,
                    'size': len(cluster_messages),
                    'representative_message': df.iloc[closest_idx]['message'][:100],
                    'sample_messages': cluster_messages['message'].head(5).tolist()
                })

        return {
            'df': df,
            'clusters': cluster_info,
            'n_clusters': actual_clusters
        }

    except Exception as e:
        print(f"Error in message clustering: {e}")
        return None


def predict_user_activity(df):
    """
    Predict which user is likely to send next message based on temporal patterns
    and conversation flow
    """
    try:
        # Filter out group notifications
        df = df[df['user'] != 'group_notification'].copy()

        # Need enough data and at least 2 different users
        unique_users = df['user'].nunique()
        if len(df) < 50 or unique_users < 2:
            return None

        # Create features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['message_length'] = df['message'].str.len()

        # Previous user (lag feature)
        df['prev_user'] = df['user'].shift(1)
        df['prev_hour'] = df['hour'].shift(1)

        # Drop NaN from lag
        df = df.dropna(subset=['prev_user', 'prev_hour'])

        if len(df) < 20:  # After dropping NaNs, still need enough data
            return None

        # Encode users
        le = LabelEncoder()
        df['user_encoded'] = le.fit_transform(df['user'])
        df['prev_user_encoded'] = le.transform(df['prev_user'])

        # Features and target
        feature_cols = ['hour_sin', 'hour_cos', 'day_of_week', 'is_weekend',
                        'message_length', 'prev_user_encoded', 'prev_hour']
        X = df[feature_cols]
        y = df['user_encoded']

        # Train-test split
        test_size = min(0.2, 10 / len(df))  # At least 10 samples or 20% for test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Random Forest Classifier
        n_estimators = min(100, len(X_train) // 2)  # Don't use too many trees for small datasets
        model = RandomForestClassifier(n_estimators=max(10, n_estimators), random_state=42, max_depth=10)
        model.fit(X_train, y_train)

        # Accuracy
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            'model': model,
            'label_encoder': le,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance
        }

    except Exception as e:
        print(f"Error in activity prediction: {e}")
        return None


def get_user_personality_insights(df, selected_user):
    """
    Generate personality insights based on messaging patterns
    """
    if selected_user == 'Overall':
        return None

    user_df = df[df['user'] == selected_user].copy()
    user_df = user_df[user_df['message'] != '<Media omitted>\n']

    if user_df.empty:
        return None

    insights = {}

    # Message length stats
    user_df['msg_length'] = user_df['message'].str.len()
    insights['avg_message_length'] = user_df['msg_length'].mean()
    insights['message_style'] = 'Verbose' if insights['avg_message_length'] > 100 else 'Concise'

    # Response time (if possible)
    user_df = user_df.sort_values('date')
    user_df['time_diff'] = user_df['date'].diff().dt.total_seconds() / 60  # minutes
    insights['avg_response_time_minutes'] = user_df['time_diff'].median()

    # Activity pattern
    hour_dist = user_df['hour'].value_counts().sort_index()
    peak_hour = hour_dist.idxmax()

    if 6 <= peak_hour < 12:
        insights['activity_type'] = 'Morning Person'
    elif 12 <= peak_hour < 17:
        insights['activity_type'] = 'Afternoon Active'
    elif 17 <= peak_hour < 22:
        insights['activity_type'] = 'Evening Active'
    else:
        insights['activity_type'] = 'Night Owl'

    # Emoji usage
    total_emojis = 0
    for message in user_df['message']:
        try:
            import emoji
            total_emojis += len([c for c in message if c in emoji.EMOJI_DATA])
        except:
            pass

    insights['emoji_per_message'] = total_emojis / len(user_df) if len(user_df) > 0 else 0
    insights['emoji_user'] = 'High' if insights['emoji_per_message'] > 1 else 'Low'

    # Engagement level
    total_messages = len(df[df['user'] != 'group_notification'])
    user_messages = len(user_df)
    insights['engagement_percentage'] = (user_messages / total_messages * 100) if total_messages > 0 else 0

    return insights
