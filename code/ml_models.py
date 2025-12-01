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


def train_emoji_classifier(df):
    """
    Train a binary classifier to predict if a message will contain emojis
    Uses ML: Logistic Regression with TF-IDF features
    """
    try:
        import emoji
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # Filter data
        df_clean = df[df['user'] != 'group_notification'].copy()
        df_clean = df_clean[df_clean['message'] != '<Media omitted>\n']

        if len(df_clean) < 20:
            return None

        # Create target variable: has emoji or not
        df_clean['has_emoji'] = df_clean['message'].apply(
            lambda x: 1 if any(c in emoji.EMOJI_DATA for c in x) else 0
        )

        # Need both classes
        if df_clean['has_emoji'].nunique() < 2:
            return None

        # Feature engineering
        df_clean['msg_length'] = df_clean['message'].str.len()
        df_clean['word_count'] = df_clean['message'].str.split().str.len()
        df_clean['has_question'] = df_clean['message'].str.contains(r'\?', regex=True).astype(int)
        df_clean['has_exclamation'] = df_clean['message'].str.contains(r'!', regex=True).astype(int)
        df_clean['hour_of_day'] = df_clean['hour']
        df_clean['is_weekend'] = df_clean['date'].dt.dayofweek.isin([5, 6]).astype(int)

        # Encode user as numeric
        df_clean['user_encoded'] = pd.factorize(df_clean['user'])[0]

        # Select features
        feature_cols = ['msg_length', 'word_count', 'has_question', 'has_exclamation',
                       'hour_of_day', 'is_weekend', 'user_encoded']
        X = df_clean[feature_cols].fillna(0)
        y = df_clean['has_emoji']

        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)

        # Emoji usage stats
        emoji_count = df_clean['has_emoji'].sum()
        emoji_percentage = (emoji_count / len(df_clean)) * 100

        return {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importance': feature_importance,
            'emoji_count': emoji_count,
            'emoji_percentage': emoji_percentage,
            'total_messages': len(df_clean),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test)
        }

    except Exception as e:
        print(f"Error in emoji classifier: {e}")
        return None


def train_message_length_predictor(df):
    """
    Train a regression model to predict message length based on user and time
    Uses ML: Random Forest Regressor
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score

        # Filter data
        df_clean = df[df['user'] != 'group_notification'].copy()
        df_clean = df_clean[df_clean['message'] != '<Media omitted>\n']

        if len(df_clean) < 20:
            return None

        # Target variable
        df_clean['msg_length'] = df_clean['message'].str.len()

        # Feature engineering
        df_clean['hour_of_day'] = df_clean['hour']
        df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
        df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
        df_clean['user_encoded'] = pd.factorize(df_clean['user'])[0]

        # Previous message length (lag feature)
        df_clean['prev_msg_length'] = df_clean['msg_length'].shift(1).fillna(df_clean['msg_length'].mean())

        # Select features
        feature_cols = ['hour_of_day', 'day_of_week', 'is_weekend', 'user_encoded', 'prev_msg_length']
        X = df_clean[feature_cols].fillna(0)
        y = df_clean['msg_length']

        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance,
            'avg_message_length': y.mean(),
            'predictions_test': y_pred_test[:10],
            'actuals_test': y_test[:10].values
        }

    except Exception as e:
        print(f"Error in message length predictor: {e}")
        return None


def train_user_classifier(df):
    """
    Train a multi-class classifier to identify user based on message style
    Uses ML: Naive Bayes with TF-IDF features
    """
    try:
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score, classification_report

        # Filter data
        df_clean = df[df['user'] != 'group_notification'].copy()
        df_clean = df_clean[df_clean['message'] != '<Media omitted>\n']

        # Need at least 2 users with 10+ messages each
        user_counts = df_clean['user'].value_counts()
        valid_users = user_counts[user_counts >= 10].index.tolist()

        if len(valid_users) < 2:
            return None

        df_clean = df_clean[df_clean['user'].isin(valid_users)]

        if len(df_clean) < 20:
            return None

        # Load stop words
        try:
            with open(STOP_WORDS_PATH, 'r') as f:
                stop_words = f.read().split('\n')
        except:
            stop_words = []

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=100, stop_words=stop_words,
                                     min_df=1, max_df=0.9, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df_clean['message'])
        y = df_clean['user']

        # Train-test split
        split_idx = int(len(X.toarray()) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        # Get top words for each user
        feature_names = vectorizer.get_feature_names_out()
        user_words = {}
        for idx, user in enumerate(model.classes_):
            top_indices = model.feature_log_prob_[idx].argsort()[-10:][::-1]
            user_words[user] = [feature_names[i] for i in top_indices]

        return {
            'model': model,
            'vectorizer': vectorizer,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'users': model.classes_.tolist(),
            'user_top_words': user_words,
            'num_users': len(model.classes_)
        }

    except Exception as e:
        print(f"Error in user classifier: {e}")
        return None
