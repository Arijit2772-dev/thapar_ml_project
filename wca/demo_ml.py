"""
Demo script to test ML features on sample WhatsApp chat
Run this to verify all ML models are working correctly
"""

import sys
sys.path.append('code')

import preprocessor
import ml_models
import pandas as pd

def test_ml_features():
    """Test all ML features with sample chat"""

    print("=" * 60)
    print("WhatsApp Chat Analyzer - ML Features Demo")
    print("=" * 60)

    # Load sample chat
    print("\n1. Loading sample chat data...")
    with open('chats/WhatsApp Chat with Jit Ghosh.txt', 'r', encoding='utf-8') as f:
        chat_data = f.read()

    # Preprocess
    print("2. Preprocessing chat data...")
    df = preprocessor.preprocess(chat_data)
    print(f"   ✓ Loaded {len(df)} messages")

    # Test 1: Sentiment Analysis
    print("\n3. Testing Sentiment Analysis...")
    sentiment_result = ml_models.sentiment_analysis(df, 'Overall')
    if sentiment_result:
        print(f"   ✓ Sentiment Analysis Complete")
        print(f"   - Positive: {sentiment_result['distribution'].get('Positive', 0)} messages")
        print(f"   - Neutral: {sentiment_result['distribution'].get('Neutral', 0)} messages")
        print(f"   - Negative: {sentiment_result['distribution'].get('Negative', 0)} messages")
        print(f"   - Average Polarity: {sentiment_result['avg_polarity']:.3f}")
    else:
        print("   ✗ Not enough data for sentiment analysis")

    # Test 2: Topic Modeling
    print("\n4. Testing Topic Modeling (LDA)...")
    topics_result = ml_models.topic_modeling(df, 'Overall', n_topics=3, method='lda')
    if topics_result:
        print(f"   ✓ Topic Modeling Complete ({topics_result['method']})")
        for topic in topics_result['topics']:
            print(f"   - Topic {topic['topic_id']}: {', '.join(topic['top_words'][:5])}")
    else:
        print("   ✗ Not enough data for topic modeling")

    # Test 3: Message Clustering
    print("\n5. Testing Message Clustering...")
    cluster_result = ml_models.message_clustering(df, 'Overall', n_clusters=3)
    if cluster_result:
        print(f"   ✓ Clustering Complete")
        for cluster in cluster_result['clusters']:
            print(f"   - Cluster {cluster['cluster_id']}: {cluster['size']} messages")
    else:
        print("   ✗ Not enough data for clustering")

    # Test 4: Activity Prediction
    print("\n6. Testing Activity Prediction Model...")
    prediction_result = ml_models.predict_user_activity(df)
    if prediction_result:
        print(f"   ✓ Prediction Model Trained")
        print(f"   - Training Accuracy: {prediction_result['train_accuracy']*100:.2f}%")
        print(f"   - Test Accuracy: {prediction_result['test_accuracy']*100:.2f}%")
        print(f"   - Top Feature: {prediction_result['feature_importance'].iloc[0]['feature']}")
    else:
        print("   ✗ Not enough data for prediction model")

    # Test 5: User Personality Insights
    print("\n7. Testing User Personality Insights...")
    users = df[df['user'] != 'group_notification']['user'].unique()[:2]
    if len(users) > 0:
        test_user = users[0]
        insights = ml_models.get_user_personality_insights(df, test_user)
        if insights:
            print(f"   ✓ Personality Insights for '{test_user}'")
            print(f"   - Message Style: {insights['message_style']}")
            print(f"   - Activity Type: {insights['activity_type']}")
            print(f"   - Emoji Usage: {insights['emoji_user']}")
            print(f"   - Engagement: {insights['engagement_percentage']:.1f}%")
        else:
            print("   ✗ Could not generate insights")

    print("\n" + "=" * 60)
    print("✓ All ML Features Tested Successfully!")
    print("=" * 60)
    print("\nTo run the full app:")
    print("  streamlit run code/app.py")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_ml_features()
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
