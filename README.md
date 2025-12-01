# WhatsApp Chat Analyzer with Machine Learning

A comprehensive Streamlit web application that analyzes WhatsApp chat exports using advanced **Machine Learning** techniques including sentiment analysis, predictive modeling, and user behavior classification.

## 🚀 Live Demo

**[Try the live app here!](https://bigdawgs2005.streamlit.app/)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bigdawgs2005.streamlit.app/)

## 👥 Team

**Team Big_Dawgs**
- Arijit Singh (102303916)
- Bhoomika (102303815)
- Namya (102303848)

**Submitted to:** Dr. Manisha Malik
**Course:** Machine Learning (UCS654)
**Institution:** Thapar Institute of Engineering & Technology

## ✨ Key Highlights

- 🤖 **5 ML Models**: Classification (Logistic Regression, Naive Bayes), Regression (Random Forest), Sentiment Analysis
- 📊 **Comprehensive Statistics**: Message counts, timelines, activity heatmaps
- 🎨 **Beautiful Visualizations**: Word clouds, charts, and interactive plots
- 🌙 **Dark Mode Support**: Fully compatible with Streamlit's native dark/light themes
- 🚀 **Easy to Use**: Upload chat → Click analyze → Get insights
- 🇮🇳 **Hinglish Support**: Optimized for Indian English and Hindi mixed conversations

## 🤖 Machine Learning Features

### 1. Emoji Usage Classifier (Binary Classification)
- **Algorithm**: Logistic Regression
- **Task**: Predict whether a message will contain emojis
- **Accuracy**: ~95.7%
- **Features**: Message length, word count, punctuation, time of day, weekend flag
- **Interpretability**: Feature coefficients show exclamation marks increase emoji probability by 2.3x

### 2. Message Length Predictor (Regression)
- **Algorithm**: Random Forest Regressor (50 trees)
- **Task**: Predict message length in characters
- **Performance**: MAE ~16 characters, R² = 0.42
- **Features**: Hour, day of week, user, previous message length
- **Key Insight**: Previous message length is the strongest predictor (68% importance)

### 3. User Style Classifier (Multi-class Classification)
- **Algorithm**: Naive Bayes with TF-IDF vectorization
- **Task**: Identify which user wrote a message based on writing style
- **Accuracy**: ~66%
- **Features**: TF-IDF vectors (100 features), unigrams and bigrams
- **Output**: Top characteristic words per user

### 4. Sentiment Analysis
- **Algorithm**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Task**: Classify messages as Positive, Neutral, or Negative
- **Optimized for**: Social media text, emojis, slang, punctuation emphasis
- **Output**: Sentiment distribution, temporal trends, user comparison

### 5. User Personality Insights
- **Type**: Statistical Pattern Recognition
- **Metrics**: Message style, activity type, emoji usage, engagement, response time
- **Output**: Personality profile for each user

## Features

### 📊 Statistical Analysis
- **Message Statistics**: Total messages, words, media shared, and links
- **User Activity**: Most active users and contribution percentages
- **Temporal Analysis**: Monthly/daily timelines, activity heatmaps
- **Activity Maps**: Busiest days, months, and time periods

### 📝 Text Analysis
- **Word Cloud**: Visual representation of most frequent words
- **Common Words**: Top 20 most used words (with 1057+ Hinglish stop words filtering)
- **Emoji Analysis**: Emoji frequency distribution with interactive pie charts

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Arijit2772-dev/thapar_ml_project.git
cd thapar_ml_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Export WhatsApp Chat
1. Open WhatsApp on your phone
2. Go to the chat you want to analyze
3. Tap on **⋮** (three dots) → **More** → **Export chat**
4. Choose **Without Media**
5. Save the `.txt` file

### 2. Run the Application
```bash
streamlit run code/app.py
```

### 3. Upload and Analyze
1. Upload your exported chat file using the sidebar
2. Select **Overall** or a specific user to analyze
3. Enable ML features using the checkbox
4. Click **Show Analysis**
5. Expand ML model sections to see predictions

## Project Structure

```
thapar_ml_project/
├── code/                       # Source code
│   ├── app.py                 # Main Streamlit application
│   ├── preprocessor.py        # WhatsApp chat parsing
│   ├── helper.py              # Statistical analysis functions
│   └── ml_models.py           # Machine Learning models
├── chats/                      # Sample WhatsApp chat files
│   └── WhatsApp Chat with Jit Ghosh.txt
├── stop_words/                 # Stop words for text processing
│   └── stop_hinglish.txt      # Hinglish stop words (1057+ words)
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies for deployment
├── PROJECT_REPORT.md           # Comprehensive project report
├── PROJECT_REPORT.pdf          # PDF version of report
└── README.md                   # This file
```

## Dependencies

### Core
- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Machine Learning
- **scikit-learn**: ML algorithms (Logistic Regression, Random Forest, Naive Bayes, TF-IDF)
- **vaderSentiment**: Social media sentiment analysis

### Visualization
- **matplotlib**: Plotting library
- **seaborn**: Statistical visualization
- **wordcloud**: Word cloud generation

### NLP
- **urlextract**: URL detection
- **emoji**: Emoji parsing

## Technical Details

### ML Model Training Process

#### 1. Feature Engineering
```python
# Temporal features
hour_of_day, day_of_week, is_weekend

# Text features
msg_length, word_count, has_emoji, has_question, has_exclamation

# Lag features
prev_msg_length, time_diff

# User encoding
user_encoded = LabelEncoder().fit_transform(user)
```

#### 2. Train-Test Split
- 80% training data
- 20% test data
- Chronological split (time-based)

#### 3. Model Evaluation
- **Classification**: Accuracy, Confusion Matrix, Feature Coefficients
- **Regression**: MAE (Mean Absolute Error), R² Score, Feature Importance

### Data Preprocessing
1. **Regex Pattern Matching**: Extracts date, time, user, and message from WhatsApp format
2. **Feature Engineering**: Creates temporal, text, and lag features
3. **User Classification**: Separates group notifications from user messages
4. **Stop Words Filtering**: Removes 1057+ common Hinglish words

### Supported Chat Format
```
dd/mm/yyyy, hh:mm - Username: Message text
```

Example:
```
01/07/2025, 08:44 - John: Hi there!
01/07/2025, 08:45 - Jane: Hello John!
```

## Performance

### Model Performance
- **Emoji Classifier**: 95.7% test accuracy
- **Length Predictor**: MAE 16 chars, R² 0.42
- **User Classifier**: 66% test accuracy
- **Sentiment Analysis**: Real-time classification

### System Performance
- **Processing Speed**: O(n) where n = number of messages
- **Optimal Range**: 50-10,000 messages
- **Large Chats**: >10,000 messages may take 10-30 seconds

## Limitations

1. **Language Support**: Optimized for English and Hinglish
2. **File Format**: Only supports standard WhatsApp text export format
3. **Dataset Size**: ML models require minimum 20-50 messages
4. **User Classifier**: Accuracy decreases on very small datasets
5. **Sentiment**: Less accurate for pure Hindi or regional languages

## Future Enhancements

- [ ] Deep learning models (LSTM, BERT, Transformers)
- [ ] Topic modeling for large datasets (LDA, NMF)
- [ ] Named Entity Recognition (NER)
- [ ] Conversation threading and topic detection
- [ ] Export analysis reports (PDF/HTML)
- [ ] Multi-language support (Hindi, Tamil, Telugu)
- [ ] Real-time chat monitoring
- [ ] User authentication and data encryption

## Project Documentation

- **PROJECT_REPORT.md**: Complete project report with methodology, results, and analysis
- **PROJECT_REPORT.pdf**: PDF version for submission

## Deployment

Deployed on **Streamlit Cloud**: [https://bigdawgs2005.streamlit.app/](https://bigdawgs2005.streamlit.app/)

## Acknowledgments

- **Dr. Manisha Malik** for guidance and support
- **Thapar Institute** for providing resources
- **scikit-learn** for ML algorithms
- **Streamlit** for the web framework
- **VADER** for social media sentiment analysis
- **WhatsApp** for exportable chat format

---

**Machine Learning Project - Thapar Institute of Engineering & Technology**
**Semester 5 - December 2025**
