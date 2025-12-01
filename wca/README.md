# WhatsApp Chat Analyzer with Machine Learning

A comprehensive Streamlit web application that analyzes WhatsApp chat exports with advanced **Machine Learning** capabilities including sentiment analysis, topic modeling, message clustering, and user behavior prediction.

## 🚀 Live Demo

**[Try the live app here!](#)** *(Add your Streamlit Cloud URL after deployment)*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](#)

## Features

### 📊 Statistical Analysis
- **Message Statistics**: Total messages, words, media shared, and links
- **User Activity**: Most active users and contribution percentages
- **Temporal Analysis**: Monthly/daily timelines, activity heatmaps
- **Activity Maps**: Busiest days, months, and time periods

### 📝 Text Analysis
- **Word Cloud**: Visual representation of most frequent words
- **Common Words**: Top 20 most used words (with Hinglish stop words filtering)
- **Emoji Analysis**: Emoji frequency distribution with interactive pie charts

### 🤖 Machine Learning Features

#### 1. Sentiment Analysis
- **Multi-method Support**: VADER (optimized for social media), TextBlob, or rule-based fallback
- **Sentiment Distribution**: Positive, Neutral, and Negative message classification
- **Polarity Scores**: Quantitative sentiment measurement (-1 to +1)
- **Temporal Trends**: Sentiment changes over time
- **User Comparison**: Most positive users in group chats

#### 2. Topic Modeling
- **LDA (Latent Dirichlet Allocation)**: Statistical topic discovery
- **NMF (Non-negative Matrix Factorization)**: Alternative topic extraction
- **Configurable Topics**: Choose 3-10 topics to discover
- **Keyword Extraction**: Top 10 keywords per topic

#### 3. Message Clustering
- **K-Means Clustering**: Group similar messages together
- **TF-IDF Vectorization**: Smart text representation
- **Representative Messages**: See typical messages from each cluster
- **Cluster Insights**: Understand conversation patterns

#### 4. User Activity Prediction
- **Random Forest Classifier**: Predicts which user will message next
- **Temporal Features**: Hour, day of week, weekend patterns
- **Conversation Flow**: Previous user patterns
- **Feature Importance**: Understand what drives user activity
- **Model Accuracy Metrics**: Training and test performance

#### 5. User Personality Insights
- **Message Style**: Verbose vs Concise communicator
- **Activity Type**: Morning person, Night owl, etc.
- **Emoji Usage**: High vs Low emoji user
- **Engagement Level**: Contribution percentage
- **Response Time**: Median response time analysis

## Installation

### 1. Clone the Repository
```bash
cd wca
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Download NLTK Data for TextBlob
```bash
python -m textblob.download_corpora
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
3. Enable/disable ML features using the checkbox
4. Click **Show Analysis**

## Project Structure

```
wca/
├── code/                       # Source code
│   ├── app.py                 # Main Streamlit application
│   ├── preprocessor.py        # WhatsApp chat parsing
│   ├── helper.py              # Statistical analysis functions
│   └── ml_models.py           # Machine Learning models (5 models)
├── docs/                       # Documentation
│   ├── DEPLOY_NOW.md          # Quick deployment guide
│   ├── DEPLOYMENT.md          # Complete deployment guide
│   ├── ML_FEATURES_SUMMARY.md # ML algorithms explained
│   └── QUICK_START.md         # Usage instructions
├── chats/                      # Sample WhatsApp chat files
│   └── WhatsApp Chat with Jit Ghosh.txt
├── stop_words/                 # Stop words for text processing
│   └── stop_hinglish.txt      # Hinglish stop words (1057+ words)
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
├── requirements.txt            # Python dependencies
├── demo_ml.py                  # ML features demo/test script
├── START_HERE.md              # Quick start guide
└── README.md                   # This file
```

## Machine Learning Models Explained

### Sentiment Analysis
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) which is specifically designed for social media text. It handles:
- Slang and informal language
- Emoticons and emojis
- Punctuation emphasis (!!!, ???)
- Capitalization for emphasis

**Algorithm**: Compound score calculation
- Score ≥ 0.05: Positive
- Score ≤ -0.05: Negative
- Otherwise: Neutral

### Topic Modeling

**LDA (Latent Dirichlet Allocation)**
- Probabilistic model that assumes documents are mixtures of topics
- Each topic is a distribution over words
- Discovers hidden thematic structure

**NMF (Non-negative Matrix Factorization)**
- Linear algebra approach to topic extraction
- Factorizes term-document matrix into two matrices
- Often produces more interpretable topics

### Message Clustering
**K-Means Algorithm**
1. Convert messages to TF-IDF vectors
2. Initialize K cluster centroids randomly
3. Assign each message to nearest centroid
4. Update centroids based on assigned messages
5. Repeat until convergence

### Activity Prediction
**Random Forest Classifier**
- Ensemble of decision trees
- Features: hour (sin/cos encoding), day of week, weekend flag, message length, previous user
- Predicts which user is likely to send the next message
- Uses temporal patterns and conversation flow

## Dependencies

### Core
- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Visualization
- **matplotlib**: Plotting library
- **seaborn**: Statistical visualization
- **wordcloud**: Word cloud generation

### NLP
- **urlextract**: URL detection
- **emoji**: Emoji parsing
- **textblob**: Sentiment analysis (optional)
- **vaderSentiment**: Social media sentiment analysis (optional)

### Machine Learning
- **scikit-learn**: ML algorithms (KMeans, LDA, NMF, RandomForest, TF-IDF)

## Technical Details

### Data Preprocessing
1. **Regex Pattern Matching**: Extracts date, time, user, and message from WhatsApp format
2. **Feature Engineering**: Creates temporal features (year, month, day, hour, period)
3. **User Classification**: Separates group notifications from user messages

### Text Processing
1. **Stop Words Filtering**: Removes common Hinglish words
2. **Tokenization**: Splits messages into words
3. **Vectorization**: TF-IDF or Count Vectorization for ML models

### Supported Chat Format
```
dd/mm/yyyy, hh:mm - Username: Message text
```

Example:
```
01/07/2025, 08:44 - John: Hi there!
01/07/2025, 08:45 - Jane: Hello John!
```

## Performance Considerations

- **Sentiment Analysis**: O(n) where n = number of messages
- **Topic Modeling**: O(n × k × i) where k = topics, i = iterations
- **Clustering**: O(n × k × i) where k = clusters, i = iterations
- **Prediction Model**: O(n × log(n)) for Random Forest

For large chats (>10,000 messages), ML operations may take 10-30 seconds.

## Limitations

1. **Language Support**: Optimized for English and Hinglish
2. **File Format**: Only supports standard WhatsApp text export format
3. **Memory**: Large chat files (>50MB) may cause memory issues
4. **Path Hardcoding**: Stop words file uses absolute path (needs configuration)

## Future Enhancements

- [ ] Language detection and multilingual support
- [ ] Named Entity Recognition (NER)
- [ ] Conversation thread detection
- [ ] Export analysis reports (PDF/HTML)
- [ ] Real-time chat monitoring
- [ ] Deep learning models (BERT, transformers)
- [ ] User authentication and data privacy
- [ ] Configurable file paths

## Contributing

This is a semester 5 ML project for Thapar University. Contributions are welcome!

## License

Educational project - feel free to use and modify.

## Acknowledgments

- **scikit-learn**: For ML algorithms
- **Streamlit**: For the amazing web framework
- **VADER**: For social media sentiment analysis
- WhatsApp for exportable chat format

---

**Built with ❤️ for ML Course - Thapar University**
