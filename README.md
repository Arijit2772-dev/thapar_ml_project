# WhatsApp Chat Analyzer with Machine Learning

A comprehensive Streamlit web application that analyzes WhatsApp chat exports with advanced **Machine Learning** capabilities including sentiment analysis and user personality insights.

## 🚀 Live Demo

**[Try the live app here!](https://bigdawgs2005.streamlit.app/)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bigdawgs2005.streamlit.app/)

## ✨ Key Highlights

- 📊 **Comprehensive Statistics**: Message counts, timelines, activity heatmaps
- 🎨 **Beautiful Visualizations**: Word clouds, charts, and interactive plots
- 🤖 **ML-Powered Analysis**: Sentiment analysis and personality insights
- 🌙 **Dark Mode Support**: Fully compatible with Streamlit's native dark/light themes
- 🚀 **Easy to Use**: Upload chat → Click analyze → Get insights
- 🇮🇳 **Hinglish Support**: Optimized for Indian English and Hindi mixed conversations

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

#### 2. User Personality Insights
- **Message Style**: Verbose vs Concise communicator
- **Activity Type**: Morning person, Night owl, etc.
- **Emoji Usage**: High vs Low emoji user
- **Engagement Level**: Contribution percentage
- **Response Time**: Median response time analysis

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
thapar_ml_project/
├── code/                       # Source code
│   ├── app.py                 # Main Streamlit application
│   ├── preprocessor.py        # WhatsApp chat parsing
│   ├── helper.py              # Statistical analysis functions
│   └── ml_models.py           # Machine Learning models (Sentiment & Personality)
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
├── packages.txt                # System dependencies for deployment
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

### User Personality Analysis
**Pattern Recognition**
- Analyzes message length patterns to determine communication style
- Identifies peak activity hours to determine activity type
- Calculates emoji usage frequency
- Measures response time patterns
- Computes engagement percentage in conversations

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
- **scikit-learn**: ML algorithms and preprocessing tools

## Technical Details

### Data Preprocessing
1. **Regex Pattern Matching**: Extracts date, time, user, and message from WhatsApp format
2. **Feature Engineering**: Creates temporal features (year, month, day, hour, period)
3. **User Classification**: Separates group notifications from user messages

### Text Processing
1. **Stop Words Filtering**: Removes common Hinglish words
2. **Tokenization**: Splits messages into words
3. **Sentiment Analysis**: Uses VADER or TextBlob for polarity scoring

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
- **Personality Analysis**: O(n) where n = number of messages
- **Statistical Features**: O(n) for most operations

The app works best with chat files containing 50-10,000 messages. Very large chats (>10,000 messages) may take 10-30 seconds to analyze.

## Limitations

1. **Language Support**: Optimized for English and Hinglish
2. **File Format**: Only supports standard WhatsApp text export format
3. **Memory**: Large chat files (>50MB) may cause memory issues
4. **Dataset Size**: Works best with 50-10,000 messages

## Future Enhancements

- [ ] Language detection and multilingual support
- [ ] Named Entity Recognition (NER)
- [ ] Export analysis reports (PDF/HTML)
- [ ] Topic modeling for very large chat datasets (500+ messages)
- [ ] Deep learning models (BERT, transformers)
- [ ] User authentication and data privacy

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
