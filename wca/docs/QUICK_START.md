# Quick Start Guide

## Installation (One-time Setup)

### Step 1: Install Required Packages
```bash
pip3 install -r requirements.txt
```

This will install:
- streamlit (web app)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)
- scikit-learn (ML algorithms)
- vaderSentiment, textblob (sentiment analysis)
- wordcloud, urlextract, emoji (text analysis)

### Step 2: Verify Installation
```bash
python3 demo_ml.py
```

**Expected output**: All ML features should test successfully ✅

---

## Running the Application

### Option 1: Using Streamlit (Recommended)
```bash
streamlit run code/app.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: Quick ML Test
```bash
python3 demo_ml.py
```

Runs ML models on sample chat without UI

---

## Using the App

### 1. Export WhatsApp Chat
- Open WhatsApp
- Go to chat → ⋮ → More → Export chat
- Choose "Without Media"
- Save the `.txt` file

### 2. Upload to App
- Click "Browse files" in sidebar
- Select your exported chat file
- Choose user to analyze (or "Overall" for everyone)
- Enable "ML Analysis" checkbox
- Click "Show Analysis"

### 3. Explore Features

**Statistical Analysis** (always shown):
- Message counts and statistics
- Activity timelines and heatmaps
- Word clouds and emoji analysis

**ML Analysis** (when enabled):
- 📊 Sentiment Analysis - Positive/Negative/Neutral classification
- 📚 Topic Discovery - Find conversation themes (LDA/NMF)
- 🔍 Message Clustering - Group similar messages
- 🎯 Activity Prediction - Predict next user to message
- 🧠 Personality Insights - User communication style

---

## Sample Commands

### Test with Existing Sample Data
```bash
# Run the app
streamlit run code/app.py

# Then upload one of these sample chats:
# - chats/WhatsApp Chat with Jit Ghosh.txt
# - chats/WhatsApp Chat with GSoC 2026.txt
```

### Run Demo Test
```bash
# Quick test of all ML features
python3 demo_ml.py
```

---

## Troubleshooting

### Issue: "command not found: streamlit"
**Solution**:
```bash
pip3 install streamlit
```

### Issue: Import errors
**Solution**:
```bash
pip3 install -r requirements.txt
```

### Issue: "File not found" for stop words
**Solution**: The stop words file path is hardcoded. Make sure you're running from the `wca/` directory.

### Issue: Streamlit opens blank page
**Solution**:
- Check if port 8501 is available
- Try: `streamlit run code/app.py --server.port 8502`

---

## File Paths

Make sure you're in the correct directory:
```bash
cd /Users/arijitsingh/Documents/thapar_sem5/ml_project/wca
```

Then run commands from here.

---

## What to Expect

### Processing Time
- Small chats (<500 messages): < 5 seconds
- Medium chats (500-2000 messages): 5-15 seconds
- Large chats (>2000 messages): 15-30 seconds

### ML Features Performance
- **Sentiment Analysis**: Works with any chat size
- **Topic Modeling**: Needs at least 50 messages
- **Message Clustering**: Needs at least 20 messages
- **Activity Prediction**: Needs at least 50 messages
- **Personality Insights**: Works with any user

---

## Example Workflow

1. **Start the app**:
   ```bash
   streamlit run code/app.py
   ```

2. **Upload**: Use `chats/WhatsApp Chat with GSoC 2026.txt`

3. **Select**: Choose "Overall" for group analysis

4. **Enable**: Check "Enable ML Analysis"

5. **Analyze**: Click "Show Analysis"

6. **Explore**:
   - View sentiment distribution
   - Click "Discover Topics" (try 5 topics, LDA method)
   - Click "Cluster Messages" (try 5 clusters)
   - Click "Train Prediction Model" (for Overall only)

7. **Individual Analysis**:
   - Select a specific user from dropdown
   - See their personality insights

---

## Tips for Best Results

✅ **Do**:
- Use chats with at least 100 messages
- Enable ML features for deeper insights
- Try both LDA and NMF for topic modeling
- Analyze individual users for personality insights

❌ **Don't**:
- Upload chats with less than 20 messages (ML won't work well)
- Expect 100% prediction accuracy (human behavior is complex)
- Upload very large files (>50MB) - may cause memory issues

---

## Next Steps

After running successfully:

1. **Experiment**: Try different parameters (number of topics, clusters)
2. **Compare**: Analyze different users and compare personalities
3. **Export**: Take screenshots of interesting insights
4. **Extend**: Modify `ml_models.py` to add more features

---

## Support

- **Documentation**: See `README.md` for detailed info
- **ML Features**: See `ML_FEATURES_SUMMARY.md` for algorithm details
- **Issues**: Check code comments in `ml_models.py`

---

**Ready to analyze your WhatsApp chats with Machine Learning! 🚀**
