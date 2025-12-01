# Deployment Guide

This guide covers deploying the WhatsApp Chat Analyzer to various platforms.

---

## Option 1: Streamlit Cloud (Recommended - FREE) 🌟

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Steps

#### 1. Push to GitHub

```bash
# Initialize git (if not already done)
cd /Users/arijitsingh/Documents/thapar_sem5/ml_project/wca
git init

# Add all files
git add .

# Commit
git commit -m "WhatsApp Chat Analyzer with ML features"

# Create a new repository on GitHub (https://github.com/new)
# Name it: whatsapp-chat-analyzer

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/whatsapp-chat-analyzer.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `YOUR_USERNAME/whatsapp-chat-analyzer`
4. Set **Main file path**: `code/app.py`
5. Click "Deploy"

**Your app will be live at**: `https://YOUR_USERNAME-whatsapp-chat-analyzer.streamlit.app`

### Deployment Time
- First deploy: 3-5 minutes
- Subsequent deploys: 1-2 minutes (automatic on git push)

---

## Option 2: Heroku (FREE Tier Available)

### Prerequisites
- Heroku account
- Heroku CLI installed

### Additional Files Needed

Create `Procfile`:
```bash
echo "web: streamlit run code/app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
```

Create `setup.sh`:
```bash
cat > setup.sh << 'EOF'
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
EOF
```

### Deploy

```bash
# Login to Heroku
heroku login

# Create app
heroku create whatsapp-analyzer-ml

# Push to Heroku
git push heroku main

# Open app
heroku open
```

---

## Option 3: Docker (Self-Hosted)

### Create Dockerfile

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "code/app.py", "--server.address", "0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t whatsapp-analyzer .

# Run container
docker run -p 8501:8501 whatsapp-analyzer

# Access at http://localhost:8501
```

---

## Option 4: Google Cloud Run

### Prerequisites
- Google Cloud account
- gcloud CLI installed

### Steps

```bash
# Build and submit
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/whatsapp-analyzer

# Deploy
gcloud run deploy whatsapp-analyzer \
  --image gcr.io/YOUR_PROJECT_ID/whatsapp-analyzer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Option 5: AWS EC2

### Steps

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04
   - Instance type: t2.micro (free tier)
   - Security group: Allow port 8501

2. **SSH and Setup**

```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip -y

# Clone your repo
git clone https://github.com/YOUR_USERNAME/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer

# Install dependencies
pip3 install -r requirements.txt

# Run with nohup (keeps running after logout)
nohup streamlit run code/app.py --server.port 8501 --server.address 0.0.0.0 &
```

3. **Access at**: `http://your-instance-ip:8501`

---

## Option 6: Local Network (Testing)

### For development/testing on local network

```bash
# Run with network access
streamlit run code/app.py --server.address 0.0.0.0

# Access from other devices on same network
# http://YOUR_LOCAL_IP:8501
# Find your IP: ifconfig (Mac/Linux) or ipconfig (Windows)
```

---

## Environment Variables (Optional)

For sensitive configurations, create `.streamlit/secrets.toml`:

```toml
# Example secrets
[general]
app_name = "WhatsApp Chat Analyzer"
version = "1.0.0"

# Add any API keys here
```

**Note**: Never commit `secrets.toml` to Git! It's in `.gitignore`.

---

## Deployment Checklist

Before deploying, ensure:

- [ ] All hardcoded paths removed ✅ (already done)
- [ ] `requirements.txt` is complete ✅
- [ ] `.gitignore` is configured ✅
- [ ] Code runs locally without errors
- [ ] Sample data is removed (or excluded in .gitignore)
- [ ] No sensitive information in code
- [ ] README.md is complete ✅

---

## Post-Deployment

### Monitor Your App

**Streamlit Cloud**:
- View logs in Streamlit dashboard
- Automatic restarts on errors
- Automatic deploys on git push

**Heroku**:
```bash
heroku logs --tail
```

**Docker**:
```bash
docker logs -f container_id
```

### Update Your App

**Streamlit Cloud**: Just push to GitHub
```bash
git add .
git commit -m "Update features"
git push origin main
# App automatically updates!
```

**Heroku**:
```bash
git push heroku main
```

**Docker**:
```bash
docker build -t whatsapp-analyzer .
docker run -p 8501:8501 whatsapp-analyzer
```

---

## Troubleshooting

### Issue: App won't start

**Check**:
- Correct main file path (`code/app.py`)
- All dependencies in `requirements.txt`
- Python version compatibility

### Issue: Import errors

**Solution**: Add missing packages to `requirements.txt`

### Issue: File not found errors

**Solution**: All absolute paths have been converted to relative paths ✅

### Issue: Memory errors on Streamlit Cloud

**Solution**:
- Streamlit Cloud free tier has 1GB RAM
- Optimize by processing smaller chunks
- Or upgrade to paid tier

### Issue: Slow performance

**Solution**:
- Cache expensive operations with `@st.cache_data`
- Reduce default number of topics/clusters
- Process only recent messages for large chats

---

## Performance Optimization

Add caching to app.py:

```python
import streamlit as st

@st.cache_data
def load_and_process_data(file_content):
    # Your preprocessing here
    return df
```

---

## Custom Domain (Optional)

### Streamlit Cloud
1. Go to app settings
2. Add custom domain
3. Follow DNS configuration instructions

### Heroku
```bash
heroku domains:add www.your-domain.com
# Configure DNS as shown
```

---

## Costs

| Platform | Free Tier | Paid |
|----------|-----------|------|
| **Streamlit Cloud** | ✅ Unlimited (1 public app) | $20/month for private |
| **Heroku** | ✅ 550-1000 hrs/month | $7/month/dyno |
| **Google Cloud Run** | ✅ 2M requests/month | Pay per use |
| **AWS EC2** | ✅ t2.micro 750 hrs/month | Varies |
| **Docker (self-hosted)** | ✅ Free | Server costs only |

---

## Recommended: Streamlit Cloud

**Why?**
- ✅ Completely free for public apps
- ✅ Automatic updates on git push
- ✅ Built-in SSL/HTTPS
- ✅ No server management
- ✅ Perfect for ML projects
- ✅ Great for portfolio

**Just push to GitHub and deploy in 2 clicks!**

---

## Share Your Deployed App

Once deployed, share the URL:
- `https://YOUR_USERNAME-whatsapp-chat-analyzer.streamlit.app`

Add to your:
- GitHub README
- Resume/Portfolio
- LinkedIn projects
- College assignments

---

## Security Notes

⚠️ **Important**:
- Users upload their own chat data
- Data is NOT stored permanently
- Each session is isolated
- Files are cleared when session ends
- No data is sent to external servers

**Privacy**: All processing happens in-app. No data is logged or stored.

---

## Next Steps

1. ✅ Fix hardcoded paths (done)
2. ✅ Create deployment configs (done)
3. 🔲 Push to GitHub
4. 🔲 Deploy to Streamlit Cloud
5. 🔲 Test deployed app
6. 🔲 Share URL!

---

**Ready to deploy! Let's get your ML project online! 🚀**
