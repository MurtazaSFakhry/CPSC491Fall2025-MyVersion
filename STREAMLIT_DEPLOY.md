# 🚀 Quick Deployment Guide for Streamlit Cloud

## Step-by-Step Instructions

### 1. Go to Streamlit Cloud
👉 Visit: **https://share.streamlit.io/**

### 2. Sign In
- Click **"Sign in"** 
- Use your GitHub account

### 3. Create New App
- Click **"New app"** button
- Fill in the form:
  - **Repository**: `MurtazaSFakhry/CPSC491Fall2025-MyVersion`
  - **Branch**: `main`
  - **Main file path**: `VectordB/streamlit_app.py`

### 4. Configure Secrets (IMPORTANT!)
Before deploying, click **"Advanced settings"** → **"Secrets"**

Add these secrets (replace with your actual keys):
```toml
OPENAI_API_KEY = "sk-proj-your-actual-openai-key-here"
SERPAPI_KEY = "your-actual-serpapi-key-here"
```

💡 **Where to get keys:**
- **OpenAI API Key**: https://platform.openai.com/api-keys
- **SerpAPI Key**: https://serpapi.com/manage-api-key (optional, for external search)

### 5. Deploy!
- Click **"Deploy!"**
- Wait 2-3 minutes for build to complete
- Your app will be live at: `https://[your-app-name].streamlit.app`

### 6. Share Your App
Once deployed, you'll get a public URL like:
```
https://emergency-alerts-chat.streamlit.app
```

---

## ✅ What's Included
- ✅ ChromaDB with 5,301 embeddings (included in repo)
- ✅ Streamlit app configuration
- ✅ All dependencies in requirements.txt
- ✅ Proper secrets management

## 🔧 Troubleshooting

**App won't start?**
- Check secrets are set correctly
- Verify OpenAI API key is valid
- Check logs in Streamlit Cloud dashboard

**ChromaDB errors?**
- The database is included in the repo (2.13 MB)
- It will automatically load on deployment

**API errors?**
- Verify your OpenAI API key has credits
- Check API key format (should start with `sk-`)

---

## 📱 Your App Features
- 🤖 AI-powered emergency alerts assistant
- 🔍 Search 5,301+ documents
- 📚 Source citations
- 💬 Chat history
- 🌐 External search integration

**Repository**: https://github.com/MurtazaSFakhry/CPSC491Fall2025-MyVersion

---

Need help? Check the full DEPLOYMENT.md guide!
