# 🚀 Deploying YT-Scraper to Vercel

This guide explains how to deploy your **DSA Video Knowledge Assistant** (`yt-scraper`) to Vercel in less than 5 minutes.

---

## 📋 Environment Variables Checklist

Before deploying, make sure you have the following 3 environment variables ready:

| Variable Name | Value Description | Where to find it |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | `gsk_...` | Groq Developer Dashboard |
| `QDRANT_URL` | `https://719b1894...cloud.qdrant.io` | Qdrant Cloud Console |
| `QDRANT_API_KEY` | `eyJhbGci...` | Qdrant Cloud Console |

---

## 🎯 Deployment Method 1: GitHub + Vercel Dashboard (Recommended)

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Deploy DSA Video Knowledge Assistant to Vercel"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/yt-scraper.git
   git push -u origin main
   ```

2. **Connect to Vercel**:
   - Go to [Vercel Dashboard](https://vercel.com/new).
   - Click **Import Repository** and select `yt-scraper`.
   - Vercel will automatically detect `vercel.json` and `app.py`.

3. **Add Environment Variables**:
   - In the **Environment Variables** section, add:
     - `GROQ_API_KEY` = your_groq_key
     - `QDRANT_URL` = your_qdrant_cloud_url
     - `QDRANT_API_KEY` = your_qdrant_cloud_api_key

4. **Click Deploy**:
   - Vercel will build your serverless function and give you a live production URL (e.g. `https://yt-scraper-xyz.vercel.app`).

---

## ⚡ Deployment Method 2: Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Deploy directly from terminal**:
   ```bash
   vercel
   ```
   Follow the prompts to link your account and project.

3. **Set Production Environment Variables**:
   ```bash
   vercel env add GROQ_API_KEY production
   vercel env add QDRANT_URL production
   vercel env add QDRANT_API_KEY production
   ```

4. **Deploy to Production**:
   ```bash
   vercel --prod
   ```

---

## ⚙️ How Production Architecture Works

- **Serverless API (`app.py`)**: Runs on Vercel Serverless Functions, routing `/` to the sleek UI and `/api/search` to the FastAPI backend.
- **Cloud Vector Store (`Qdrant Cloud`)**: Houses all transcript vector embeddings safely in the cloud, allowing instant high-speed vector retrieval across cold starts.
- **LLM Engine (`Groq API`)**: Generates grounded, 3-5 point concise answers with exact timestamp selection.

