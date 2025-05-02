# 🏡 Real Estate Price Predictor

This is an advanced real estate analytics web application that predicts house prices using machine learning and provides rich insights like market trends, investment ROI, EMI calculators, weather, job market snapshots, and more — all in a visually interactive interface.

> 🔍 Built to assist home buyers, investors, and real estate analysts make smarter decisions.

---

## 🚀 Key Features

- 🧠 **Machine Learning Models** – XGBoost, Random Forest, and Linear Regression
- 📍 **City Snapshot** – Real-time map view, pricing, trends, popularity score
- 📊 **Market Analytics** – Prediction vs average comparison, multi-model outputs
- 💡 **AI-powered Suggestions** – Dynamic insights based on selected cities
- 📰 **Real Estate News** – Country-filtered, category-based interactive feed
- 📈 **ROI & Loan Calculator** – Estimate returns, EMI breakdown, generate PDF
- ☁️ **Live Weather API** – Get current weather for any selected city
- 💼 **Job Market View** – Understand employment opportunities in each region
- 🎉 **Community Tab & Listings** – Simulated property listings and chat
- 🌐 Hosted on **Streamlit Cloud** and fully open-source

---

## 🧠 Tech Stack

- **Frontend**: `Streamlit`, `Matplotlib`, `Seaborn`
- **Backend**: `Scikit-learn`, `XGBoost`, `Pandas`, `Joblib`
- **APIs Used**:  
  - OpenWeather API  
  - GNews API  
  - Ticketmaster Events API  
  - Dummy Job Listings & Property Listings (JSON)

---

📦 Repo: [GitHub](https://github.com/vibhutirohan/real-estate-price-predictor)

---


## 📄 Getting Started (Local Setup)

```bash
git clone https://github.com/vibhutirohan/real-estate-price-predictor.git
cd real-estate-price-predictor
pip install -r requirements.txt
streamlit run app.py
