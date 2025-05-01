import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import io
import base64
import requests
from fpdf import FPDF
from datetime import datetime
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

st.set_page_config(page_title="Explore, Predict & Invest Smartly", layout="wide")

# Load OpenWeather API key
weather_api_key = st.secrets["openweather"]["api_key"] if "openweather" in st.secrets else os.getenv("OPENWEATHER_API_KEY")

@st.cache_data
def load_city_data():
    return pd.read_csv("data/avg_price_by_city.csv")

@st.cache_resource
def load_models():
    return {
        "XGBoost": joblib.load("models/xgboost_model.pkl"),
        "Random Forest": joblib.load("models/randomforest_model.pkl"),
        "Linear Regression": joblib.load("models/linear_model.pkl")
    }

city_data = load_city_data()
models = load_models()

# Weather fetch function
def get_weather(city_name):
    api_key = weather_api_key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather = {
                "temp": data["main"]["temp"],
                "desc": data["weather"][0]["description"].capitalize(),
                "humidity": data["main"]["humidity"],
                "wind": data["wind"]["speed"]
            }
            return weather
        else:
            return None
    except Exception as e:
        return None


show_sidebar = st.sidebar.checkbox("Show sidebar", value=True)
if show_sidebar:
    st.sidebar.title("📌 Input Features")
    location_search = st.sidebar.text_input("🔍 Search City by Name")
    if location_search:
        filtered = city_data[city_data['City'].str.contains(location_search, case=False)]
        if not filtered.empty:
            city = filtered.iloc[0]['City']
            st.sidebar.success(f"✅ Found city: {city}")
        else:
            st.sidebar.error("City not found. Please check spelling.")
            city = st.sidebar.selectbox("City", city_data["City"].unique())
    else:
        city = st.sidebar.selectbox("City", city_data["City"].unique())

    model_choice = st.sidebar.selectbox("Model", list(models.keys()))
    area = st.sidebar.number_input("Area (sq ft)", value=2000)
    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
    gr_liv_area = st.sidebar.number_input("Above Ground Living Area", value=1800)
    garage_cars = st.sidebar.slider("Garage Capacity", 0, 5, 2)
    full_bath = st.sidebar.slider("Full Bathrooms", 0, 5, 2)
    bsmt = st.sidebar.number_input("Basement Area (sq ft)", value=1000)
    overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)
    year_built = st.sidebar.number_input("Year Built", 1900, 2025, value=2010)
    trend = st.sidebar.slider("Adjust market trend (%)", -10, 10, 3)
    currency_toggle = st.sidebar.radio("Preferred Currency", ("INR", "USD"))
else:
    st.sidebar.info("Sidebar is hidden. Enable it above.")
    city = "Mumbai"
    model_choice = "XGBoost"
    area, bedrooms, bathrooms, gr_liv_area = 2000, 3, 2, 1800
    garage_cars, full_bath, bsmt, overall_qual, year_built = 2, 2, 1000, 6, 2010
    trend = 3
    currency_toggle = "INR"

exchange_rate = 0.012 if currency_toggle == "USD" else 1
currency_symbol = "$" if currency_toggle == "USD" else "₹"
input_data = pd.DataFrame([{
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "GrLivArea": gr_liv_area,
    "GarageCars": garage_cars,
    "FullBath": full_bath,
    "TotalBsmtSF": bsmt,
    "OverallQual": overall_qual,
    "YearBuilt": year_built
}])



# Tabs setup
tab1, tab2, tab3, tab4, tab5, tab6, tab7,tab8,tab9 = st.tabs(["🏠 Prediction", "📈 Analytics", "📰 News", "🏙️ City Snapshot", "🔍 Explore City", "💰 Investment Score", "🏦 Loan & EMI Calculator","💬 Community Chat", "📊 Price Forecast Dashboard"])

# --- Tab 1: Prediction ---
with tab1:
    st.header("🏠 House Price Prediction")
    if st.button("Predict Price 💰"):
        with st.spinner("🔍 Predicting house price..."):
            time.sleep(1)  # simulate delay
            selected_model = models[model_choice]
            pred_price = selected_model.predict(input_data)[0] * exchange_rate
            city_row = city_data[city_data["City"] == city].iloc[0]
            rate = city_row["PricePerSqFt"] * exchange_rate
            lat, lon = city_row["Lat"], city_row["Lon"]
            avg_price = area * rate
            lower, upper = pred_price * 0.9, pred_price * 1.1

            st.session_state.pred_price = pred_price
            st.session_state.avg_price = avg_price
            st.session_state.lat = lat
            st.session_state.lon = lon

        st.success(f"**Predicted Price: {currency_symbol}{pred_price:,.2f}**")
        st.info(f"📍 Avg Price in {city}: {currency_symbol}{avg_price:,.2f}")
        st.warning(f"🔎 Confidence Range: {currency_symbol}{lower:,.2f} – {currency_symbol}{upper:,.2f}")
        st.toast("🎉 Prediction Complete!", icon="🎯")

# --- Tab 2: Analytics ---
with tab2:
    st.header("📊 Market Analytics")
    if 'pred_price' in st.session_state:
        pred_price = st.session_state.pred_price
        avg_price = st.session_state.avg_price
        lat = st.session_state.lat
        lon = st.session_state.lon

        st.subheader("📊 Price Comparison")
        st.bar_chart(pd.DataFrame({"Predicted": [pred_price], "Average": [avg_price]}))

        st.subheader("🔁 Model Prediction Comparison")
        all_predictions = {name: model.predict(input_data)[0] * exchange_rate for name, model in models.items()}
        comparison_df = pd.DataFrame.from_dict(all_predictions, orient='index', columns=['Predicted Price'])
        st.bar_chart(comparison_df)

        st.subheader("📍 Location Map")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

        st.subheader("📈 Adjusted Price Trend Over the Years")
        years = np.arange(2015, 2026)
        trend_prices = [avg_price * (1 + (trend / 100) * (year - 2015)) for year in years]
        trend_df = pd.DataFrame({"Year": years, "Estimated Avg Price": trend_prices})
        st.line_chart(trend_df.set_index("Year"))
    else:
        st.info("⚠️ Please make a prediction in the '🏠 Prediction' tab first.")

# --- Tab 3: News ---
with tab3:
    st.header("🗞️ Real Estate News Highlights")

    def fetch_news(region, category=None, max_age_days=7):
        api_key = st.secrets["gnews"]["api_key"]
        base_query = "real+estate"
        query = base_query if category is None else f"{base_query}+{category}"
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&country={region}&max=10&token={api_key}&max_age={max_age_days}"
        r = requests.get(url)
        articles = r.json().get("articles", [])

        if not articles and category is not None:
            fallback_url = f"https://gnews.io/api/v4/search?q={base_query}&lang=en&country={region}&max=10&token={api_key}&max_age={max_age_days}"
            r = requests.get(fallback_url)
            articles = r.json().get("articles", [])

        return articles

    def render_news_list(title, news_list):
        st.subheader(title)
        if news_list:
            for article in news_list:
                st.markdown(f"""
                <div style='padding: 1rem; border-radius: 10px; margin-bottom: 1rem; background-color: #f0f2f6;'>
                    <h5>📰 <a href="{article['url']}" target="_blank" style='text-decoration: none;'>{article['title']}</a></h5>
                    <p style='font-size: 14px;'>🗞️ {article['source']['name']} | 🕒 {article['publishedAt'][:10]}</p>
                    <p style='color: #555;'>{article.get('description', 'No description available.')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No news found even after fallback. Please try a different category.")

    st.subheader("🎯 News Filters")
    category = st.radio("Choose a news category:", ["all", "residential", "commercial", "investment", "trends"], horizontal=True)
    category_filter = None if category == "all" else category

    st.subheader("🌍 Region & Date Range")
    selected_countries = st.multiselect("Choose countries to fetch news for:", ["in", "us", "gb"], default=["in", "us"])
    age_days = st.selectbox("Show news from the last:", [1, 7, 30], index=1)

    with st.spinner("📰 Fetching news stories..."):
        for region in selected_countries:
            label = {
                "India": "🇮🇳 India Real Estate News",
                "United States": "🇺🇸 USA Real Estate News",
                "Global": "🌍 Global Real Estate News"
            }.get(region, f"🌐 News - {region.upper()}")

            region_news = fetch_news(region, category_filter, max_age_days=age_days)
            render_news_list(label, region_news)

    st.success("✅ News loaded successfully!")
# --- Tab 4: City Snapshot ---
with tab4:
    st.header(f"📊 Snapshot: {city}")

    col_input, col_compare = st.columns([2, 1])
    with col_compare:
        compare_city = st.selectbox("Compare with another city", city_data["City"].unique(), index=1)

    for selected_city in [city, compare_city]:
        st.subheader(f"🏙️ City: {selected_city}")

        city_row = city_data[city_data["City"] == selected_city].iloc[0]
        price = city_row["PricePerSqFt"]
        trend_tag = np.random.choice(["Rising 🔼", "Stable 🟢", "Falling 🔽"])
        popularity = np.random.randint(7, 10) + round(np.random.rand(), 1)
        insight = {
            "Mumbai": "Great for long-term investment, premium market.",
            "Bangalore": "Ideal for tech professionals and startups.",
            "Delhi": "Central hub with a mix of old & new real estate.",
            "Hyderabad": "Emerging market with strong ROI potential.",
            "Pune": "Affordable tech city with a growing market."
        }.get(selected_city, "Popular city with promising growth.")

        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("📍 City", selected_city)
            st.metric("💸 Avg Price/Sqft", f"{currency_symbol}{price * exchange_rate:,.2f}")
            st.metric("📈 Market Trend", trend_tag)
            st.metric("🌟 Popularity Score", f"{popularity}/10")

        with col2:
            st.success(f"💡 *Insight*: {insight}")
            st.markdown("📌 **Nearby Features**\n- 🏫 Top schools\n- 🚇 Metro connectivity\n- 🏥 Hospitals\n- 🛒 Malls\n- 🏢 Tech parks")

        st.markdown("\n📈 **Mini Trend Chart**")
        mini_years = np.arange(2019, 2025)
        mini_prices = [price * (1 + 0.03 * (i - 2019)) * area * exchange_rate for i in mini_years]
        mini_df = pd.DataFrame({"Year": mini_years, "Estimated Price": mini_prices})
        st.line_chart(mini_df.set_index("Year"))

        

# ... [previous setup and imports remain unchanged]

# --- Tab 5: Explore Any City ---
with tab5:
    st.header("🔍 Explore Any City")

    # Descriptions and price adjustments based on type
    city_descriptions = {
        "Mumbai": "Mumbai is India’s financial capital with a robust housing market offering high ROI opportunities.",
        "Delhi": "Delhi combines government and commercial spaces. Old Delhi sees redevelopment, while South Delhi remains elite.",
        "Bangalore": "Bangalore is the Silicon Valley of India, favored for tech hubs and good rental yields.",
        "Hyderabad": "Hyderabad offers affordable real estate with booming IT zones like HITECH city.",
        "Pune": "Known for education and IT, Pune is a balanced real estate market for investors and families.",
        "Chennai": "Chennai balances affordability with industrial and IT expansion in OMR zones.",
        "Ahmedabad": "A fast-growing metro with smart city infrastructure and expanding residential outskirts.",
        "New York": "New York offers premium global real estate but demands high investment with long-term appreciation.",
        "Los Angeles": "LA combines glamour and real estate growth, especially in suburban tech zones.",
        "San Francisco": "SF is a tech capital with very high housing prices and strong ROI over time.",
        "Chicago": "Chicago offers steady real estate, a strong rental market, and affordable suburban zones.",
        "Austin": "Austin is booming for tech professionals, offering great appreciation and rental demand."
    }

    city_coords = {
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Bangalore": (12.9716, 77.5946),
        "Hyderabad": (17.3850, 78.4867),
        "Pune": (18.5204, 73.8567),
        "Chennai": (13.0827, 80.2707),
        "Ahmedabad": (23.0225, 72.5714),
        "New York": (40.7128, -74.0060),
        "Los Angeles": (34.0522, -118.2437),
        "San Francisco": (37.7749, -122.4194),
        "Chicago": (41.8781, -87.6298),
        "Austin": (30.2672, -97.7431)
    }

    real_estate_types = {
        "Residential": 1.0,
        "Commercial": 1.6,
        "Plots": 0.8,
        "Luxury": 2.2
    }

    all_cities = sorted(city_descriptions.keys())
    selected_city = st.selectbox("Select a city to explore", all_cities)
    selected_type = st.radio("Select real estate type", list(real_estate_types.keys()), horizontal=True)

    st.subheader(f"📍 {selected_city} — {selected_type}")
    st.write(f"🧠 **Overview**: {city_descriptions[selected_city]}")

    # Simulated base price & multiplier
    base_price = 5000 if selected_city in ["Mumbai", "Delhi", "Bangalore"] else 3000
    multiplier = real_estate_types[selected_type]

    # Mini chart
    years = np.arange(2019, 2025)
    prices = [base_price * multiplier * (1 + 0.05 * (i - 2019)) for i in years]
    trend_df = pd.DataFrame({"Year": years, "Estimated Price per Sqft": prices})
    st.line_chart(trend_df.set_index("Year"))

    # Weather Section
    st.subheader("☁️ Real-Time Weather")
    lat, lon = city_coords[selected_city]
    if weather_api_key:
        try:
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={weather_api_key}"
            res = requests.get(weather_url)
            weather = res.json()
            temp = weather['main']['temp']
            desc = weather['weather'][0]['description'].capitalize()
            humidity = weather['main']['humidity']
            wind = weather['wind']['speed']
            st.success(f"🌡️ {temp}°C | {desc}")
            st.caption(f"💧 Humidity: {humidity}% | 🌬️ Wind: {wind} m/s")
        except:
            st.warning("⚠️ Unable to fetch weather info right now.")
    else:
        st.warning("⚠️ Weather API key not set.")

    # Map
    st.subheader("🗺️ City Location")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

    # --- Best Time to Buy Section ---
    st.subheader("🕰️ Best Time to Buy")

    # Simulated seasonal price index (lower = better time to buy)
    monthly_factors = {
        "January": 1.05, "February": 1.04, "March": 1.03, "April": 1.02,
        "May": 1.00, "June": 0.97, "July": 0.96, "August": 0.98,
        "September": 1.01, "October": 1.02, "November": 1.03, "December": 1.05
    }
    price_base = 10000
    monthly_prices = {month: price_base * factor for month, factor in monthly_factors.items()}
    best_month = min(monthly_prices, key=monthly_prices.get)

    seasonal_df = pd.DataFrame({
        "Month": list(monthly_prices.keys()),
        "Avg Price": list(monthly_prices.values())
    })

    st.line_chart(seasonal_df.set_index("Month"))

    st.success(f"💡 Based on seasonal trends, **{best_month}** is typically the best time to buy property in **{selected_city}**.")

    # --- Best Time to Sell Section ---


# --- Tab 6: Investment Score ---
with tab6:
    st.header("💰 Investment Score & ROI Calculator")
    st.markdown("""
    Use this tool to evaluate the potential return on investment (ROI) for a property.
    Just enter a few details, and we’ll crunch the numbers for you!
    """)

    col1, col2 = st.columns(2)
    with col1:
        property_price = st.number_input("Property Purchase Price (₹ or $)", value=7500000)
        expected_rent = st.number_input("Expected Monthly Rent", value=25000)
    with col2:
        years = st.slider("Planned Holding Period (Years)", 1, 30, 10)
        annual_appreciation = st.slider("Expected Annual Price Appreciation (%)", 0, 20, 6)

    if st.button("📊 Calculate Investment ROI"):
        total_rent_income = expected_rent * 12 * years
        future_price = property_price * ((1 + annual_appreciation / 100) ** years)
        gain_from_sale = future_price - property_price
        total_profit = total_rent_income + gain_from_sale
        roi_percent = (total_profit / property_price) * 100

        st.subheader("📈 Results")
        st.metric("Total Rental Income", f"{currency_symbol}{total_rent_income * exchange_rate:,.2f}")
        st.metric("Estimated Future Property Value", f"{currency_symbol}{future_price * exchange_rate:,.2f}")
        st.metric("Total Profit (Rent + Sale)", f"{currency_symbol}{total_profit * exchange_rate:,.2f}")
        st.metric("💹 ROI", f"{roi_percent:.2f}%")

        if roi_percent >= 100:
            st.success("🔥 Excellent Investment! ROI is very strong.")
        elif roi_percent >= 50:
            st.info("👍 Good Investment. Reasonable ROI.")
        else:
            st.warning("⚠️ Consider reviewing — ROI is on the lower side.")

# --- Tab 7: Loan Eligibility and EMI Calculator ---
with tab7:
    st.header("🏦 Loan Eligibility & EMI Calculator")
    st.markdown("""
    Estimate your home loan EMI and check how much loan you may be eligible for.
    Enter your salary, interest rate, and tenure below.
    """)

    user_name = st.text_input("Enter your name", value="")
    

    col1, col2 = st.columns(2)
    with col1:
        monthly_salary = st.number_input("Monthly Take-Home Salary", value=60000)
        interest_rate = st.number_input("Expected Interest Rate (%)", value=8.5, step=0.1)
    with col2:
        loan_tenure_years = st.slider("Loan Tenure (in years)", 5, 30, 20)
        existing_emi = st.number_input("Existing Monthly EMIs", value=0)

    def format_currency(val):
        return f"INR {val:,.2f}" if currency_symbol == '₹' else f"${val:,.2f}"

    if st.button("📉 Calculate EMI"):
        max_emi_affordable = 0.45 * monthly_salary
        eligible_emi = max_emi_affordable - existing_emi

        r = (interest_rate / 12) / 100
        n = loan_tenure_years * 12

        if r == 0:
            loan_amount = eligible_emi * n
        else:
            loan_amount = eligible_emi * ((1 + r)**n - 1) / (r * (1 + r)**n)

        st.subheader("💼 Loan Details")
        st.metric("Eligible Monthly EMI", format_currency(eligible_emi))
        st.metric("Eligible Loan Amount", format_currency(loan_amount))
        st.metric("Loan Tenure", f"{loan_tenure_years} years")
        st.metric("Interest Rate", f"{interest_rate:.2f}%")

        if loan_amount < 1000000:
            remark = "Loan amount may be low. Try increasing tenure or reducing EMIs."
            st.warning("⚠️ " + remark)
        else:
            remark = "You are eligible for a good loan amount."
            st.success("✅ " + remark)

        # --- Charts ---
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.bar(["EMI", "Loan Amount"], [eligible_emi, loan_amount], color=["skyblue", "green"])
        ax1.set_ylabel("Amount")
        ax1.set_title("Loan Summary")
        chart_buf = io.BytesIO()
        fig1.savefig(chart_buf, format="PNG")
        chart_buf.seek(0)

        # Pie Chart
        total_payment = eligible_emi * n
        total_interest = total_payment - loan_amount
        pie_labels = ['Principal', 'Interest']
        pie_sizes = [loan_amount, total_interest]
        pie_colors = ['#1f77b4', '#ff7f0e']
        fig2, ax2 = plt.subplots()
        ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=pie_colors)
        ax2.set_title("Principal vs Interest Breakdown")
        pie_buf = io.BytesIO()
        fig2.savefig(pie_buf, format="PNG")
        pie_buf.seek(0)

        # EMI vs Salary
        fig3, ax3 = plt.subplots()
        ax3.bar(["Monthly Salary", "EMI"], [monthly_salary, eligible_emi], color=["#2ca02c", "#d62728"])
        ax3.set_title("EMI vs Salary")
        bar_buf = io.BytesIO()
        fig3.savefig(bar_buf, format="PNG")
        bar_buf.seek(0)

        # ROI Estimate
        roi_percent = round((loan_amount * 0.06) / total_payment * 100, 2)
        fig4, ax4 = plt.subplots()
        ax4.bar(["ROI %"], [roi_percent], color="purple")
        ax4.set_title("Estimated ROI")
        roi_buf = io.BytesIO()
        fig4.savefig(roi_buf, format="PNG")
        roi_buf.seek(0)

        # Rate Trend
        years = list(range(2018, 2025))
        rates = [9.5, 8.75, 8.6, 8.2, 7.8, 8.1, 8.5]
        fig5, ax5 = plt.subplots()
        ax5.plot(years, rates, marker='o', color='darkorange')
        ax5.set_title("Historical Rate Trend")
        rate_buf = io.BytesIO()
        fig5.savefig(rate_buf, format="PNG")
        rate_buf.seek(0)

        # --- PDF Report Generation ---
        pdf = FPDF()

        # Cover page
        pdf.add_page()
        try:
            pdf.image("logo.png", x=80, y=30, w=50)
        except:
            pass
        pdf.set_font("Arial", "B", 20)
        pdf.ln(100)
        pdf.cell(0, 10, "Smart House Price Predictor", ln=True, align='C')
        pdf.set_font("Arial", "", 14)
        pdf.cell(0, 10, "Loan & Investment Report", ln=True, align='C')
        pdf.set_font("Arial", "I", 12)
        pdf.cell(0, 10, f"User: {user_name if user_name else 'N/A'}", ln=True, align='C')
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')

            
        pdf = FPDF()

        # Cover page
        pdf.add_page()
        try:
            pdf.image("logo.png", x=80, y=30, w=50)
        except:
            pass
        pdf.set_font("Arial", "B", 20)
        pdf.ln(100)
        pdf.cell(0, 10, "Smart House Price Predictor", ln=True, align='C')
        pdf.set_font("Arial", "", 14)
        pdf.cell(0, 10, "Loan & Investment Report", ln=True, align='C')
        pdf.set_font("Arial", "I", 12)
        pdf.cell(0, 10, f"User: {user_name if user_name else 'N/A'}", ln=True, align='C')
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')

        # Report Page
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="Loan Eligibility Summary", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Monthly Salary: {format_currency(monthly_salary)}", ln=True)
        pdf.cell(200, 10, txt=f"Interest Rate: {interest_rate:.2f}%", ln=True)
        pdf.cell(200, 10, txt=f"Loan Tenure: {loan_tenure_years} years", ln=True)
        pdf.cell(200, 10, txt=f"Eligible EMI: {format_currency(eligible_emi)}", ln=True)
        pdf.cell(200, 10, txt=f"Eligible Loan Amount: {format_currency(loan_amount)}", ln=True)
        pdf.multi_cell(0, 10, txt=f"Summary: {remark}")

        for buf, name in zip(
            [chart_buf, pie_buf, bar_buf, roi_buf, rate_buf],
            ["loan_summary.png", "pie_chart.png", "bar_chart.png", "roi_chart.png", "rate_chart.png"]):
            with open(name, "wb") as f:
                f.write(buf.getbuffer())

            # Check if current y-position is near the bottom, then add a new page
            if pdf.get_y() > 240:
                pdf.add_page()

            pdf.image(name, x=25, w=160)

        # Footer, only if not too low
        if pdf.get_y() < 270:
            pdf.set_y(-20)
            pdf.set_font("Arial", "I", 9)
            pdf.cell(0, 10, txt="Generated by Smart House Price Predictor", align='C')

        buffer = io.BytesIO()
        buffer.write(pdf.output(dest="S").encode("latin-1"))
        buffer.seek(0)

        st.download_button(
            label="📥 Download Report as PDF",
            data=buffer,
            file_name="loan_report.pdf",
            mime="application/pdf"
        )

with tab8:
    st.header("💬 Real Estate Community Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_input = st.text_input("💬 Your message", key="chat_box")
    if st.button("Send"):
        if chat_input:
            st.session_state.messages.append(("You", chat_input))

    st.subheader("📜 Chat Thread")
    for user, msg in reversed(st.session_state.messages):
        st.markdown(f"**{user}**: {msg}")

# --- Tab 10: Real Estate Price Forecast Dashboard ---
with tab9:
    st.header("📊 Price Forecast Dashboard")
    city_selected = st.selectbox("Choose a city", [
        "Mumbai", "Bangalore", "Delhi", "Hyderabad", "Pune",
        "New York", "San Francisco", "Los Angeles", "Chicago", "Austin"
    ])

    base_price = 6000 if city_selected in ["Mumbai", "Bangalore", "Delhi"] else 10000
    growth_rate = st.slider("Expected Annual Growth Rate (%)", 1, 10, 5)

    forecast_years = np.arange(2024, 2031)
    forecast_prices = [base_price * ((1 + growth_rate / 100) ** i) for i in range(len(forecast_years))]
    forecast_df = pd.DataFrame({"Year": forecast_years, "Forecasted Price": forecast_prices})

    st.line_chart(forecast_df.set_index("Year"))
    st.success(f"🏙️ Forecast complete for {city_selected}. Estimated price in 2030: ₹{forecast_prices[-1]:,.2f}")
