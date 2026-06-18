import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components






# ////////////////// page name //////////////////
st.set_page_config(
    page_title="BuySense",
    page_icon="🛍️")

# ////////////////// load dataset //////////////////
df = pd.read_csv("buysense_dataset.csv")

# ////////////////// sidebar //////////////////
with st.sidebar:
    selected = option_menu(
        menu_title="BuySense",
        options=["Home", "Dataset", "Prediction", "About"],
        icons=["house", "table", "robot", "info-circle"],
        default_index=0
    )

# ================= HOME
if selected == "Home":
    st.set_page_config(page_title="BuySense", layout="wide")

    st.set_page_config(page_title="BuySense Dashboard", layout="wide")

    st.title("BuySense Power BI Dashboard")

    powerbi_iframe = """
    <iframe 
        title="aaditya" 
        width="100%" 
        height="800" 
        src="https://app.powerbi.com/reportEmbed?reportId=b9a44f14-0f9e-4a9f-b76b-7553a5a81f54&autoAuth=true&embeddedDemo=true" 
        frameborder="0" 
        allowFullScreen="true">
    </iframe>
    """

    components.html(powerbi_iframe, height=850)

# ================= DATASET
elif selected == "Dataset":
    st.title("Dataset View")
    st.dataframe(df)

# ///////////data records
    Total_records=len(df)
    Avg_order_value =round(df["avg_order_value"].mean(), 2)
    Total_product = df["recommended_category"].nunique()
    col1,col2,col3 = st.columns(3)
    col1.metric("Total record", Total_records)
    col2.metric("Avg_order_value", Avg_order_value)
    col3.metric("Total_product", Total_product)

#////////////////data shape
    st.subheader("Dataset Shape")
    st.write("Rows :",df.shape[0])
    st.write("Columes",df.shape[1])

    st.subheader("Stastical Summary:")
    st.dataframe(df.describe())


# ================= PREDICTION
elif selected == "Prediction":

    st.markdown("## 🧠 Recommended Category Prediction")

    st.write(
        "RandomForestClassifier model jo customer behaviour ke basis pe "
        "recommended_category predict karta hai."
    )

    st.metric("Model Accuracy", "41.67%")

    st.markdown("### Try a Prediction")

    # Load model
    model = joblib.load("buysense_model.pkl")
    encoders = joblib.load("encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")

    col1, col2 = st.columns(2)

    with col1:
        age_group = st.selectbox(
            "Age Group",
            ["18-24", "25-34", "35-44", "45-54"]
        )

        city_tier = st.selectbox(
            "City Tier",
            encoders["city_tier"].classes_
        )

        discount_sensitivity = st.selectbox(
            "Discount Sensitivity",
            ["Low", "Medium", "High"]
        )

    with col2:

        browsing_minutes = st.slider(
            "Browsing Minutes",
            0, 100, 30
        )

        pages_viewed = st.slider(
            "Pages Viewed",
            0, 50, 10
        )

        past_purchases = st.slider(
            "Past Purchases",
            0, 20, 5
        )

        avg_order_value = st.slider(
            "Avg Order Value",
            0, 1000, 300
        )

    # Hidden fields (default values)
    preferred_device = "Mobile"
    purchase_time = "Evening"

    interest_electronics = 5
    interest_fashion = 5
    interest_home_decor = 5
    interest_books = 5
    interest_fitness = 5
    interest_beauty = 5

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Recommended Category"):

        input_data = pd.DataFrame([{
            "age_group": encoders["age_group"].transform([age_group])[0],
            "city_tier": encoders["city_tier"].transform([city_tier])[0],
            "browsing_minutes": browsing_minutes,
            "pages_viewed": pages_viewed,
            "past_purchases": past_purchases,
            "avg_order_value": avg_order_value,
            "discount_sensitivity": encoders["discount_sensitivity"].transform([discount_sensitivity])[0],
            "preferred_device": encoders["preferred_device"].transform([preferred_device])[0],
            "purchase_time": encoders["purchase_time"].transform([purchase_time])[0],
            "interest_electronics": interest_electronics,
            "interest_fashion": interest_fashion,
            "interest_home_decor": interest_home_decor,
            "interest_books": interest_books,
            "interest_fitness": interest_fitness,
            "interest_beauty": interest_beauty
        }])

        prediction = model.predict(input_data)

        result = target_encoder.inverse_transform(prediction)

        st.success(f"🎯 Recommended Category: {result[0]}")
# ================= ABOUT =================
elif selected == "About":
    st.markdown(
        "<h2 style='color:green;'> About projcet </h2>",
        unsafe_allow_html=True
    )
    st.subheader("⭐️ BuySense is a machine learning based product recommendation system.")
    st.write("""
    ->  BuySense is a Machine Learning-based Product Recommendation System designed to analyze customer data and recommend the most suitable product category based on user preferences and purchasing behavior.
    
    ->  The system uses advanced Machine Learning algorithms to process historical customer information, identify patterns, and generate accurate product recommendations. This helps businesses understand customer needs and improve the shopping experience.
    
    """)
    st.subheader("⭐️  Project Objectives")
    st.write("""
    -> Analyze customer demographic and purchasing data.
    
    -> Predict the most relevant product category for a customer.
    
    -> Improve recommendation accuracy using Machine Learning techniques.
    
    -> Visualize customer insights through interactive dashboards.
    
    -> Provide a user-friendly interface for prediction and analysis.
    """)
    st.subheader("⭐️  Technologies Used")
    st.subheader(" 👉🏻Programming Language ")
    st.write(">Python")
    st.subheader("⭐️  Libraries")
    st.write("""
    Pandas
    
    -> NumPy
    
    -> Scikit-learn (sklearn)
    
    -> LabelEncoder
    
    -> Random Forest Classifier
    
    -> Joblib
    
    -> Streamlit
    
    """)
    st.subheader("⭐️  Data Visualization")
    st.write("""
    -> Power BI
    
    >Streamlit Charts
    """)
    st.subheader("⭐️  Machine Learning Workflow")
    st.write("""
    1.Data Collection and Loading
    
    2.Data Preprocessing and Cleaning
    
    3.Encoding Categorical Data using LabelEncoder
    
    4.Feature Selection
    
    5.Train-Test Split
    
    6.Model Training using Random Forest Classifier
    
    7.Model Evaluation and Accuracy Testing
    
    8.Product Category Prediction
    
    9.Recommendation Generation
    
    10.Dashboard Visualization
    """)
    st.subheader("⭐️  Key Features")
    st.write("""
    ✅ Customer Data Analysis
    
    ✅ Machine Learning-Based Prediction
    
    ✅ Product Recommendation Engine
    
    ✅ Interactive Streamlit Dashboard
    
    ✅ Power BI Analytics Dashboard
    
    ✅ Fast and User-Friendly Interface
    
    ✅ Accurate Category Prediction
    """)
    st.subheader("⭐️  How BuySense Works")
    st.write("""
    1.User enters customer information.
    
    2.The system processes the input data.
    
    3.The trained Machine Learning model analyzes the data.
    
    4.The model predicts the most suitable product category.
    
    5.Recommended products are displayed to the user.
    
    6.Power BI dashboards provide detailed business insights and analytics.
    """)

    st.subheader("⭐️  Benefits")
    st.write("""
    -> Personalized Product Recommendations
    
    -> Better Customer Experience
    
    -> Data-Driven Decision Making
    
    -> Improved Marketing Strategy
    
    -> Enhanced Sales Opportunities
    """)

    st.subheader("⭐️  Conclusion")
    st.write("""BuySense is an intelligent recommendation system that combines Machine Learning, Streamlit, and Power BI to deliver personalized product suggestions and business insights. The project demonstrates how data analytics and predictive modeling can be used to improve customer satisfaction and support smarter business decisions.""")