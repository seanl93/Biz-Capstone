import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Sample data (Replace with your actual dataset)
data = {
    "order_id": [101, 102, 103, 104, 105],
    "product_name": ["Shirt", "Shoes", "Watch", "Backpack", "Headphones"],
    "rating": [3.5, 2.0, 4.0, 1.5, 5.0],
    "review_text": [
        "Good quality but slow delivery",
        "Terrible product, broke in a week",
        "Loved it, works perfectly",
        "Very disappointed, not what I expected",
        "Excellent value and fast shipping"
    ],
    "price": [29.99, 59.99, 199.99, 49.99, 89.99],
    "category": ["Clothing", "Footwear", "Accessories", "Bags", "Electronics"],
    "cancelled": [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Add sentiment score
df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Encode categorical feature
df['category_encoded'] = LabelEncoder().fit_transform(df['category'])

# Prepare features and model
features = df[['rating', 'sentiment', 'price', 'category_encoded']]
labels = df['cancelled']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = RandomForestClassifier().fit(X_train, y_train)

df['risk_score'] = model.predict_proba(features)[:, 1]

# Streamlit UI
st.title("Order Cancellation Risk Analysis Dashboard")

threshold = st.slider("Set Risk Threshold", 0.0, 1.0, 0.5, 0.05)
high_risk = df[df['risk_score'] >= threshold]

st.subheader("High-Risk Orders")
st.dataframe(high_risk[['order_id', 'product_name', 'rating', 'risk_score']])

def recommend_action(risk):
    if risk > 0.8:
        return "Offer expedited shipping or manual review"
    elif risk > 0.6:
        return "Send reassurance email with FAQs"
    else:
        return "No action needed"

high_risk['AI_Suggestion'] = high_risk['risk_score'].apply(recommend_action)

st.subheader("AI-Suggested Actions")
st.dataframe(high_risk[['order_id', 'product_name', 'risk_score', 'AI_Suggestion']])
