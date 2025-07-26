import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
import numpy as np

# Load external data
uploaded_file = st.file_uploader("Upload your order dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove any extra spaces from column names

    st.write("### Columns in your file:", df.columns.tolist())

    # Drop rows with missing values in important columns (if they exist)
    expected_cols = ['Status', 'ship-service-level', 'Item Total']
    available_cols = [col for col in expected_cols if col in df.columns]
    if available_cols:
        df = df.dropna(subset=available_cols)

    # Add cancellation flag
    if 'Status' in df.columns:
        df['cancelled'] = df['Status'].apply(lambda x: 1 if 'CANCELLED' in str(x).upper() else 0)
    else:
        st.error("Missing 'Status' column needed to define cancellation.")

    # Compute sentiment if review text is available
    if 'review_text' in df.columns:
        df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    else:
        df['sentiment'] = 0.0  # fallback

    # Encode categorical variables
    for col in ['Category', 'ship-service-level']:
        if col in df.columns:
            df[col + '_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))

    # Prepare features
    feature_cols = ['rating', 'sentiment', 'Item Total', 'Category_encoded', 'ship-service-level_encoded']
    available_features = [col for col in feature_cols if col in df.columns]
    if not available_features:
        st.error("No valid feature columns found in your data.")
    else:
        features = df[available_features].fillna(0)
        labels = df['cancelled']

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        df['risk_score'] = model.predict_proba(features)[:, 1]

        # Calculate dollar_amount column from 'Amount'
        if 'Amount' in df.columns:
            df['dollar_amount'] = df['Amount']

        # Streamlit UI
        st.title("Order Cancellation Risk Analysis Dashboard")

        st.subheader("📊 All Orders Preview")
        st.dataframe(df[['Order ID', 'Status', 'risk_score', 'dollar_amount']].head(10))

        st.write("Min Risk Score:", df['risk_score'].min())
        st.write("Max Risk Score:", df['risk_score'].max())

        threshold = st.sidebar.slider("Set Risk Threshold", 0.0, 1.0, 0.3, 0.05)
        filtered = df[df['risk_score'] >= threshold]

        st.subheader("📦 High-Risk Orders")
        cols_to_show = [col for col in ['Order ID', 'Category', 'ship-service-level', 'dollar_amount', 'risk_score'] if col in df.columns]
        if filtered.empty:
            st.warning("No high-risk orders found at the selected threshold. Try lowering the threshold.")
        else:
            st.dataframe(filtered[cols_to_show])

        def recommend_action(risk):
            if risk > 0.8:
                return "Offer expedited shipping or manual review"
            elif risk > 0.6:
                return "Send reassurance email with FAQs"
            else:
                return "No action needed"

        filtered['AI_Suggestion'] = filtered['risk_score'].apply(recommend_action)

        st.subheader("🧠 AI-Suggested Actions")
        cols_to_show = [col for col in ['Order ID', 'ship-service-level', 'dollar_amount', 'risk_score', 'AI_Suggestion'] if col in filtered.columns]
        st.dataframe(filtered[cols_to_show])

        # Feature Importance
        st.subheader("🔍 Top 5 Features Driving Cancellations")
        try:
            rfe = RFE(model, n_features_to_select=min(5, len(available_features)))
            rfe.fit(X_train, y_train)
            top_features = [available_features[i] for i in range(len(available_features)) if rfe.support_[i]]
            if top_features:
                st.write("Top Features:", top_features)
            else:
                st.warning("No features selected by RFE. Try using more data or different features.")
        except Exception as e:
            st.error(f"Feature importance calculation failed: {e}")

        # Overall sentiment
        st.subheader("💬 Overall Review Sentiment")
        if 'sentiment' in df.columns:
            avg_sentiment = df['sentiment'].mean()
            st.metric(label="Average Sentiment Score", value=round(avg_sentiment, 3))
        else:
            st.info("No review text found, so sentiment could not be calculated.")

else:
    st.info("Please upload a CSV file to begin.")
