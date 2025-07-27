import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Order Risk Dashboard", layout="wide")

# Load external data
uploaded_file = st.file_uploader("Upload your order dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Drop rows with missing values in important columns
    expected_cols = ['Status', 'ship-service-level', 'Item Total']
    df.dropna(subset=[col for col in expected_cols if col in df.columns], inplace=True)

    # Cancellation flag
    if 'Status' in df.columns:
        df['cancelled'] = df['Status'].apply(lambda x: 1 if 'CANCELLED' in str(x).upper() else 0)
    else:
        st.error("Missing 'Status' column.")
        st.stop()

    # Sentiment
    df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity) if 'review_text' in df.columns else 0.0

    # Encode categoricals
    for col in ['Category', 'ship-service-level']:
        if col in df.columns:
            df[col + '_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))

    # Feature preparation
    feature_cols = ['rating', 'sentiment', 'Item Total', 'Category_encoded', 'ship-service-level_encoded']
    available_features = [f for f in feature_cols if f in df.columns]

    if not available_features:
        st.error("No usable features found.")
        st.stop()

    X = df[available_features].fillna(0)
    y = df['cancelled']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    df['risk_score'] = model.predict_proba(X)[:, 1]

    # Add dollar_amount
    df['dollar_amount'] = df['Amount'] if 'Amount' in df.columns else df.get('Item Total', 0)

    # Main dashboard
    st.title("📊 Order Cancellation Risk Dashboard")

    threshold = st.sidebar.slider("Set Risk Threshold", 0.0, 1.0, 0.3, 0.05)
    df['risk_level'] = df['risk_score'].apply(
        lambda r: 'High' if r > 0.8 else ('Medium' if r > 0.6 else ('Low' if r > threshold else 'Below Threshold'))
    )
    filtered = df[df['risk_level'].isin(['High', 'Medium', 'Low']) & (df['risk_score'] >= threshold)]

    st.subheader("📦 High-Risk Orders (Medium and High) with AI Suggestions")
    if filtered.empty:
        st.warning("No orders above the threshold.")
    else:
        def ai_suggestion(row):
            if row['risk_level'] == 'High':
                return "🚨 Suggest Manual Review / Offer Express Shipping"
            elif row['risk_level'] == 'Medium':
                return "⚠️ Suggest Reassurance Email or Discount"
            else:
                return "No action needed"

        filtered['AI_Suggestion'] = filtered.apply(ai_suggestion, axis=1)
        for _, row in filtered.iterrows():
            with st.expander(f"Order {row.get('Order ID', 'Unknown')} | Risk: {row['risk_level']} | Score: {row['risk_score']:.2f}"):
                st.markdown(f"**Ship Level:** {row.get('ship-service-level', '-')}")
                st.markdown(f"**Amount:** ${row['dollar_amount']:.2f}")
                st.markdown(f"**Suggested Action:** {row['AI_Suggestion']}")
                approve = st.checkbox(f"✅ Approve Action for Order {row.get('Order ID', 'N/A')}", key=f"approve_{row.name}")
                if approve:
                    st.success("Action approved ✅")
                else:
                    st.info("Awaiting action...")

    # AI Metrics
    st.subheader("📈 Model Performance Metrics")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    st.metric("ROC AUC Score", round(auc, 3))

    st.write("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Cancelled', 'Cancelled'], yticklabels=['Not Cancelled', 'Cancelled'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature Importance
    st.subheader("🔍 Top 5 Features Driving Cancellations")
    try:
        rfe = RFE(model, n_features_to_select=min(5, len(available_features)))
        rfe.fit(X_train, y_train)
        top_features = [available_features[i] for i, selected in enumerate(rfe.support_) if selected]
        st.write("Top Features:", top_features)
    except Exception as e:
        st.error(f"Feature selection failed: {e}")

    # Sentiment overview
    st.subheader("💬 Average Sentiment")
    avg_sentiment = df['sentiment'].mean() if 'sentiment' in df.columns else 0.0
    st.metric("Average Review Sentiment", round(avg_sentiment, 3))

else:
    st.info("Please upload a CSV file to begin.")
