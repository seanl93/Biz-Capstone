""import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load external data
st.set_page_config(page_title="Order Cancellation Risk Dashboard", layout="wide")
st.title("🚨 Enhanced Order Risk Detection Dashboard")

uploaded_file = st.file_uploader("Upload your order dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Add cancellation flag
    if 'Status' in df.columns:
        df['cancelled'] = df['Status'].apply(lambda x: 1 if 'CANCELLED' in str(x).upper() else 0)
    else:
        st.error("Missing 'Status' column needed to define cancellation.")
        st.stop()

    # Sentiment
    if 'review_text' in df.columns:
        df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    else:
        df['sentiment'] = 0.0

    # Encode categorical features
    for col in ['Category', 'ship-service-level']:
        if col in df.columns:
            df[col + '_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))

    # Prepare features
    feature_cols = ['rating', 'sentiment', 'Item Total', 'Category_encoded', 'ship-service-level_encoded']
    available_features = [col for col in feature_cols if col in df.columns]

    if not available_features:
        st.error("No valid features available for modeling.")
        st.stop()

    features = df[available_features].fillna(0)
    labels = df['cancelled']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Predict risk score
    df['risk_score'] = model.predict_proba(scaler.transform(features))[:, 1]

    # Redistribute risk levels on non-cancelled
    non_cancelled = df[df['cancelled'] == 0].copy()
    sorted_nc = non_cancelled.sort_values('risk_score', ascending=False).reset_index(drop=True)

    total_nc = len(sorted_nc)
    high_cut = int(total_nc * 0.15)
    medium_cut = int(total_nc * 0.55)

    high_thresh = sorted_nc.loc[high_cut - 1, 'risk_score'] if total_nc > high_cut else 1.0
    medium_thresh = sorted_nc.loc[medium_cut - 1, 'risk_score'] if total_nc > medium_cut else 0.0

    def assign_risk(score):
        if score >= high_thresh:
            return 'High'
        elif score >= medium_thresh:
            return 'Medium'
        else:
            return 'Low'

    df['risk_level'] = df['risk_score'].apply(assign_risk)

    # Summary of distribution
    st.subheader("📊 Risk Level Distribution (Non-Cancelled Orders)")
    distribution = df[df['cancelled'] == 0]['risk_level'].value_counts(normalize=True).round(3) * 100
    st.write(distribution.to_frame("Percentage (%)"))

    # Show only Medium/High risk non-cancelled orders
    filtered = df[(df['risk_level'].isin(['Medium', 'High'])) & (df['cancelled'] == 0)].copy()

    def recommend_action(row):
        if row['risk_level'] == 'High':
            return "🔴 Offer expedited shipping or manual review"
        elif row['risk_level'] == 'Medium':
            return "🟠 Send reassurance email with FAQs"
        return "🟢 No action needed"

    filtered['AI_Suggestion'] = filtered.apply(recommend_action, axis=1)

    # Display filtered orders
    st.subheader("📦 Medium & High Risk Orders with AI Suggestions")
    display_cols = ['Order ID', 'Status', 'risk_score', 'risk_level', 'dollar_amount', 'AI_Suggestion']
    display_cols = [col for col in display_cols if col in filtered.columns]
    st.dataframe(filtered[display_cols].sort_values(by='risk_score', ascending=False))

    # Logistic regression metrics
    st.subheader("📈 Model Performance")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    st.write("**ROC AUC Score:**", round(roc_auc_score(y_test, y_prob), 3))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Cancel', 'Cancel'], yticklabels=['No Cancel', 'Cancel'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin analysis.")
