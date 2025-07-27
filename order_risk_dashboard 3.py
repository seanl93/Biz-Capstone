import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Order Risk Dashboard", layout="wide")
st.title("📦 Enhanced Order Cancellation Risk Dashboard")

uploaded_file = st.file_uploader("Upload your order dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Cancellation Flag
    if 'Status' in df.columns:
        df['cancelled'] = df['Status'].apply(lambda x: 1 if 'CANCELLED' in str(x).upper() else 0)
    else:
        st.error("Missing 'Status' column.")
        st.stop()

    # Sentiment
    df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity) if 'review_text' in df.columns else 0.0

    # Encode categorical
    for col in ['Category', 'ship-service-level']:
        if col in df.columns:
            df[col + '_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))

    # Feature preparation
    feature_cols = ['rating', 'sentiment', 'Item Total', 'Category_encoded', 'ship-service-level_encoded']
    available_features = [col for col in feature_cols if col in df.columns]
    if not available_features:
        st.error("No valid features available.")
        st.stop()

    features = df[available_features].fillna(0)
    labels = df['cancelled']

    # Model
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    df['risk_score'] = model.predict_proba(features)[:, 1]

    # Reassign risk levels based on non-cancelled distribution
    non_cancelled = df[df['cancelled'] == 0].copy()
    non_cancelled_sorted = non_cancelled.sort_values('risk_score', ascending=False).reset_index(drop=True)
    total = len(non_cancelled_sorted)
    high_cutoff = int(total * 0.15)
    medium_cutoff = int(total * 0.55)
    
    high_threshold = non_cancelled_sorted.loc[high_cutoff - 1, 'risk_score'] if total > high_cutoff else 1.0
    medium_threshold = non_cancelled_sorted.loc[medium_cutoff - 1, 'risk_score'] if total > medium_cutoff else 0.0

    def reassign_risk(score):
        if score >= high_threshold:
            return "High"
        elif score >= medium_threshold:
            return "Medium"
        else:
            return "Low"

    df['risk_level'] = df['risk_score'].apply(reassign_risk)

    st.subheader("📊 Risk Level Distribution (Non-Cancelled Orders)")
    distribution = df[df['cancelled'] == 0]['risk_level'].value_counts(normalize=True).round(3) * 100
    st.write(distribution.to_frame("Percentage (%)"))

    # Show Medium and High Risk Orders Only
    filtered = df[df['risk_level'].isin(['Medium', 'High'])].copy()

    def recommend_action(row):
        if row['risk_level'] == 'High':
            return "🔴 Review or offer discount"
        elif row['risk_level'] == 'Medium':
            return "🟠 Send reassurance email"
        else:
            return "🟢 No action needed"

    filtered['AI_Suggestion'] = filtered.apply(recommend_action, axis=1)

    st.subheader("🛠️ Medium and High Risk Orders with Suggested Actions")
    st.dataframe(filtered[[
        'Order ID', 'Status', 'risk_score', 'risk_level', 
        'Item Total', 'ship-service-level', 'AI_Suggestion'
    ]].sort_values('risk_score', ascending=False))

    # Confusion Matrix & AUC
    st.subheader("📈 Model Evaluation")
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, y_pred)

    st.metric("ROC AUC Score", f"{auc:.3f}")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Cancelled', 'Cancelled'], yticklabels=['Not Cancelled', 'Cancelled'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start.")
    st.markdown("""
    **Note:** Required columns: `Order ID`, `Status`, `Item Total`, `ship-service-level`, `review_text`, `rating`, `Category`
    
    Optional: `Amount` for dollar-based filters.
    """)
