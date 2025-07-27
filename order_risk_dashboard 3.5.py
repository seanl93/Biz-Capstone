import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(page_title="Order Cancellation Risk Dashboard", layout="wide")
st.title("🚨 Advanced Order Risk Analysis Dashboard")

# Load external data
uploaded_file = st.file_uploader("Upload your order dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove any extra spaces from column names

    # Drop rows with missing values in important columns
    expected_cols = ['Status', 'ship-service-level', 'Item Total']
    available_cols = [col for col in expected_cols if col in df.columns]
    if available_cols:
        df = df.dropna(subset=available_cols)

    # Add cancellation flag
    if 'Status' in df.columns:
        df['cancelled'] = df['Status'].apply(lambda x: 1 if 'CANCELLED' in str(x).upper() else 0)
    else:
        st.error("Missing 'Status' column needed to define cancellation.")
        st.stop()

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
        st.stop()

    features = df[available_features].fillna(0)
    labels = df['cancelled']

    # Train model with standardization
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Calculate risk scores
    df['risk_score'] = model.predict_proba(scaler.transform(features))[:, 1]

    # Redistribute risk levels to ensure 10% of non-cancelled are high risk
    non_cancelled = df[df['cancelled'] == 0].copy()
    sorted_nc = non_cancelled.sort_values('risk_score', ascending=False).reset_index(drop=True)

    total_nc = len(sorted_nc)
    high_cut = int(total_nc * 0.10)  # 10% as high risk
    medium_cut = int(total_nc * 0.40)  # Next 30% as medium risk (10% + 30% = 40%)

    high_thresh = sorted_nc.loc[high_cut - 1, 'risk_score'] if total_nc > high_cut else 1.0
    medium_thresh = sorted_nc.loc[medium_cut - 1, 'risk_score'] if total_nc > medium_cut else 0.0

    def assign_risk_level(score):
        if score >= high_thresh:
            return "High"
        elif score >= medium_thresh:
            return "Medium"
        else:
            return "Low"

    df['risk_level'] = df['risk_score'].apply(assign_risk_level)

    # AI recommendation function
    def recommend_action(row):
        if row['risk_level'] == 'High':
            return "🔴 Immediate: Offer expedited shipping or manual review"
        elif row['risk_level'] == 'Medium':
            return "🟠 Proactive: Send reassurance email with FAQs"
        else:
            return "🟢 Monitor: No immediate action needed"

    df['AI_Suggestion'] = df.apply(recommend_action, axis=1)

    # Streamlit UI
    st.title("Order Cancellation Risk Analysis Dashboard")

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Orders", len(df))
    with col2:
        st.metric("Cancelled Orders", df['cancelled'].sum())
    with col3:
        st.metric("Cancellation Rate", f"{df['cancelled'].mean()*100:.1f}%")

    # Risk level distribution
    st.subheader("📊 Risk Level Distribution (Non-Cancelled Orders)")
    risk_dist = df[df['cancelled'] == 0]['risk_level'].value_counts(normalize=True).mul(100).round(1)
    st.bar_chart(risk_dist)

    # Risk threshold slider in sidebar
    with st.sidebar:
        st.header("Risk Filters")
        threshold = st.slider(
            "Minimum Risk Score to Display",
            min_value=float(df['risk_score'].min()),
            max_value=float(df['risk_score'].max()),
            value=0.5,
            step=0.01
        )
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=['High', 'Medium', 'Low'],
            default=['High', 'Medium']
        )

    # Filtered orders display
    st.subheader("📦 Risk Orders with AI Recommendations")
    
    filtered = df[
        (df['risk_level'].isin(risk_filter)) & 
        (df['risk_score'] >= threshold)
    ].sort_values('risk_score', ascending=False)
    
    # Display all relevant columns automatically
    display_cols = [col for col in df.columns if col not in ['sentiment', 'Category_encoded', 'ship-service-level_encoded']]
    st.dataframe(filtered[display_cols])

    # Model evaluation
    st.subheader("📈 Model Performance Evaluation")
    
    # Calculate predictions and probabilities
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # ROC AUC Score
    st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_prob):.3f}")
    
    # Confusion Matrix
    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Cancel', 'Cancel'],
                yticklabels=['No Cancel', 'Cancel'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature Importance
    st.subheader("🔍 Top Predictive Features")
    try:
        rfe = RFE(model, n_features_to_select=min(5, len(available_features)))
        rfe.fit(X_train_scaled, y_train)
        top_features = pd.DataFrame({
            'Feature': available_features,
            'Selected': rfe.support_,
            'Ranking': rfe.ranking_
        }).sort_values('Ranking')
        
        st.dataframe(top_features[top_features['Selected'] == True])
    except Exception as e:
        st.error(f"Feature importance calculation failed: {e}")

else:
    st.info("Please upload a CSV file to begin analysis.")
