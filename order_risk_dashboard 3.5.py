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

    # Define risk levels based on percentiles
    def assign_risk_level(score):
        if score >= 0.8:
            return "High"
        elif score >= 0.5:
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
    st.subheader("📊 Risk Level Distribution")
    risk_dist = df['risk_level'].value_counts()
    st.bar_chart(risk_dist)

    # Filtered orders display
    st.subheader("📦 Risk Orders with AI Recommendations")
    
    # Filters
    risk_filter = st.multiselect(
        "Filter by Risk Level",
        options=['High', 'Medium', 'Low'],
        default=['High', 'Medium']
    )
    
    filtered = df[df['risk_level'].isin(risk_filter)].sort_values('risk_score', ascending=False)
    
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

    # ROC Curve
    st.write("**ROC Curve:**")
    roc_display = RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Chance')
    plt.legend()
    st.pyplot(plt)

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
