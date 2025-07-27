import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(page_title="Order Cancellation Risk Dashboard", layout="wide")
st.title("🚨 Order Cancellation Risk Prediction Dashboard")

# Load external data
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

    # Check if we have both cancelled and non-cancelled orders
    if len(df['cancelled'].unique()) == 1:
        st.warning("Warning: Dataset contains only one class (all orders are {}). Using dummy risk scores.".format(
            "cancelled" if df['cancelled'].iloc[0] == 1 else "non-cancelled"
        ))
        # Assign dummy risk scores (0.5 for non-cancelled, 1.0 for cancelled)
        df['risk_score'] = np.where(df['cancelled'] == 1, 1.0, 0.5)
    else:
        # Proceed with normal modeling if we have both classes
        non_cancelled = df[df['cancelled'] == 0].copy()
        
        # Compute sentiment if review text is available
        if 'review_text' in df.columns:
            non_cancelled['sentiment'] = non_cancelled['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        else:
            non_cancelled['sentiment'] = 0.0

        # Encode categorical variables
        for col in ['Category', 'ship-service-level']:
            if col in non_cancelled.columns:
                non_cancelled[col + '_encoded'] = LabelEncoder().fit_transform(non_cancelled[col].astype(str))

        # Prepare features
        feature_cols = ['rating', 'sentiment', 'Item Total', 'Category_encoded', 'ship-service-level_encoded']
        available_features = [col for col in feature_cols if col in non_cancelled.columns]
        
        if not available_features:
            st.error("No valid feature columns found in your data.")
            st.stop()

        features = non_cancelled[available_features].fillna(0)
        labels = non_cancelled['cancelled']  # Should be all 0s

        # Train model with standardization
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        # Calculate risk scores
        non_cancelled['risk_score'] = model.predict_proba(scaler.transform(features))[:, 1]

    # Define risk levels (10% High, 30% Medium, 60% Low) for non-cancelled orders
    non_cancelled = df[df['cancelled'] == 0].copy()
    if 'risk_score' not in non_cancelled.columns:
        non_cancelled['risk_score'] = 0.5  # Default if modeling wasn't run
        
    sorted_scores = non_cancelled['risk_score'].sort_values(ascending=False)
    high_thresh = sorted_scores.quantile(0.10) if len(sorted_scores) > 0 else 0.8
    medium_thresh = sorted_scores.quantile(0.40) if len(sorted_scores) > 0 else 0.5

    non_cancelled['risk_level'] = np.where(
        non_cancelled['risk_score'] >= high_thresh, 'High',
        np.where(non_cancelled['risk_score'] >= medium_thresh, 'Medium', 'Low')
    )

    # AI recommendation function
    def recommend_action(row):
        if row['risk_level'] == 'High':
            return "🔴 Immediate: Offer expedited shipping or manual review"
        elif row['risk_level'] == 'Medium':
            return "🟠 Proactive: Send reassurance email with FAQs"
        else:
            return "🟢 Monitor: No immediate action needed"

    non_cancelled['AI_Suggestion'] = non_cancelled.apply(recommend_action, axis=1)

    # Merge back with original data
    df = pd.concat([df[df['cancelled'] == 1], non_cancelled], ignore_index=True)

    # Streamlit UI
    st.title("Order Cancellation Risk Analysis")

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Orders", len(df))
    with col2:
        st.metric("Cancelled Orders", df['cancelled'].sum())
    with col3:
        st.metric("Cancellation Rate", f"{df['cancelled'].mean()*100:.1f}%")

    # Risk level distribution (non-cancelled only)
    st.subheader("📊 Risk Level Distribution (Non-Cancelled Orders)")
    risk_dist = non_cancelled['risk_level'].value_counts(normalize=True).mul(100).round(1)
    st.bar_chart(risk_dist)

    # Filters in sidebar
    with st.sidebar:
        st.header("🔍 Filters")
        risk_filter = st.multiselect(
            "Risk Levels",
            options=['High', 'Medium', 'Low'],
            default=['High', 'Medium']
        )
        min_score = st.slider(
            "Minimum Risk Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )

    # Filtered orders display (non-cancelled only)
    filtered = non_cancelled[
        (non_cancelled['risk_level'].isin(risk_filter)) & 
        (non_cancelled['risk_score'] >= min_score)
    ].sort_values('risk_score', ascending=False)

    st.subheader("📦 At-Risk Orders with AI Recommendations")
    st.caption(f"Showing {len(filtered)} non-cancelled orders needing attention")
    
    # Display columns - showing requested fields
    display_cols = [
        'Order ID', 'Date', 'Status', 'Fulfilment', 'Sales Channel',
        'ship-service-level', 'Category', 'Size', 'Courier Status',
        'Amount', 'risk_score', 'risk_level', 'AI_Suggestion'
    ]
    
    # Only include columns that exist in the dataframe
    display_cols = [col for col in display_cols if col in filtered.columns]
    
    # Format numeric columns
    def format_df(df):
        if 'Amount' in df.columns:
            df['Amount'] = df['Amount'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else '')
        if 'risk_score' in df.columns:
            df['risk_score'] = df['risk_score'].round(4)
        return df

    st.dataframe(format_df(filtered[display_cols]))

    # Only show model evaluation if we had both classes
    if len(df['cancelled'].unique()) > 1:
        st.subheader("📈 Model Performance")
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC AUC Score", f"{roc_auc_score(y_test, y_prob):.3f}")
        with col2:
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
        rfe = RFE(model, n_features_to_select=min(5, len(available_features)))
        rfe.fit(X_train_scaled, y_train)
        top_features = pd.DataFrame({
            'Feature': available_features,
            'Importance': rfe.support_
        }).sort_values('Importance', ascending=False)
        st.dataframe(top_features[top_features['Importance'] == True])
    else:
        st.warning("Model evaluation not available - dataset contains only one class of orders")

else:
    st.info("Please upload a CSV file to begin analysis.")
