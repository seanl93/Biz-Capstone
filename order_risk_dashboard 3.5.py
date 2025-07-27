import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(page_title="Order Cancellation Risk Dashboard", layout="wide")
st.title("🚨 Order Cancellation Risk Dashboard")

def main():
    uploaded_file = st.file_uploader("Upload your order dataset", type=["csv"])
    
    if not uploaded_file:
        st.info("Please upload a CSV file to begin analysis.")
        return
    
    # Load and preprocess data
    df = load_and_preprocess_data(uploaded_file)
    
    # Check if we have both cancelled and non-cancelled orders
    if len(df['cancelled'].unique()) < 2:
        handle_single_class_case(df)
        return
    
    # Train model and calculate risk scores
    df, model = calculate_risk_scores(df)
    
    # Reclassify risk levels
    df = reclassify_risk_levels(df)
    
    # Display dashboard
    show_dashboard(df, model)

def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the uploaded data"""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

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
        df['sentiment'] = 0.0

    # Encode categorical variables
    for col in ['Category', 'ship-service-level']:
        if col in df.columns:
            df[col + '_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))
    
    return df

def handle_single_class_case(df):
    """Handle case where all orders are either cancelled or non-cancelled"""
    if df['cancelled'].all():
        st.warning("All orders in your dataset are already cancelled. No risk prediction needed.")
    else:
        st.warning("No cancelled orders found in your dataset. Using simple heuristics for risk assessment.")
        
        # Create dummy risk scores
        non_cancelled = df[df['cancelled'] == 0].copy()
        non_cancelled['risk_score'] = np.random.uniform(0.1, 0.7, len(non_cancelled))
        
        # Define risk levels
        non_cancelled['risk_level'] = pd.qcut(non_cancelled['risk_score'], 
                                            q=[0, 0.6, 0.9, 1], 
                                            labels=['Low', 'Medium', 'High'])
        
        # Add AI suggestions
        non_cancelled['AI_Suggestion'] = non_cancelled['risk_level'].apply(
            lambda x: "🟠 Monitor" if x == 'Low' else 
                     "🟡 Check inventory" if x == 'Medium' else 
                     "🔴 Verify payment")
        
        # Show results
        show_dashboard(pd.concat([df[df['cancelled'] == 1], non_cancelled]), None)

def calculate_risk_scores(df):
    """Train model and calculate risk scores"""
    # Use only non-cancelled orders for features
    non_cancelled = df[df['cancelled'] == 0].copy()
    
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
    
    # Calculate risk scores for non-cancelled orders only
    non_cancelled['risk_score'] = model.predict_proba(scaler.transform(features))[:, 1]
    
    # Merge back with cancelled orders
    df = pd.concat([df[df['cancelled'] == 1], non_cancelled])
    
    return df, model

def reclassify_risk_levels(df):
    """Reclassify risk levels based on distribution"""
    non_cancelled = df[df['cancelled'] == 0].copy()
    
    # Define risk levels (10% High, 30% Medium, 60% Low)
    if len(non_cancelled) > 0:
        non_cancelled['risk_level'] = pd.qcut(non_cancelled['risk_score'], 
                                            q=[0, 0.6, 0.9, 1], 
                                            labels=['Low', 'Medium', 'High'])
    else:
        non_cancelled['risk_level'] = 'Low'
    
    # Add AI recommendations
    non_cancelled['AI_Suggestion'] = non_cancelled['risk_level'].apply(
        lambda x: "🟢 No action needed" if x == 'Low' else 
                 "🟠 Send reassurance email" if x == 'Medium' else 
                 "🔴 Expedited shipping")
    
    # Merge back with cancelled orders
    return pd.concat([df[df['cancelled'] == 1], non_cancelled])

def show_dashboard(df, model):
    """Display all dashboard components"""
    # Summary statistics
    st.subheader("📊 Risk Distribution Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Orders", len(df))
    with col2:
        st.metric("Cancelled Orders", df['cancelled'].sum())
    with col3:
        st.metric("Cancellation Rate", f"{df['cancelled'].mean()*100:.1f}%")
    
    # Risk level distribution (non-cancelled only)
    if 'risk_level' in df.columns:
        st.subheader("Risk Level Distribution (Non-Cancelled Orders)")
        risk_dist = df[df['cancelled'] == 0]['risk_level'].value_counts(normalize=True)
        st.bar_chart(risk_dist)

    # Filters
    with st.sidebar:
        st.header("🔍 Filters")
        if 'risk_level' in df.columns:
            risk_filter = st.multiselect(
                "Risk Levels",
                options=['High', 'Medium', 'Low'],
                default=['High', 'Medium']
            )
        min_score = st.slider(
            "Minimum Risk Score",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01
        )

    # Filtered orders display
    st.subheader("📦 At-Risk Orders")
    
    # Get display columns that exist in the dataframe
    possible_cols = [
        'Order ID', 'Date', 'Status', 'Fulfilment', 'Sales Channel',
        'ship-service-level', 'Category', 'Size', 'Courier Status',
        'Amount', 'risk_score', 'risk_level', 'AI_Suggestion'
    ]
    display_cols = [col for col in possible_cols if col in df.columns]
    
    # Filter non-cancelled orders with risk scores
    filtered = df[(df['cancelled'] == 0) & (df['risk_score'] >= min_score)]
    if 'risk_level' in df.columns:
        filtered = filtered[filtered['risk_level'].isin(risk_filter)]
    
    # Format dataframe
    if 'Amount' in display_cols:
        filtered['Amount'] = filtered['Amount'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else '')
    if 'risk_score' in display_cols:
        filtered['risk_score'] = filtered['risk_score'].round(4)
    
    st.dataframe(filtered[display_cols].sort_values('risk_score', ascending=False))

    # Model evaluation (only if model exists)
    if model is not None:
        st.subheader("📈 Model Performance")
        try:
            features = df[[col for col in df.columns if col.endswith('_encoded') or col in ['rating', 'sentiment', 'Item Total']]]
            X = scaler.transform(features.fillna(0))
            y = df['cancelled']
            
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROC AUC Score", f"{roc_auc_score(y, y_prob):.3f}")
            with col2:
                cm = confusion_matrix(y, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['No Cancel', 'Cancel'],
                            yticklabels=['No Cancel', 'Cancel'])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not display model metrics: {str(e)}")

if __name__ == "__main__":
    main()
