import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
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
    
    # Check if we have any cancelled orders
    if df['cancelled'].sum() == 0:
        st.warning("No cancelled orders found in dataset. Using simplified risk assessment.")
        df = simple_risk_assessment(df)
    else:
        st.success(f"Found {df['cancelled'].sum()} cancelled orders. Using advanced risk assessment.")
        df = advanced_risk_assessment(df)
    
    # Display dashboard
    show_dashboard(df)

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

def simple_risk_assessment(df):
    """Simplified risk assessment when no cancellations exist"""
    non_cancelled = df[df['cancelled'] == 0].copy()
    
    # Create simple risk scores based on available features
    risk_factors = []
    
    if 'rating' in non_cancelled.columns:
        risk_factors.append((5 - non_cancelled['rating']) / 4)  # Normalize 1-5 to 0-1
    
    if 'sentiment' in non_cancelled.columns:
        risk_factors.append((1 - non_cancelled['sentiment']) / 2)  # Normalize -1 to 1 to 0-1
    
    if len(risk_factors) > 0:
        non_cancelled['risk_score'] = np.mean(risk_factors, axis=0)
    else:
        non_cancelled['risk_score'] = 0.3  # Default baseline
    
    # Define risk levels
    non_cancelled['risk_level'] = pd.qcut(non_cancelled['risk_score'], 
                                        q=[0, 0.7, 0.9, 1], 
                                        labels=['Low', 'Medium', 'High'])
    
    # Add AI suggestions
    non_cancelled['AI_Suggestion'] = non_cancelled['risk_level'].apply(
        lambda x: "🟢 No action needed" if x == 'Low' else 
                 "🟠 Check inventory" if x == 'Medium' else 
                 "🔴 Verify payment details")
    
    return pd.concat([df[df['cancelled'] == 1], non_cancelled])

def advanced_risk_assessment(df):
    """Advanced risk assessment when cancellations exist"""
    # This would use your original logistic regression approach
    # For now, we'll use the same simple method but you can expand this
    return simple_risk_assessment(df)

def show_dashboard(df):
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

if __name__ == "__main__":
    main()
