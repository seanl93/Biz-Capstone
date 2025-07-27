import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Check for SMOTE availability
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    st.warning("SMOTE not available (install with: pip install imbalanced-learn). Using class weights instead.")

# Configure page
st.set_page_config(page_title="Enhanced Cancellation Risk Dashboard", layout="wide")
st.title("🚨 Enhanced Order Cancellation Risk Analysis")

# Load external data
uploaded_file = st.file_uploader("Upload your order dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Data Preparation
    st.write("### Data Preparation")
    
    # Create cancellation flag - more inclusive definition
    if 'Status' in df.columns:
        status_upper = df['Status'].str.upper()
        df['cancelled'] = (status_upper.str.contains('CANCEL') | 
                          status_upper.str.contains('REFUND') | 
                          status_upper.str.contains('RETURN')).astype(int)
        cancellation_rate = df['cancelled'].mean()
        st.sidebar.metric("Cancellation Rate", f"{cancellation_rate*100:.1f}%")
        
        if cancellation_rate < 0.05:
            st.warning(f"⚠️ Very low cancellation rate ({cancellation_rate*100:.1f}%). Model may need special handling.")
    else:
        st.error("Missing 'Status' column")
        st.stop()

    # Feature Engineering
    if 'review_text' in_
