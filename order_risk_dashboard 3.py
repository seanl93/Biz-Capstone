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
    if 'review_text' in df.columns:
        df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    else:
        df['sentiment'] = 0.0

    # Encode categorical variables
    for col in ['Category', 'ship-service-level']:
        if col in df.columns:
            df[col + '_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))

    # Prepare features
    feature_cols = [
        'rating', 'sentiment', 'Item Total', 'Quantity', 
        'Category_encoded', 'ship-service-level_encoded'
    ]
    available_features = [col for col in feature_cols if col in df.columns]
    
    if not available_features:
        st.error("No valid features found")
        st.stop()
    
    features = df[available_features].fillna(0)
    labels = df['cancelled']

    # Handle class imbalance
    st.write("### Handling Class Imbalance")
    st.write(f"Class distribution: {Counter(labels)}")
    
    # Apply SMOTE if available and cancellation rate is low
    if SMOTE_AVAILABLE and cancellation_rate < 0.2:
        try:
            sm = SMOTE(random_state=42)
            features_res, labels_res = sm.fit_resample(features, labels)
            st.write(f"After SMOTE resampling: {Counter(labels_res)}")
        except Exception as e:
            st.warning(f"SMOTE failed: {str(e)}. Using original data.")
            features_res, labels_res = features, labels
    else:
        features_res, labels_res = features, labels
        if not SMOTE_AVAILABLE:
            st.info("Using class_weight='balanced' instead of SMOTE")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_res)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels_res, test_size=0.2, random_state=42
    )

    # Train model with appropriate class handling
    model_params = {
        'class_weight': 'balanced',
        'max_iter': 1000,
        'solver': 'liblinear'
    }
    
    if SMOTE_AVAILABLE and cancellation_rate < 0.2:
        # If we used SMOTE, we don't need class weights
        model_params['class_weight'] = None
    
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)

    # Generate risk scores for original data
    df['risk_score'] = model.predict_proba(scaler.transform(features))[:, 1]

    # Dynamic threshold calculation
    st.write("### Risk Score Distribution")
    risk_scores = df['risk_score']
    
    # Calculate thresholds based on percentiles
    threshold_options = {
        'Auto (80th %ile)': np.percentile(risk_scores, 80),
        'Conservative (90th %ile)': np.percentile(risk_scores, 90),
        'Moderate (75th %ile)': np.percentile(risk_scores, 75),
        'Aggressive (50th %ile)': np.percentile(risk_scores, 50)
    }
    
    selected_threshold = st.selectbox(
        "Select threshold strategy:",
        list(threshold_options.keys()),
        index=0
    )
    threshold = threshold_options[selected_threshold]
    
    # Show risk score statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Minimum Risk", f"{risk_scores.min():.3f}")
    with col2:
        st.metric("Average Risk", f"{risk_scores.mean():.3f}")
    with col3:
        st.metric("Maximum Risk", f"{risk_scores.max():.3f}")
    
    st.write(f"Current threshold: {threshold:.3f} ({selected_threshold})")

    # 📊 Histogram of Risk Scores
    st.subheader("📊 Risk Score Histogram")
    fig, ax = plt.subplots()
    sns.histplot(risk_scores, bins=30, kde=True, ax=ax, color='skyblue')
    ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
    ax.set_title("Distribution of Predicted Risk Scores")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # Enhanced recommendation system
    def recommend_action(row):
        risk = row['risk_score']
        amount = row.get('dollar_amount', row.get('Item Total', 0))
        
        if risk > threshold:
            if amount > 100:
                return "🔴 CRITICAL: High-value cancellation risk - immediate review"
            elif risk > threshold * 1.5:
                return "🔴 HIGH: Probable cancellation - offer discount"
            else:
                return "🟠 MEDIUM: Potential cancellation - monitor"
        elif risk > threshold * 0.7:
            return "🟡 WATCH: Slight risk - standard process"
        else:
            return "🟢 OK: Normal order"

    # Display high-risk orders
    high_risk = df[df['risk_score'] >= threshold]
    if not high_risk.empty:
        high_risk['Recommendation'] = high_risk.apply(recommend_action, axis=1)
        
        st.subheader(f"🚨 High-Risk Orders (≥{threshold:.3f}) - {len(high_risk)} found")
        cols_to_show = [
            'Order ID', 'Status', 'risk_score', 'Item Total', 
            'Category', 'ship-service-level', 'Recommendation'
        ]
        cols_to_show = [c for c in cols_to_show if c in high_risk.columns]
        st.dataframe(
            high_risk[cols_to_show]
            .sort_values('risk_score', ascending=False)
            .head(50)
        )
        
        if st.button("Export High-Risk Orders"):
            csv = high_risk.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="high_risk_orders.csv",
                mime="text/csv"
            )
    else:
        st.warning(f"No orders above current threshold ({threshold:.3f})")
        st.info("Try selecting a less aggressive threshold strategy")

    # Model diagnostics
    st.subheader("🔍 Model Diagnostics")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

    # 📉 Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("📉 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Not Cancelled', 'Cancelled'],
                yticklabels=['Not Cancelled', 'Cancelled'])
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    # Feature importance
    if hasattr(model, 'coef_'):
        importance_df = pd.DataFrame({
            'Feature': available_features,
            'Coefficient': model.coef_[0],
            'Absolute_Impact': np.abs(model.coef_[0])
        }).sort_values('Absolute_Impact', ascending=False)
        
        st.subheader("Feature Importance")
        st.dataframe(importance_df)

else:
    st.info("Please upload a CSV file to begin analysis")
    st.markdown("""
    **Installation notes:**
    - Required: `pip install streamlit pandas textblob scikit-learn numpy`
    - For advanced features: `pip install imbalanced-learn matplotlib seaborn`
    """)
