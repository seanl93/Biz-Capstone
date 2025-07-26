import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from collections import Counter

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
        df['cancelled'] = df['Status'].str.upper().str.contains('CANCEL|REFUND|RETURN').astype(int)
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

    # Prepare features - adding more potential predictors
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
    st.write(f"Before resampling: {Counter(labels)}")
    
    # Apply SMOTE oversampling only if cancellation rate is low
    if cancellation_rate < 0.2:
        try:
            sm = SMOTE(random_state=42)
            features_res, labels_res = sm.fit_resample(features, labels)
            st.write(f"After SMOTE resampling: {Counter(labels_res)}")
        except:
            st.warning("SMOTE failed, proceeding with original data")
            features_res, labels_res = features, labels
    else:
        features_res, labels_res = features, labels

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_res)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels_res, test_size=0.2, random_state=42
    )

    # Train model with class weighting
    model = LogisticRegression(
        class_weight='balanced',  # Automatically adjusts for imbalanced classes
        max_iter=1000,
        solver='liblinear'  # Better for small datasets
    )
    model.fit(X_train, y_train)

    # Generate risk scores for original data
    df['risk_score'] = model.predict_proba(scaler.transform(features))[:, 1]

    # Dynamic threshold calculation
    st.write("### Risk Score Distribution")
    risk_scores = df['risk_score']
    
    # Calculate dynamic threshold based on percentiles
    threshold_options = {
        'Conservative (90th %ile)': np.percentile(risk_scores, 90),
        'Moderate (75th %ile)': np.percentile(risk_scores, 75),
        'Aggressive (50th %ile)': np.percentile(risk_scores, 50)
    }
    
    selected_threshold = st.selectbox(
        "Select threshold strategy:",
        list(threshold_options.keys()),
        index=1
    )
    threshold = threshold_options[selected_threshold]
    
    # Visualization of risk scores
    st.write(f"Current threshold: {threshold:.3f}")
    st.write(f"Risk scores range: {risk_scores.min():.3f} to {risk_scores.max():.3f}")
    
    # Enhanced recommendation system
    def recommend_action(row):
        risk = row['risk_score']
        amount = row.get('dollar_amount', row.get('Item Total', 0))
        
        if risk > threshold:
            if amount > 100:
                return "🔴 CRITICAL: High-value cancellation risk - immediate manager review"
            elif risk > threshold * 1.5:
                return "🔴 HIGH: Probable cancellation - offer discount/upgrade"
            else:
                return "🟠 MEDIUM: Potential cancellation - send reassurance email"
        elif risk > threshold * 0.7:
            return "🟡 WATCH: Monitor for changes"
        else:
            return "🟢 OK: Normal order"

    # Display high-risk orders
    high_risk = df[df['risk_score'] >= threshold]
    if not high_risk.empty:
        high_risk['Recommendation'] = high_risk.apply(recommend_action, axis=1)
        
        st.subheader(f"🚨 High-Risk Orders (≥{threshold:.3f})")
        cols_to_show = [
            'Order ID', 'Status', 'risk_score', 'Item Total', 
            'Category', 'ship-service-level', 'Recommendation'
        ]
        cols_to_show = [c for c in cols_to_show if c in high_risk.columns]
        st.dataframe(
            high_risk[cols_to_show]
            .sort_values('risk_score', ascending=False)
            .style.background_gradient(subset=['risk_score'], cmap='Reds')
        )
        
        # Show risk distribution by category
        if 'Category' in high_risk.columns:
            st.subheader("High-Risk Orders by Category")
            category_risk = high_risk.groupby('Category').size()
            st.bar_chart(category_risk)
    else:
        st.warning(f"No orders above current threshold ({threshold:.3f})")

    # Model diagnostics
    st.subheader("🔍 Model Diagnostics")
    
    # Performance metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
    
    # Feature importance
    if hasattr(model, 'coef_'):
        importance_df = pd.DataFrame({
            'Feature': available_features,
            'Importance': np.abs(model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        st.subheader("Feature Importance")
        st.dataframe(importance_df)
        
        # Show top drivers of cancellations
        top_feature = importance_df.iloc[0]['Feature']
        st.write(f"Top cancellation driver: **{top_feature}**")
        
        if top_feature in df.columns:
            st.write(f"Distribution of {top_feature} in cancellations vs non-cancellations:")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='cancelled', y=top_feature, ax=ax)
            st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin analysis")
    st.markdown("""
    **Expected columns:**
    - `Status` (to identify cancellations)
    - `Item Total` or `Amount` (order value)
    - `Category` (product category)
    - Optional: `review_text` for sentiment analysis
    """)
