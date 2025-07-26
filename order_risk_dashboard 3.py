import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(page_title="Cancellation Risk Dashboard", layout="wide")
st.title("📊 Order Cancellation Risk Analysis Dashboard")

# Load external data
uploaded_file = st.file_uploader("Upload your order dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove any extra spaces from column names

    st.write("### Columns in your file:", df.columns.tolist())

    # Drop rows with missing values in important columns
    expected_cols = ['Status', 'ship-service-level', 'Item Total']
    available_cols = [col for col in expected_cols if col in df.columns]
    if available_cols:
        df = df.dropna(subset=available_cols)

    # Add cancellation flag
    if 'Status' in df.columns:
        df['cancelled'] = df['Status'].apply(lambda x: 1 if 'CANCELLED' in str(x).upper() else 0)
        cancellation_rate = df['cancelled'].mean()
        st.sidebar.metric("Overall Cancellation Rate", f"{cancellation_rate*100:.1f}%")
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

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    df['risk_score'] = model.predict_proba(features)[:, 1]

    # Calculate dollar_amount
    if 'Amount' in df.columns:
        df['dollar_amount'] = df['Amount']
    elif 'Item Total' in df.columns:
        df['dollar_amount'] = df['Item Total']
    else:
        df['dollar_amount'] = 0  # fallback

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.3, 0.05)
        show_technical = st.checkbox("Show Technical Details", False)

    # Main dashboard layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Order Overview")
        st.write(f"Total orders: {len(df)}")
        st.write(f"High-risk orders: {len(df[df['risk_score'] >= threshold])}")
        st.write(f"Min Risk Score: {df['risk_score'].min():.3f}")
        st.write(f"Max Risk Score: {df['risk_score'].max():.3f}")

    with col2:
        st.subheader("💬 Review Sentiment")
        if 'sentiment' in df.columns:
            avg_sentiment = df['sentiment'].mean()
            st.metric(label="Average Sentiment", value=f"{avg_sentiment:.2f}")
            st.progress((avg_sentiment + 1)/2)  # Scale from -1 to 1 to 0-1
        else:
            st.info("No review text available")

    # Risk score distribution
    st.subheader("📈 Risk Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['risk_score'], bins=20, kde=True, ax=ax)
    ax.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    ax.legend()
    st.pyplot(fig)

    # Improved AI recommendation function
    def recommend_action(row):
        risk = row['risk_score']
        dollar_amount = row.get('dollar_amount', 0)
        shipping_level = str(row.get('ship-service-level', 'standard')).lower()
        
        if risk > 0.8:
            if dollar_amount > 100:
                return "🔴 High Risk + High Value: Manual review + expedited shipping offer + discount coupon"
            else:
                return "🔴 High Risk: Manual review + expedited shipping offer"
        elif risk > 0.6:
            if 'standard' in shipping_level:
                return "🟠 Medium Risk: Upgrade shipping + reassurance email"
            else:
                return "🟠 Medium Risk: Personal follow-up call + FAQ email"
        elif risk > 0.4:
            return "🟡 Low Risk: Automated reassurance email"
        else:
            return "🟢 Very Low Risk: No action needed"

    # High-risk orders
    filtered = df[df['risk_score'] >= threshold]
    if not filtered.empty:
        filtered['AI_Suggestion'] = filtered.apply(recommend_action, axis=1)
        
        st.subheader("🚨 High-Risk Orders")
        cols_to_show = [col for col in ['Order ID', 'Category', 'ship-service-level', 
                                      'dollar_amount', 'risk_score', 'AI_Suggestion'] 
                       if col in filtered.columns]
        st.dataframe(filtered[cols_to_show].sort_values('risk_score', ascending=False))
    else:
        st.warning("No high-risk orders found at the selected threshold.")

    # Feature Importance Analysis
    st.subheader("🔍 What Drives Cancellation Risk?")
    
    tab1, tab2, tab3 = st.tabs(["Technical Analysis", "Business Insights", "Model Performance"])
    
    with tab1:
        st.markdown("#### Model Coefficients")
        if hasattr(model, 'coef_'):
            coef_df = pd.DataFrame({
                'Feature': available_features,
                'Coefficient': model.coef_[0],
                'Absolute_Impact': np.abs(model.coef_[0])
            }).sort_values('Absolute_Impact', ascending=False)
            st.dataframe(coef_df.style.background_gradient(cmap='RdBu', subset=['Coefficient']))
        
        st.markdown("#### Permutation Importance")
        try:
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            perm_df = pd.DataFrame({
                'Feature': available_features,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(perm_df)
            fig, ax = plt.subplots()
            sns.barplot(data=perm_df, x='Importance', y='Feature', ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Couldn't calculate permutation importance: {str(e)}")
    
    with tab2:
        st.markdown("#### Business Patterns in High-Risk Orders")
        
        if not filtered.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Category' in df.columns:
                    st.write("Top Categories at Risk:")
                    top_cats = filtered['Category'].value_counts().head(5)
                    st.bar_chart(top_cats)
                
                if 'dollar_amount' in df.columns:
                    st.write("Price Distribution of High-Risk Orders:")
                    st.write(filtered['dollar_amount'].describe())
            
            with col2:
                if 'ship-service-level' in df.columns:
                    st.write("Risk by Shipping Level:")
                    shipping_risk = df.groupby('ship-service-level')['risk_score'].mean().sort_values(ascending=False)
                    st.bar_chart(shipping_risk)
                
                if 'rating' in df.columns:
                    st.write("Average Rating by Risk Level:")
                    df['risk_level'] = pd.cut(df['risk_score'], bins=[0, 0.3, 0.6, 1], 
                                            labels=['Low', 'Medium', 'High'])
                    rating_risk = df.groupby('risk_level')['rating'].mean()
                    st.bar_chart(rating_risk)
    
    with tab3:
        st.markdown("#### Model Evaluation Metrics")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        st.text(classification_report(y_test, y_pred))
        st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Not Cancelled', 'Cancelled'],
                   yticklabels=['Not Cancelled', 'Cancelled'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

else:
    st.info("👋 Please upload a CSV file to begin analysis. Expected columns include order status, shipping level, and item total.")
    st.image("https://via.placeholder.com/600x200?text=Upload+Your+Order+Data+CSV", use_column_width=True)
