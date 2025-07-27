import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(page_title="Order Cancellation Risk Dashboard", layout="wide")
st.title("🚨 Advanced Order Risk Analysis Dashboard")

def main():
    uploaded_file = st.file_uploader("Upload your order dataset", type=["csv"])
    
    if not uploaded_file:
        st.info("Please upload a CSV file to begin analysis.")
        return
    
    # Load and preprocess data
    df = load_and_preprocess_data(uploaded_file)
    
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

    st.write("### Columns in your file:", df.columns.tolist())

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
    
    return df

def calculate_risk_scores(df):
    """Train model and calculate risk scores"""
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
    
    # Calculate dollar_amount if available
    if 'Amount' in df.columns:
        df['dollar_amount'] = df['Amount']
    
    return df, model

def reclassify_risk_levels(df):
    """Reclassify risk levels based on distribution"""
    non_cancelled = df[df['cancelled'] == 0].copy()
    non_cancelled_sorted = non_cancelled.sort_values(by='risk_score', ascending=False).reset_index(drop=True)

    total = len(non_cancelled_sorted)
    high_cutoff = int(total * 0.15)
    medium_cutoff = int(total * 0.55)  # 15% + 40%

    # Set boundaries based on sorted scores
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
    return df

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
    
    # Risk level distribution
    st.write("### Risk Level Distribution (Non-Cancelled Orders)")
    dist = df[df['cancelled'] == 0]['risk_level'].value_counts(normalize=True).mul(100).round(1)
    st.bar_chart(dist)

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
            min_value=float(df['risk_score'].min()),
            max_value=float(df['risk_score'].max()),
            value=0.5,
            step=0.01
        )
        max_results = st.selectbox(
            "Max orders to show",
            options=[50, 100, 200, 500],
            index=1
        )

    # Apply filters
    filtered = df[
        (df['risk_level'].isin(risk_filter)) & 
        (df['risk_score'] >= min_score)
    ].copy()

    # Add AI recommendations (display only - no checkboxes)
    def recommend_action(row):
        if row['risk_level'] == 'High':
            return "🔴 Offer expedited shipping or manual review"
        elif row['risk_level'] == 'Medium':
            return "🟠 Send reassurance email with FAQs"
        return "🟢 No action needed"

    filtered['AI_Suggestion'] = filtered.apply(recommend_action, axis=1)

    # Display filtered results
    st.subheader("📦 Risk Orders with AI Recommendations")
    st.caption(f"Showing {min(len(filtered), max_results)} of {len(filtered)} filtered orders")
    
    display_cols = ['Order ID', 'Status', 'risk_score', 'risk_level', 'dollar_amount', 'AI_Suggestion']
    display_cols = [col for col in display_cols if col in filtered.columns]
    
    # Paginated display without approval checkboxes
    def show_dataframe(df, page_size=10):
        pages = len(df) // page_size + (1 if len(df) % page_size else 0)
        page = st.number_input('Page', min_value=1, max_value=pages, value=1)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Style the dataframe
        styled_df = df.iloc[start_idx:end_idx].style \
            .applymap(lambda x: 'background-color: #ffcccc' if x == 'High' else 
                                ('background-color: #fff3cd' if x == 'Medium' else ''), 
                     subset=['risk_level']) \
            .format({'risk_score': '{:.3f}', 'dollar_amount': '${:,.2f}'})
        
        st.dataframe(styled_df)

    show_dataframe(filtered[display_cols].sort_values('risk_score', ascending=False).head(max_results))

    # Feature Importance
    st.subheader("🔍 Top Predictive Features")
    try:
        features = df[[col for col in df.columns if col.endswith('_encoded') or col in ['rating', 'sentiment', 'Item Total']]]
        if not features.empty:
            rfe = RFE(model, n_features_to_select=min(5, len(features.columns)))
            rfe.fit(StandardScaler().fit_transform(features.fillna(0)), df['cancelled'])
            top_features = pd.DataFrame({
                'Feature': features.columns[rfe.support_],
                'Ranking': rfe.ranking_[rfe.support_]
            }).sort_values('Ranking')
            st.dataframe(top_features)
        else:
            st.warning("No features available for importance analysis")
    except Exception as e:
        st.error(f"Feature importance analysis failed: {str(e)}")

    # Model Performance
    st.subheader("📈 Model Evaluation")
    X = df[[col for col in df.columns if col.endswith('_encoded') or col in ['rating', 'sentiment', 'Item Total']]]
    y = df['cancelled']
    
    if not X.empty:
        X_scaled = StandardScaler().fit_transform(X.fillna(0))
        y_prob = model.predict_proba(X_scaled)[:, 1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC AUC Score", f"{roc_auc_score(y, y_prob):.3f}")
        
        with col2:
            y_pred = model.predict(X_scaled)
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['No Cancel', 'Cancel'], 
                        yticklabels=['No Cancel', 'Cancel'])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
    else:
        st.warning("Insufficient features to evaluate model performance")

    # Sentiment Analysis
    if 'sentiment' in df.columns:
        st.subheader("💬 Customer Sentiment Analysis")
        avg_sentiment = df['sentiment'].mean()
        st.metric("Average Sentiment Score", f"{avg_sentiment:.3f}")
        st.caption("Positive values indicate positive sentiment, negative values indicate negative sentiment")

if __name__ == "__main__":
    main()
