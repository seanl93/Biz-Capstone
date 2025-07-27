import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(page_title="Order Cancellation Risk Dashboard", layout="wide")
st.title("🚨 Enhanced Order Risk Detection Dashboard")

# Main function
def main():
    uploaded_file = st.file_uploader("Upload your order dataset", type=["csv"])
    
    if not uploaded_file:
        st.info("Please upload a CSV file to begin analysis.")
        return
    
    # Data processing
    df = load_and_preprocess_data(uploaded_file)
    
    # Model training
    model, scaler, features = train_logistic_regression(df)
    
    # Risk assessment
    df = assess_risk_levels(df, model, scaler, features)
    
    # Display results
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

    # Sentiment analysis
    df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity) if 'review_text' in df.columns else 0.0

    # Encode categorical features
    for col in ['Category', 'ship-service-level']:
        if col in df.columns:
            df[col + '_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))
    
    return df

def train_logistic_regression(df):
    """Train the logistic regression model"""
    feature_cols = ['rating', 'sentiment', 'Item Total', 'Category_encoded', 'ship-service-level_encoded']
    available_features = [col for col in feature_cols if col in df.columns]

    if not available_features:
        st.error("No valid features available for modeling.")
        st.stop()

    features = df[available_features].fillna(0)
    labels = df['cancelled']

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, features

def assess_risk_levels(df, model, scaler, features):
    """Calculate risk scores and assign risk levels"""
    df['risk_score'] = model.predict_proba(scaler.transform(features))[:, 1]

    # Redistribute risk levels on non-cancelled orders
    non_cancelled = df[df['cancelled'] == 0].copy()
    sorted_nc = non_cancelled.sort_values('risk_score', ascending=False).reset_index(drop=True)

    total_nc = len(sorted_nc)
    high_cut = int(total_nc * 0.15)
    medium_cut = int(total_nc * 0.55)

    high_thresh = sorted_nc.loc[high_cut - 1, 'risk_score'] if total_nc > high_cut else 1.0
    medium_thresh = sorted_nc.loc[medium_cut - 1, 'risk_score'] if total_nc > medium_cut else 0.0

    def assign_risk(score):
        if score >= high_thresh:
            return 'High'
        elif score >= medium_thresh:
            return 'Medium'
        else:
            return 'Low'

    df['risk_level'] = df['risk_score'].apply(assign_risk)
    
    return df

def show_dashboard(df):
    """Display all dashboard components"""
    # Summary of distribution
    st.subheader("📊 Risk Level Distribution (Non-Cancelled Orders)")
    distribution = df[df['cancelled'] == 0]['risk_level'].value_counts(normalize=True).round(3) * 100
    st.write(distribution.to_frame("Percentage (%)"))

    # Filtered orders display
    st.subheader("📦 Risk Orders with AI Suggestions")
    
    # Filter controls in sidebar
    with st.sidebar:
        st.header("Filters")
        risk_filter = st.multiselect(
            "Risk Levels",
            options=['High', 'Medium'],
            default=['High', 'Medium']
        )
        min_score = st.slider(
            "Minimum Risk Score",
            min_value=0.0,
            max_value=1.0,
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
        (df['cancelled'] == 0) &
        (df['risk_score'] >= min_score)
    ].copy()

    # Add AI suggestions
    def recommend_action(row):
        if row['risk_level'] == 'High':
            return "🔴 Offer expedited shipping or manual review"
        elif row['risk_level'] == 'Medium':
            return "🟠 Send reassurance email with FAQs"
        return "🟢 No action needed"

    filtered['AI_Suggestion'] = filtered.apply(recommend_action, axis=1)
    
    # Sort and limit results
    filtered_display = filtered.sort_values('risk_score', ascending=False).head(max_results)
    
    # Display results with pagination
    display_cols = ['Order ID', 'Status', 'risk_score', 'risk_level', 'Item Total', 'AI_Suggestion']
    display_cols = [col for col in display_cols if col in filtered_display.columns]
    
    # Show summary stats
    st.caption(f"Showing {len(filtered_display)} of {len(filtered)} filtered orders (from {len(df[df['cancelled'] == 0])} total non-cancelled orders)")
    
    # Paginated table
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
            .format({'risk_score': '{:.3f}', 'Item Total': '${:,.2f}'})
        
        st.dataframe(styled_df)

    show_dataframe(filtered_display[display_cols])

    # Model performance
    st.subheader("📈 Model Performance")
    X = df[[col for col in df.columns if col in ['rating', 'sentiment', 'Item Total', 'Category_encoded', 'ship-service-level_encoded']]].fillna(0)
    y = df['cancelled']
    
    if len(X.columns) > 0:
        X_scaled = StandardScaler().fit_transform(X)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        st.write("**Overall ROC AUC Score:**", round(roc_auc_score(y, y_prob), 3))
        
        # Confusion Matrix
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
        st.warning("Insufficient features to show model performance")

if __name__ == "__main__":
    main()
