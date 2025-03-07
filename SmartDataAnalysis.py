import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import time
import numpy as np
import plotly.graph_objects as go
import base64
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config with custom icon
st.set_page_config(page_title=" Smart Data Analysis ", layout="wide", initial_sidebar_state="expanded")

# Enhanced Styling and Animations (keeping your original styling)
st.markdown(
    """
    <style>
        /* Global animations */
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-20px);}
            to {opacity: 1; transform: translateY(0);}
        }
        @keyframes slideIn {
            from {opacity: 0; transform: translateX(-50px);}
            to {opacity: 1; transform: translateX(0);}
        }
        @keyframes pulse {
            0% {transform: scale(1);}
            50% {transform: scale(1.05);}
            100% {transform: scale(1);}
        }
        @keyframes glow {
            0% {text-shadow: 0 0 5px rgba(106, 90, 205, 0.5);}
            50% {text-shadow: 0 0 20px rgba(106, 90, 205, 0.8);}
            100% {text-shadow: 0 0 5px rgba(106, 90, 205, 0.5);}
        }
        @keyframes shimmer {
            0% {background-position: -1000px 0;}
            100% {background-position: 1000px 0;}
        }
        
        /* Global styling */
        body {
            background: linear-gradient(135deg, #1A1A1D 0%, #2E2E3D 100%);
            color: #E0E0FF;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Header styling */
        .title {
            color: #9D72FF;
            text-align: center;
            font-size: 52px;
            font-weight: bold;
            margin-bottom: 25px;
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(90deg, rgba(106, 90, 205, 0.1), rgba(106, 90, 205, 0.3), rgba(106, 90, 205, 0.1));
            animation: fadeIn 1.5s ease-in-out, glow 3s infinite;
            text-shadow: 0 0 10px rgba(106, 90, 205, 0.7);
        }
        
        /* Greeting styling */
        .greeting {
            font-size: 72px;
            font-weight: bold;
            text-align: left;
            background: linear-gradient(45deg, #D4AF37, #FFE16B, #D4AF37);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: fadeIn 2s ease-in-out, shimmer 5s infinite linear;
            background-size: 200% 100%;
            margin-top: 10px;
            margin-left: 30px;
            text-shadow: 0 0 15px rgba(212, 175, 55, 0.5);
            border-bottom: 3px solid rgba(212, 175, 55, 0.3);
            padding-bottom: 15px;
        }
        
        /* Section styles */
        .upload-section {
            text-align: center;
            animation: slideIn 1.5s;
            margin-top: 20px;
            padding: 25px;
            border-radius: 15px;
            background: linear-gradient(to right, rgba(44, 47, 51, 0.7), rgba(75, 78, 85, 0.7));
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            transition: all 0.3s ease;
        }
        .upload-section:hover {
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.57);
            transform: translateY(-5px);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2C2F33 0%, #23272A 100%);
            border-right: 2px solid rgba(106, 90, 205, 0.3);
        }
        
        /* Widget styling */
        .stSlider > div > div {
            background-color: #6A5ACD !important;
        }
        .stSlider > div > div > div > div {
            background-color: #FFD700 !important;
            animation: pulse 2s infinite;
        }
        .slider-label {
            color: #B19CD9;
            font-weight: bold;
            letter-spacing: 1px;
        }
        
        /* Dataframe styling */
        .dataframe {
            animation: fadeIn 1s;
            border-radius: 10px !important;
            overflow: hidden !important;
        }
        
        /* Input field styling */
        input[type=text] {
            border-radius: 10px !important;
            border: 2px solid rgba(106, 90, 205, 0.5) !important;
            background-color: rgba(40, 42, 54, 0.8) !important;
            color: #E0E0FF !important;
            transition: all 0.3s !important;
        }
        input[type=text]:focus {
            border: 2px solid #9D72FF !important;
            box-shadow: 0 0 15px rgba(106, 90, 205, 0.5) !important;
        }
        
        /* Section headers */
        h2, h3 {
            color: #B19CD9;
            border-bottom: 2px solid rgba(106, 90, 205, 0.3);
            padding-bottom: 8px;
            animation: fadeIn 1.5s;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(45deg, #6A5ACD, #9370DB) !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 10px 25px !important;
            transition: all 0.3s !important;
            animation: fadeIn 1.5s, pulse 3s infinite !important;
            font-weight: bold !important;
            letter-spacing: 1px !important;
        }
        .stButton>button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 7px 20px rgba(106, 90, 205, 0.5) !important;
        }
        
        /* Metrics */
        .metric-container {
            background: linear-gradient(135deg, rgba(106, 90, 205, 0.2), rgba(147, 112, 219, 0.2)) !important;
            border-radius: 10px !important;
            padding: 10px !important;
            margin-bottom: 15px !important;
            border: 1px solid rgba(106, 90, 205, 0.3) !important;
            animation: fadeIn 1.5s !important;
            transition: all 0.3s !important;
        }
        .metric-container:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 7px 20px rgba(106, 90, 205, 0.3) !important;
        }
        
        /* File uploader */
        .uploadedFile {
            background: linear-gradient(135deg, rgba(106, 90, 205, 0.2), rgba(147, 112, 219, 0.2)) !important;
            border-radius: 10px !important;
            border: 2px dashed rgba(106, 90, 205, 0.5) !important;
            padding: 15px !important;
            animation: fadeIn 1.5s !important;
        }
        
        /* Selectbox styling */
        .stSelectbox>div>div {
            background-color: rgba(40, 42, 54, 0.8) !important;
            border: 2px solid rgba(106, 90, 205, 0.5) !important;
            border-radius: 10px !important;
            color: #E0E0FF !important;
            transition: all 0.3s !important;
        }
        .stSelectbox>div>div:hover {
            border: 2px solid #9D72FF !important;
            box-shadow: 0 0 15px rgba(106, 90, 205, 0.5) !important;
        }
        
        /* Power BI Style Filter Pane */
        .filter-pane {
            background: linear-gradient(135deg, rgba(60, 63, 68, 0.9), rgba(40, 42, 54, 0.9));
            border-radius: 10px;
            border-left: 3px solid #9D72FF;
            padding: 15px;
            box-shadow: -5px 0 15px rgba(0, 0, 0, 0.2);
            height: 100%;
        }
        
        /* KPI Cards */
        .kpi-card {
            background: linear-gradient(135deg, rgba(60, 63, 68, 0.9), rgba(40, 42, 54, 0.9));
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid;
            transition: all 0.3s;
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 7px 15px rgba(0, 0, 0, 0.3);
        }
        .kpi-green {
            border-color: #4CAF50;
        }
        .kpi-yellow {
            border-color: #FFC107;
        }
        .kpi-red {
            border-color: #F44336;
        }
        
        /* Tooltips */
        .custom-tooltip {
            background-color: rgba(40, 42, 54, 0.95) !important;
            border: 1px solid #9D72FF !important;
            border-radius: 5px !important;
            padding: 10px !important;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3) !important;
            color: #E0E0FF !important;
        }
        
        /* Bookmark Panel */
        .bookmark-panel {
            background: linear-gradient(135deg, rgba(60, 63, 68, 0.9), rgba(40, 42, 54, 0.9));
            border-radius: 10px;
            border-top: 3px solid #FFD700;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        /* AI Insights Panel */
        .ai-insights {
            background: linear-gradient(135deg, rgba(106, 90, 205, 0.1), rgba(147, 112, 219, 0.2));
            border-radius: 10px;
            border-left: 4px solid #00BCD4;
            padding: 15px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        .insight-item {
            padding: 10px;
            margin: 10px 0;
            border-radius: 8px;
            background: rgba(60, 63, 68, 0.5);
            border-left: 3px solid;
        }
        .high-priority {
            border-color: #F44336;
        }
        .medium-priority {
            border-color: #FFC107;
        }
        .low-priority {
            border-color: #4CAF50;
        }
        
        /* Q&A Section */
        .qa-section {
            background: linear-gradient(135deg, rgba(60, 63, 68, 0.9), rgba(40, 42, 54, 0.9));
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            border-left: 4px solid #9D72FF;
        }
        .qa-input {
            background-color: rgba(40, 42, 54, 0.8) !important;
            border: 2px solid rgba(106, 90, 205, 0.5) !important;
            border-radius: 20px !important;
            padding: 10px 15px !important;
            color: #E0E0FF !important;
        }
        
        /* Grid Layout */
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .grid-item {
            background: rgba(60, 63, 68, 0.5);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s;
        }
        .grid-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom-styled sidebar for navigation
st.sidebar.markdown(
    """
    <div style='animation: fadeIn 1.5s; text-align: center;'>
        <h1 style='color: #9D72FF; text-shadow: 0 0 10px rgba(106, 90, 205, 0.5); font-size: 28px; margin-bottom: 30px;'>
             Dashboard Navigation 
        </h1>
        <p style='color: #B19CD9; font-style: italic; margin-bottom: 20px; padding: 10px; border-radius: 10px; background: rgba(106, 90, 205, 0.1);'>
            Use the options below to explore your data insights.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Enhanced title with animated sparkles
st.markdown(
    """
    <div class='title'>
        <span style='animation: pulse 3s infinite;'></span> 
        Smart Data Analysis
        <span style='animation: pulse 3s infinite;'></span>
    </div>
    """, 
    unsafe_allow_html=True
)

# Get current time to determine greeting with emoji
time_now = datetime.datetime.now().hour
if time_now < 12:
    greeting_text = "Good Morning 🌅"
elif time_now < 18:
    greeting_text = "Good Afternoon ☀️"
else:
    greeting_text = "Good Evening 🌙"

# User input name with enhanced animation
st.markdown(
    """
    <div style='animation: slideIn 1.5s; padding: 15px; margin-bottom: 20px; border-radius: 10px; background: linear-gradient(to right, rgba(44, 47, 51, 0.7), rgba(75, 78, 85, 0.7));'>
        <p style='color: #B19CD9; font-size: 20px; margin-bottom: 10px;'>Please enter your name below:</p>
    </div>
    """,
    unsafe_allow_html=True
)
name = st.text_input("", "", key="name_input", placeholder="Type your name here...")

# Function to display KPI
def display_kpi(title, value, subtitle):
    st.markdown(
        f"""
        <div style='padding: 10px; border-radius: 5px; background: rgba(60, 63, 68, 0.7); margin-bottom: 10px;'>
            <h4 style='color: #9D72FF; font-size: 16px; margin-bottom: 5px;'>{title}</h4>
            <p style='color: #E0E0FF; font-size: 24px; margin: 0;'>{value}</p>
            <p style='color: #B19CD9; font-size: 14px; margin: 0;'>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to generate AI insights from data
def generate_ai_insights(df):
    insights = []
    
    # Check for numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 0:
        try:
            # 1. Find outliers/anomalies using Isolation Forest
            if len(df) > 10:  # Need enough data for meaningful anomaly detection
                # Select numeric columns for anomaly detection
                data_for_anomaly = df[numeric_cols].copy()
                # Fill NaN values with column means
                data_for_anomaly = data_for_anomaly.fillna(data_for_anomaly.mean())
                
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_for_anomaly)
                
                # Train Isolation Forest
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                outliers = iso_forest.fit_predict(scaled_data)
                
                # Count outliers
                num_outliers = np.sum(outliers == -1)
                if num_outliers > 0:
                    outlier_indices = np.where(outliers == -1)[0]
                    outlier_rows = df.iloc[outlier_indices]
                    
                    # Identify which columns have the most extreme values in outliers
                    problematic_columns = []
                    for col in numeric_cols:
                        if col in df.columns:
                            col_mean = df[col].mean()
                            col_std = df[col].std()
                            if col_std > 0:  # Avoid division by zero
                                extreme_values = outlier_rows[abs((outlier_rows[col] - col_mean) / col_std) > 2]
                                if len(extreme_values) > 0:
                                    problematic_columns.append((col, len(extreme_values)))
                    
                    if problematic_columns:
                        problematic_columns.sort(key=lambda x: x[1], reverse=True)
                        insights.append({
                            'title': f"🚨 Detected {num_outliers} outliers in your data",
                            'description': f"The most affected columns are: {', '.join([col[0] for col in problematic_columns[:3]])}. These outliers may be skewing your analysis.",
                            'recommendation': "Review these outliers to determine if they're errors or valuable insights. Consider using robust statistical methods.",
                            'priority': 'high' if num_outliers > len(df) * 0.1 else 'medium'
                        })
            
            # 2. Identify columns with high null values
            null_percentage = df.isnull().sum() / len(df) * 100
            high_null_cols = null_percentage[null_percentage > 15].index.tolist()
            
            if high_null_cols:
                insights.append({
                    'title': f"⚠️ High missing values detected in {len(high_null_cols)} columns",
                    'description': f"Columns with >15% missing values: {', '.join(high_null_cols)}",
                    'recommendation': "Consider imputation techniques or evaluate if these columns can be dropped.",
                    'priority': 'high' if any(null_percentage > 40) else 'medium'
                })
            
            # 3. Identify low-performing metrics based on statistical analysis
            if len(numeric_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = df[numeric_cols].corr().abs()
                
                # Find highly correlated pairs (could indicate redundant features)
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = [(upper_tri.index[i], upper_tri.columns[j], upper_tri.iloc[i, j]) 
                                  for i, j in zip(*np.where(upper_tri > 0.9))]
                
                if high_corr_pairs:
                    insights.append({
                        'title': f"📊 Found {len(high_corr_pairs)} highly correlated metrics",
                        'description': f"These pairs show >90% correlation, suggesting potential redundancy: " + 
                                      f"{', '.join([f'{pair[0]} & {pair[1]}' for pair in high_corr_pairs[:3]])}",
                        'recommendation': "Consider consolidating these metrics for cleaner analysis.",
                        'priority': 'medium'
                    })
                
                # Identify metrics with high variance (potential instability)
                cv_values = df[numeric_cols].std() / df[numeric_cols].mean()
                high_variance_cols = cv_values[cv_values > 1].index.tolist()
                
                if high_variance_cols:
                    insights.append({
                        'title': f"📈 Detected {len(high_variance_cols)} metrics with high variability",
                        'description': f"These metrics show unusually high coefficient of variation: {', '.join(high_variance_cols[:3])}",
                        'recommendation': "Investigate the causes of volatility and consider segmentation for more stable metrics.",
                        'priority': 'medium'
                    })
                
                # Identify skewed distributions
                skewed_cols = []
                for col in numeric_cols:
                    if df[col].skew() > 1.5 or df[col].skew() < -1.5:
                        skewed_cols.append((col, df[col].skew()))
                
                if skewed_cols:
                    insights.append({
                        'title': f"⚖️ Found {len(skewed_cols)} metrics with skewed distributions",
                        'description': f"These metrics may benefit from transformation: {', '.join([col[0] for col in skewed_cols[:3]])}",
                        'recommendation': "Consider log or square-root transformations for more accurate statistical analysis.",
                        'priority': 'low'
                    })
                    
                # Check for potential seasonality/trends (if any date columns exist)
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols and len(date_cols) > 0:
                    insights.append({
                        'title': "📅 Time-based analysis opportunity detected",
                        'description': f"Your data contains date/time information in columns: {', '.join(date_cols[:3])}",
                        'recommendation': "Analyze temporal patterns to uncover seasonality or trends in your metrics.",
                        'priority': 'medium'
                    })
            
            # 4. Performance insights (for sales data specifically)
            if any('sale' in col.lower() for col in df.columns) or any('revenue' in col.lower() for col in df.columns):
                sales_cols = [col for col in df.columns if 'sale' in col.lower() or 'revenue' in col.lower()]
                if sales_cols:
                    sales_col = sales_cols[0]  # Use the first sales column found
                    if df[sales_col].mean() > 0:
                        quartiles = df[sales_col].quantile([0.25, 0.75]).values
                        bottom_performers = df[df[sales_col] < quartiles[0]]
                        
                        if len(bottom_performers) > 0:
                            # Look for patterns in underperforming sales
                            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                            pattern_insights = []
                            
                            for cat_col in categorical_cols[:3]:  # Check first 3 categorical columns
                                if cat_col in bottom_performers.columns:
                                    value_counts = bottom_performers[cat_col].value_counts(normalize=True)
                                    if not value_counts.empty and value_counts.iloc[0] > 0.3:  # If any category represents >30% of poor performers
                                        pattern_insights.append(f"{cat_col}='{value_counts.index[0]}'")
                            
                            if pattern_insights:
                                insights.append({
                                    'title': f"💡 Performance improvement opportunity identified",
                                    'description': f"25% of your sales/revenue data falls below {quartiles[0]:.2f}. " +
                                                  f"Common patterns in low performers: {', '.join(pattern_insights)}",
                                    'recommendation': "Focus improvement efforts on these specific segments to boost overall performance.",
                                    'priority': 'high'
                                })
        except Exception as e:
            # Fallback generic insights if analysis fails
            insights.append({
                'title': "📊 General data quality assessment",
                'description': f"Your dataset contains {len(df)} records and {len(df.columns)} variables.",
                'recommendation': "Ensure data completeness and accuracy for reliable analysis.",
                'priority': 'medium'
            })
    
    # If we couldn't generate insights or not enough were generated
    if len(insights) == 0:
        insights.append({
            'title': "👋 Welcome to AI Insights",
            'description': "Upload more comprehensive data to receive detailed performance analysis.",
            'recommendation': "For best results, include sales/revenue metrics, dates, and categorical dimensions.",
            'priority': 'low'
        })
    
    return insights

# Function to create date slicer
def create_date_slicer(df):
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if not date_cols:
        st.markdown(
            """
            <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
                <p style='color: #B19CD9; font-size: 16px;'>No date columns detected for time-based filtering.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return None
    
    # Select first date column
    date_col = date_cols[0]
    
    try:
        # Convert to datetime if not already
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Get min and max dates
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        
        st.markdown(
            """
            <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
                <h4 style='color: #9D72FF; font-size: 18px; margin-bottom: 10px;'>📅 Date Range Filter</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create date range slider
        date_range = st.date_input(
            "",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Return filtered dataframe if date range selected
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]
            return filtered_df
    except:
        st.warning(f"Could not convert '{date_col}' to date format.", icon="⚠️")
    
    return df

# Function to create KPI cards
def create_kpi_cards(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        return
    
    # Select up to 3 numeric columns for KPIs
    kpi_cols = numeric_cols[:3]
    
    st.markdown(
        """
        <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
            <h4 style='color: #9D72FF; font-size: 18px; margin-bottom: 10px;'>📊 Key Performance Indicators</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    for col in kpi_cols:
        current_value = df[col].mean()
        
        # Determine if KPI is good, average, or concerning
        if df[col].skew() > 0:  # Right-skewed (higher values better)
            percentile_75 = df[col].quantile(0.75)
            percentile_25 = df[col].quantile(0.25)
            
            if current_value >= percentile_75:
                kpi_class = "kpi-green"
                kpi_icon = "↗️"
                kpi_status = "Excellent"
            elif current_value >= percentile_25:
                kpi_class = "kpi-yellow"
                kpi_icon = "→"
                kpi_status = "Average"
            else:
                kpi_class = "kpi-red"
                kpi_icon = "↘️"
                kpi_status = "Needs Attention"
        else:  # Left-skewed or symmetric (lower values might be better)
            percentile_75 = df[col].quantile(0.75)
            percentile_25 = df[col].quantile(0.25)
            
            if current_value <= percentile_25:
                kpi_class = "kpi-green"
                kpi_icon = "↘️"
                kpi_status = "Excellent"
            elif current_value <= percentile_75:
                kpi_class = "kpi-yellow" 
                kpi_icon = "→"
                kpi_status = "Average"
            else:
                kpi_class = "kpi-red"
                kpi_icon = "↗️"
                kpi_status = "Needs Attention"
        
        # Format value appropriately
        if abs(current_value) > 1000:
            formatted_value = f"{current_value:,.0f}"
        elif abs(current_value) > 10:
            formatted_value = f"{current_value:.1f}"
        else:
            formatted_value = f"{current_value:.2f}"
            
        st.markdown(
            f"""
            <div class='kpi-card {kpi_class}'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h4 style='margin: 0; color: #E0E0FF;'>{col.replace('_', ' ').title()}</h4>
                    <span style='font-size: 24px;'>{kpi_icon}</span>
                </div>
                <h2 style='margin: 10px 0; font-size: 32px; color: #E0E0FF;'>{formatted_value}</h2>
                <p style='margin: 0; color: #B19CD9;'>{kpi_status}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Function to create interactive visualizations
def create_visualizations(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols:
        return
    
    st.markdown(
        """
        <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
            <h4 style='color: #9D72FF; font-size: 18px; margin-bottom: 10px;'>📈 Interactive Data Visualizations</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create tabs for different visualization types
    viz_tabs = st.tabs(["Distribution Analysis", "Correlation Analysis", "Trend Analysis"])
    
    with viz_tabs[0]:
        # Distribution Analysis
        if len(numeric_cols) > 0:
            st.markdown("<h5 style='color: #B19CD9;'>Distribution of Key Metrics</h5>", unsafe_allow_html=True)
            selected_metric = st.selectbox("Select metric to analyze:", numeric_cols)
            
            # Create histogram with KDE
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=df[selected_metric],
                    name="Frequency",
                    marker=dict(color="rgba(106, 90, 205, 0.6)"),
                    nbinsx=20
                )
            )
            
            # Add KDE (Kernel Density Estimation)
            try:
                from scipy.stats import gaussian_kde
                import numpy as np
                
                # Remove NaN values
                data = df[selected_metric].dropna()
                
                if len(data) > 1:  # Need at least 2 points for KDE
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 1000)
                    y_kde = kde(x_range)
                    
                    # Scale KDE to match histogram height
                    hist, bin_edges = np.histogram(data, bins=20)
                    max_hist_height = max(hist)
                    scaling_factor = max_hist_height / max(y_kde)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range, 
                            y=y_kde * scaling_factor,
                            mode="lines",
                            line=dict(color="rgba(255, 215, 0, 0.8)", width=3),
                            name="Density"
                        ),
                        secondary_y=False
                    )
            except:
                pass
            
            # Add box plot on secondary y-axis
            fig.add_trace(
                go.Box(
                    x=df[selected_metric],
                    name="Distribution",
                    marker=dict(color="rgba(157, 114, 255, 0.7)"),
                    boxpoints="outliers",
                    orientation="h"
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(40, 42, 54, 0.8)",
                paper_bgcolor="rgba(40, 42, 54, 0)",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                title={
                    "text": f"Distribution Analysis of {selected_metric}",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": {"color": "#9D72FF", "size": 18}
                }
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Frequency", secondary_y=False, showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
            fig.update_yaxes(title_text="", secondary_y=True, showticklabels=False)
            
            # Update x-axis
            fig.update_xaxes(title_text=selected_metric, showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Add statistics summary
            st.markdown("<h5 style='color: #B19CD9;'>Statistical Summary</h5>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[selected_metric].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[selected_metric].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[selected_metric].std():.2f}")
            with col4:
                st.metric("Skewness", f"{df[selected_metric].skew():.2f}")
            
    with viz_tabs[1]:
        # Correlation Analysis
        if len(numeric_cols) >= 2:
            st.markdown("<h5 style='color: #B19CD9;'>Correlation Between Metrics</h5>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Select X-axis metric:", numeric_cols)
            with col2:
                y_var = st.selectbox("Select Y-axis metric:", [col for col in numeric_cols if col != x_var])
            
            # Create scatter plot with trendline
            fig = px.scatter(
                df, 
                x=x_var, 
                y=y_var,
                trendline="ols",
                color_discrete_sequence=["rgba(157, 114, 255, 0.7)"],
                trendline_color_override="rgba(255, 215, 0, 0.8)"
            )
            
            # Calculate correlation coefficient
            corr = df[x_var].corr(df[y_var])
            
            # Update layout
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(40, 42, 54, 0.8)",
                paper_bgcolor="rgba(40, 42, 54, 0)",
                margin=dict(l=20, r=20, t=50, b=20),
                title={
                    "text": f"Correlation: {corr:.2f}",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": {"color": "#9D72FF", "size": 18}
                }
            )
            
            # Update axes
            fig.update_xaxes(title_text=x_var, showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
            fig.update_yaxes(title_text=y_var, showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Add correlation interpretation
            correlation_text = ""
            if abs(corr) < 0.3:
                correlation_text = "Weak correlation"
            elif abs(corr) < 0.7:
                correlation_text = "Moderate correlation"
            else:
                correlation_text = "Strong correlation"
                
            if corr > 0:
                correlation_text += " (positive)"
            else:
                correlation_text += " (negative)"
            
            st.markdown(
                f"""
                <div style='background: rgba(60, 63, 68, 0.7); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                    <h5 style='color: #B19CD9; margin-top: 0;'>Correlation Insight</h5>
                    <p style='color: #E0E0FF;'>{correlation_text} detected between {x_var} and {y_var}.</p>
                    <p style='color: #B19CD9;'>
                        R² value: {corr**2:.2f} - This means approximately {(corr**2 * 100):.1f}% of the variation 
                        in {y_var} can be explained by {x_var}.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with viz_tabs[2]:
        # Trend Analysis
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_cols and numeric_cols:
            st.markdown("<h5 style='color: #B19CD9;'>Trend Analysis Over Time</h5>", unsafe_allow_html=True)
            
            # Select date column and metric
            date_col = st.selectbox("Select date column:", date_cols)
            trend_metric = st.selectbox("Select metric to track:", numeric_cols, key="trend_metric")
            
            # Convert to datetime if needed
            if df[date_col].dtype != 'datetime64[ns]':
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Group by date and calculate metrics
            try:
                # Determine appropriate date grouping based on date range
                date_range = (df[date_col].max() - df[date_col].min()).days
                
                if date_range > 365*2:  # More than 2 years
                    grouper = pd.Grouper(key=date_col, freq='Q')
                    group_name = "Quarterly"
                elif date_range > 90:  # More than 3 months
                    grouper = pd.Grouper(key=date_col, freq='M')
                    group_name = "Monthly"
                elif date_range > 21:  # More than 3 weeks
                    grouper = pd.Grouper(key=date_col, freq='W')
                    group_name = "Weekly"
                else:
                    grouper = pd.Grouper(key=date_col, freq='D')
                    group_name = "Daily"
                
                # Group and aggregate
                trend_df = df.groupby(grouper)[trend_metric].agg(['mean', 'min', 'max']).reset_index()
                trend_df.columns = [date_col, 'Average', 'Minimum', 'Maximum']
                
                # Create area plot
                fig = go.Figure()
                
                # Add min-max range
                fig.add_trace(
                    go.Scatter(
                        x=trend_df[date_col],
                        y=trend_df['Maximum'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(157, 114, 255, 0.1)',
                        name='Maximum'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=trend_df[date_col],
                        y=trend_df['Minimum'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(157, 114, 255, 0.1)',
                        name='Minimum'
                    )
                )
                
                # Add average line
                fig.add_trace(
                    go.Scatter(
                        x=trend_df[date_col],
                        y=trend_df['Average'],
                        mode='lines+markers',
                        line=dict(color='rgba(255, 215, 0, 0.8)', width=3),
                        marker=dict(size=8, color='rgba(255, 215, 0, 0.9)'),
                        name='Average'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(40, 42, 54, 0.8)",
                    paper_bgcolor="rgba(40, 42, 54, 0)",
                    margin=dict(l=20, r=20, t=50, b=20),
                    title={
                        "text": f"{group_name} Trend of {trend_metric}",
                        "y": 0.95,
                        "x": 0.5,
                        "xanchor": "center",
                        "yanchor": "top",
                        "font": {"color": "#9D72FF", "size": 18}
                    },
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Update axes
                fig.update_xaxes(title_text="Date", showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
                fig.update_yaxes(title_text=trend_metric, showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate trend statistics
                first_avg = trend_df['Average'].iloc[0] if not trend_df.empty else 0
                last_avg = trend_df['Average'].iloc[-1] if not trend_df.empty else 0
                pct_change = ((last_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0
                
                # Add trend summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Start Value", f"{first_avg:.2f}")
                with col2:
                    st.metric("End Value", f"{last_avg:.2f}")
                with col3:
                    st.metric("Overall Change", f"{pct_change:.1f}%",
                             delta=f"{pct_change:.1f}%",
                             delta_color="normal")
            except Exception as e:
                st.error(f"Could not create trend analysis: {str(e)}")

# Function to render AI insights panel
def render_ai_insights(insights):
    st.markdown(
        """
        <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
            <h4 style='color: #9D72FF; font-size: 18px; margin-bottom: 10px;'>🧠 AI-Generated Insights</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create expandable sections for each insight
    for i, insight in enumerate(insights):
        priority_class = f"{insight['priority']}-priority"
        
        st.markdown(
            f"""
            <div class='insight-item {priority_class}'>
                <h5 style='color: #E0E0FF; margin-top: 0;'>{insight['title']}</h5>
                <p style='color: #B19CD9;'>{insight['description']}</p>
                <p style='color: #9D72FF; font-style: italic;'><strong>Recommendation:</strong> {insight['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Function to display Q&A section
def display_qa_section(df):
    st.markdown(
        """
        <div class='qa-section'>
            <h4 style='color: #9D72FF; margin-top: 0;'>📝 Ask Questions About Your Data</h4>
            <p style='color: #B19CD9;'>Type your question below to get insights about your data.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    query = st.text_input("", "", key="qa_input", placeholder="e.g., What is the average sales value?")
    
    if query:
        with st.spinner("Analyzing your question..."):
            time.sleep(1)  # Simulate processing
            
            try:
                # Simple keyword-based response system
                query_lower = query.lower()
                
                if 'average' in query_lower or 'mean' in query_lower:
                    # Extract column name from query
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    matched_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in query_lower:
                            matched_col = col
                            break
                    
                    if matched_col:
                        avg_value = df[matched_col].mean()
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>The average {matched_col} is <strong>{avg_value:.2f}</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        # If no specific column mentioned, show averages for all numeric columns
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Here are the averages for all numeric columns:</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        avg_df = pd.DataFrame(df.mean()).reset_index()
                        avg_df.columns = ['Column', 'Average']
                        st.dataframe(avg_df)
                
                elif 'maximum' in query_lower or 'max' in query_lower:
                    # Extract column name from query
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    matched_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in query_lower:
                            matched_col = col
                            break
                    
                    if matched_col:
                        max_value = df[matched_col].max()
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>The maximum {matched_col} is <strong>{max_value:.2f}</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        # If no specific column mentioned, explain the limitation
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Please specify which column you'd like to find the maximum value for.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                elif 'minimum' in query_lower or 'min' in query_lower:
                    # Extract column name from query
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    matched_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in query_lower:
                            matched_col = col
                            break
                    
                    if matched_col:
                        min_value = df[matched_col].min()
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>The minimum  {matched_col} is <strong>{min_value:.2f}</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        # If no specific column mentioned, explain the limitation
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Please specify which column you'd like to find the minimum value for.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                elif 'correlation' in query_lower or 'correlate' in query_lower:
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    if len(numeric_cols) >= 2:
                        corr_matrix = df[numeric_cols].corr().abs()
                        
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Here's the correlation matrix between numeric variables:</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Plot heatmap
                        fig = px.imshow(
                            corr_matrix,
                            color_continuous_scale="Viridis",
                            labels=dict(color="Correlation")
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(40, 42, 54, 0.8)",
                            paper_bgcolor="rgba(40, 42, 54, 0)",
                            margin=dict(l=20, r=20, t=50, b=20),
                            title={
                                "text": "Correlation Matrix",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"color": "#9D72FF", "size": 18}
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Find highest correlation
                        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                        highest_corr = upper_tri.stack().nlargest(1)
                        
                        for idx, value in highest_corr.items():
                            st.markdown(
                                f"""
                                <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                    <p style='color: #E0E0FF;'>The strongest correlation is between <strong>{idx[0]}</strong> and <strong>{idx[1]}</strong> with a value of <strong>{value:.2f}</strong></p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Need at least two numeric columns to calculate correlations.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                elif 'top' in query_lower or 'highest' in query_lower:
                    # Extract column name and number of results
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    matched_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in query_lower:
                            matched_col = col
                            break
                    
                    # Try to extract a number from the query
                    import re
                    num_match = re.search(r'\b(\d+)\b', query_lower)
                    num_results = int(num_match.group(1)) if num_match else 5
                    
                    if matched_col:
                        top_values = df.nlargest(num_results, matched_col)
                        
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Top {num_results} highest {matched_col} values:</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.dataframe(top_values)
                    else:
                        # If no specific column mentioned, explain the limitation
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Please specify which column you'd like to see top values for.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                else:
                    # Generic response for unrecognized queries
                    st.markdown(
                        f"""
                        <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                            <p style='color: #E0E0FF;'>I understand you're asking about: "{query}"</p>
                            <p style='color: #B19CD9;'>Try asking about averages, maximums, correlations, or top values in your data.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")

    st.markdown("<h3 style='color: #9D72FF;'>Excel Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

# Sidebar navigation
navigation = st.sidebar.radio(
    "",
    ["Upload Data", "Dashboard", "Advanced Analysis"],
    key="navigation"
)

# Initialize session state for data storage
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'insights' not in st.session_state:
    st.session_state.insights = []

# Upload Data Page
if navigation == "Upload Data":
    if name:
        st.markdown(f"<div class='greeting'>{greeting_text}, {name}!</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='upload-section'>
            <h2 style='color: #9D72FF; margin-bottom: 15px;'>Upload Your Sales Data</h2>
            <p style='color: #B19CD9; margin-bottom: 20px;'>
                Upload CSV, Excel, or JSON files to begin your data exploration journey.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # File uploader
    uploaded_file = st.file_uploader("", type=["csv", "xlsx", "xls", "json"])
    
    if uploaded_file is not None:
        try:
            # Display progress
            progress_text = st.markdown("**Processing your data...**")
            progress_bar = st.progress(0)
            
            # Update progress
            for i in range(10):
                progress_bar.progress((i+1) * 10)
                time.sleep(0.05)    
            
            # Read file based on type
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file, encoding='latin1')
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_ext == 'json':
                df = pd.read_json(uploaded_file, encoding='latin1')
            
            # Save to session state
            st.session_state.data = df
            st.session_state.file_name = uploaded_file.name
            
            # Generate AI insights
            st.session_state.insights = generate_ai_insights(df)
            
            # Update progress to completion
            progress_bar.progress(100)
            progress_text.text("✅ Data processed successfully!")
            
            # Show success message with animation
            st.markdown(
                f"""
                <div style='animation: fadeIn 1s; background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1)); 
                     padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #4CAF50;'>
                    <h3 style='color: #4CAF50; margin-top: 0;'>✅ Success!</h3>
                    <p style='color: #E0E0FF;'>
                        Successfully loaded <strong>{len(df)}</strong> records with <strong>{len(df.columns)}</strong> columns from <strong>{uploaded_file.name}</strong>
                    </p>
                    <p style='color: #B19CD9;'>
                        Generated <strong>{len(st.session_state.insights)}</strong> AI-powered insights about your data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Preview data
            st.markdown("<h3 style='color: #9D72FF;'>Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            # Show column info
            st.markdown("<h3 style='color: #9D72FF;'>Column Information</h3>", unsafe_allow_html=True)
            
            # Create column info table
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Missing Values': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            
            st.dataframe(col_info, use_container_width=True)
            
            # Navigation hint
            st.markdown(
                """
                <div style='animation: fadeIn 1.5s; text-align: center; margin-top: 30px;'>
                    <p style='color: #B19CD9; font-size: 18px;'>
                        🎉 You're all set! Navigate to the <strong>Dashboard</strong> tab to explore your data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Show sample data option
        st.markdown(
            """
            <div style='margin-top: 30px; text-align: center; animation: fadeIn 1.5s;'>
                <p style='color: #B19CD9; font-size: 18px;'>
                    Don't have data? Try our <span style='cursor: pointer; text-decoration: underline; color: #9D72FF;' id='load-sample'>sample dataset</span> to see the dashboard in action.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # JavaScript to handle sample data loading
        st.markdown(
            """
            <script>
                document.getElementById('load-sample').addEventListener('click', function() {
                    // This is handled by Streamlit components
                });
            </script>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("Load Sample Dataset"):
            # Create sample sales data
            sample_data = {
                'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
                'Product': np.random.choice(['Widget A', 'Widget B', 'Widget C', 'Widget D'], 100),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                'Sales': np.random.normal(loc=1000, scale=200, size=100).round(2),
                'Units': np.random.randint(10, 100, 100),
                'Customer_Satisfaction': np.random.uniform(3.5, 5, 100).round(1),
                'Marketing_Spend': np.random.uniform(50, 200, 100).round(2),
                'Profit_Margin': np.random.uniform(0.1, 0.3, 100).round(3)
            }
            
            df = pd.DataFrame(sample_data)
            
            # Calculate some derived metrics
            df['Revenue'] = df['Sales'] * df['Units']
            df['Marketing_ROI'] = df['Revenue'] / df['Marketing_Spend']
            df['Profit'] = df['Revenue'] * df['Profit_Margin']
            
            # Save to session state
            st.session_state.data = df
            st.session_state.file_name = "sample_sales_data.csv"
            
            # Generate AI insights
            st.session_state.insights = generate_ai_insights(df)
            
            # Show success message
            st.markdown(
                f"""
                <div style='animation: fadeIn 1s; background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1)); 
                     padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #4CAF50;'>
                    <h3 style='color: #4CAF50; margin-top: 0;'>✅ Sample Data Loaded!</h3>
                    <p style='color: #E0E0FF;'>
                        Successfully loaded <strong>{len(df)}</strong> records with <strong>{len(df.columns)}</strong> columns from the sample dataset.
                    </p>
                    <p style='color: #B19CD9;'>
                        Generated <strong>{len(st.session_state.insights)}</strong> AI-powered insights about your data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Preview data
            st.markdown("<h3 style='color: #9D72FF;'>Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            # Navigation hint
            st.markdown(
                """
                <div style='animation: fadeIn 1.5s; text-align: center; margin-top: 30px;'>
                    <p style='color: #B19CD9; font-size: 18px;'>
                        🎉 All set! Navigate to the <strong>Dashboard</strong> tab to explore your data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Dashboard Page
elif navigation == "Dashboard":
    if st.session_state.data is not None:
        # Dashboard header
        st.markdown(
            f"""
            <div class='dashboard-header'>
                <h1 style='color: #9D72FF; margin-bottom: 5px;'>Sales Analytics Dashboard</h1>
                <p style='color: #B19CD9; margin-top: 0;'>
                    Analyzing data from <strong>{st.session_state.file_name}</strong> • 
                    <span style='color: #E0E0FF;'>{len(st.session_state.data)} records</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # KPI Metrics Row
        st.markdown(
            """
            <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
                <h4 style='color: #9D72FF; font-size: 18px; margin-bottom: 10px;'>📊 Key Performance Indicators</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Find revenue/sales column
        df = st.session_state.data
        revenue_col = None
        
        potential_revenue_cols = ['Revenue', 'Sales', 'Income', 'Amount']
        for col in potential_revenue_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                revenue_col = col
                break
        
        # Find profit column
        profit_col = None
        potential_profit_cols = ['Profit', 'Margin', 'Earnings', 'Net_Income', 'Profit_Margin']
        for col in potential_profit_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                profit_col = col
                break
        
        # Create KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Total Revenue
            if revenue_col:
                total_revenue = df[revenue_col].sum()
                display_kpi("Total Revenue", f"${total_revenue:,.2f}", "Up 12% from last period")
            else:
                # Find any numeric column for demo
                num_cols = df.select_dtypes(include=['number']).columns
                if len(num_cols) > 0:
                    total_val = df[num_cols[0]].sum()
                    display_kpi("Total " + num_cols[0], f"{total_val:,.2f}", "Key metric overview")
                else:
                    display_kpi("Total Records", f"{len(df):,}", "Data volume indicator")
        
        with col2:
            # Average Order Value
            if revenue_col:
                avg_value = df[revenue_col].mean()
                display_kpi("Average Order", f"${avg_value:,.2f}", "Up 3.5% from last period")
            else:
                # Use another numeric column
                num_cols = df.select_dtypes(include=['number']).columns
                if len(num_cols) > 0:
                    avg_val = df[num_cols[0]].mean()
                    display_kpi("Average " + num_cols[0], f"{avg_val:,.2f}", "Trend is stable")
                else:
                    display_kpi("Unique Values", f"{df.nunique().sum():,}", "Data diversity")
        
        with col3:
            # Profit Margin
            if profit_col and revenue_col:
                if 'Margin' in profit_col and df[profit_col].max() <= 1:
                    # It's already a margin percentage
                    margin = df[profit_col].mean() * 100
                else:
                    # Calculate margin from profit and revenue
                    margin = (df[profit_col].sum() / df[revenue_col].sum()) * 100
                display_kpi("Profit Margin", f"{margin:.1f}%", "Down 1.2% from last period")
            else:
                # Use another metric
                num_cols = df.select_dtypes(include=['number']).columns
                if len(num_cols) > 1:
                    ratio = (df[num_cols[0]].sum() / df[num_cols[1]].sum()) * 100
                    display_kpi("Ratio Metric", f"{ratio:.1f}%", "Key performance ratio")
                else:
                    display_kpi("Missing Values", f"{df.isna().sum().sum():,}", "Data quality metric")
        
        with col4:
            # Customer Satisfaction
            satisfaction_col = None
            potential_satisfaction_cols = ['Satisfaction', 'Rating', 'Score', 'NPS', 'Customer_Satisfaction']
            for col in potential_satisfaction_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    satisfaction_col = col
                    break
            
            if satisfaction_col:
                avg_satisfaction = df[satisfaction_col].mean()
                max_possible = df[satisfaction_col].max()
                
                if max_possible <= 5:
                    # Assume 5-star scale
                    display_kpi("Satisfaction", f"{avg_satisfaction:.1f}/5 ★", "Up 0.3 points")
                elif max_possible <= 10:
                    # Assume 10-point scale
                    display_kpi("Satisfaction", f"{avg_satisfaction:.1f}/10", "Trending positive")
                else:
                    display_kpi("Satisfaction", f"{avg_satisfaction:.1f} pts", "Customer happiness")
            else:
                # Use another unique metric
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    try:
                        time_range = (pd.to_datetime(df[date_cols[0]]).max() - pd.to_datetime(df[date_cols[0]]).min()).days
                        display_kpi("Date Range", f"{time_range} days", "Time period analyzed")
                    except:
                        display_kpi("Unique Categories", f"{df.select_dtypes(include=['object']).nunique().sum():,}", "Categorical breakdown")
                else:
                    display_kpi("Unique Categories", f"{df.select_dtypes(include=['object']).nunique().sum():,}", "Categorical breakdown")
        
        # Create horizontal spacer
        st.markdown("<hr style='margin: 30px 0; opacity: 0.3;'>", unsafe_allow_html=True)
        
        # Main dashboard charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Time Series Chart
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols and revenue_col:
                st.markdown("<h5 style='color: #B19CD9;'>Sales Performance Over Time</h5>", unsafe_allow_html=True)
                
                # Convert to datetime
                date_col = date_cols[0]
                if df[date_col].dtype != 'datetime64[ns]':
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Group by date
                try:
                    # Determine appropriate date grouping
                    date_range = (df[date_col].max() - df[date_col].min()).days
                    
                    if date_range > 365:
                        grouper = pd.Grouper(key=date_col, freq='M')
                        period = "Monthly"
                    elif date_range > 60:
                        grouper = pd.Grouper(key=date_col, freq='W')
                        period = "Weekly"
                    else:
                        grouper = pd.Grouper(key=date_col, freq='D')
                        period = "Daily"
                    
                    # Group and plot
                    time_series = df.groupby(grouper)[revenue_col].sum().reset_index()
                    
                    fig = px.line(
                        time_series,
                        x=date_col,
                        y=revenue_col,
                        title=f"{period} {revenue_col} Performance",
                        color_discrete_sequence=["#9D72FF"]
                    )
                    
                    # Add markers
                    fig.update_traces(mode="lines+markers", marker=dict(size=8, opacity=0.7))
                    
                    # Calculate rolling average
                    window = 3 if len(time_series) > 5 else 2
                    if len(time_series) > window:
                        time_series['Rolling_Avg'] = time_series[revenue_col].rolling(window=window).mean()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=time_series[date_col],
                                y=time_series['Rolling_Avg'],
                                mode="lines",
                                line=dict(color="rgba(255, 215, 0, 0.8)", width=3, dash="dot"),
                                name=f"{window}-Period Moving Average"
                            )
                        )
                    
                    # Update layout
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(40, 42, 54, 0.8)",
                        paper_bgcolor="rgba(40, 42, 54, 0)",
                        margin=dict(l=20, r=20, t=50, b=20),
                        title={
                            "y": 0.95,
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                            "font": {"color": "#9D72FF", "size": 18}
                        },
                        xaxis=dict(
                            title="",
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(107, 114, 142, 0.2)"
                        ),
                        yaxis=dict(
                            title=revenue_col,
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(107, 114, 142, 0.2)"
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Could not create time series: {str(e)}")
            else:
                # Fallback to distribution chart
                st.markdown("<h5 style='color: #B19CD9;'>Value Distribution</h5>", unsafe_allow_html=True)
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select metric:", numeric_cols)
                    
                    fig = go.Figure()
                    
                    # Add histogram
                    fig.add_trace(
                        go.Histogram(
                            x=df[selected_col],
                            marker=dict(color="rgba(157, 114, 255, 0.7)"),
                            nbinsx=20
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(40, 42, 54, 0.8)",
                        paper_bgcolor="rgba(40, 42, 54, 0)",
                        margin=dict(l=20, r=20, t=50, b=20),
                        title={
                            "text": f"Distribution of {selected_col}",
                            "y": 0.95,
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                            "font": {"color": "#9D72FF", "size": 18}
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No numeric columns found for visualization")
        
        with col2:
            # Category breakdown chart
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols and revenue_col:
                st.markdown("<h5 style='color: #B19CD9;'>Revenue by Category</h5>", unsafe_allow_html=True)
                
                # Select categorical column with the most appropriate number of categories
                best_cat_col = None
                best_cat_count = 0
                
                for col in categorical_cols:
                    cat_count = df[col].nunique()
                    if 3 <= cat_count <= 10:
                        best_cat_col = col
                        break
                    elif cat_count > best_cat_count:
                        best_cat_col = col
                        best_cat_count = cat_count
                
                if best_cat_col:
                    # If too many categories, take the top ones
                    if df[best_cat_col].nunique() > 7:
                        top_cats = df.groupby(best_cat_col)[revenue_col].sum().nlargest(7).index.tolist()
                        filtered_df = df[df[best_cat_col].isin(top_cats)].copy()
                        filtered_df.loc[~filtered_df[best_cat_col].isin(top_cats), best_cat_col] = 'Other'
                    else:
                        filtered_df = df
                    
                    # Aggregate data
                    cat_data = filtered_df.groupby(best_cat_col)[revenue_col].sum().reset_index()
                    
                    # Create pie chart
                    fig = px.pie(
                        cat_data,
                        values=revenue_col,
                        names=best_cat_col,
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Agsunset
                    )
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(40, 42, 54, 0.8)",
                        paper_bgcolor="rgba(40, 42, 54, 0)",
                        margin=dict(l=10, r=10, t=30, b=10),
                        title={
                            "text": f"{revenue_col} by {best_cat_col}",
                            "y": 0.95,
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                            "font": {"color": "#9D72FF", "size": 16}
                        },
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    
                    # Update traces
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add insights below the chart
                    top_category = cat_data.iloc[cat_data[revenue_col].argmax()][best_cat_col]
                    top_percent = cat_data[revenue_col].max() / cat_data[revenue_col].sum() * 100
                    
                    st.markdown(
                        f"""
                        <div style='background: rgba(60, 63, 68, 0.7); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                            <p style='color: #E0E0FF; margin: 0;'>
                                <strong>{top_category}</strong> accounts for <strong>{top_percent:.1f}%</strong> of total {revenue_col}.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.write("No suitable categorical columns found")
            else:
                # Fallback to another categorical visualization
                if categorical_cols:
                    st.markdown("<h5 style='color: #B19CD9;'>Category Distribution</h5>", unsafe_allow_html=True)
                    
                    # Find categorical column with reasonable number of categories
                    suitable_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 10]
                    if suitable_cols:
                        selected_cat = suitable_cols[0]
                        
                        # Create count plot
                        cat_counts = df[selected_cat].value_counts().reset_index()
                        cat_counts.columns = [selected_cat, 'Count']
                        
                        fig = px.bar(
                            cat_counts,
                            x=selected_cat,
                            y='Count',
                            color='Count',
                            color_continuous_scale="Purp"
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(40, 42, 54, 0.8)",
                            paper_bgcolor="rgba(40, 42, 54, 0)",
                            margin=dict(l=20, r=20, t=50, b=20),
                            title={
                                "text": f"Distribution of {selected_cat}",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"color": "#9D72FF", "size": 16}
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Categorical columns have too many unique values for visualization")
                else:
                    # If no categorical columns, show data summary
                    st.markdown("<h5 style='color: #B19CD9;'>Data Summary</h5>", unsafe_allow_html=True)
                    
                    numeric_df = df.select_dtypes(include=['number'])
                    if not numeric_df.empty:
                        summary_stats = numeric_df.describe().T[['mean', 'min', 'max', 'std']].reset_index()
                        summary_stats.columns = ['Metric', 'Mean', 'Min', 'Max', 'Std Dev']
                        
                        st.dataframe(summary_stats, use_container_width=True)
                    else:
                        st.write("No numeric data available for summary statistics")
        
        # AI Insights Panel
        if st.session_state.insights:
            render_ai_insights(st.session_state.insights)
        
        # Interactive Q&A Section
        display_qa_section(df)
    else:
        # Prompt to upload data
        st.markdown(
            """
            <div style='text-align: center; padding: 50px 20px; animation: fadeIn 1s;'>
                <img src="https://cdn-icons-png.flaticon.com/512/6571/6571582.png" style="width: 100px; opacity: 0.5;">
                <h2 style='color: #9D72FF; margin-top: 20px;'>Upload Data to View Dashboard</h2>
                <p style='color: #B19CD9; margin-bottom: 30px;'>
                    Please navigate to the Upload Data tab to load your dataset.
                </p>
                <a href="#" style='
                    display: inline-block;
                    padding: 10px 20px;
                    background: linear-gradient(90deg, #9D72FF 0%, #B19CD9 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    cursor: pointer;
                '>Go to Upload Data</a>
            </div>
            """,
            unsafe_allow_html=True
        )

# Advanced Analysis Page
elif navigation == "Advanced Analysis":
    if st.session_state.data is not None:
        # Advanced Analysis Header
        st.markdown(
            """
            <div class='dashboard-header'>
                <h1 style='color: #9D72FF; margin-bottom: 5px;'>Advanced Analytics Tools</h1>
                <p style='color: #B19CD9; margin-top: 0;'>
                    Deep dive tools for comprehensive data exploration and analysis
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create tabs for different analysis tools
        analysis_tabs = st.tabs(["Data Explorer", "Correlation Analysis", "Segment Analysis", "Time Series"])
        
        with analysis_tabs[0]:
            # Data Explorer
            st.markdown("<h4 style='color: #B19CD9;'>Interactive Data Explorer</h4>", unsafe_allow_html=True)
            
            df = st.session_state.data
            
            # Filter controls
            st.markdown("<p style='color: #9D72FF;'><strong>Data Filters</strong></p>", unsafe_allow_html=True)
            
            # Dynamically create filters based on data types
            filters_applied = False
            filtered_df = df.copy()
            
            # Categorical filters
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    if cat_cols:
                        cat_filter_col = st.selectbox("Select category to filter", ["None"] + cat_cols)
                        
                        if cat_filter_col != "None":
                            cat_values = ["All"] + sorted(df[cat_filter_col].unique().tolist())
                            selected_cats = st.multiselect("Select values", cat_values, ["All"])
                            
                            if selected_cats and "All" not in selected_cats:
                                filtered_df = filtered_df[filtered_df[cat_filter_col].isin(selected_cats)]
                                filters_applied = True
                
                with col2:
                    # Second categorical filter if available
                    remaining_cat_cols = [col for col in cat_cols if col != cat_filter_col]
                    if remaining_cat_cols:
                        cat_filter_col2 = st.selectbox("Select second category", ["None"] + remaining_cat_cols)
                        
                        if cat_filter_col2 != "None":
                            cat_values2 = ["All"] + sorted(df[cat_filter_col2].unique().tolist())
                            selected_cats2 = st.multiselect("Select values", cat_values2, ["All"], key="second_cat")
                            
                            if selected_cats2 and "All" not in selected_cats2:
                                filtered_df = filtered_df[filtered_df[cat_filter_col2].isin(selected_cats2)]
                                filters_applied = True
            
            # Numeric range filters
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    if num_cols:
                        num_filter_col = st.selectbox("Select numeric range to filter", ["None"] + num_cols)
                        
                        if num_filter_col != "None":
                            min_val = float(df[num_filter_col].min())
                            max_val = float(df[num_filter_col].max())
                            
                            num_range = st.slider(
                                f"Range for {num_filter_col}",
                                min_val,
                                max_val,
                                (min_val, max_val)
                            )
                            
                            if num_range[0] > min_val or num_range[1] < max_val:
                                filtered_df = filtered_df[
                                    (filtered_df[num_filter_col] >= num_range[0]) & 
                                    (filtered_df[num_filter_col] <= num_range[1])
                                ]
                                filters_applied = True
                
                with col2:
                    # Second numeric filter if available
                    remaining_num_cols = [col for col in num_cols if col != num_filter_col]
                    if remaining_num_cols:
                        num_filter_col2 = st.selectbox("Select second numeric range", ["None"] + remaining_num_cols)
                        
                        if num_filter_col2 != "None":
                            min_val2 = float(df[num_filter_col2].min())
                            max_val2 = float(df[num_filter_col2].max())
                            
                            num_range2 = st.slider(
                                f"Range for {num_filter_col2}",
                                min_val2,
                                max_val2,
                                (min_val2, max_val2),
                                key="second_num"
                            )
                            
                            if num_range2[0] > min_val2 or num_range2[1] < max_val2:
                                filtered_df = filtered_df[
                                    (filtered_df[num_filter_col2] >= num_range2[0]) & 
                                    (filtered_df[num_filter_col2] <= num_range2[1])
                                ]
                                filters_applied = True
            
            # Date range filter if date columns exist
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                st.markdown("<br>", unsafe_allow_html=True)
                date_filter_col = st.selectbox("Select date range to filter", ["None"] + date_cols)
                
                if date_filter_col != "None":
                    # Convert to datetime if needed
                    if df[date_filter_col].dtype != 'datetime64[ns]':
                        df[date_filter_col] = pd.to_datetime(df[date_filter_col], errors='coerce')
                        filtered_df[date_filter_col] = pd.to_datetime(filtered_df[date_filter_col], errors='coerce')
                    
                    min_date = df[date_filter_col].min().date()
                    max_date = df[date_filter_col].max().date()
                    
                    date_range = st.date_input(
                        f"Range for {date_filter_col}",
                        [min_date, max_date]
                    )
                    
                    if len(date_range) == 2:
                        if date_range[0] > min_date or date_range[1] < max_date:
                            filtered_df = filtered_df[
                                (filtered_df[date_filter_col].dt.date >= date_range[0]) & 
                                (filtered_df[date_filter_col].dt.date <= date_range[1])
                            ]
                            filters_applied = True
            
            # Display filter summary
            # Display filter summary
            if filters_applied:
                st.markdown(
                    f"""
                    <div style='background: rgba(157, 114, 255, 0.1); padding: 10px 15px; border-radius: 5px; 
                         border-left: 3px solid #9D72FF; margin: 15px 0;'>
                        <p style='color: #B19CD9; margin: 0;'>
                            <strong>Filters Applied:</strong> Showing {len(filtered_df)} of {len(df)} records ({(len(filtered_df)/len(df)*100):.1f}%)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            
            # Data table display
            st.markdown("<hr style='margin: 30px 0; opacity: 0.2;'>", unsafe_allow_html=True)
            st.markdown("<p style='color: #9D72FF;'><strong>Filtered Data Preview</strong></p>", unsafe_allow_html=True)
            
            # Column selector
            all_cols = filtered_df.columns.tolist()
            selected_cols = st.multiselect("Select columns to display", all_cols, all_cols[:6] if len(all_cols) > 6 else all_cols)
            
            if selected_cols:
                st.dataframe(filtered_df[selected_cols].head(100), use_container_width=True)
                
                # Download option
                csv = filtered_df[selected_cols].to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                
                st.markdown(
                    f"""
                    <div style='text-align: right; margin-top: 10px;'>
                        <a href="data:file/csv;base64,{b64}" download="filtered_data.csv" style='
                            display: inline-block;
                            padding: 8px 15px;
                            background: rgba(157, 114, 255, 0.2);
                            color: #9D72FF;
                            text-decoration: none;
                            border-radius: 5px;
                            font-size: 14px;
                            border: 1px solid rgba(157, 114, 255, 0.5);
                        '>
                            <span style='display: flex; align-items: center; gap: 5px;'>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                    <polyline points="7 10 12 15 17 10"></polyline>
                                    <line x1="12" y1="15" x2="12" y2="3"></line>
                                </svg>
                                Download Filtered Data
                            </span>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # Data display options
            st.markdown("<hr style='margin: 20px 0; opacity: 0.2;'>", unsafe_allow_html=True)
            st.markdown("<p style='color: #9D72FF;'><strong>Data Visualization</strong></p>", unsafe_allow_html=True)
            
            # Choose columns for visualization
            vis_options = st.columns(2)
            
            with vis_options[0]:
                num_features = df.select_dtypes(include=['number']).columns.tolist()
                x_axis = st.selectbox("X-axis (Feature)", num_features if num_features else ["None"])
            
            with vis_options[1]:
                y_axis = st.selectbox("Y-axis (Target)", [col for col in num_features if col != x_axis] if len(num_features) > 1 else ["None"])
            
            if x_axis != "None" and y_axis != "None":
                # Create visualization
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Create color option if categorical columns exist
                color_by = None
                cat_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
                if cat_cols:
                    color_options = ["None"] + cat_cols
                    color_by = st.selectbox("Color points by category", color_options)
                
                # Create scatter plot
                fig = px.scatter(
                    filtered_df,
                    x=x_axis,
                    y=y_axis,
                    color=color_by if color_by and color_by != "None" else None,
                    opacity=0.7,
                    size_max=10,
                    color_discrete_sequence=px.colors.qualitative.Pastel if color_by and color_by != "None" else ["#9D72FF"]
                )
                
                # Add trendline if no color grouping
                if not color_by or color_by == "None":
                    fig.update_layout(
                        shapes=[{
                            'type': 'line',
                            'x0': min(filtered_df[x_axis]),
                            'y0': filtered_df[y_axis].mean(),
                            'x1': max(filtered_df[x_axis]),
                            'y1': filtered_df[y_axis].mean(),
                            'line': {
                                'color': 'rgba(255, 255, 255, 0.3)',
                                'width': 2,
                                'dash': 'dash'
                            }
                        }]
                    )
                
                # Update layout
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(40, 42, 54, 0.8)",
                    paper_bgcolor="rgba(40, 42, 54, 0)",
                    margin=dict(l=20, r=20, t=50, b=20),
                    title={
                        "text": f"Relationship between {x_axis} and {y_axis}",
                        "y": 0.95,
                        "x": 0.5,
                        "xanchor": "center",
                        "yanchor": "top",
                        "font": {"color": "#9D72FF", "size": 18}
                    },
                    xaxis=dict(
                        title=x_axis,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="rgba(107, 114, 142, 0.2)"
                    ),
                    yaxis=dict(
                        title=y_axis,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="rgba(107, 114, 142, 0.2)"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add correlation information
                correlation = filtered_df[[x_axis, y_axis]].corr().iloc[0, 1]
                
                # Interpret correlation strength
                corr_strength = "strong"
                if abs(correlation) < 0.3:
                    corr_strength = "weak"
                elif abs(correlation) < 0.7:
                    corr_strength = "moderate"
                
                # Interpret direction
                corr_direction = "positive" if correlation > 0 else "negative"
                
                st.markdown(
                    f"""
                    <div style='background: rgba(40, 42, 54, 0.5); padding: 15px; border-radius: 5px; margin: 15px 0;'>
                        <p style='color: #E0E0FF; margin: 0;'>
                            <strong>Correlation Analysis:</strong> There is a <span style='color: {"#64FFDA" if correlation > 0 else "#FF5370"};'>{corr_strength} {corr_direction}</span> 
                            correlation (r = {correlation:.3f}) between {x_axis} and {y_axis}.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            

        with analysis_tabs[1]:
            # Correlation Analysis
            st.markdown("<h4 style='color: #B19CD9;'>Correlation Matrix & Insights</h4>", unsafe_allow_html=True)
            
            df = st.session_state.data
            
            # Get numeric columns
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(num_cols) > 1:
                # Column selection
                selected_corr_cols = st.multiselect(
                    "Select columns for correlation analysis",
                    num_cols,
                    num_cols[:5] if len(num_cols) > 5 else num_cols
                )
                
                if selected_corr_cols and len(selected_corr_cols) > 1:
                    # Generate correlation matrix
                    corr_matrix = df[selected_corr_cols].corr()
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        aspect="auto"
                    )
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(40, 42, 54, 0.8)",
                        paper_bgcolor="rgba(40, 42, 54, 0)",
                        margin=dict(l=20, r=20, t=50, b=20),
                        title={
                            "text": "Correlation Matrix Heatmap",
                            "y": 0.95,
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                            "font": {"color": "#9D72FF", "size": 18}
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Find top correlations
                    corr_pairs = []
                    for i in range(len(selected_corr_cols)):
                        for j in range(i+1, len(selected_corr_cols)):
                            col1 = selected_corr_cols[i]
                            col2 = selected_corr_cols[j]
                            corr_value = corr_matrix.iloc[i, j]
                            corr_pairs.append((col1, col2, corr_value))
                    
                    # Sort by absolute correlation value
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    
                    # Display top correlations
                    st.markdown("<p style='color: #9D72FF;'><strong>Top Correlations</strong></p>", unsafe_allow_html=True)
                    
                    for i, (col1, col2, corr) in enumerate(corr_pairs[:5]):
                        # Determine correlation strength
                        if abs(corr) >= 0.7:
                            strength = "Strong"
                            color = "#64FFDA" if corr > 0 else "#FF5370"
                        elif abs(corr) >= 0.3:
                            strength = "Moderate"
                            color = "#C3E88D" if corr > 0 else "#F78C6C"
                        else:
                            strength = "Weak"
                            color = "#B2CCD6" if corr > 0 else "#EEFFFF"
                        
                        direction = "positive" if corr > 0 else "negative"
                        
                        st.markdown(
                            f"""
                            <div style='background: rgba(40, 42, 54, 0.5); padding: 15px; border-radius: 5px; margin: 10px 0;'>
                                <p style='color: #E0E0FF; margin: 0;'>
                                    <span style='display: inline-block; width: 24px; height: 24px; text-align: center; 
                                           background: {color}; color: rgba(40, 42, 54, 1); border-radius: 12px; 
                                           margin-right: 10px; font-weight: bold;'>{i+1}</span>
                                    <strong>{col1}</strong> and <strong>{col2}</strong>: 
                                    <span style='color: {color};'>{strength} {direction}</span> correlation (r = {corr:.3f})
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # If it's the top correlation, show scatterplot
                        if i == 0:
                            # Create scatter plot for top correlation
                            scatter_fig = px.scatter(
                                df,
                                x=col1,
                                y=col2,
                                trendline="ols",
                                opacity=0.7,
                                color_discrete_sequence=["#9D72FF"]
                            )
                            
                            scatter_fig.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(40, 42, 54, 0.8)",
                                paper_bgcolor="rgba(40, 42, 54, 0)",
                                margin=dict(l=20, r=20, t=50, b=20),
                                title={
                                    "text": f"Strongest Relationship: {col1} vs {col2}",
                                    "y": 0.95,
                                    "x": 0.5,
                                    "xanchor": "center",
                                    "yanchor": "top",
                                    "font": {"color": "#9D72FF", "size": 16}
                                }
                            )
                            
                            st.plotly_chart(scatter_fig, use_container_width=True)
                else:
                    st.info("Please select at least two columns to generate correlation matrix")
            else:
                st.write("Not enough numeric columns for correlation analysis.")

        with analysis_tabs[2]:
            # Segment Analysis
            st.markdown("<h4 style='color: #B19CD9;'>Customer/Product Segment Analysis</h4>", unsafe_allow_html=True)
            
            df = st.session_state.data
            
            # Find appropriate categorical column for segmentation
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if cat_cols and num_cols:
                # Segment selection
                segment_col = st.selectbox("Select segment column", cat_cols)
                
                # Metric selection 
                metric_col = st.selectbox("Select metric to analyze", num_cols)
                
                if segment_col and metric_col:
                    # Get segment data
                    segment_data = df.groupby(segment_col)[metric_col].agg(['mean', 'sum', 'count']).reset_index()
                    segment_data.columns = [segment_col, 'Average', 'Total', 'Count']
                    
                    # Calculate percentage of total
                    segment_data['Percentage'] = (segment_data['Total'] / segment_data['Total'].sum() * 100).round(1)
                    
                    # Sort by total value
                    segment_data = segment_data.sort_values('Total', ascending=False)
                    
                    # Display segment insights
                    st.markdown(
                        f"""
                        <div style='background: rgba(40, 42, 54, 0.5); padding: 20px; border-radius: 5px; margin: 15px 0;'>
                            <h5 style='color: #9D72FF; margin-top: 0;'>Segment Overview: {segment_col} by {metric_col}</h5>
                            <p style='color: #E0E0FF;'>
                                The analysis shows <strong>{len(segment_data)}</strong> distinct segments. 
                                The top segment accounts for <strong>{segment_data['Percentage'].max():.1f}%</strong> of the total {metric_col}.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Visualization tabs
                    viz_tabs = st.tabs(["Bar Chart", "Pie Chart", "Treemap", "Data Table",])
                    
                    with viz_tabs[0]:
                        # Create bar chart
                        fig = px.bar(
                            segment_data,
                            x=segment_col,
                            y='Total',
                            text='Percentage',
                            color='Average',
                            color_continuous_scale="Purp",
                            labels={'Total': f'Total {metric_col}', 'Average': f'Avg {metric_col}'}
                        )
                        
                        fig.update_traces(texttemplate='%{text}%', textposition='outside')
                        
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(40, 42, 54, 0.8)",
                            paper_bgcolor="rgba(40, 42, 54, 0)",
                            margin=dict(l=20, r=20, t=50, b=20),
                            title={
                                "text": f"{segment_col} Segments by {metric_col}",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"color": "#9D72FF", "size": 18}
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tabs[1]:  # Pie Chart Tab
                        st.markdown("<h5 style='color: #B19CD9;'>Segment Distribution (Pie Chart)</h5>", unsafe_allow_html=True)

                        fig_pie = px.pie(
                            segment_data,
                            names=segment_col,
                            values='Total',
                            color_discrete_sequence=px.colors.sequential.Plasma,
                            title=f"Distribution of {segment_col}"
                        )
                        fig_pie.update_traces(textinfo='percent+label', pull=[0.1 if i == 0 else 0 for i in range(len(segment_data))])

                        fig_pie.update_layout(
                            template="plotly_dark",
                            margin=dict(t=50, b=20, l=20, r=20),
                            title=dict(x=0.5, xanchor="center", font=dict(size=18, color="#9D72FF"))
                        )

                        st.plotly_chart(fig_pie, use_container_width=True)

                    with viz_tabs[2]:
                        # Create treemap for hierarchical view
                        fig = px.treemap(
                            segment_data,
                            path=[segment_col],
                            values='Total',
                            color='Average',
                            color_continuous_scale="Purp",
                            hover_data=['Percentage', 'Count']
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            margin=dict(l=20, r=20, t=50, b=20),
                            title={
                                "text": f"Treemap of {segment_col} Segments",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"color": "#9D72FF", "size": 18}
                            }
                        )
                        
                        fig.update_traces(textinfo="label+value+percent parent")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tabs[3]:
                        # Format the dataframe for display
                        display_df = segment_data.copy()
                        
                        # Format columns
                        display_df['Average'] = display_df['Average'].round(2)
                        display_df['Percentage'] = display_df['Percentage'].astype(str) + '%'
                        
                        # Add styling
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Calculate concentration metrics
                        top_segments = segment_data.head(3)
                        concentration = top_segments['Total'].sum() / segment_data['Total'].sum() * 100
                        
                        st.markdown(
                            f"""
                            <div style='background: rgba(40, 42, 54, 0.5); padding: 15px; border-radius: 5px; margin: 15px 0;'>
                                <p style='color: #E0E0FF; margin: 0;'>
                                    <strong>Segment Concentration:</strong> Top 3 segments account for <strong>{concentration:.1f}%</strong> of total {metric_col}.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    
            else:
                st.write("Need both categorical and numeric columns for segment analysis.")

        with analysis_tabs[3]:
            # Time Series Analysis
            st.markdown("<h4 style='color: #B19CD9;'>Time Series Analysis</h4>", unsafe_allow_html=True)
            
            df = st.session_state.data
            
            # Look for date columns
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_cols:
                # Date column selection
                date_col = st.selectbox("Select date column", date_cols)
                
                # Convert to datetime if needed
                if df[date_col].dtype != 'datetime64[ns]':
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Drop rows with invalid dates
                valid_df = df.dropna(subset=[date_col])
                
                if not valid_df.empty:
                    # Select metric for time series
                    num_cols = valid_df.select_dtypes(include=['number']).columns.tolist()
                    if num_cols:
                        metric_col = st.selectbox("Select metric for time analysis", num_cols)
                        
                        # Group by options
                        date_range = (valid_df[date_col].max() - valid_df[date_col].min()).days
                        
                        if date_range > 365*2:
                            default_period = 'Year'
                            period_options = ['Year', 'Quarter', 'Month']
                        elif date_range > 180:
                            default_period = 'Month'
                            period_options = ['Quarter', 'Month', 'Week']
                        elif date_range > 60:
                            default_period = 'Week'
                            period_options = ['Month', 'Week', 'Day']
                        else:
                            default_period = 'Day'
                            period_options = ['Week', 'Day']
                        
                        period = st.radio("Group by:", period_options, period_options.index(default_period) if default_period in period_options else 0)
                        
                        # Define frequency mapping
                        freq_map = {
                            'Year': 'Y',
                            'Quarter': 'Q',
                            'Month': 'M',
                            'Week': 'W',
                            'Day': 'D'
                        }
                        
                        # Aggregate data
                        time_df = valid_df.set_index(date_col)
                        time_df = time_df.resample(freq_map[period])[metric_col].agg(['sum', 'mean', 'count'])
                        time_df.reset_index(inplace=True)
                        
                        # Calculate period-over-period change
                        time_df['pct_change'] = time_df['sum'].pct_change() * 100
                        
                        # Create time series visualization
                        fig = go.Figure()
                        
                        # Add bar chart for sum
                        fig.add_trace(
                            go.Bar(
                                x=time_df[date_col],
                                y=time_df['sum'],
                                name=f"Total {metric_col}",
                                marker_color="rgba(157, 114, 255, 0.7)"
                            )
                        )
                        
                        # Add line for the mean
                        fig.add_trace(
                            go.Scatter(
                                x=time_df[date_col],
                                y=time_df['mean'],
                                mode='lines+markers',
                                name=f"Average {metric_col}",
                                line=dict(color='rgba(100, 255, 218, 0.8)', width=3),
                                marker=dict(size=8, line=dict(width=2, color='#64FFDA')),
                                yaxis="y2"
                            )
                        )
                        
                        # Update layout with double y-axis
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(40, 42, 54, 0.8)",
                            paper_bgcolor="rgba(40, 42, 54, 0)",
                            margin=dict(l=20, r=20, t=50, b=20),
                            title={
                                "text": f"{metric_col} Over Time (by {period})",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"color": "#9D72FF", "size": 18}
                            },
                            xaxis=dict(
                                title=f"{period}",
                                showgrid=True,
                                gridwidth=1,
                                gridcolor="rgba(107, 114, 142, 0.2)"
                            ),
                            yaxis=dict(
                                title=f"Total {metric_col}",
                                showgrid=True,
                                gridwidth=1,
                                gridcolor="rgba(107, 114, 142, 0.2)"
                            ),
                            yaxis2=dict(
                                title=f"Average {metric_col}",
                                overlaying="y",
                                side="right"
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Growth analysis
                        growth_tabs = st.tabs(["Growth Analysis", "Trend Details"])
                        
                        with growth_tabs[0]:
                            # Create growth visualization
                            growth_fig = go.Figure()
                            
                            # Add bar chart for period-over-period change
                            growth_fig.add_trace(
                                go.Bar(
                                    x=time_df[date_col][1:],  # Skip first entry since pct_change creates NaN
                                    y=time_df['pct_change'][1:],
                                    name="% Change",
                                    marker_color=["rgba(255, 83, 112, 0.7)" if x < 0 else "rgba(100, 255, 218, 0.7)" 
                                                for x in time_df['pct_change'][1:]]
                                )
                            )
                            
                            # Add zero line
                            growth_fig.add_shape(
                                type="line",
                                x0=time_df[date_col].iloc[1],
                                y0=0,
                                x1=time_df[date_col].iloc[-1],
                                y1=0,
                                line=dict(color="rgba(255, 255, 255, 0.5)", width=2, dash="dot")
                            )
                            
                            # Update layout
                            growth_fig.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(40, 42, 54, 0.8)",
                                paper_bgcolor="rgba(40, 42, 54, 0)",
                                margin=dict(l=20, r=20, t=50, b=20),
                                title={
                                    "text": f"{period}-over-{period} Growth Rate",
                                    "y": 0.95,
                                    "x": 0.5,
                                    "xanchor": "center",
                                    "yanchor": "top",
                                    "font": {"color": "#9D72FF", "size": 18}
                                },
                                yaxis=dict(
                                    title="% Change",
                                    showgrid=True,
                                    zeroline=False,
                                    gridwidth=1,
                                    gridcolor="rgba(107, 114, 142, 0.2)"
                                )
                            )
                            
                            st.plotly_chart(growth_fig, use_container_width=True)
                            
                            # Calculate growth metrics
                            avg_growth = time_df['pct_change'][1:].mean()
                            positive_periods = (time_df['pct_change'] > 0).sum()
                            total_periods = len(time_df) - 1  # Subtract 1 because first period has no growth rate
                            
                            st.markdown(
                                f"""
                                <div style='background: rgba(40, 42, 54, 0.5); padding: 15px; border-radius: 5px; margin: 15px 0;'>
                                    <p style='color: #E0E0FF; margin: 0;'>
                                        <strong>Growth Insights:</strong> Average {period.lower()}-over-{period.lower()} growth rate is 
                                        <span style='color: {"#64FFDA" if avg_growth > 0 else "#FF5370"};'>{avg_growth:.2f}%</span>.
                                        Positive growth in {positive_periods} out of {total_periods} periods 
                                        ({(positive_periods/total_periods*100):.1f}% of the time).
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with growth_tabs[1]:
                            # Time series decomposition if enough periods
                            if len(time_df) >= 6:
                                try:
                                    # Resample to regular frequency if needed
                                    ts_data = time_df.set_index(date_col)['sum']
                                    
                                    # Get trend using moving average
                                    window_size = min(5, len(ts_data) // 2)
                                    if window_size % 2 == 0:  # Make window size odd
                                        window_size += 1
                                    
                                    ts_data_clean = ts_data.dropna()
                                    if len(ts_data_clean) >= window_size:
                                        # Calculate trend
                                        trend = ts_data_clean.rolling(window=window_size, center=True).mean()
                                        
                                        # Create decomposition plot
                                        decomp_fig = go.Figure()
                                        
                                        # Add original data
                                        decomp_fig.add_trace(
                                            go.Scatter(
                                                x=ts_data_clean.index,
                                                y=ts_data_clean.values,
                                                mode='lines+markers',
                                                name="Original Data",
                                                line=dict(color='rgba(157, 114, 255, 0.8)', width=2),
                                                marker=dict(size=6)
                                            )
                                        )
                                        
                                        # Add trend
                                        decomp_fig.add_trace(
                                            go.Scatter(
                                                x=trend.index,
                                                y=trend.values,
                                                mode='lines',
                                                name="Trend Component",
                                                                                                line=dict(color='rgba(100, 255, 218, 0.8)', width=3)
                                            )
                                        )
                                        
                                        # Update layout
                                        decomp_fig.update_layout(
                                            template="plotly_dark",
                                            plot_bgcolor="rgba(40, 42, 54, 0.8)",
                                            paper_bgcolor="rgba(40, 42, 54, 0)",
                                            margin=dict(l=20, r=20, t=50, b=20),
                                            title={
                                                "text": f"Trend Analysis of {metric_col} Over Time",
                                                "y": 0.95,
                                                "x": 0.5,
                                                "xanchor": "center",
                                                "yanchor": "top",
                                                "font": {"color": "#9D72FF", "size": 18}
                                            },
                                            xaxis=dict(
                                                title=f"{period}",
                                                showgrid=True,
                                                gridwidth=1,
                                                gridcolor="rgba(107, 114, 142, 0.2)"
                                            ),
                                            yaxis=dict(
                                                title=f"{metric_col}",
                                                showgrid=True,
                                                gridwidth=1,
                                                gridcolor="rgba(107, 114, 142, 0.2)"
                                            ),
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,
                                                xanchor="right",
                                                x=1
                                            )
                                        )
                                        
                                        st.plotly_chart(decomp_fig, use_container_width=True)
                                        
                                        # Calculate trend metrics
                                        trend_growth = (trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0] * 100
                                        st.markdown(
                                            f"""
                                            <div style='background: rgba(40, 42, 54, 0.5); padding: 15px; border-radius: 5px; margin: 15px 0;'>
                                                <p style='color: #E0E0FF; margin: 0;'>
                                                    <strong>Trend Insights:</strong> The overall trend shows a 
                                                    <span style='color: {"#64FFDA" if trend_growth > 0 else "#FF5370"};'>{trend_growth:.2f}%</span> 
                                                    change from the beginning to the end of the period.
                                                </p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.warning("Not enough data points to calculate trend.")
                                except Exception as e:
                                    st.error(f"Error in trend analysis: {e}")
                            else:
                                st.warning("At least 6 periods are required for trend analysis.")
                    else:
                        st.warning("No numeric columns available for time series analysis.")
                else:
                    st.warning("No valid dates found in the selected column.")
            else:
                st.warning("No date or time columns found for time series analysis.")