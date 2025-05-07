import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import re
import base64
import joblib
import shap
import os
import shutil
import zipfile
import tempfile
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_object_dtype, is_categorical_dtype, is_string_dtype
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Imbalanced-learn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Modeling
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
import xgboost as xgb

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve

# Explainability
from lime.lime_tabular import LimeTabularExplainer
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

# Optuna for HPO
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
pio.templates.default = "plotly_dark"

# --- Constants and Configuration ---
APP_TITLE = "ClarifAI: Automate. Explore. Model. Explain. Dominate."
MAX_ROWS_DISPLAY = 10000
MAX_COLS_DISPLAY = 50
DEFAULT_PLOTLY_THEME = "plotly_dark"

# --- Helper Functions ---
def get_cleaned_df_copy():
    df = st.session_state.get("cleaned_df", None)
    return df.copy() if df is not None else None

def get_original_df_copy():
    df = st.session_state.get("original_df", None)
    return df.copy() if df is not None else None

def get_clickable_markdown_download_link(df, text="Download Processed Dataset", filename="cleaned_dataset.csv"):
    if df is None:
        return ""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"""
    <a href="data:file/csv;base64,{b64}" 
       download="{filename}" 
       style="text-decoration: none; font-size: 1.1rem; color: #00cfff; padding: 8px 12px; border: 1px solid #00cfff; border-radius: 5px; margin-top: 10px; display: inline-block;">
       {text}
    </a>
    """
    return href

def inject_custom_css():
    custom_css = """
        <style>
            /* General Styles */
            body {
                font-family: 'Arial', sans-serif;
            }
            .stApp {
                background-color: #1E1E1E; /* Dark background */
                color: #E0E0E0; /* Light text */
            }
            .stButton>button {
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                border: 2px solid #00cfff;
                background-color: transparent;
                color: #00cfff;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #00cfff;
                color: #1E1E1E;
            }
            .stSelectbox, .stMultiselect, .stTextInput, .stTextArea, .stSlider {
                 border-radius: 5px;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #00cfff; /* Accent color for headers */
                font-weight: bold;
            }
            .stDataFrame {
                border: 1px solid #444;
                border-radius: 5px;
            }
            .stExpander {
                border: 1px solid #444;
                border-radius: 8px;
                background-color: #2a2a2a;
            }
            .stExpander header {
                font-size: 1.1em;
                font-weight: bold;
                color: #00cfff;
            }
            /* Custom class for success messages */
            .success-message {
                color: #4CAF50; /* Green */
                font-weight: bold;
            }
            /* Custom class for warning messages */
            .warning-message {
                color: #FFC107; /* Amber */
            }
            /* Custom class for error messages */
            .error-message {
                color: #F44336; /* Red */
            }
            /* Fixed navigation buttons */
            .fixed-nav {
                position: fixed;
                bottom: 20px;
                width: calc(100% - 40px); /* Adjust for padding */
                display: flex;
                justify-content: space-between;
                padding: 0 20px;
                z-index: 999;
            }
            .fixed-nav .stButton>button {
                min-width: 100px;
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def clean_column_names_for_model(df):
    df_copy = df.copy()
    # Replace special characters with underscores, ensure names are valid Python identifiers
    df_copy.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in df_copy.columns]
    # Ensure names don't start with a number
    df_copy.columns = ['_' + col if col[0].isdigit() else col for col in df_copy.columns]
    return df_copy

# --- Main Application Logic ---
def main():
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🤖"
    )
    inject_custom_css()

    # --- Session State Initialization ---
    if "started" not in st.session_state:
        st.session_state.started = False
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = None 
    if "target_column" not in st.session_state:
        st.session_state.target_column = None
    if "task_type" not in st.session_state:
        st.session_state.task_type = None
    if "preprocessing_pipeline" not in st.session_state:
        st.session_state.preprocessing_pipeline = [] # Stores tuples of (step_name, transformer_object_or_details)

    # --- Landing Page ---
    if not st.session_state.started:
        st.markdown(
            f"""
            <div style='text-align: center; padding-top: 10vh;'>
                <h1 style='font-size: 5em; font-weight: bold; color: #00cfff;'>ClarifAI</h1>
                <p style='font-size: 1.5em; max-width: 700px; margin: 20px auto; color: #E0E0E0;'>
                    Your Ultimate AI Co-Pilot: Automate. Explore. Model. Explain. Dominate.
                </p>
                <p style='font-size: 1.1em; max-width: 600px; margin: 10px auto; color: #B0B0B0;'>
                    Unleash the power of your data with an intuitive, end-to-end machine learning workbench.
                </p>
            </div>
            """, unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([2, 1.5, 2])
        with col2:
            if st.button("🚀 Launch ClarifAI", use_container_width=True, key="get_started_button"):
                st.session_state.started = True
                st.rerun()

        st.markdown("""
            <div style='position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
                    color: #777; font-size: 0.9em; text-align: center;'>
                © Enhanced by  Varun Sripad Kota
            </div>
        """, unsafe_allow_html=True)
        return # Stop further execution for landing page

    # --- Main Application Interface (Tabs) ---
    if st.session_state.started:
        # Home button to return to landing page
        if st.sidebar.button("🏠 Home", use_container_width=True):
            # Clear relevant session state if going back to home to allow fresh start
            keys_to_clear = ["started", "active_tab", "original_df", "cleaned_df", "uploaded_file_name",
                             "X_train", "X_test", "y_train", "y_test", "trained_model", "model_name",
                             "target_column", "task_type", "preprocessing_pipeline"]
            for key in keys_to_clear:
                if key in st.session_state: # Check if key exists before deleting
                    del st.session_state[key]
            st.rerun()

    tab_titles = [
        "⚙️ Upload & Pre-Clean",
        "📊 Exploratory Data Analysis (EDA)",
        "🛠️ Advanced Preprocessing",
        "🧠 Modeling & Hyperparameter Optimization",
        "💡 Model Explainability (XAI)"
    ]

    # Use st.radio for sidebar navigation for a cleaner look
    st.sidebar.title("Navigation")
    current_tab_title = st.sidebar.radio("Go to", tab_titles, index=st.session_state.active_tab, key="sidebar_nav")
    st.session_state.active_tab = tab_titles.index(current_tab_title)

    st.markdown(f"<h1 style='text-align: center; color: #00cfff;'>{tab_titles[st.session_state.active_tab]}</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #00cfff; margin-bottom: 20px;'>", unsafe_allow_html=True)

    # --- Tab 0: Upload & Pre-Clean ---
    if st.session_state.active_tab == 0:
        render_tab0()

    # --- Tab 1: Exploratory Data Analysis (EDA) ---
    elif st.session_state.active_tab == 1:
        render_tab1()

    # --- Tab 2: Advanced Preprocessing ---
    elif st.session_state.active_tab == 2:
        render_tab2()

    # --- Tab 3: Modeling & Hyperparameter Optimization ---
    elif st.session_state.active_tab == 3:
        render_tab3()

    # --- Tab 4: Model Explainability (XAI) ---
    elif st.session_state.active_tab == 4:
        render_tab4()

    # --- Fixed Navigation Buttons ---
    # This is a conceptual placement. Streamlit's native layout might be better.
    # For true fixed buttons, you might need more complex HTML/CSS injection.
    # The sidebar radio buttons serve as the primary navigation now.

    # --- Footer Signature ---
    st.markdown("""
        <div style='position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
                 color: #777; font-size: 0.9em; text-align: center;'>
            ClarifAI © Enhanced by Varun Sripad Kota
        </div>
    """, unsafe_allow_html=True)


# --- TAB 0: UPLOAD & PRE-CLEAN --- 
def render_tab0():
    st.markdown("### 📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], key="file_uploader")

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.get("uploaded_file_name"): # New file uploaded
            st.session_state.uploaded_file_name = uploaded_file.name
            try:
                with st.spinner(f"Loading {uploaded_file.name}..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    else:
                        st.error("Unsupported file type. Please upload a CSV or Excel file.")
                        return

                if df.empty:
                    st.error("The uploaded file is empty.")
                    return
                
                df = df.convert_dtypes().infer_objects(copy=False)
                st.session_state.original_df = df.copy()
                st.session_state.cleaned_df = df.copy()
                st.session_state.preprocessing_pipeline = [] # Reset pipeline
                st.success(f"🎉 Dataset '{uploaded_file.name}' loaded successfully!")

                if df.shape[0] > MAX_ROWS_DISPLAY:
                    st.warning(f"Dataset has {df.shape[0]} rows. Previews will be limited to {MAX_ROWS_DISPLAY} rows for performance.")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                st.session_state.original_df = None
                st.session_state.cleaned_df = None
                return
    
    df = get_original_df_copy()
    df_cleaned = get_cleaned_df_copy()

    if df is None:
        st.info("👋 Welcome to ClarifAI! Upload a dataset to begin your journey.")
        return

    st.markdown("--- ")
    st.markdown("### 📋 Initial Data Preview")
    col1, col2 = st.columns([3,1])
    with col1:
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    with col2:
        show_full_initial = st.checkbox("Show Full Initial Dataset", False, key="show_full_initial")
    
    st.dataframe(df.head(MAX_ROWS_DISPLAY) if not show_full_initial else df, height=300)
    
    st.markdown("--- ") 
    st.markdown("### ✨ Basic Cleaning Operations")
    if df_cleaned is None: return

    cleaning_expander = st.expander("Perform Basic Cleaning", expanded=True)
    with cleaning_expander:
        cleaning_options = [
            "No Action",
            "Handle Duplicate Rows",
            "Drop Columns",
            "Rename Columns",
            "Standardize Null-like Text to NaN",
            "Convert Comma-Separated Numbers",
            "Trim & Standardize Text Case"
        ]
        selected_cleaning_task = st.selectbox("Select a cleaning task:", cleaning_options, key="cleaning_task_selector")

        if selected_cleaning_task == "Handle Duplicate Rows":
            handle_duplicates(df_cleaned)
        elif selected_cleaning_task == "Drop Columns":
            drop_columns(df_cleaned)
        elif selected_cleaning_task == "Rename Columns":
            rename_columns(df_cleaned)
        elif selected_cleaning_task == "Standardize Null-like Text to NaN":
            standardize_null_like(df_cleaned)
        elif selected_cleaning_task == "Convert Comma-Separated Numbers":
            convert_comma_numbers(df_cleaned)
        elif selected_cleaning_task == "Trim & Standardize Text Case":
            trim_standardize_text(df_cleaned)

        if selected_cleaning_task != "No Action":
            st.markdown("#### Preview of Cleaned Data (First 50 rows)")
            st.dataframe(st.session_state.cleaned_df.head(50) if st.session_state.cleaned_df is not None else pd.DataFrame(), height=250)
            st.markdown(get_clickable_markdown_download_link(st.session_state.cleaned_df, "Download Current Cleaned Data", "pre_cleaned_data.csv"), unsafe_allow_html=True)

    if st.button("➡️ Proceed to EDA", use_container_width=True, key="proceed_to_eda"):
        st.session_state.active_tab = 1
        st.rerun()

# Helper functions for Tab 0 cleaning operations
def handle_duplicates(df_cleaned_ref):
    total_dupes = df_cleaned_ref.duplicated().sum()
    if total_dupes == 0:
        st.success("✅ No exact duplicate rows found.")
        return

    st.warning(f"⚠️ Found {total_dupes} exact duplicate rows.")
    if st.checkbox("Preview duplicate rows", key="preview_dupes_tab0"):
        st.dataframe(df_cleaned_ref[df_cleaned_ref.duplicated(keep=False)].head(10))

    dup_mode = st.radio("Method:", ["Remove exact duplicates (all columns)", "Remove based on specific columns"], key="dup_mode_tab0")
    keep_option = st.selectbox("Keep which duplicate?", ["first", "last", "none (drop all)"], index=0, key="keep_option_tab0")
    keep_value = None if keep_option == "none (drop all)" else keep_option

    if dup_mode == "Remove exact duplicates (all columns)":
        if st.button("Remove Duplicates", key="remove_exact_dupes_tab0"):
            before_shape = df_cleaned_ref.shape[0]
            df_cleaned_ref.drop_duplicates(keep=keep_value, inplace=True)
            st.session_state.cleaned_df = df_cleaned_ref # Update session state
            st.success(f"✅ Removed {before_shape - df_cleaned_ref.shape[0]} duplicate rows.")
            st.session_state.preprocessing_pipeline.append(("Removed Duplicates (all cols)", {"keep": keep_value}))
            st.rerun()
    else:
        subset_cols = st.multiselect("Select columns to define duplicates:", df_cleaned_ref.columns, key="subset_cols_dupes_tab0")
        if subset_cols and st.button("Remove Duplicates by Subset", key="remove_subset_dupes_tab0"):
            before_shape = df_cleaned_ref.shape[0]
            df_cleaned_ref.drop_duplicates(subset=subset_cols, keep=keep_value, inplace=True)
            st.session_state.cleaned_df = df_cleaned_ref # Update session state
            st.success(f"✅ Removed {before_shape - df_cleaned_ref.shape[0]} duplicates based on {subset_cols}.")
            st.session_state.preprocessing_pipeline.append(("Removed Duplicates (subset)", {"subset": subset_cols, "keep": keep_value}))
            st.rerun()

def drop_columns(df_cleaned_ref):
    cols_to_drop = st.multiselect("Select columns to drop:", df_cleaned_ref.columns, key="drop_cols_tab0")
    if cols_to_drop and st.button("Drop Selected Columns", key="drop_button_tab0"):
        df_cleaned_ref.drop(columns=cols_to_drop, inplace=True)
        st.session_state.cleaned_df = df_cleaned_ref
        st.success(f"✅ Dropped columns: {', '.join(cols_to_drop)}. New shape: {df_cleaned_ref.shape}")
        st.session_state.preprocessing_pipeline.append(("Dropped Columns", {"columns": cols_to_drop}))
        st.rerun()

def rename_columns(df_cleaned_ref):
    st.markdown("Edit column names below (press Enter after each change):")
    rename_map = {}
    cols_per_row = 3
    col_chunks = [df_cleaned_ref.columns[i:i + cols_per_row] for i in range(0, len(df_cleaned_ref.columns), cols_per_row)]
    
    for chunk in col_chunks:
        cols = st.columns(cols_per_row)
        for i, col_name in enumerate(chunk):
            with cols[i]:
                new_name = st.text_input(f"`{col_name}` to:", value=col_name, key=f"rename_{col_name}_tab0")
                if new_name and new_name != col_name:
                    rename_map[col_name] = new_name
    
    if rename_map and st.button("Apply Renaming", key="apply_rename_tab0"):
        df_cleaned_ref.rename(columns=rename_map, inplace=True)
        st.session_state.cleaned_df = df_cleaned_ref
        st.success("✅ Column names updated.")
        st.session_state.preprocessing_pipeline.append(("Renamed Columns", {"map": rename_map}))
        st.rerun()

def standardize_null_like(df_cleaned_ref):
    text_cols = df_cleaned_ref.select_dtypes(include=['object', 'string']).columns.tolist()
    if not text_cols:
        st.info("No text columns found to standardize null-like values.")
        return
    selected_cols = st.multiselect("Select text columns to standardize nulls:", text_cols, default=text_cols, key="std_null_cols_tab0")
    null_tokens_str = st.text_area("Null-like tokens (comma-separated):", 
                                   value="None, none, NA, N/A, na, null, NULL, --, '', NaN, nan", 
                                   key="null_tokens_tab0")
    if selected_cols and st.button("Replace Null Tokens with Actual NaN", key="replace_null_tab0"):
        null_tokens = [token.strip() for token in null_tokens_str.split(',')]
        replaced_count = 0
        for col in selected_cols:
            initial_nulls = df_cleaned_ref[col].isnull().sum()
            df_cleaned_ref[col] = df_cleaned_ref[col].replace(null_tokens, np.nan)
            replaced_count += (df_cleaned_ref[col].isnull().sum() - initial_nulls)
        st.session_state.cleaned_df = df_cleaned_ref
        st.success(f"✅ Replaced {replaced_count} null-like tokens with NaN across selected columns.")
        st.session_state.preprocessing_pipeline.append(("Standardized Nulls", {"columns": selected_cols, "tokens": null_tokens}))
        st.rerun()

def convert_comma_numbers(df_cleaned_ref):
    candidate_cols = df_cleaned_ref.select_dtypes(include=['object', 'string']).columns.tolist()
    if not candidate_cols:
        st.info("No potential string columns found for comma-separated number conversion.")
        return
    selected_cols = st.multiselect("Select columns to convert comma-separated numbers:", candidate_cols, key="comma_num_cols_tab0")
    if selected_cols and st.button("Convert Selected to Numeric", key="convert_comma_num_tab0"):
        converted_cols = []
        for col in selected_cols:
            try:
                # Ensure it's string type before str.replace
                df_cleaned_ref[col] = df_cleaned_ref[col].astype(str).str.replace(",", "", regex=False)
                df_cleaned_ref[col] = pd.to_numeric(df_cleaned_ref[col], errors='coerce')
                converted_cols.append(col)
            except Exception as e:
                st.warning(f"⚠️ Could not convert column '{col}': {e}")
        st.session_state.cleaned_df = df_cleaned_ref
        if converted_cols:
            st.success(f"✅ Successfully attempted conversion for: {', '.join(converted_cols)}.")
            st.session_state.preprocessing_pipeline.append(("Converted Comma Numbers", {"columns": converted_cols}))
        else:
            st.info("No columns were converted.")
        st.rerun()

def trim_standardize_text(df_cleaned_ref):
    text_cols = df_cleaned_ref.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    if not text_cols:
        st.info("No text-like columns found for trimming or case standardization.")
        return
    selected_cols = st.multiselect("Select text columns to clean:", text_cols, default=text_cols, key="trim_text_cols_tab0")
    case_option = st.selectbox("Case transformation:", ["No Change", "lowercase", "UPPERCASE", "Capitalize Words", "Sentence case"], key="case_option_tab0")
    trim_whitespace = st.checkbox("Trim leading/trailing whitespace", value=True, key="trim_ws_tab0")

    if selected_cols and st.button("Apply Text Cleaning", key="apply_text_clean_tab0"):
        for col in selected_cols:
            if pd.api.types.is_categorical_dtype(df_cleaned_ref[col]):
                 # Convert to object for string operations, then back to category if needed
                df_cleaned_ref[col] = df_cleaned_ref[col].astype(str)
            
            if trim_whitespace:
                df_cleaned_ref[col] = df_cleaned_ref[col].str.strip()
            
            if case_option == "lowercase":
                df_cleaned_ref[col] = df_cleaned_ref[col].str.lower()
            elif case_option == "UPPERCASE":
                df_cleaned_ref[col] = df_cleaned_ref[col].str.upper()
            elif case_option == "Capitalize Words":
                df_cleaned_ref[col] = df_cleaned_ref[col].str.title()
            elif case_option == "Sentence case":
                df_cleaned_ref[col] = df_cleaned_ref[col].str.capitalize() # Capitalizes first letter of string

            # Optionally convert back to category if it was originally
            # This might change category orders, handle with care if order is important
            # if pd.api.types.is_categorical_dtype(st.session_state.original_df[col]):
            #    df_cleaned_ref[col] = pd.Categorical(df_cleaned_ref[col])

        st.session_state.cleaned_df = df_cleaned_ref
        st.success("✅ Text cleaning applied to selected columns.")
        st.session_state.preprocessing_pipeline.append(("Trimmed/Standardized Text", {"columns": selected_cols, "case": case_option, "trim": trim_whitespace}))
        st.rerun()

# --- TAB 1: EXPLORATORY DATA ANALYSIS (EDA) ---
def render_tab1():
    df_cleaned = get_cleaned_df_copy()
    if df_cleaned is None:
        st.warning("⚠️ Please upload and pre-clean a dataset in Tab 0 to perform EDA.")
        if st.button("Go to Upload Tab"):
            st.session_state.active_tab = 0
            st.rerun()
        return

    st.markdown("### 📈 Dataset Overview & Statistics")
    with st.expander("Dataset Summary", expanded=True):
        # Data Types
        st.markdown("**Data Types:**")
        dtypes_df = df_cleaned.dtypes.astype(str).reset_index()
        dtypes_df.columns = ['Feature', 'Data Type']
        st.dataframe(dtypes_df, height=200)

        # Missing Values Summary
        st.markdown("**Missing Values:**")
        missing_summary = df_cleaned.isnull().sum().reset_index()
        missing_summary.columns = ['Feature', 'Missing Count']
        missing_summary['Missing (%)'] = (missing_summary['Missing Count'] / len(df_cleaned) * 100).round(2)
        missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values(by='Missing (%)', ascending=False)
        if not missing_summary.empty:
            fig_missing = px.bar(missing_summary, x='Feature', y='Missing (%)', title='Percentage of Missing Values per Feature',
                                 labels={'Missing (%)':'Percentage Missing', 'Feature':'Feature Name'},
                                 color='Missing (%)', color_continuous_scale=px.colors.sequential.Viridis)
            fig_missing.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing values found in the current dataset!")

        # Numerical and Categorical Columns Breakdown
        st.markdown("**Column Types Breakdown:**")
        num_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df_cleaned.select_dtypes(include=['object', 'category', 'string', 'boolean']).columns.tolist()
        st.write(f"Numeric Columns ({len(num_cols)}): `{', '.join(num_cols) if num_cols else 'None'}`")
        st.write(f"Categorical/Text Columns ({len(cat_cols)}): `{', '.join(cat_cols) if cat_cols else 'None'}`")

        # Descriptive Statistics
        st.markdown("**Descriptive Statistics (Numerical Features):**")
        if num_cols:
            st.dataframe(df_cleaned[num_cols].describe().T.round(3))
        else:
            st.info("No numerical features to describe.")
        
        st.markdown("**Descriptive Statistics (Categorical/Text Features):**")
        if cat_cols:
            st.dataframe(df_cleaned[cat_cols].describe(include=['object', 'category', 'string', 'boolean']).T)
        else:
            st.info("No categorical/text features to describe.")

    st.markdown("--- ")
    st.markdown("### 📊 Visual Exploratory Data Analysis")
    
    eda_options = [
        "Distribution Analysis (Histogram/Density Plot)", 
        "Box Plot (Outlier Detection)", 
        "Count Plot (Categorical Frequencies)",
        "Scatter Plot (Relationship between two numerical features)",
        "Correlation Heatmap (Numerical Features)",
        "Pair Plot (Relationships among multiple numerical features)",
        "Categorical vs Numerical Plot (e.g., Box plot by category)"
    ]
    selected_eda_plot = st.selectbox("Select a type of visualization:", eda_options, key="eda_plot_selector")

    num_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_cleaned.select_dtypes(include=['object', 'category', 'string', 'boolean']).columns.tolist()

    if selected_eda_plot == "Distribution Analysis (Histogram/Density Plot)":
        if not num_cols:
            st.warning("No numerical columns available for distribution plots.")
        else:
            col_to_plot = st.selectbox("Select a numerical column:", num_cols, key="dist_col_select")
            plot_type = st.radio("Plot type:", ("Histogram", "Density Plot"), key="dist_plot_type")
            show_kde = st.checkbox("Show KDE (for Histogram)", value=True, disabled=(plot_type=="Density Plot"))
            if col_to_plot:
                fig = px.histogram(df_cleaned, x=col_to_plot, marginal="box", 
                                   title=f'{plot_type} of {col_to_plot}', 
                                   histnorm='probability density' if plot_type == "Density Plot" or show_kde else None,
                                   opacity=0.75)
                if plot_type == "Density Plot" or show_kde:
                     # For true density, px.density_contour or px.violin might be alternatives, or using a different library
                     # For now, using histnorm for histogram and relying on seaborn for pure density if needed.
                     # A simple density with plotly express:
                     fig_density = px.density_contour(df_cleaned, x=col_to_plot, title=f'Density of {col_to_plot}')
                     st.plotly_chart(fig_density, use_container_width=True)
                else:
                    st.plotly_chart(fig, use_container_width=True)
    
    elif selected_eda_plot == "Box Plot (Outlier Detection)":
        if not num_cols:
            st.warning("No numerical columns available for box plots.")
        else:
            col_to_plot = st.selectbox("Select a numerical column for Box Plot:", num_cols, key="boxplot_col_select")
            if col_to_plot:
                fig = px.box(df_cleaned, y=col_to_plot, title=f'Box Plot of {col_to_plot}', points="outliers")
                st.plotly_chart(fig, use_container_width=True)

    elif selected_eda_plot == "Count Plot (Categorical Frequencies)":
        if not cat_cols:
            st.warning("No categorical columns available for count plots.")
        else:
            col_to_plot = st.selectbox("Select a categorical column:", cat_cols, key="countplot_col_select")
            if col_to_plot:
                if df_cleaned[col_to_plot].nunique() > 50:
                    st.warning(f"Column '{col_to_plot}' has {df_cleaned[col_to_plot].nunique()} unique values. Plot might be cluttered.")
                value_counts = df_cleaned[col_to_plot].value_counts().reset_index()
                value_counts.columns = [col_to_plot, 'count']
                fig = px.bar(value_counts, x=col_to_plot, y='count', title=f'Count Plot of {col_to_plot}',
                             color=col_to_plot)
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    elif selected_eda_plot == "Scatter Plot (Relationship between two numerical features)":
        if len(num_cols) < 2:
            st.warning("Need at least two numerical columns for a scatter plot.")
        else:
            x_col = st.selectbox("Select X-axis (numerical):", num_cols, index=0, key="scatter_x")
            y_col = st.selectbox("Select Y-axis (numerical):", num_cols, index=1 if len(num_cols)>1 else 0, key="scatter_y")
            color_col_options = ["None"] + cat_cols + num_cols
            color_col = st.selectbox("Optional: Color by column:", color_col_options, key="scatter_color")
            
            if x_col and y_col:
                fig = px.scatter(df_cleaned, x=x_col, y=y_col, title=f'Scatter Plot: {x_col} vs {y_col}',
                                 color=color_col if color_col != "None" else None,
                                 trendline="ols" if df_cleaned[x_col].nunique() > 1 and df_cleaned[y_col].nunique() > 1 else None,
                                 marginal_y="violin", marginal_x="box", opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)

    elif selected_eda_plot == "Correlation Heatmap (Numerical Features)":
        if len(num_cols) < 2:
            st.warning("Need at least two numerical columns for a correlation heatmap.")
        else:
            corr_matrix = df_cleaned[num_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                            title="Correlation Heatmap of Numerical Features")
            st.plotly_chart(fig, use_container_width=True)

    elif selected_eda_plot == "Pair Plot (Relationships among multiple numerical features)":
        if len(num_cols) < 2:
            st.warning("Need at least two numerical columns for a pair plot.")
        else:
            st.info("Pair plots can be computationally intensive for many features.")
            selected_pair_cols = st.multiselect("Select numerical columns for Pair Plot (max 5 recommended):", 
                                                num_cols, default=num_cols[:min(len(num_cols), 4)], key="pairplot_cols")
            if len(selected_pair_cols) > 8:
                st.error("Too many columns selected for pair plot. Please select fewer than 8.")
            elif selected_pair_cols and len(selected_pair_cols) >=2:
                color_col_options_pair = ["None"] + cat_cols
                color_col_pair = st.selectbox("Optional: Color by categorical column:", color_col_options_pair, key="pairplot_color_col")
                
                with st.spinner("Generating Pair Plot..."):
                    fig = px.scatter_matrix(df_cleaned, dimensions=selected_pair_cols, 
                                            color=color_col_pair if color_col_pair != "None" else None,
                                            title="Pair Plot of Selected Numerical Features")
                    fig.update_traces(diagonal_visible=False, showupperhalf=False)
                    fig.update_layout(height=800)
                    st.plotly_chart(fig, use_container_width=True)
            elif not selected_pair_cols:
                 st.info("Select at least two columns for the pair plot.")

    elif selected_eda_plot == "Categorical vs Numerical Plot (e.g., Box plot by category)":
        if not num_cols or not cat_cols:
            st.warning("Need at least one numerical and one categorical column for this plot.")
        else:
            num_col_cvn = st.selectbox("Select Numerical Column:", num_cols, key="cvn_num")
            cat_col_cvn = st.selectbox("Select Categorical Column:", cat_cols, key="cvn_cat")
            plot_type_cvn = st.radio("Plot Type:", ("Box Plot", "Violin Plot", "Bar Plot (Aggregated)"), key="cvn_plot_type")

            if num_col_cvn and cat_col_cvn:
                if df_cleaned[cat_col_cvn].nunique() > 20 and plot_type_cvn != "Bar Plot (Aggregated)":
                    st.warning(f"Categorical column '{cat_col_cvn}' has many unique values. Plot might be cluttered.")
                
                if plot_type_cvn == "Box Plot":
                    fig = px.box(df_cleaned, x=cat_col_cvn, y=num_col_cvn, color=cat_col_cvn, 
                                 title=f'Box Plot of {num_col_cvn} by {cat_col_cvn}')
                elif plot_type_cvn == "Violin Plot":
                    fig = px.violin(df_cleaned, x=cat_col_cvn, y=num_col_cvn, color=cat_col_cvn, box=True, points="all",
                                    title=f'Violin Plot of {num_col_cvn} by {cat_col_cvn}')
                elif plot_type_cvn == "Bar Plot (Aggregated)":
                    agg_func = st.selectbox("Aggregation function:", ['mean', 'median', 'sum', 'count'], key="cvn_agg_func")
                    agg_df = df_cleaned.groupby(cat_col_cvn)[num_col_cvn].agg(agg_func).reset_index()
                    fig = px.bar(agg_df, x=cat_col_cvn, y=num_col_cvn, color=cat_col_cvn,
                                 title=f'{agg_func.capitalize()} of {num_col_cvn} by {cat_col_cvn}',
                                 labels={num_col_cvn: f'{agg_func.capitalize()} of {num_col_cvn}'})
                
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    if st.button("➡️ Proceed to Advanced Preprocessing", use_container_width=True, key="proceed_to_preprocessing"):
        st.session_state.active_tab = 2
        st.rerun()

# --- TAB 2: ADVANCED PREPROCESSING ---
def render_tab2():
    df_cleaned = get_cleaned_df_copy()
    if df_cleaned is None:
        st.warning("⚠️ Please upload and process a dataset in Tab 0 & 1 to perform Advanced Preprocessing.")
        if st.button("Go to Upload Tab"):
            st.session_state.active_tab = 0
            st.rerun()
        return

    st.markdown("### 🧩 Advanced Data Transformation & Feature Engineering")
    
    # Display current preprocessing steps
    if st.session_state.preprocessing_pipeline:
        with st.expander("Applied Preprocessing Steps (So Far)", expanded=False):
            for i, (step_name, params) in enumerate(st.session_state.preprocessing_pipeline):
                st.write(f"{i+1}. **{step_name}**: {params if params else ''}")
    else:
        st.info("No preprocessing steps applied yet from previous tabs.")

    # --- Section Select --- #
    section = st.radio("Choose Preprocessing Section:", 
                       ("Handle Missing Values", "Feature Scaling", "Categorical Encoding", 
                        "Outlier Detection & Handling", "Feature Engineering", 
                        "Dimensionality Reduction", "Class Imbalance Handling (for Classification)"), 
                       horizontal=True, key="preprocess_section_radio")

    # --- Handle Missing Values ---
    if section == "Handle Missing Values":
        with st.expander("Advanced Missing Value Imputation", expanded=True):
            missing_values_summary(df_cleaned)
            handle_missing_values_advanced(df_cleaned)

    # --- Feature Scaling ---
    elif section == "Feature Scaling":
        with st.expander("Feature Scaling Techniques", expanded=True):
            apply_feature_scaling(df_cleaned)

    # --- Categorical Encoding ---
    elif section == "Categorical Encoding":
        with st.expander("Advanced Categorical Encoding", expanded=True):
            apply_categorical_encoding(df_cleaned)
    
    # --- Outlier Detection & Handling ---
    elif section == "Outlier Detection & Handling":
        with st.expander("Outlier Detection and Treatment", expanded=True):
            handle_outliers(df_cleaned)

    # --- Feature Engineering ---
    elif section == "Feature Engineering":
        with st.expander("Feature Creation & Transformation", expanded=True):
            perform_feature_engineering(df_cleaned)

    # --- Dimensionality Reduction ---
    elif section == "Dimensionality Reduction":
        with st.expander("Dimensionality Reduction Techniques (PCA/t-SNE)", expanded=True):
            apply_dimensionality_reduction(df_cleaned)

    # --- Class Imbalance Handling ---
    elif section == "Class Imbalance Handling (for Classification)":
        with st.expander("Handle Class Imbalance (Target Variable)", expanded=True):
            handle_class_imbalance(df_cleaned)

    st.markdown("--- ")
    st.markdown("### 💾 Current Processed Dataset Preview & Download")
    st.dataframe(st.session_state.cleaned_df.head(50) if st.session_state.cleaned_df is not None else pd.DataFrame(), height=300)
    st.markdown(get_clickable_markdown_download_link(st.session_state.cleaned_df, "Download Fully Processed Dataset", "fully_processed_data.csv"), unsafe_allow_html=True)

    if st.button("➡️ Proceed to Modeling", use_container_width=True, key="proceed_to_modeling"):
        if st.session_state.cleaned_df is None or st.session_state.cleaned_df.empty:
            st.error("Cannot proceed to modeling. The dataset is empty or not loaded.")
        elif st.session_state.cleaned_df.isnull().sum().sum() > 0:
            st.warning("⚠️ Your dataset still contains missing values. It's highly recommended to handle them before modeling. Proceed with caution.")
            if st.button("Proceed Anyway", key="proceed_model_warn"):
                 st.session_state.active_tab = 3
                 st.rerun()
        else:
            st.session_state.active_tab = 3
            st.rerun()

# Helper functions for Tab 2
def missing_values_summary(df_ref):
    st.markdown("**Current Missing Values:**")
    missing_summary = df_ref.isnull().sum().reset_index()
    missing_summary.columns = ['Feature', 'Missing Count']
    missing_summary['Missing (%)'] = (missing_summary['Missing Count'] / len(df_ref) * 100).round(2)
    missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values(by='Missing (%)', ascending=False)
    if not missing_summary.empty:
        st.dataframe(missing_summary)
    else:
        st.success("✅ No missing values currently in the dataset!")

def handle_missing_values_advanced(df_ref):
    cols_with_missing = df_ref.columns[df_ref.isnull().any()].tolist()
    if not cols_with_missing:
        st.info("No columns currently have missing values.")
        return

    selected_cols_impute = st.multiselect("Select columns to impute missing values:", cols_with_missing, default=cols_with_missing, key="impute_cols_select")
    
    imputation_strategy_num = st.selectbox("Imputation strategy for NUMERICAL columns:", 
                                         ("Mean", "Median", "Mode (Most Frequent)", "Constant"), key="num_impute_strat")
    constant_val_num = None
    if imputation_strategy_num == "Constant":
        constant_val_num = st.number_input("Enter constant value for numerical imputation:", value=0, key="num_impute_const")

    imputation_strategy_cat = st.selectbox("Imputation strategy for CATEGORICAL columns:", 
                                         ("Mode (Most Frequent)", "Constant"), key="cat_impute_strat")
    constant_val_cat = None
    if imputation_strategy_cat == "Constant":
        constant_val_cat = st.text_input("Enter constant value for categorical imputation:", value="Missing", key="cat_impute_const")

    drop_rows_thresh_pct = st.slider("Or, drop rows if missing values exceed X% of columns (0 to disable):", 0, 100, 0, 5, key="drop_row_thresh")
    drop_cols_thresh_pct = st.slider("Or, drop columns if missing values exceed X% of rows (0 to disable):", 0, 100, 0, 5, key="drop_col_thresh")

    if st.button("Apply Missing Value Treatment", key="apply_mv_treatment"):
        df_processed = df_ref.copy()
        imputation_details = []

        if drop_cols_thresh_pct > 0:
            thresh = len(df_processed) * (drop_cols_thresh_pct / 100)
            cols_to_drop_mv = df_processed.columns[df_processed.isnull().sum() > thresh].tolist()
            if cols_to_drop_mv:
                df_processed.drop(columns=cols_to_drop_mv, inplace=True)
                st.success(f"Dropped columns with >{drop_cols_thresh_pct}% missing: {cols_to_drop_mv}")
                st.session_state.preprocessing_pipeline.append(("Dropped Columns (High MV %)", {"columns": cols_to_drop_mv, "threshold_pct": drop_cols_thresh_pct}))
                selected_cols_impute = [col for col in selected_cols_impute if col not in cols_to_drop_mv]
        
        if drop_rows_thresh_pct > 0:
            thresh = len(df_processed.columns) * (drop_rows_thresh_pct / 100)
            initial_rows = len(df_processed)
            df_processed.dropna(thresh=len(df_processed.columns) - thresh + 1, inplace=True) # Keep rows with at least 'thresh' non-NA values
            rows_dropped = initial_rows - len(df_processed)
            if rows_dropped > 0:
                st.success(f"Dropped {rows_dropped} rows with >{drop_rows_thresh_pct}% missing values.")
                st.session_state.preprocessing_pipeline.append(("Dropped Rows (High MV %)", {"rows_dropped": rows_dropped, "threshold_pct": drop_rows_thresh_pct}))

        for col in selected_cols_impute:
            if col not in df_processed.columns: continue # Column might have been dropped
            if df_processed[col].isnull().sum() == 0: continue

            if is_numeric_dtype(df_processed[col]):
                strategy = imputation_strategy_num.lower()
                if strategy == "mode (most frequent)": strategy = "most_frequent"
                fill_value_num = constant_val_num if strategy == "constant" else None
                imputer = SimpleImputer(strategy=strategy, fill_value=fill_value_num)
                df_processed[col] = imputer.fit_transform(df_processed[[col]])
                imputation_details.append((col, f"Numerical Imputation: {strategy.capitalize()}" + (f" with {fill_value_num}" if fill_value_num is not None else "")))
            else: # Categorical
                strategy = imputation_strategy_cat.lower()
                if strategy == "mode (most frequent)": strategy = "most_frequent"
                fill_value_cat = constant_val_cat if strategy == "constant" else None
                imputer = SimpleImputer(strategy=strategy, fill_value=fill_value_cat)
                df_processed[col] = imputer.fit_transform(df_processed[[col]]).ravel()
                imputation_details.append((col, f"Categorical Imputation: {strategy.capitalize()}" + (f" with {fill_value_cat}" if fill_value_cat is not None else "")))
        
        st.session_state.cleaned_df = df_processed
        if imputation_details:
            st.success("Missing value imputation applied.")
            for col, detail in imputation_details:
                st.write(f"  - For '{col}': {detail}")
            st.session_state.preprocessing_pipeline.append(("Missing Value Imputation", {"details": imputation_details}))
        st.rerun()

def apply_feature_scaling(df_ref):
    num_cols = df_ref.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.info("No numerical columns available for scaling.")
        return

    selected_cols_scale = st.multiselect("Select numerical columns to scale:", num_cols, default=num_cols, key="scale_cols_select")
    scaling_method = st.selectbox("Choose a scaling method:", 
                                  ("StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer (row-wise)"), 
                                  key="scaling_method_select")

    if selected_cols_scale and st.button("Apply Scaling", key="apply_scaling_button"):
        df_processed = df_ref.copy()
        scaler_map = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "MaxAbsScaler": MaxAbsScaler(),
            "Normalizer (row-wise)": Normalizer()
        }
        scaler = scaler_map[scaling_method]
        
        try:
            df_processed[selected_cols_scale] = scaler.fit_transform(df_processed[selected_cols_scale])
            st.session_state.cleaned_df = df_processed
            st.success(f"Applied {scaling_method} to: {', '.join(selected_cols_scale)}.")
            st.session_state.preprocessing_pipeline.append(("Feature Scaling", {"method": scaling_method, "columns": selected_cols_scale, "scaler_obj": scaler}))
            st.rerun()
        except Exception as e:
            st.error(f"Scaling failed: {e}")

def apply_categorical_encoding(df_ref):
    cat_cols = df_ref.select_dtypes(include=['object', 'category', 'string', 'boolean']).columns.tolist()
    # Exclude already numerically encoded boolean columns if they are not meant for further encoding like OHE
    cat_cols = [col for col in cat_cols if not (is_numeric_dtype(df_ref[col]) and df_ref[col].nunique() <=2)] 

    if not cat_cols:
        st.info("No suitable categorical columns detected for encoding.")
        return

    selected_cols_encode = st.multiselect("Select categorical columns to encode:", cat_cols, default=cat_cols, key="encode_cols_select")
    encoding_method = st.selectbox("Select encoding method:", 
                                   ("Label Encoding", "One-Hot Encoding", "Ordinal Encoding"), 
                                   key="encoding_method_select")

    drop_first_ohe = True
    if encoding_method == "One-Hot Encoding":
        drop_first_ohe = st.checkbox("Drop first category (to avoid multicollinearity)", value=True, key="ohe_drop_first")

    ordinal_orders = {}
    if encoding_method == "Ordinal Encoding":
        st.markdown("Define order for Ordinal Encoding (comma-separated, lowest to highest). Leave blank for default alphabetical/numerical sort.")
        for col in selected_cols_encode:
            unique_vals = sorted(df_ref[col].dropna().unique().astype(str))
            order_str = st.text_input(f"Order for '{col}' (e.g., {','.join(unique_vals[:3])}...):", key=f"ordinal_order_{col}")
            if order_str:
                ordinal_orders[col] = [s.strip() for s in order_str.split(',')]
            else:
                ordinal_orders[col] = unique_vals # Default order
    
    if selected_cols_encode and st.button("Apply Encoding", key="apply_encoding_button"):
        df_processed = df_ref.copy()
        encoding_details = []
        try:
            for col in selected_cols_encode:
                if is_numeric_dtype(df_processed[col]): # Skip if somehow already numeric
                    st.warning(f"Column '{col}' is already numeric. Skipping encoding.")
                    continue

                if encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    encoding_details.append((col, "Label Encoded", {"encoder_obj": le}))
                
                elif encoding_method == "One-Hot Encoding":
                    if df_processed[col].nunique() > 50:
                        st.warning(f"Column '{col}' has >50 unique values. OHE might create too many features. Skipping.")
                        continue
                    # Ensure no NaN conflicts with get_dummies, fill if necessary or ensure it's handled
                    # df_processed[col] = df_processed[col].astype(str).fillna('MISSING_CAT') # Example handling
                    dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=drop_first_ohe, dummy_na=False) # dummy_na=False is safer
                    df_processed = pd.concat([df_processed.drop(columns=[col]), dummies], axis=1)
                    encoding_details.append((col, f"One-Hot Encoded (drop_first={drop_first_ohe})", {"original_col": col, "dummy_cols": dummies.columns.tolist()}))
                
                elif encoding_method == "Ordinal Encoding":
                    oe = OrdinalEncoder(categories=[ordinal_orders[col]])
                    df_processed[col] = oe.fit_transform(df_processed[[col]].astype(str))
                    encoding_details.append((col, "Ordinal Encoded", {"order": ordinal_orders[col], "encoder_obj": oe}))
            
            st.session_state.cleaned_df = df_processed
            st.success(f"Applied {encoding_method} to selected columns.")
            for col_name, detail_str, params_dict in encoding_details:
                st.session_state.preprocessing_pipeline.append((f"{detail_str} for {col_name}", params_dict))
            st.rerun()
        except Exception as e:
            st.error(f"Encoding failed: {e}")

def handle_outliers(df_ref):
    num_cols = df_ref.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.info("No numerical columns available for outlier detection.")
        return

    st.markdown("**Note:** IQR is robust to skewed data. Z-score assumes near-normal distribution.")
    selected_cols_outlier = st.multiselect("Select numerical columns to check for outliers:", num_cols, default=num_cols, key="outlier_cols_select")
    method = st.radio("Outlier detection method:", ("IQR (Interquartile Range)", "Z-Score"), key="outlier_method")
    action = st.radio("Action for outliers:", ("Cap (Clip)", "Remove Rows", "Mark (add boolean column)"), key="outlier_action")

    if selected_cols_outlier and st.button("Handle Outliers", key="apply_outlier_button"):
        df_processed = df_ref.copy()
        outlier_handling_details = []
        try:
            for col in selected_cols_outlier:
                if method == "IQR (Interquartile Range)":
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR_val = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR_val
                    upper_bound = Q3 + 1.5 * IQR_val
                    outliers = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                elif method == "Z-Score":
                    z_threshold = st.slider("Z-score threshold:", 1.0, 5.0, 3.0, 0.1, key=f"z_thresh_{col}")
                    z_scores = np.abs(stats.zscore(df_processed[col].dropna())) # dropna before zscore
                    # Align z_scores back to original df_processed index if there were NaNs
                    # This is complex. Simpler: operate on non-NaN part, then put back.
                    # For now, this might misalign if NaNs exist and action is 'Remove Rows' or 'Mark'
                    # A robust way: calculate bounds on non-na, then apply to original column.
                    mean_val = df_processed[col].dropna().mean()
                    std_val = df_processed[col].dropna().std()
                    lower_bound = mean_val - z_threshold * std_val
                    upper_bound = mean_val + z_threshold * std_val
                    outliers = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                
                num_outliers = outliers.sum()
                if num_outliers > 0:
                    st.write(f"Found {num_outliers} outliers in '{col}' using {method} (Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]).")
                    if action == "Cap (Clip)":
                        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                        outlier_handling_details.append((col, f"Capped using {method}"))
                    elif action == "Remove Rows":
                        df_processed = df_processed[~outliers]
                        outlier_handling_details.append((col, f"Rows removed using {method}"))
                    elif action == "Mark (add boolean column)":
                        df_processed[f'{col}_is_outlier'] = outliers
                        outlier_handling_details.append((col, f"Marked as new column using {method}"))
                else:
                    st.write(f"No outliers detected in '{col}' with current settings.")
            
            st.session_state.cleaned_df = df_processed
            st.success("Outlier handling applied.")
            for col_name, detail_str in outlier_handling_details:
                 st.session_state.preprocessing_pipeline.append((f"Outlier Handling for {col_name}", {"detail": detail_str}))
            st.rerun()
        except Exception as e:
            st.error(f"Outlier handling failed: {e}")

def perform_feature_engineering(df_ref):
    st.markdown("Create new features from existing ones.")
    num_cols = df_ref.select_dtypes(include=np.number).columns.tolist()
    # date_cols = df_ref.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    # For simplicity, assume date columns are already pd.to_datetime or need explicit conversion step first

    # Option 1: Polynomial Features
    if num_cols:
        st.markdown("**Polynomial Features (for numerical columns):**")
        poly_cols = st.multiselect("Select numerical columns for polynomial features:", num_cols, key="poly_feat_cols")
        degree = st.slider("Polynomial degree:", 2, 4, 2, key="poly_degree")
        interaction_only = st.checkbox("Interaction terms only?", key="poly_interaction_only")
        include_bias = st.checkbox("Include bias (intercept column)?", value=False, key="poly_include_bias")

        if poly_cols and st.button("Generate Polynomial Features", key="gen_poly_feats"):
            from sklearn.preprocessing import PolynomialFeatures
            df_processed = df_ref.copy()
            poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
            poly_features = poly.fit_transform(df_processed[poly_cols])
            poly_feature_names = poly.get_feature_names_out(poly_cols)
            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_processed.index)
            # Drop original columns used if not interaction_only to avoid redundancy, or user choice
            # For now, just add new ones. User can drop later.
            df_processed = pd.concat([df_processed, poly_df.drop(columns=poly_cols if not interaction_only and degree >1 else [])], axis=1)
            st.session_state.cleaned_df = df_processed
            st.success(f"Generated {len(poly_feature_names)} polynomial features.")
            st.session_state.preprocessing_pipeline.append(("Polynomial Features", {"columns": poly_cols, "degree": degree, "interaction_only": interaction_only, "new_cols": poly_feature_names.tolist()}))
            st.rerun()

    # Option 2: Simple Interactions (Numerical)
    if len(num_cols) >= 2:
        st.markdown("**Simple Interaction Terms (Numerical x Numerical):**")
        interact_col1 = st.selectbox("Select first numerical column for interaction:", num_cols, key="interact_col1")
        interact_col2 = st.selectbox("Select second numerical column for interaction:", num_cols, index=1 if len(num_cols)>1 else 0, key="interact_col2")
        if interact_col1 and interact_col2 and interact_col1 != interact_col2:
            if st.button(f"Create Interaction: {interact_col1} * {interact_col2}", key="create_num_interact"):
                df_processed = df_ref.copy()
                new_col_name = f"{interact_col1}_x_{interact_col2}"
                df_processed[new_col_name] = df_processed[interact_col1] * df_processed[interact_col2]
                st.session_state.cleaned_df = df_processed
                st.success(f"Created interaction feature: {new_col_name}")
                st.session_state.preprocessing_pipeline.append(("Numerical Interaction", {"cols": [interact_col1, interact_col2], "new_col": new_col_name}))
                st.rerun()
    
    # Option 3: Binning Numerical Features
    if num_cols:
        st.markdown("**Binning Numerical Features (Discretization):**")
        bin_col = st.selectbox("Select numerical column to bin:", num_cols, key="bin_col_select")
        num_bins = st.slider("Number of bins:", 2, 20, 5, key="num_bins_slider")
        bin_strategy = st.radio("Binning strategy:", ('uniform', 'quantile', 'kmeans'), key="bin_strategy_radio")
        if bin_col and st.button("Create Binned Feature", key="create_binned_feat"):
            from sklearn.preprocessing import KBinsDiscretizer
            df_processed = df_ref.copy()
            discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy=bin_strategy, subsample=None if bin_strategy != 'kmeans' else 200_000, random_state=42 if bin_strategy == 'kmeans' else None)
            try:
                # KBinsDiscretizer expects 2D array
                df_processed[f"{bin_col}_binned"] = discretizer.fit_transform(df_processed[[bin_col]])
                st.session_state.cleaned_df = df_processed
                st.success(f"Created binned feature for '{bin_col}'.")
                st.session_state.preprocessing_pipeline.append(("Binned Numerical Feature", {"column": bin_col, "bins": num_bins, "strategy": bin_strategy}))
                st.rerun()
            except Exception as e:
                st.error(f"Binning failed for {bin_col}: {e}")

def apply_dimensionality_reduction(df_ref):
    num_cols = df_ref.select_dtypes(include=np.number).columns.tolist()
    if not num_cols or len(num_cols) < 2:
        st.info("Need at least two numerical features for dimensionality reduction.")
        return

    st.markdown("Reduce the number of features while preserving important information.")
    method = st.radio("Select Dimensionality Reduction Method:", ("PCA (Principal Component Analysis)", "t-SNE (t-distributed Stochastic Neighbor Embedding)"), key="dim_reduc_method")

    if method == "PCA (Principal Component Analysis)":
        n_components_pca = st.slider("Number of Principal Components to keep:", 1, len(num_cols)-1 if len(num_cols)>1 else 1, min(5, len(num_cols)-1 if len(num_cols)>1 else 1), key="pca_n_components")
        if st.button("Apply PCA", key="apply_pca_button"):
            df_processed = df_ref.copy()
            pca_data = df_processed[num_cols].fillna(df_processed[num_cols].mean()) # PCA needs no NaNs
            pca = PCA(n_components=n_components_pca)
            principal_components = pca.fit_transform(pca_data)
            pc_cols = [f"PC{i+1}" for i in range(n_components_pca)]
            pc_df = pd.DataFrame(data=principal_components, columns=pc_cols, index=df_processed.index)
            # Decide whether to drop original num_cols or keep them
            df_processed = pd.concat([df_processed.drop(columns=num_cols), pc_df], axis=1)
            st.session_state.cleaned_df = df_processed
            st.success(f"PCA applied. Reduced to {n_components_pca} components. Original numerical columns replaced.")
            st.write(f"Explained Variance Ratio by selected components: {pca.explained_variance_ratio_}")
            st.write(f"Total Explained Variance: {np.sum(pca.explained_variance_ratio_):.3f}")
            st.session_state.preprocessing_pipeline.append(("PCA", {"n_components": n_components_pca, "original_num_cols_dropped": num_cols, "explained_variance": pca.explained_variance_ratio_.tolist()}))
            st.rerun()
    
    elif method == "t-SNE (t-distributed Stochastic Neighbor Embedding)":
        st.info("t-SNE is primarily for visualization and can be computationally intensive.")
        n_components_tsne = st.radio("Number of t-SNE Components (usually 2 or 3 for viz):", (2, 3), key="tsne_n_components")
        perplexity_tsne = st.slider("Perplexity (related to num nearest neighbors):", 5, 50, 30, key="tsne_perplexity")
        if st.button("Apply t-SNE", key="apply_tsne_button"):
            with st.spinner("Applying t-SNE... This may take a while."):
                df_processed = df_ref.copy()
                tsne_data = df_processed[num_cols].fillna(df_processed[num_cols].mean()) # t-SNE needs no NaNs
                tsne = TSNE(n_components=n_components_tsne, perplexity=perplexity_tsne, random_state=42, n_iter=300)
                tsne_results = tsne.fit_transform(tsne_data)
                tsne_cols = [f"tSNE{i+1}" for i in range(n_components_tsne)]
                tsne_df = pd.DataFrame(data=tsne_results, columns=tsne_cols, index=df_processed.index)
                # Typically t-SNE results are added, originals not dropped unless specified
                df_processed = pd.concat([df_processed, tsne_df], axis=1)
                st.session_state.cleaned_df = df_processed
                st.success(f"t-SNE applied. Added {n_components_tsne} t-SNE components.")
                st.session_state.preprocessing_pipeline.append(("t-SNE", {"n_components": n_components_tsne, "perplexity": perplexity_tsne, "new_cols": tsne_cols}))
                st.rerun()

def handle_class_imbalance(df_ref):
    st.markdown("Balance class distribution for a categorical target variable.")
    cat_cols_target = [col for col in df_ref.columns if df_ref[col].nunique() < 20 and not is_numeric_dtype(df_ref[col])] # Heuristic for potential target
    all_cols_target = df_ref.columns.tolist()

    potential_targets = cat_cols_target + [col for col in all_cols_target if df_ref[col].nunique() < 10 and is_numeric_dtype(df_ref[col])] # also allow low-cardinality numeric
    if not potential_targets:
        st.warning("No suitable categorical or low-cardinality numerical target column found for imbalance handling.")
        return

    target_col_imb = st.selectbox("Select your TARGET column for imbalance handling:", potential_targets, key="target_col_imbalance")
    if not target_col_imb:
        return
    
    if df_ref[target_col_imb].isnull().any():
        st.error(f"Target column '{target_col_imb}' contains missing values. Please handle them first.")
        return

    st.markdown("**Current Class Distribution:**")
    class_counts = df_ref[target_col_imb].value_counts()
    fig_dist = px.bar(class_counts, x=class_counts.index, y=class_counts.values, labels={'x':'Class', 'y':'Count'}, title=f"Class Distribution of '{target_col_imb}'")
    st.plotly_chart(fig_dist, use_container_width=True)

    if class_counts.min() / class_counts.max() > 0.3: # Arbitrary threshold for 'balanced enough'
        st.info("Classes seem relatively balanced. You may not need imbalance handling.")

    imbalance_method = st.selectbox("Choose balancing method:", 
                                    ("SMOTE (Oversampling)", "RandomOverSampler", "RandomUnderSampler"), 
                                    key="imbalance_method_select")

    if st.button("Apply Class Balancing", key="apply_imbalance_button"):
        with st.spinner(f"Applying {imbalance_method}..."):
            df_processed = df_ref.copy()
            X = df_processed.drop(columns=[target_col_imb])
            y = df_processed[target_col_imb]

            # Ensure X is purely numeric for SMOTE, etc.
            # This is a critical step: SMOTE needs numeric features. If not, encode them first.
            # For simplicity, we'll assume previous steps handled encoding. If not, SMOTE will fail.
            numeric_X_check = X.select_dtypes(include=np.number)
            if numeric_X_check.shape[1] != X.shape[1]:
                st.error("Class balancing methods like SMOTE require all features (X) to be numeric. Please encode categorical features first.")
                return
            if X.isnull().sum().sum() > 0:
                st.error("Features (X) contain missing values. Please impute them before applying class balancing.")
                return

            try:
                if imbalance_method == "SMOTE (Oversampling)":
                    # k_neighbors for SMOTE must be less than the number of samples in the smallest class
                    min_class_count = y.value_counts().min()
                    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
                    if k_neighbors < 1:
                        st.error(f"Smallest class for SMOTE has {min_class_count} samples. Cannot apply SMOTE. Try RandomOverSampler or ensure more samples.")
                        return
                    sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
                elif imbalance_method == "RandomOverSampler":
                    sampler = RandomOverSampler(random_state=42)
                elif imbalance_method == "RandomUnderSampler":
                    sampler = RandomUnderSampler(random_state=42)
                
                X_res, y_res = sampler.fit_resample(X, y)
                df_resampled = pd.DataFrame(X_res, columns=X.columns)
                df_resampled[target_col_imb] = y_res

                st.session_state.cleaned_df = df_resampled
                st.success(f"{imbalance_method} applied successfully.")
                st.markdown("**New Class Distribution:**")
                new_class_counts = df_resampled[target_col_imb].value_counts()
                fig_new_dist = px.bar(new_class_counts, x=new_class_counts.index, y=new_class_counts.values, labels={'x':'Class', 'y':'Count'}, title=f"New Distribution of '{target_col_imb}'")
                st.plotly_chart(fig_new_dist, use_container_width=True)
                st.session_state.preprocessing_pipeline.append(("Class Imbalance Handling", {"method": imbalance_method, "target": target_col_imb}))
                st.rerun()
            except Exception as e:
                st.error(f"Class balancing failed: {e}. Ensure features are numeric and no NaNs.")

# --- TAB 3: MODELING & HYPERPARAMETER OPTIMIZATION ---
def render_tab3():
    df_model_ready = get_cleaned_df_copy()
    if df_model_ready is None or df_model_ready.empty:
        st.warning("⚠️ Dataset not available or empty. Please complete previous steps.")
        if st.button("Go to Upload Tab"):
            st.session_state.active_tab = 0
            st.rerun()
        return
    
    if df_model_ready.isnull().sum().sum() > 0:
        st.error("🚨 Your dataset contains missing values! Models will likely fail. Please go back to 'Advanced Preprocessing' and handle them.")
        if st.button("Go to Preprocessing"):
            st.session_state.active_tab = 2
            st.rerun()
        return

    st.markdown("### 🎯 Target Variable & Task Definition")
    all_cols = df_model_ready.columns.tolist()
    # Try to infer target from session state if previously set
    default_target_idx = all_cols.index(st.session_state.target_column) if st.session_state.target_column in all_cols else 0
    target_col = st.selectbox("Select your TARGET variable:", all_cols, index=default_target_idx, key="model_target_select")
    st.session_state.target_column = target_col

    if df_model_ready[target_col].nunique() == 1:
        st.error(f"Target column '{target_col}' has only one unique value. Cannot be used for modeling.")
        return
    
    # Task Type Inference
    if is_numeric_dtype(df_model_ready[target_col]) and df_model_ready[target_col].nunique() > 15: # Heuristic
        inferred_task = "Regression"
    else:
        inferred_task = "Classification"
    
    task_type = st.radio("Select Task Type:", ("Classification", "Regression"), 
                         index=0 if inferred_task == "Classification" else 1, key="task_type_radio")
    st.session_state.task_type = task_type
    st.info(f"🤖 Inferred Task Type: **{task_type}** based on target '{target_col}'.")

    # Feature Selection (X) and Target (y)
    X = df_model_ready.drop(columns=[target_col])
    y = df_model_ready[target_col]

    # Ensure X is purely numeric - this is a common pitfall
    non_numeric_X_cols = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_X_cols.empty:
        st.error(f"🚨 Features (X) contain non-numeric columns: {', '.join(non_numeric_X_cols)}. All features must be numeric for modeling. Please go back to 'Advanced Preprocessing' and encode them.")
        if st.button("Go to Preprocessing to Encode"):
            st.session_state.active_tab = 2
            st.rerun()
        return
    
    # Clean column names for models like XGBoost
    X = clean_column_names_for_model(X)

    # Train-Test Split
    st.markdown("### 🔪 Train-Test Split")
    test_size = st.slider("Test Set Size (%):", 10, 50, 20, 5, key="test_size_slider")
    random_state_split = st.number_input("Random State for Split:", value=42, key="random_state_split")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state_split, stratify=y if task_type == "Classification" and y.nunique() > 1 else None)
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test
        st.write(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples.")
    except Exception as e:
        st.error(f"Error during train-test split: {e}. This might be due to too few samples in a class for stratification.")
        return

    st.markdown("--- ")
    st.markdown("### 🤖 Model Selection & Training")
    
    if task_type == "Classification":
        model_options = {
            "Logistic Regression": LogisticRegression(random_state=random_state_split, solver='liblinear'),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state_split),
            "Random Forest Classifier": RandomForestClassifier(random_state=random_state_split),
            "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=random_state_split),
            "AdaBoost Classifier": AdaBoostClassifier(random_state=random_state_split),
            "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "Support Vector Classifier (SVC)": SVC(probability=True, random_state=random_state_split),
            "XGBoost Classifier": xgb.XGBClassifier(random_state=random_state_split, use_label_encoder=False, eval_metric='logloss' if y_train.nunique() == 2 else 'mlogloss'),
            "Explainable Boosting Classifier (EBM)": ExplainableBoostingClassifier(random_state=random_state_split) if ExplainableBoostingClassifier else None
        }
    else: # Regression
        model_options = {
            "Linear Regression": LinearRegression(),
            "Lasso Regression": Lasso(random_state=random_state_split),
            "Ridge Regression": Ridge(random_state=random_state_split),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state_split),
            "Random Forest Regressor": RandomForestRegressor(random_state=random_state_split),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=random_state_split),
            "AdaBoost Regressor": AdaBoostRegressor(random_state=random_state_split),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
            "Support Vector Regressor (SVR)": SVR(),
            "XGBoost Regressor": xgb.XGBRegressor(random_state=random_state_split),
            "Explainable Boosting Regressor (EBM)": ExplainableBoostingRegressor(random_state=random_state_split) if ExplainableBoostingRegressor else None
        }
    model_options = {k: v for k, v in model_options.items() if v is not None} # Filter out None EBMs
    
    selected_model_name = st.selectbox("Select a Model:", list(model_options.keys()), key="model_selector")
    model = model_options[selected_model_name]

    # Hyperparameter Optimization (Optuna)
    use_hpo = st.checkbox("Tune Hyperparameters with Optuna? (Recommended, but slower)", value=False, key="use_hpo_checkbox")
    n_trials_optuna = 10
    if use_hpo:
        n_trials_optuna = st.slider("Number of Optuna Trials:", 5, 100, 20, 5, key="optuna_trials")

    if st.button(f"🚀 Train {selected_model_name}", use_container_width=True, key="train_model_button"):
        with st.spinner(f"Training {selected_model_name}..."):
            try:
                if use_hpo:
                    st.write(f"🔥 Optimizing {selected_model_name} with Optuna ({n_trials_optuna} trials)... This might take a while!")
                    study = optuna.create_study(direction="maximize" if task_type == "Classification" else "minimize")
                    
                    # Define objective function for Optuna
                    def objective(trial):
                        # HPO logic here - this needs to be model specific
                        # For simplicity, let's do RandomForest. This section needs to be expanded for each model.
                        if "Random Forest" in selected_model_name:
                            n_estimators = trial.suggest_int("n_estimators", 50, 300)
                            max_depth = trial.suggest_int("max_depth", 3, 20, log=True)
                            min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
                            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
                            model_hpo = model_options[selected_model_name].set_params(
                                n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf
                            )
                        elif "XGBoost" in selected_model_name:
                            param = {
                                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
                                'subsample': trial.suggest_categorical('subsample', [0.5,0.6,0.7,0.8,1.0]),
                                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                                'max_depth': trial.suggest_int('max_depth', 3, 12),
                                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                            }
                            if task_type == "Classification":
                                model_hpo = xgb.XGBClassifier(**param, random_state=random_state_split, use_label_encoder=False, eval_metric='logloss' if y_train.nunique() == 2 else 'mlogloss')
                            else:
                                model_hpo = xgb.XGBRegressor(**param, random_state=random_state_split)
                        # Add more model HPO configs here
                        else:
                            st.warning(f"HPO not yet configured for {selected_model_name}. Using default parameters.")
                            model_hpo = model # Fallback to default if no HPO config
                        
                        model_hpo.fit(X_train, y_train)
                        preds = model_hpo.predict(X_test)
                        if task_type == "Classification":
                            # Use F1-score for classification, or AUC if binary and probabilities available
                            if y_train.nunique() == 2 and hasattr(model_hpo, "predict_proba"):
                                probas = model_hpo.predict_proba(X_test)[:, 1]
                                return roc_auc_score(y_test, probas)
                            return f1_score(y_test, preds, average='weighted' if y_train.nunique() > 2 else 'binary', zero_division=0)
                        else:
                            return -mean_squared_error(y_test, preds) # Optuna minimizes, so negate MSE for maximization context or use direction='minimize'
                    
                    study.optimize(objective, n_trials=n_trials_optuna)
                    st.success(f"Optuna study complete! Best trial: {study.best_trial.value:.4f}")
                    st.write("Best hyperparameters:", study.best_params)
                    model.set_params(**study.best_params) # Set the main model to best params
                
                # Final model training (either with default or best HPO params)
                model.fit(X_train, y_train)
                st.session_state.trained_model = model
                st.session_state.model_name = selected_model_name
                st.success(f"✅ Model '{selected_model_name}' trained successfully!")

                # --- Evaluation ---
                st.markdown("### 📊 Model Evaluation on Test Set")
                y_pred = model.predict(X_test)

                if task_type == "Classification":
                    eval_classification(y_test, y_pred, model, X_test)
                else: # Regression
                    eval_regression(y_test, y_pred)
                
                # --- Prediction Preview ---
                st.markdown("### 🔮 Prediction Preview (First 20 Test Samples)")
                preview_df = pd.DataFrame({'Actual': y_test[:20].values, 'Predicted': y_pred[:20]})
                st.dataframe(preview_df)

                # --- Download Model ---
                st.markdown("### 💾 Download Trained Model")
                model_filename = f"{selected_model_name.replace(' ', '_').lower()}_model.pkl"
                st.download_button(
                    label=f"Download {model_filename}",
                    data=joblib.dump(model, model_filename, compress=3)[0], # joblib.dump returns a list of filenames
                    file_name=model_filename,
                    mime="application/octet-stream"
                )
                # Clean up the created file after preparing download data
                if os.path.exists(model_filename): os.remove(model_filename)

                # --- SHAP Summary (if applicable) ---
                if hasattr(model, 'predict') and not isinstance(model, (ExplainableBoostingClassifier, ExplainableBoostingRegressor)):
                    try:
                        with st.spinner("Generating SHAP summary plot..."):
                            explainer = shap.Explainer(model, X_train) # Use X_train for explainer for consistency
                            shap_values = explainer(X_test)
                            st.markdown("#### SHAP Feature Importance (Test Set)")
                            fig_shap, ax_shap = plt.subplots()
                            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                            st.pyplot(fig_shap)
                            plt.close(fig_shap) # Close plot to free memory
                    except Exception as e_shap:
                        st.warning(f"Could not generate SHAP summary plot: {e_shap}")

            except Exception as e_train:
                st.error(f"💥 Model training failed: {e_train}")
                # For debugging, print traceback
                # import traceback
                # st.code(traceback.format_exc())

    if st.button("➡️ Proceed to Model Explainability", use_container_width=True, key="proceed_to_xai"):
        if st.session_state.trained_model is None:
            st.error("No model has been trained yet. Please train a model first.")
        else:
            st.session_state.active_tab = 4
            st.rerun()

def eval_classification(y_true, y_pred, model, X_test_eval):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted' if y_true.nunique() > 2 else 'binary', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted' if y_true.nunique() > 2 else 'binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted' if y_true.nunique() > 2 else 'binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.markdown(f"- **Accuracy:** `{acc:.4f}`")
    st.markdown(f"- **F1-Score (Weighted):** `{f1:.4f}`")
    st.markdown(f"- **Precision (Weighted):** `{prec:.4f}`")
    st.markdown(f"- **Recall (Weighted):** `{rec:.4f}`")
    st.markdown(f"- **Matthews Corr Coef:** `{mcc:.4f}`")

    if hasattr(model, "predict_proba") and y_true.nunique() == 2: # ROC for binary classification
        y_proba = model.predict_proba(X_test_eval)[:, 1]
        roc_auc = roc_auc_score(y_true, y_proba)
        st.markdown(f"- **ROC AUC Score:** `{roc_auc:.4f}`")
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={roc_auc:.2f})', labels=dict(x='False Positive Rate', y='True Positive Rate'))
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_roc.update_xaxes(constrain='domain')
        st.plotly_chart(fig_roc, use_container_width=True)

    with st.expander("Classification Report Details"):
        report = classification_report(y_true, y_pred, zero_division=0)
        st.code(report)
    
    with st.expander("Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(y_true.unique())
        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=labels, y=labels, text_auto=True, aspect="auto",
                           color_continuous_scale=px.colors.sequential.Blues)
        fig_cm.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

def eval_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    st.markdown(f"- **Mean Absolute Error (MAE):** `{mae:.4f}`")
    st.markdown(f"- **Mean Squared Error (MSE):** `{mse:.4f}`")
    st.markdown(f"- **Root Mean Squared Error (RMSE):** `{rmse:.4f}`")
    st.markdown(f"- **R-squared (R²):** `{r2:.4f}`")

    # Scatter plot of Actual vs. Predicted
    fig_scatter = px.scatter(x=y_true, y=y_pred, labels={'x':'Actual Values', 'y':'Predicted Values'}, 
                             title='Actual vs. Predicted Values for Regression')
    fig_scatter.add_shape(type='line', x0=y_true.min(), y0=y_true.min(), x1=y_true.max(), y1=y_true.max(), line=dict(color='red', dash='dash'))
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 4: MODEL EXPLAINABILITY (XAI) ---
def render_tab4():
    st.markdown("### 🔍 Interpreting Your Trained Model")
    
    model = st.session_state.get("trained_model")
    model_name = st.session_state.get("model_name")
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train") # For LIME class names
    task_type = st.session_state.get("task_type")

    if model is None or X_train is None or X_test is None or y_train is None:
        st.warning("⚠️ Model and data not found. Please train a model in Tab 3 first, or ensure data is loaded.")
        if st.button("Go to Modeling Tab"):
            st.session_state.active_tab = 3
            st.rerun()
        return

    st.info(f"Explaining model: **{model_name}**")

    # EBM Specific Explanations
    if isinstance(model, (ExplainableBoostingClassifier, ExplainableBoostingRegressor)):
        st.subheader("🌟 Explainable Boosting Machine (EBM) Insights")
        try:
            with st.spinner("Generating EBM global explanations..."):
                ebm_global = model.explain_global(name=f'{model_name} Global Explanation')
                # The show function in interpret library might try to open a new browser window/tab or use a dashboard.
                # For Streamlit, we might need to capture plots if possible or guide user.
                # show(ebm_global) # This might not work well directly in Streamlit cloud
                st.markdown("EBM Global Explanations (feature importance and main effects):")
                # Try to render plots if visualize method returns plotly figures
                global_plots = ebm_global.visualize()
                if isinstance(global_plots, go.Figure):
                    st.plotly_chart(global_plots, use_container_width=True)
                elif isinstance(global_plots, list): # List of figures
                     for i, fig_item in enumerate(global_plots):
                        if isinstance(fig_item, go.Figure):
                            st.plotly_chart(fig_item, use_container_width=True)
                        else:
                            st.write(f"Plot item {i+1} is not a direct Plotly figure. Interpret's `show()` might be needed locally.")
                else:
                    st.write("EBM global explanation visualization might require running `show(ebm_global)` in a local environment if plots don't render here.")

            if st.checkbox("Show EBM Local Explanation (select a sample)", key="ebm_local_check"):
                 sample_idx_ebm = st.slider("Select a sample index from the test set for EBM local explanation:", 0, len(X_test) - 1, 0, key="ebm_sample_idx")
                 if sample_idx_ebm < len(X_test):
                    with st.spinner("Generating EBM local explanation..."):
                        ebm_local = model.explain_local(X_test.iloc[[sample_idx_ebm]], y_test.iloc[[sample_idx_ebm]] if y_test is not None else None, name=f'{model_name} Local Explanation')
                        # show(ebm_local)
                        local_plot = ebm_local.visualize()
                        if isinstance(local_plot, go.Figure):
                             st.plotly_chart(local_plot, use_container_width=True)
                        else:
                            st.write("EBM local explanation visualization might require `show(ebm_local)` locally.")
        except Exception as e_ebm:
            st.error(f"EBM explanation failed: {e_ebm}")
        return # EBM has its own rich explanations

    # SHAP Explanations (for other models)
    st.subheader("🌍 SHAP (SHapley Additive exPlanations)")
    try:
        with st.spinner("Calculating SHAP values... This can take time for complex models or large datasets."):
            # For tree models, TreeExplainer is faster. For others, KernelExplainer or just Explainer.
            if any(m_type in str(type(model)).lower() for m_type in ['xgb', 'forest', 'tree', 'gbm', 'lightgbm']):
                explainer = shap.TreeExplainer(model, X_train) # Pass X_train as background for TreeExplainer
            else:
                # KernelExplainer can be slow. Sample background data.
                background_data = shap.sample(X_train, min(100, X_train.shape[0])) 
                explainer = shap.KernelExplainer(model.predict, background_data)
            
            shap_values_test = explainer(X_test) # Get SHAP values for test set
            # For multi-class classification, shap_values_test might be a list of arrays
            # We need to handle this for plotting. Usually, we pick a class or sum absolute values.

        # Determine if shap_values_test is a list (multi-class) or single array (binary/regression)
        is_multiclass_shap = isinstance(shap_values_test.values, list) or (isinstance(shap_values_test.values, np.ndarray) and shap_values_test.values.ndim == 3)

        st.markdown("**Global Feature Importance (SHAP Summary Bar Plot):**")
        fig_shap_bar, ax_shap_bar = plt.subplots()
        if is_multiclass_shap:
            # For multiclass, shap.summary_plot often takes the SHAP values for a specific class, or sums abs values across classes
            # Simplest is to use the base_values and values for one class or average impact
            # Or, use shap_values.abs.mean(0) for global importance if it's a new explainer object
            # For now, let's try with the raw shap_values object, summary_plot might handle it.
            shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
        st.pyplot(fig_shap_bar)
        plt.close(fig_shap_bar)

        st.markdown("**SHAP Summary Dot Plot (Distribution of SHAP values):**")
        fig_shap_dot, ax_shap_dot = plt.subplots()
        shap.summary_plot(shap_values_test, X_test, plot_type="dot", show=False)
        st.pyplot(fig_shap_dot)
        plt.close(fig_shap_dot)

        st.markdown("**Individual Prediction Explanation (SHAP Waterfall/Force Plot):**")
        sample_idx_shap = st.slider("Select a sample index from the test set:", 0, len(X_test) - 1, 0, key="shap_sample_idx")
        
        if sample_idx_shap < len(X_test):
            shap_values_sample = shap_values_test[sample_idx_shap]
            
            st.markdown("Waterfall Plot:")
            fig_waterfall, ax_waterfall = plt.subplots()
            shap.waterfall_plot(shap_values_sample, show=False)
            st.pyplot(fig_waterfall)
            plt.close(fig_waterfall)

            # Force plot requires JS, st.html can be used
            # st.markdown("Force Plot (HTML - may require scrolling):")
            # force_plot_html = shap.force_plot(explainer.expected_value, shap_values_test.values[sample_idx_shap,:], X_test.iloc[sample_idx_shap,:], matplotlib=False, show=False)
            # st.components.v1.html(force_plot_html.html(), height=200, scrolling=True)
            # Matplotlib version of force plot:
            st.markdown("Force Plot (Matplotlib - single prediction):")
            fig_force_mpl, ax_force_mpl = plt.subplots(figsize=(10,3))
            shap.force_plot(explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value, 
                            shap_values_test.values[sample_idx_shap,:], 
                            X_test.iloc[sample_idx_shap,:], 
                            matplotlib=True, show=False)
            st.pyplot(fig_force_mpl)
            plt.close(fig_force_mpl)

    except Exception as e_shap:
        st.error(f"SHAP explanation failed: {e_shap}")
        # import traceback
        # st.code(traceback.format_exc())

    # LIME Explanations
    st.subheader("🍋 LIME (Local Interpretable Model-agnostic Explanations)")
    if not hasattr(model, "predict_proba") and task_type == "Classification":
        st.warning("Selected model does not have 'predict_proba' method, required by LIME for classification. LIME might not work as expected.")
    
    sample_idx_lime = st.slider("Select a sample index from the test set for LIME:", 0, len(X_test) - 1, 0, key="lime_sample_idx")
    if sample_idx_lime < len(X_test):
        try:
            with st.spinner("Generating LIME explanation..."):
                lime_explainer = LimeTabularExplainer(
                    training_data=X_train.values, # LIME uses numpy arrays
                    feature_names=X_train.columns.tolist(),
                    class_names=np.unique(y_train).astype(str).tolist() if task_type == "Classification" else None,
                    mode=task_type.lower(),
                    discretize_continuous=True
                )
                
                instance_to_explain = X_test.iloc[sample_idx_lime].values
                
                if task_type == "Classification" and hasattr(model, "predict_proba"):
                    lime_exp = lime_explainer.explain_instance(instance_to_explain, model.predict_proba, num_features=10, top_labels=1 if y_train.nunique() > 2 else None)
                elif task_type == "Classification": # No predict_proba, try predict (less ideal for LIME classification)
                     lime_exp = lime_explainer.explain_instance(instance_to_explain, lambda x: np.eye(len(np.unique(y_train)))[model.predict(x)], num_features=10, top_labels=1 if y_train.nunique() > 2 else None)
                else: # Regression
                    lime_exp = lime_explainer.explain_instance(instance_to_explain, model.predict, num_features=10)

                st.markdown(f"**LIME Explanation for Sample {sample_idx_shap} (Predicted: {model.predict(instance_to_explain.reshape(1, -1))[0]})**")
                # Show as HTML plot
                st.components.v1.html(lime_exp.as_html(), height=400, scrolling=True)
                # Or as matplotlib figure
                # fig_lime, ax_lime = lime_exp.as_pyplot_figure()
                # st.pyplot(fig_lime)
        except Exception as e_lime:
            st.error(f"LIME explanation failed: {e_lime}")
            # import traceback
            # st.code(traceback.format_exc())

    st.markdown("--- ")
    if st.button("🏠 Back to Home/Upload New Data", use_container_width=True, key="back_to_home_xai"):
        keys_to_clear = ["started", "active_tab", "original_df", "cleaned_df", "uploaded_file_name",
                             "X_train", "X_test", "y_train", "y_test", "trained_model", "model_name",
                             "target_column", "task_type", "preprocessing_pipeline"]
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()


