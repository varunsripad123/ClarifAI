# ClarifAI: Your Ultimate AI Co-Pilot

ClarifAI is an advanced, intuitive, end-to-end machine learning workbench designed to empower users to automate, explore, model, and explain their data with unprecedented ease and power. Built with Streamlit, it provides a comprehensive suite of tools for the entire data science lifecycle.

## Key Features

*   **Modern & Interactive UI:** A polished dark theme with interactive Plotly visualizations for an engaging user experience.
*   **Seamless Data Handling:** Upload CSV or Excel files, perform initial data cleaning (duplicates, renaming, null standardization, text processing).
*   **Advanced Exploratory Data Analysis (EDA):** 
    *   Interactive Plotly charts: Histograms, density plots, box plots, count plots, scatter plots, correlation heatmaps, and pair plots.
    *   Detailed dataset summaries: Data types, missing value analysis, and descriptive statistics for numerical and categorical features.
*   **Comprehensive Preprocessing Suite:**
    *   Advanced missing value imputation strategies.
    *   Multiple feature scaling techniques (StandardScaler, MinMaxScaler, RobustScaler, etc.).
    *   Flexible categorical encoding (Label, One-Hot, Ordinal).
    *   Outlier detection (IQR, Z-Score) and handling (cap, remove, mark).
    *   Feature engineering: Polynomial features, numerical interactions, and numerical feature binning.
    *   Dimensionality reduction: PCA and t-SNE.
    *   Class imbalance handling for classification tasks (SMOTE, RandomOverSampler, RandomUnderSampler).
    *   Trackable preprocessing pipeline.
*   **Powerful Modeling & Optimization:**
    *   Wide range of classification and regression models (Logistic/Linear Regression, Decision Trees, Random Forests, Gradient Boosting, AdaBoost, KNN, Naive Bayes, SVC/SVR, XGBoost, EBM).
    *   **Hyperparameter Optimization with Optuna:** Automatically tune model hyperparameters for optimal performance.
    *   Clear train-test split and model evaluation metrics (Accuracy, F1, Precision, Recall, MCC, ROC AUC for classification; MAE, MSE, RMSE, R² for regression).
    *   Interactive evaluation plots (ROC curves, Actual vs. Predicted).
    *   Download trained models.
*   **In-depth Model Explainability (XAI):**
    *   **SHAP (SHapley Additive exPlanations):** Global and local feature importance, summary plots (bar, dot), waterfall plots, and force plots.
    *   **LIME (Local Interpretable Model-agnostic Explanations):** Local explanations for individual predictions.
    *   **EBM (Explainable Boosting Machine) Insights:** Specific global and local explanation plots for EBM models.

## How to Run ClarifAI

1.  **Prerequisites:**
    *   Python 3.8+ installed.
    *   A virtual environment is highly recommended.

2.  **Setup:**
    *   Clone this repository or download the `enhanced_autonexus_v2.py` (rename to `clarifai_app.py` or similar if you wish) and `requirements.txt` files into a directory.
    *   Navigate to the directory in your terminal.
    *   Create and activate a virtual environment:
        ```bash
        python -m venv venv
        # On Windows
        venv\Scripts\activate
        # On macOS/Linux
        source venv/bin/activate
        ```
    *   Install the required dependencies:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Launch the Application:**
    ```bash
    streamlit run clarifai_app.py
    ```

    The application will open in your default web browser.

## Dependencies

All major dependencies are listed in `requirements.txt`. Key libraries include:
*   Streamlit
*   Pandas, NumPy
*   Scikit-learn
*   Plotly, Matplotlib, Seaborn
*   XGBoost
*   SHAP
*   LIME
*   Optuna
*   Interpret (for EBM)
*   Imbalanced-learn
