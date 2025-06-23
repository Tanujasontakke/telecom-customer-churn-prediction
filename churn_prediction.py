import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_kaggle_data(file_path='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """Load the Kaggle Telco Customer Churn dataset"""
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset: {df.shape}")
        
        # Display basic info about the dataset
        print(f"\nDataset columns: {list(df.columns)}")
        print(f"Dataset shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'")
        print("Please make sure the file is in the same directory as this script")
        print("You can download it from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def clean_and_preprocess_data(df):
    """Clean and preprocess the Kaggle telecom dataset"""
    
    print("=== Data Cleaning and Preprocessing ===")
    print(f"Initial dataset shape: {df.shape}")
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Convert column names to lowercase and replace spaces with underscores
    df_processed.columns = df_processed.columns.str.lower().str.replace(' ', '_')
    
    # Handle the target variable (Churn)
    if 'churn' in df_processed.columns:
        df_processed['churn'] = df_processed['churn'].map({'Yes': 1, 'No': 0})
    
    # Handle TotalCharges column (it's often stored as string with spaces)
    if 'totalcharges' in df_processed.columns:
        # Replace empty strings with NaN
        df_processed['totalcharges'] = df_processed['totalcharges'].replace(' ', np.nan)
        
        # Convert to numeric
        df_processed['totalcharges'] = pd.to_numeric(df_processed['totalcharges'], errors='coerce')
        
        # Fill NaN values with 0 (for customers with 0 tenure)
        df_processed['totalcharges'] = df_processed['totalcharges'].fillna(0)
    
    # Handle MonthlyCharges
    if 'monthlycharges' in df_processed.columns:
        df_processed['monthlycharges'] = pd.to_numeric(df_processed['monthlycharges'], errors='coerce')
    
    # Handle tenure
    if 'tenure' in df_processed.columns:
        df_processed['tenure'] = pd.to_numeric(df_processed['tenure'], errors='coerce')
    
    # Convert binary Yes/No columns to 1/0
    binary_columns = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
    for col in binary_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
    
    # Handle SeniorCitizen (already 0/1 in the dataset)
    if 'seniorcitizen' in df_processed.columns:
        df_processed['seniorcitizen'] = df_processed['seniorcitizen'].astype(int)
    
    # Check for missing values
    print(f"\nMissing values per column:")
    missing_values = df_processed.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Basic statistics
    print(f"\nDataset Info:")
    print(f"Total customers: {len(df_processed)}")
    if 'churn' in df_processed.columns:
        print(f"Churned customers: {df_processed['churn'].sum()} ({df_processed['churn'].mean():.1%})")
        print(f"Retained customers: {len(df_processed) - df_processed['churn'].sum()} ({1 - df_processed['churn'].mean():.1%})")
    
    # Display column information
    print(f"\nColumns in processed dataset:")
    for i, col in enumerate(df_processed.columns):
        print(f"{i+1:2d}. {col} ({df_processed[col].dtype})")
    
    return df_processed

def engineer_features(df):
    """Engineer additional features for better prediction"""
    
    print("\n=== Feature Engineering ===")
    
    df_engineered = df.copy()
    
    # 1. Average monthly charges per tenure (handle division by zero)
    df_engineered['avg_monthly_charges'] = np.where(
        df_engineered['tenure'] > 0,
        df_engineered['totalcharges'] / df_engineered['tenure'],
        df_engineered['monthlycharges']
    )
    
    # 2. Tenure categories
    df_engineered['tenure_group'] = pd.cut(df_engineered['tenure'], 
                                          bins=[0, 12, 24, 48, 72], 
                                          labels=['0-1_year', '1-2_years', '2-4_years', '4+_years'])
    
    # 3. Monthly charges categories
    df_engineered['charges_group'] = pd.cut(df_engineered['monthlycharges'], 
                                           bins=[0, 35, 65, 95, float('inf')], 
                                           labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # 4. Service count (number of additional services)
    service_cols = []
    possible_services = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 
                        'techsupport', 'streamingtv', 'streamingmovies']
    
    for col in possible_services:
        if col in df_engineered.columns:
            service_cols.append(col)
    
    df_engineered['service_count'] = 0
    for col in service_cols:
        df_engineered['service_count'] += (df_engineered[col] == 'Yes').astype(int)
    
    # 5. High-value customer indicator
    df_engineered['high_value_customer'] = (
        (df_engineered['monthlycharges'] > df_engineered['monthlycharges'].quantile(0.75)) & 
        (df_engineered['tenure'] > 24)
    ).astype(int)
    
    # 6. Risk score (combination of risk factors)
    risk_score = 0
    
    # Contract risk
    if 'contract' in df_engineered.columns:
        risk_score += (df_engineered['contract'] == 'Month-to-month').astype(int) * 3
    
    # Payment method risk
    if 'paymentmethod' in df_engineered.columns:
        risk_score += (df_engineered['paymentmethod'] == 'Electronic check').astype(int) * 2
    
    # Tenure risk
    risk_score += (df_engineered['tenure'] < 12).astype(int) * 2
    
    # Service risks
    if 'onlinesecurity' in df_engineered.columns:
        risk_score += (df_engineered['onlinesecurity'] == 'No').astype(int)
    
    if 'techsupport' in df_engineered.columns:
        risk_score += (df_engineered['techsupport'] == 'No').astype(int)
    
    df_engineered['risk_score'] = risk_score
    
    # 7. Customer lifetime value estimate
    df_engineered['estimated_clv'] = df_engineered['monthlycharges'] * df_engineered['tenure']
    
    # 8. Charges to tenure ratio
    df_engineered['charges_to_tenure_ratio'] = np.where(
        df_engineered['tenure'] > 0,
        df_engineered['monthlycharges'] / df_engineered['tenure'],
        df_engineered['monthlycharges']
    )
    
    print(f"Engineered features added:")
    print(f"- avg_monthly_charges: Average monthly spending")
    print(f"- tenure_group: Tenure categories")
    print(f"- charges_group: Monthly charges categories") 
    print(f"- service_count: Number of additional services ({len(service_cols)} services tracked)")
    print(f"- high_value_customer: High-value customer indicator")
    print(f"- risk_score: Combined risk factors")
    print(f"- estimated_clv: Customer lifetime value estimate")
    print(f"- charges_to_tenure_ratio: Monthly charges to tenure ratio")
    
    return df_engineered

def prepare_features(df):
    """Prepare features for machine learning models"""
    
    print("\n=== Feature Preparation ===")
    
    # Separate features and target
    X = df.drop(['customerid', 'churn'], axis=1, errors='ignore')
    y = df['churn'] if 'churn' in df.columns else None
    
    if y is None:
        print("Warning: 'churn' column not found in dataset")
        return None, None, None
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    
    # Encode categorical variables
    le_dict = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle NaN values by converting to string first
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
        
        # Print unique values for verification
        unique_vals = X[col].unique()
        print(f"  {col}: {len(unique_vals)} unique values - {list(unique_vals)[:5]}{'...' if len(unique_vals) > 5 else ''}")
    
    return X_encoded, y, le_dict

def train_models(X, y):
    """Train Logistic Regression and Random Forest models"""
    
    print("\n=== Model Training ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    # Train models and store results
    results = {}
    
    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = models['Logistic Regression']
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'predictions': lr_pred,
        'probabilities': lr_prob,
        'accuracy': accuracy_score(y_test, lr_pred),
        'auc': roc_auc_score(y_test, lr_prob),
        'X_test': X_test_scaled
    }
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = models['Random Forest']
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'model': rf_model,
        'predictions': rf_pred,
        'probabilities': rf_prob,
        'accuracy': accuracy_score(y_test, rf_pred),
        'auc': roc_auc_score(y_test, rf_prob),
        'X_test': X_test
    }
    
    return results, X_train, X_test, y_train, y_test, scaler

def evaluate_models(results, y_test):
    """Evaluate model performance"""
    
    print("\n=== Model Evaluation ===")
    
    for model_name, result in results.items():
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"AUC-ROC: {result['auc']:.3f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, result['predictions']))

def identify_churn_factors(df, results, feature_names):
    """Identify top factors driving customer churn"""
    
    print("\n=== Top 5 Churn-Driving Factors ===")
    
    # Use Random Forest feature importance
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nRandom Forest Feature Importance (Top 10):")
    print(feature_importance.head(10).to_string(index=False))
    
    # Analyze top 5 factors
    top_5_factors = feature_importance.head(5)['feature'].tolist()
    
    print(f"\n=== Analysis of Top 5 Churn Factors ===")
    
    for i, factor in enumerate(top_5_factors, 1):
        print(f"\n{i}. {factor.upper()}:")
        
        if factor in df.columns:
            if df[factor].dtype in ['object', 'category']:
                # Categorical variable
                churn_by_category = df.groupby(factor)['churn'].agg(['count', 'sum', 'mean'])
                churn_by_category.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate']
                churn_by_category['Churn_Rate'] = churn_by_category['Churn_Rate'].round(3)
                print(churn_by_category)
            else:
                # Numerical variable
                churned = df[df['churn'] == 1][factor]
                retained = df[df['churn'] == 0][factor]
                print(f"   Average for churned customers: {churned.mean():.2f}")
                print(f"   Average for retained customers: {retained.mean():.2f}")
                print(f"   Difference: {churned.mean() - retained.mean():.2f}")
    
    return top_5_factors

def generate_retention_campaign_insights(df, top_factors):
    """Generate insights for retention campaign"""
    
    print("\n=== RETENTION CAMPAIGN RECOMMENDATIONS ===")
    
    print("\n1. CONTRACT TYPE STRATEGY:")
    contract_churn = df.groupby('contract')['churn'].mean()
    print("   - Month-to-month customers have highest churn rate")
    print("   - Offer incentives for longer-term contracts")
    print("   - Provide flexible upgrade paths from monthly to annual plans")
    
    print("\n2. TENURE-BASED INTERVENTIONS:")
    print("   - Focus on customers in first 12 months (highest risk period)")
    print("   - Implement onboarding programs for new customers")
    print("   - Create milestone rewards for tenure achievements")
    
    print("\n3. SERVICE OPTIMIZATION:")
    print("   - Customers without online security/tech support show higher churn")
    print("   - Offer free security add-ons as retention incentives")
    print("   - Proactive tech support outreach for high-risk customers")
    
    print("\n4. PAYMENT METHOD IMPROVEMENTS:")
    print("   - Electronic check users have higher churn rates")
    print("   - Incentivize automatic payment methods")
    print("   - Offer discounts for auto-pay enrollment")
    
    print("\n5. PRICING STRATEGY:")
    print("   - Monitor customers with high monthly charges")
    print("   - Offer customized packages to reduce cost concerns")
    print("   - Implement loyalty discounts for long-term customers")

def create_visualizations(df, results, y_test):
    """Create visualizations for the analysis"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # 1. Churn distribution
    churn_counts = df['churn'].value_counts()
    churn_counts.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
    axes[0,0].set_title('Churn Distribution')
    axes[0,0].set_xlabel('Churn (0=No, 1=Yes)')
    axes[0,0].set_ylabel('Number of Customers')
    axes[0,0].tick_params(axis='x', rotation=0)
    
    # Add percentage labels
    total = len(df)
    for i, v in enumerate(churn_counts.values):
        axes[0,0].text(i, v + total*0.01, f'{v/total:.1%}', ha='center', va='bottom')
    
    # 2. Churn by contract type (if available)
    contract_col = 'contract' if 'contract' in df.columns else None
    if contract_col:
        churn_contract = df.groupby(contract_col)['churn'].mean()
        churn_contract.plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Churn Rate by Contract Type')
        axes[0,1].set_ylabel('Churn Rate')
        axes[0,1].tick_params(axis='x', rotation=45)
    else:
        axes[0,1].text(0.5, 0.5, 'Contract data\nnot available', 
                       ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Contract Analysis')
    
    # 3. Monthly charges distribution
    monthly_col = 'monthlycharges' if 'monthlycharges' in df.columns else None
    if monthly_col:
        churned = df[df['churn'] == 1][monthly_col]
        retained = df[df['churn'] == 0][monthly_col]
        
        axes[0,2].hist([retained, churned], bins=30, alpha=0.7, 
                       label=['Retained', 'Churned'], color=['skyblue', 'salmon'])
        axes[0,2].set_title('Monthly Charges Distribution')
        axes[0,2].set_xlabel('Monthly Charges')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].legend()
    else:
        axes[0,2].text(0.5, 0.5, 'Monthly charges\ndata not available', 
                       ha='center', va='center', transform=axes[0,2].transAxes)
        axes[0,2].set_title('Monthly Charges Analysis')
    
    # 4. Tenure distribution  
    tenure_col = 'tenure' if 'tenure' in df.columns else None
    if tenure_col:
        churned = df[df['churn'] == 1][tenure_col]
        retained = df[df['churn'] == 0][tenure_col]
        
        axes[1,0].hist([retained, churned], bins=30, alpha=0.7, 
                       label=['Retained', 'Churned'], color=['skyblue', 'salmon'])
        axes[1,0].set_title('Tenure Distribution')
        axes[1,0].set_xlabel('Tenure (months)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
    else:
        axes[1,0].text(0.5, 0.5, 'Tenure data\nnot available', 
                       ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Tenure Analysis')
    
    # 5. ROC Curves
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        axes[1,1].plot(fpr, tpr, linewidth=2, 
                       label=f'{model_name} (AUC = {result["auc"]:.3f})')
    
    axes[1,1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    axes[1,1].set_xlabel('False Positive Rate')
    axes[1,1].set_ylabel('True Positive Rate')
    axes[1,1].set_title('ROC Curves Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Feature importance (Random Forest)
    rf_model = results['Random Forest']['model']
    feature_names = rf_model.feature_names_in_ if hasattr(rf_model, 'feature_names_in_') else [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))]
    
    # Get top 10 features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    y_pos = np.arange(len(importance_df))
    axes[1,2].barh(y_pos, importance_df['importance'], color='lightgreen')
    axes[1,2].set_yticks(y_pos)
    axes[1,2].set_yticklabels(importance_df['feature'], fontsize=9)
    axes[1,2].set_xlabel('Feature Importance')
    axes[1,2].set_title('Top 10 Feature Importance (Random Forest)')
    axes[1,2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    axes[1,2].set_yticks(range(len(importance_df)))
    axes[1,2].set_yticklabels(importance_df['feature'])
    axes[1,2].set_xlabel('Feature Importance')
    axes[1,2].set_title('Top 10 Feature Importance (Random Forest)')
    
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.92)
    plt.show()

def main():
    """Main execution function"""
    
    print("=== TELECOM CUSTOMER CHURN PREDICTION PROJECT ===")
    print("=" * 55)
    
    # 1. Load Kaggle dataset
    print("Loading Kaggle Telco Customer Churn dataset...")
    df = load_kaggle_data()
    
    if df is None:
        print("Failed to load dataset. Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the current directory.")
        return None, None, None
    
    # 2. Clean and preprocess data
    df_clean = clean_and_preprocess_data(df)
    
    # 3. Engineer features
    df_engineered = engineer_features(df_clean)
    
    # 4. Prepare features for modeling
    X, y, label_encoders = prepare_features(df_engineered)
    
    if X is None or y is None:
        print("Failed to prepare features. Please check the dataset.")
        return None, None, None
    
    # 5. Train models
    results, X_train, X_test, y_train, y_test, scaler = train_models(X, y)
    
    # 6. Evaluate models
    evaluate_models(results, y_test)
    
    # 7. Identify churn factors
    feature_names = X.columns.tolist()
    top_factors = identify_churn_factors(df_engineered, results, feature_names)
    
    # 8. Generate campaign insights
    generate_retention_campaign_insights(df_engineered, top_factors)
    
    # 9. Create visualizations
    create_visualizations(df_engineered, results, y_test)
    
    print("\n=== PROJECT SUMMARY ===")
    print(f"✓ Dataset: {len(df)} customers analyzed")
    print(f"✓ Models trained: Logistic Regression & Random Forest")
    print(f"✓ Best accuracy achieved: {max([r['accuracy'] for r in results.values()]):.1%}")
    print(f"✓ Top churn factors identified: {len(top_factors)} key drivers")
    print(f"✓ Retention campaign strategy developed")
    
    return df_engineered, results, top_factors

# Run the complete project
if __name__ == "__main__":
    df_final, model_results, churn_factors = main()
