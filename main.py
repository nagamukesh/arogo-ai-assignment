import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Check for NaN values
    print("NaN values in the dataset:")
    print(df.isna().sum())
    
    # Identify numeric and categorical columns
    numeric_features = ['school_year', 'age', 'bmi', 'phq_score', 'gad_score', 'epworth_score']
    categorical_features = ['gender', 'who_bmi', 'depression_severity', 'depression_diagnosis', 
                            'depression_treatment', 'anxiety_severity', 'anxiety_diagnosis', 
                            'anxiety_treatment']
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return df, preprocessor

def engineer_features(df):
    # Convert boolean columns to string
    boolean_columns = ['depression_diagnosis', 'depression_treatment', 'anxiety_diagnosis', 'anxiety_treatment']
    for col in boolean_columns:
        df[col] = df[col].astype(str)
    
    # Handle missing values in the target variable
    df = df.dropna(subset=['depressiveness'])
    
    # Ensure 'depressiveness' is treated as a categorical variable
    df['depressiveness'] = df['depressiveness'].astype('category')
    
    # For this example, we'll use 'depressiveness' as our target variable
    X = df.drop(['id', 'depressiveness', 'suicidal', 'anxiousness', 'sleepiness'], axis=1)
    y = df['depressiveness']
    
    return X, y

def train_and_evaluate_models(X, y, preprocessor):
    # Create a pipeline with preprocessor and classifier
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted'),
        'roc_auc': make_scorer(roc_auc_score, average='weighted', multi_class='ovr')
    }
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
    
    print("Cross-validation results:")
    for metric, scores in cv_results.items():
        if metric.startswith('test_'):
            print(f"{metric[5:]}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return model

# Main execution
if __name__ == "__main__":
    file_path = 'depression_anxiety_data.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Please check the file path.")
    else:
        # Load and preprocess data
        data, preprocessor = load_and_preprocess_data(file_path)
        
        # Engineer features
        X, y = engineer_features(data)
        
        # Check if there are enough samples
        if len(X) < 10:
            print("Error: Not enough samples. Please check your data.")
        else:
            # Train and evaluate models
            model = train_and_evaluate_models(X, y, preprocessor)
            
            print("Model training complete.")



