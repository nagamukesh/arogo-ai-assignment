import joblib
import pandas as pd
import numpy as np
import os

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Path to your saved model
MODEL_PATH = "/home/nagamukesh/arogo-ai-assignment/model/mental_health_model.pkl"
print(MODEL_PATH)
def calculate_bmi(height, weight):
    """Calculate BMI from height (in cm) and weight (in kg)"""
    height_m = height / 100  # convert cm to m
    bmi = weight / (height_m * height_m)
    return round(bmi, 2)

def get_who_bmi_category(bmi):
    """Determine WHO BMI category"""
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

def preprocess_data(data):
    """
    Preprocess the input data to match the format used during training
    """
    try:
        # Calculate BMI and WHO BMI category
        bmi = calculate_bmi(data['height'], data['weight'])
        who_bmi = get_who_bmi_category(bmi)
        
        # Create a DataFrame with all required features
        processed_data = {
            'school_year': data['school_year'],
            'age': data['age'],
            'gender': data['gender'],
            'bmi': bmi,
            'who_bmi': who_bmi,
            'phq_score': data['phq_score'],
            'gad_score': data['gad_score'],
            'epworth_score': data['epworth_score'],
            'depression_severity': 'Moderate' if data['phq_score'] >= 10 else 'Mild',
            'depression_diagnosis': 'True' if data['phq_score'] >= 15 else 'False',
            'depression_treatment': 'True' if data['phq_score'] >= 15 else 'False',
            'anxiety_severity': 'Moderate' if data['gad_score'] >= 10 else 'Mild',
            'anxiety_diagnosis': 'True' if data['gad_score'] >= 15 else 'False',
            'anxiety_treatment': 'True' if data['gad_score'] >= 15 else 'False'
        }
        
        # Create DataFrame
        df = pd.DataFrame([processed_data])
        
        return df
        
    except Exception as e:
        raise Exception(f"Data preprocessing error: {str(e)}")

def predict_mental_health(data):
    """
    Make predictions using the loaded model
    """
    try:
        logger.debug(f"Received data: {data}")
        
        # Ensure data is in the correct format (DataFrame)
        X = preprocess_data(data)
        logger.debug(f"Preprocessed data: {X}")
        
        model_path = MODEL_PATH
        logger.debug(f"Attempting to load model from: {model_path}")
        
        if os.path.exists(model_path):
            logger.info(f"Model file found at {model_path}")
            pipeline = joblib.load(model_path)
            logger.debug("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Make prediction using the loaded pipeline
        logger.debug("Making prediction")
        prediction = pipeline.predict(X)[0]
        logger.debug(f"Raw prediction: {prediction}")
        
        # Convert prediction to boolean
        is_depressive = bool(prediction)
        logger.debug(f"Converted prediction: {is_depressive}")
        
        result = {
            'depressiveness': is_depressive,
            'phq_severity': 'Moderate to Severe' if data['phq_score'] >= 10 else 'Mild',
            'gad_severity': 'Moderate to Severe' if data['gad_score'] >= 10 else 'Mild',
            'sleep_concern': 'Yes' if data['epworth_score'] >= 10 else 'No'
        }
        logger.info(f"Prediction result: {result}")
        
        return result
        
    except Exception as e:
        logger.exception(f"An error occurred during prediction: {str(e)}")
        raise
