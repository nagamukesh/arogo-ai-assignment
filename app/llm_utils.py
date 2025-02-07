def explain_mental_health_prediction(data, prediction_result):
    """
    Generate an explanation for the mental health prediction
    """
    try:
        explanation_parts = []
        
        # Add risk level explanation
        explanation_parts.append(f"Based on your responses, you show {prediction_result['risk_level'].lower()} "
                               f"indicators of mental health concerns (confidence: {prediction_result['confidence']:.1f}%).")
        
        # Add specific score interpretations
        if prediction_result['phq_severity'] == 'Moderate to Severe':
            explanation_parts.append("Your depression screening score suggests moderate to severe symptoms.")
        
        if prediction_result['gad_severity'] == 'Moderate to Severe':
            explanation_parts.append("Your anxiety screening score indicates moderate to severe symptoms.")
        
        if prediction_result['sleep_concern'] == 'Yes':
            explanation_parts.append("Your responses suggest you may be experiencing excessive daytime sleepiness.")
            
        # Add recommendation
        explanation_parts.append("\nPlease note that this is not a medical diagnosis. "
                               "If you're experiencing mental health concerns, it's recommended to consult with "
                               "a qualified healthcare professional for a proper evaluation and support.")
        
        return " ".join(explanation_parts)
        
    except Exception as e:
        raise Exception(f"Explanation generation error: {str(e)}")
