import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
model_name = "google/flan-t5-base"  # Replace with your fine-tuned medical LLM if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def explain_mental_health_prediction(input_data):
    # Construct the prompt
    prompt = f"""
    Analyze the following mental health assessment data and provide a detailed explanation:

    PHQ-9 score: {input_data['phq_score']}
    Depression severity: {input_data['depression_severity']}
    GAD-7 score: {input_data['gad_score']}
    Anxiety severity: {input_data['anxiety_severity']}
    Reported anxiousness: {input_data['anxiousness']}
    Depressiveness (model prediction): {input_data['depressiveness']}

    Please provide:
    1. Interpretation of the PHQ-9 and GAD-7 scores
    2. Analysis of depression and anxiety indicators
    3. Explanation of the model's prediction for depressiveness
    4. Recommendations for further evaluation

    Important: Emphasize that this is not a clinical diagnosis and recommend professional evaluation.
    """

    # Tokenize the input and generate the explanation
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=500, num_return_sequences=1, temperature=0.7)
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return explanation

# Example usage
input_data = {
    'phq_score': 9,
    'depression_severity': 'Mild',
    'gad_score': 11,
    'anxiety_severity': 'Moderate',
    'anxiousness': True,
    'depressiveness': False  # This would be the prediction from your trained model
}

explanation = explain_mental_health_prediction(input_data)
print(explanation)

# Function to suggest coping mechanisms
# def suggest_coping_mechanisms(condition):
#     prompt = f"""
#     Suggest some coping mechanisms and potential next steps for someone showing signs of {condition}. 
#     Include both immediate self-help strategies and recommendations for professional support.
#     """

#     inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
#     outputs = model.generate(**inputs, max_length=300, num_return_sequences=1, temperature=0.7)
#     suggestions = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     return suggestions

# # Example usage of coping mechanisms suggestion
# condition = "mild depression and moderate anxiety"
# coping_suggestions = suggest_coping_mechanisms(condition)
# print("\nSuggested coping mechanisms and next steps:")
# print(coping_suggestions)
