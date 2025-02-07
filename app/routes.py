from flask import render_template, request, jsonify
from app import app
from app.model_utils import predict_mental_health
from app.llm_utils import explain_mental_health_prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        prediction_result = predict_mental_health(data)
        explanation = explain_mental_health_prediction(data, prediction_result)
        
        return jsonify({
            'success': True,
            'prediction': prediction_result['risk_level'],
            'confidence': prediction_result['confidence'],
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

