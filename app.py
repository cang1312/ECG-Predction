from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import pandas as pd
from io import StringIO
from scipy.io import loadmat
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = keras.models.load_model('Hybrid_ClassWeighted_BEST.keras')

# ECG classes (adjust based on your model)
ECG_CLASSES = [
    'Normal',
    'Atrial Fibrillation',
    'Atrial Flutter', 
    'Ventricular Tachycardia',
    'Ventricular Fibrillation'
]

@app.route('/predict', methods=['POST'])
def predict_ecg():
    try:
        data = request.get_json()
        
        # Expect ECG data as array
        ecg_data = np.array(data['ecg_data'])
        
        # Reshape if needed (adjust based on your model input shape)
        if len(ecg_data.shape) == 1:
            ecg_data = ecg_data.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(ecg_data)
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        result = {
            'predicted_class': ECG_CLASSES[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                ECG_CLASSES[i]: float(prediction[0][i]) 
                for i in range(len(ECG_CLASSES))
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file.filename.endswith('.mat'):
            # Handle .mat files
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mat')
            try:
                file.save(temp_file.name)
                temp_file.close()  # Close file before reading
                mat_data = loadmat(temp_file.name)
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                
                # Try common ECG data keys
                ecg_data = None
                for key in ['ecg', 'data', 'signal', 'val']:
                    if key in mat_data:
                        ecg_data = mat_data[key].flatten().tolist()
                        break
                
                if ecg_data is None:
                    # Use first numeric array found
                    for key, value in mat_data.items():
                        if not key.startswith('__') and isinstance(value, np.ndarray):
                            ecg_data = value.flatten().tolist()
                            break
                
                if ecg_data is None:
                    return jsonify({'error': 'No ECG data found in .mat file'}), 400
        else:
            content = file.read().decode('utf-8')
            
            if file.filename.endswith('.csv'):
                # Parse CSV - extract first column only
                df = pd.read_csv(StringIO(content), header=None)
                ecg_data = df.iloc[:, 0].values.tolist() if len(df) > 0 else []
            elif file.filename.endswith('.json'):
                # Parse JSON
                data = json.loads(content)
                ecg_data = data if isinstance(data, list) else data.get('data', [])
            else:
                # Parse as comma-separated text
                ecg_data = [float(x.strip()) for x in content.split(',') if x.strip()]
        
        # Make prediction
        ecg_array = np.array(ecg_data).reshape(1, -1)
        prediction = model.predict(ecg_array)
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        result = {
            'predicted_class': ECG_CLASSES[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                ECG_CLASSES[i]: float(prediction[0][i]) 
                for i in range(len(ECG_CLASSES))
            },
            'data_points': len(ecg_data)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)