import React, { useState } from 'react';
import axios from 'axios';

interface PredictionResult {
  predicted_class: string;
  confidence: number;
  all_probabilities: Record<string, number>;
}

const ECGPredictor: React.FC = () => {
  const [ecgData, setEcgData] = useState<string>('');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [fileName, setFileName] = useState<string>('');

  const handlePredict = async () => {
    if (!ecgData.trim()) {
      setError('Please enter ECG data');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const dataArray = ecgData.split(',').map(val => parseFloat(val.trim()));
      
      const response = await axios.post('http://localhost:5000/predict', {
        ecg_data: dataArray
      });
      
      setPrediction(response.data);
    } catch {
      setError('Prediction failed. Please check your data format.');
    } finally {
      setLoading(false);
    }
  };

  const generateSampleData = () => {
    // Generate more realistic ECG-like data
    const sampleData = [];
    const length = 187;
    
    for (let i = 0; i < length; i++) {
      // Create ECG-like pattern with P, QRS, T waves
      const t = (i / length) * 4 * Math.PI; // 2 heartbeats
      let value = 0;
      
      // P wave
      value += 0.1 * Math.sin(t * 0.8) * Math.exp(-Math.pow((t % (2 * Math.PI) - 0.5), 2) / 0.1);
      
      // QRS complex
      value += 0.8 * Math.sin(t * 3) * Math.exp(-Math.pow((t % (2 * Math.PI) - Math.PI), 2) / 0.05);
      
      // T wave
      value += 0.2 * Math.sin(t * 0.6) * Math.exp(-Math.pow((t % (2 * Math.PI) - 1.8 * Math.PI), 2) / 0.2);
      
      // Add some noise
      value += (Math.random() - 0.5) * 0.05;
      
      sampleData.push(value);
    }
    
    setEcgData(sampleData.map(v => v.toFixed(4)).join(', '));
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    
    if (file.name.endsWith('.mat')) {
      // Handle .mat files via backend upload
      const formData = new FormData();
      formData.append('file', file);
      
      setLoading(true);
      axios.post('http://localhost:5000/upload', formData)
        .then(response => {
          setPrediction(response.data);
          setError('');
        })
        .catch(err => {
          const errorMsg = err.response?.data?.error || 'Failed to process .mat file.';
          setError(`Error: ${errorMsg}`);
          console.error('Upload error:', err.response?.data);
        })
        .finally(() => {
          setLoading(false);
        });
      return;
    }
    
    const reader = new FileReader();
    
    reader.onload = (e) => {
      const content = e.target?.result as string;
      
      try {
        if (file.name.endsWith('.json')) {
          const jsonData = JSON.parse(content);
          const dataArray = Array.isArray(jsonData) ? jsonData : jsonData.data;
          setEcgData(dataArray.join(', '));
        } else if (file.name.endsWith('.csv')) {
          const lines = content.split('\n').filter(line => line.trim());
          const values = lines.map(line => {
            const firstValue = line.split(',')[0].trim();
            return parseFloat(firstValue);
          }).filter(val => !isNaN(val));
          setEcgData(values.join(', '));
        } else {
          setEcgData(content.trim());
        }
        setError('');
      } catch {
        setError('Failed to parse file. Please check format.');
      }
    };
    
    reader.readAsText(file);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h2>ECG Arrhythmia Prediction</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <label>
          ECG Data (comma-separated values):
          <textarea
            value={ecgData}
            onChange={(e) => setEcgData(e.target.value)}
            rows={5}
            cols={80}
            style={{ 
              width: '100%', 
              marginTop: '10px',
              padding: '10px',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
            placeholder="Enter ECG data as comma-separated values..."
          />
        </label>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>Upload ECG File (CSV/JSON/TXT/MAT):</label>
          <input
            type="file"
            accept=".csv,.json,.txt,.mat"
            onChange={handleFileUpload}
            style={{
              padding: '8px',
              border: '1px solid #ccc',
              borderRadius: '4px',
              marginRight: '10px'
            }}
          />
          {fileName && <span style={{ color: '#28a745' }}>✓ {fileName}</span>}
        </div>
        
        <button 
          onClick={generateSampleData}
          style={{
            padding: '10px 20px',
            marginRight: '10px',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Generate ECG Sample
        </button>
        
        <button 
          onClick={handlePredict}
          disabled={loading}
          style={{
            padding: '10px 20px',
            backgroundColor: loading ? '#ccc' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>

      {error && (
        <div style={{ 
          color: 'red', 
          marginBottom: '20px',
          padding: '10px',
          backgroundColor: '#f8d7da',
          border: '1px solid #f5c6cb',
          borderRadius: '4px'
        }}>
          {error}
        </div>
      )}

      {prediction && (
        <div style={{ 
          marginTop: '20px',
          padding: '20px',
          backgroundColor: '#161718ff',
          border: '1px solid #dee2e6',
          borderRadius: '4px'
        }}>
          <h3>Prediction Result</h3>
          <p><strong>Predicted Class:</strong> {prediction.predicted_class}</p>
          <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
          
          <h4>All Probabilities:</h4>
          <ul>
            {Object.entries(prediction.all_probabilities).map(([className, prob]) => (
              <li key={className}>
                {className}: {(prob * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ECGPredictor;