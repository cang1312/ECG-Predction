# ECG Arrhythmia Prediction Setup

## Backend Setup (Python)

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run Flask server:
```bash
python app.py
```
Server akan berjalan di http://localhost:5000

## Frontend Setup (React + TypeScript)

1. Install Node.js dependencies:
```bash
npm install
```

2. Run development server:
```bash
npm run dev
```
Frontend akan berjalan di http://localhost:5173

## Cara Penggunaan

1. Jalankan backend Python terlebih dahulu
2. Jalankan frontend React
3. Buka browser ke http://localhost:5173
4. Klik "Generate Sample Data" untuk data contoh atau masukkan data ECG manual
5. Klik "Predict" untuk mendapatkan hasil prediksi

## Format Data ECG

Data ECG harus berupa angka yang dipisahkan koma, contoh:
```
0.1, -0.2, 0.3, -0.1, 0.5, ...
```

Model mengharapkan input dengan panjang tertentu (sesuaikan dengan model Keras Anda).