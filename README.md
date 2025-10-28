# ECG Arrhythmia Prediction App

Aplikasi web untuk prediksi aritmia EKG menggunakan machine learning dengan React + TypeScript frontend dan Python Flask backend.

## 🚀 Fitur

- **Upload File**: Mendukung format CSV, JSON, TXT, dan MAT
- **Input Manual**: Masukkan data EKG secara manual
- **Generate Sample**: Buat data EKG sintetis untuk testing
- **Real-time Prediction**: Prediksi menggunakan model Keras
- **Confidence Score**: Tampilkan tingkat kepercayaan prediksi
- **Multiple Classes**: Deteksi berbagai jenis aritmia

## 🏗️ Teknologi

**Frontend:**
- React 19 + TypeScript
- Vite (build tool)
- Axios (HTTP client)

**Backend:**
- Python Flask
- TensorFlow/Keras
- NumPy, Pandas, SciPy

## 📦 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/cang1312/ECG-Predction.git
cd ECG-Predction
```

### 2. Setup Backend (Python)
```bash
pip install -r requirements.txt
```

### 3. Setup Frontend (Node.js)
```bash
npm install
```

## 🚀 Menjalankan Aplikasi

### Otomatis (Windows)
```bash
start.bat
```

### Manual
**Terminal 1 (Backend):**
```bash
python app.py
```

**Terminal 2 (Frontend):**
```bash
npm run dev
```

## 📊 Format Data

**CSV (1 kolom):**
```
0.1
-0.2
0.3
```

**JSON:**
```json
{"data": [0.1, -0.2, 0.3, ...]}
```

**MAT (MATLAB):**
- Key: `ecg`, `data`, `signal`, atau `val`

## 🎯 Kelas Prediksi

- Normal
- Atrial Fibrillation
- Atrial Flutter
- Ventricular Tachycardia
- Ventricular Fibrillation

## 📝 Lisensi

MIT License
