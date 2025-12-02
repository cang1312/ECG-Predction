import streamlit as st
import wfdb
import numpy as np
import onnxruntime as ort
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import plotly.graph_objects as go
import json
from datetime import datetime
import io
import base64
from scipy.signal import find_peaks, resample, welch, butter, filtfilt
from collections import deque
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# ===============================
# LOAD MODEL ONNX
# ===============================
@st.cache_resource
def load_onnx_model():
    model_files = [
        ('onnx_model/fine_tuned_ecg_model.onnx', 'onnx_model/fine_tuned_label_encoder.pkl', " Fine-tuned ONNX Model (91.43% Acc)"),
    ]
    
    for model_file, encoder_file, msg in model_files:
        if os.path.exists(model_file) and os.path.exists(encoder_file):
            try:
                session = ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
                encoder = joblib.load(encoder_file)
                st.sidebar.success(msg)
                return session, encoder
            except Exception as e:
                st.sidebar.error(f"❌ Model load error: {e}")
                continue
    
    st.sidebar.error("❌ No ONNX model found!")
    return None, None

# ===============================
# GET RECORD PATH
# ===============================
def get_record_path(record_id, dataset_type):
    if dataset_type == "MIT-BIH Arrhythmia (100-234)":
        db = "selected_records/mit-bih-arrhythmia/"
    else:
        db = "selected_records/mit-bih-supraventricular/"
    
    record_path = f"{db}{record_id}"

    for ext in [".hea", ".dat", ".atr"]:
        if not os.path.exists(record_path + ext):
            raise FileNotFoundError(f"File missing: {record_path+ext}")

    return record_path

# ===============================
# LOAD ECG DATA
# ===============================
@st.cache_data
def load_ecg_data(record_path):
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    return record, annotation

# ===============================
# EXPORT FUNCTIONS
# ===============================
def export_ecg_data(record, annotation, selected_record):
    """Export ECG raw data to CSV"""
    signal = record.p_signal[:, 0]
    time_axis = np.arange(len(signal)) / record.fs
    
    df = pd.DataFrame({
        'Time (s)': time_axis,
        'ECG Signal (mV)': signal,
        'Sample': np.arange(len(signal))
    })
    
    # Add R-peaks if available
    if annotation is not None:
        r_peak_times = annotation.sample / record.fs
        r_peak_df = pd.DataFrame({
            'R_Peak_Time (s)': r_peak_times,
            'R_Peak_Sample': annotation.sample,
            'Annotation': annotation.symbol
        })
        return df.to_csv(index=False), r_peak_df.to_csv(index=False)
    
    return df.to_csv(index=False), None

def export_analysis_results(hr_stats, rr_intervals, pred_labels, confidence, selected_record):
    """Export analysis results to JSON"""
    results = {
        'record_id': selected_record,
        'timestamp': datetime.now().isoformat(),
        'heart_rate_analysis': {
            'avg_hr': float(hr_stats['avg_hr']),
            'min_hr': float(hr_stats['min_hr']),
            'max_hr': float(hr_stats['max_hr']),
            'rr_mean_ms': float(hr_stats['rr_mean']),
            'rr_std_ms': float(hr_stats['rr_std']),
            'total_beats': int(hr_stats['total_beats'])
        },
        'rr_intervals': [float(x) for x in rr_intervals],
        'ai_analysis': {
            'predictions': pred_labels.tolist(),
            'confidence_scores': [float(x) for x in confidence],
            'avg_confidence': float(np.mean(confidence)),
            'total_beats': len(pred_labels),
            'abnormal_beats': len([p for p in pred_labels if p != "Normal"])
        },
        'arrhythmia_distribution': {str(k): int(v) for k, v in dict(zip(*np.unique(pred_labels, return_counts=True))).items()}
    }
    
    return json.dumps(results, indent=2)

def create_summary_report(record, hr_stats, pred_labels, confidence, selected_record):
    """Create summary report text"""
    unique, counts = np.unique(pred_labels, return_counts=True)
    normal_pct = (counts[unique == "Normal"][0] / len(pred_labels) * 100) if "Normal" in unique else 0
    
    report = f"""
# ECG ANALYSIS REPORT

**Record ID:** {selected_record}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Duration:** {len(record.p_signal)/record.fs:.1f} seconds
**Sampling Rate:** {record.fs} Hz

## HEART RATE ANALYSIS
- **Average HR:** {hr_stats['avg_hr']:.1f} BPM
- **Min HR:** {hr_stats['min_hr']:.1f} BPM  
- **Max HR:** {hr_stats['max_hr']:.1f} BPM
- **RR Mean:** {hr_stats['rr_mean']:.1f} ms
- **RR Std (SDNN):** {hr_stats['rr_std']:.1f} ms
- **Total Beats:** {hr_stats['total_beats']}

## AI ARRHYTHMIA ANALYSIS
- **Total Beats Analyzed:** {len(pred_labels)}
- **Average Confidence:** {np.mean(confidence):.3f}
- **Normal Beats:** {normal_pct:.1f}%
- **Abnormal Beats:** {100-normal_pct:.1f}%

## ARRHYTHMIA DISTRIBUTION
"""
    
    for arrhythmia, count in zip(unique, counts):
        percentage = count / len(pred_labels) * 100
        report += f"- **{arrhythmia}:** {count} beats ({percentage:.1f}%)\n"
    
    report += f"""

## MEDICAL ASSESSMENT
"""
    
    if normal_pct > 90:
        report += "✅ **NORMAL RHYTHM** - No immediate concern\n"
        report += "📋 **Recommendation:** Regular monitoring, maintain healthy lifestyle\n"
    elif normal_pct > 70:
        report += "⚠️ **MILD ARRHYTHMIA** - Monitoring recommended\n"
        report += "📋 **Recommendation:** Consult cardiologist, avoid excessive caffeine/stress\n"
    else:
        report += "🚨 **SIGNIFICANT ARRHYTHMIA** - Medical consultation required\n"
        report += "📋 **Recommendation:** Immediate medical attention, detailed cardiac evaluation\n"
    
    report += "\n⚠️ **Disclaimer:** For educational purposes only. Not for clinical use.\n"
    
    return report

# ===============================
# REAL-TIME R-PEAK DETECTION
# ===============================
def detect_rpeak_realtime(signal_window, fs, threshold=0.6):
    """Real-time R-peak detection using adaptive threshold"""
    if len(signal_window) < fs // 4:  # Need at least 0.25s of data
        return []
    
    # Simple peak detection with adaptive threshold
    peaks, _ = find_peaks(signal_window, 
                         height=threshold * np.max(signal_window),
                         distance=int(0.4 * fs))  # Min 0.4s between peaks
    return peaks

def extract_beat_realtime(signal, peak_idx, fs, window_size=0.4):
    """Extract single beat around R-peak for real-time inference"""
    half_window = int(window_size * fs / 2)
    start = max(0, peak_idx - half_window)
    end = min(len(signal), peak_idx + half_window)
    
    beat = signal[start:end]
    
    # Resample to target length (144 samples)
    if len(beat) > 0:
        beat = resample(beat, 144)
    
    return beat

def predict_beat_realtime(beat, session, encoder):
    """Real-time beat classification using ONNX model"""
    if session is None or len(beat) != 144:
        return "Unknown", 0.0
    
    try:
        X = beat.reshape(1, 144, 1).astype(np.float32)
        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: X})[0]
        
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred, axis=1)[0]
        label = encoder.inverse_transform([pred_class])[0]
        
        return label, confidence
    except:
        return "Error", 0.0

# Initialize real-time buffers
if "realtime_buffer" not in st.session_state:
    st.session_state.realtime_buffer = deque(maxlen=1000)
if "rpeak_buffer" not in st.session_state:
    st.session_state.rpeak_buffer = deque(maxlen=50)
if "beat_predictions" not in st.session_state:
    st.session_state.beat_predictions = deque(maxlen=100)

# ===============================
# REAL-TIME ECG WITH R-PEAK & AI
# ===============================
def smooth_realtime_ecg_with_ai():
    if "ecg_data" not in st.session_state:
        st.error("Please load ECG data first")
        return
    
    signal = st.session_state.ecg_data["signal"]
    fs = st.session_state.ecg_data["fs"]
    
    # Load AI model for real-time inference
    session, encoder = load_onnx_model()
    
    # Initialize state with memory management
    if "rt_running" not in st.session_state:
        st.session_state.rt_running = False
    if "rt_index" not in st.session_state:
        st.session_state.rt_index = 0
    if "perf_mode" not in st.session_state:
        st.session_state.perf_mode = "balanced"
    if "frame_times" not in st.session_state:
        st.session_state.frame_times = []
    if "data_buffer" not in st.session_state:
        st.session_state.data_buffer = {}
    if "chart_template" not in st.session_state:
        st.session_state.chart_template = None
    if "chart_initialized" not in st.session_state:
        st.session_state.chart_initialized = False
    
    # Controls with performance selector
    col1, col2, col3, col4, col5 = st.columns(5)
    
    start_btn = col1.button("▶️ Start")
    pause_btn = col2.button("⏸️ Pause") 
    reset_btn = col3.button("🔄 Reset")
    speed = col4.slider("Speed", 0.5, 3.0, 1.0, 0.1)
    perf_mode = col5.selectbox("Quality", ["fast", "balanced", "high"], index=1)
    
    # Clear buffer when mode changes
    if st.session_state.perf_mode != perf_mode:
        st.session_state.data_buffer.clear()
        st.session_state.frame_times.clear()
        st.session_state.chart_template = None
        st.session_state.chart_initialized = False
    st.session_state.perf_mode = perf_mode
    
    if start_btn:
        st.session_state.rt_running = True
        st.session_state.frame_times.clear()
    if pause_btn:
        st.session_state.rt_running = False
    if reset_btn:
        st.session_state.rt_running = False
        st.session_state.rt_index = 0
        st.session_state.data_buffer.clear()
        st.session_state.frame_times.clear()
        st.session_state.chart_template = None
        st.session_state.chart_initialized = False
    
    # Enhanced performance settings
    perf_config = {
        "fast": {"window": 2, "downsample": 4, "line_width": 1, "grid": False, "height": 350, "target_fps": 15, "step_mult": 1.5},
        "balanced": {"window": 3, "downsample": 2, "line_width": 2, "grid": True, "height": 400, "target_fps": 12, "step_mult": 1.0},
        "high": {"window": 4, "downsample": 1, "line_width": 2, "grid": True, "height": 450, "target_fps": 10, "step_mult": 0.8}
    }
    
    config = perf_config[perf_mode]
    
    # Separate containers to prevent full page rerender
    chart_container = st.empty()
    stats_container = st.empty()
    progress_container = st.empty()
    predictions_container = st.empty()
    
    # Real-time loop with frequency tuning and memory management
    if st.session_state.rt_running:
        frame_start = time.time()
        
        # Adaptive step size based on performance
        base_step = int(fs * 0.08 * config["step_mult"] / speed)
        
        # Adjust based on actual FPS
        if len(st.session_state.frame_times) > 5:
            avg_time = np.mean(st.session_state.frame_times[-5:])
            actual_fps = 1 / avg_time if avg_time > 0 else config["target_fps"]
            
            if actual_fps < config["target_fps"] * 0.8:
                step_size = int(base_step * 1.2)
            elif actual_fps > config["target_fps"] * 1.2:
                step_size = int(base_step * 0.8)
            else:
                step_size = base_step
        else:
            step_size = base_step
        
        step_size = max(1, step_size)
        window_size = int(config["window"] * fs)
        
        if st.session_state.rt_index < len(signal) - window_size:
            start_idx = st.session_state.rt_index
            end_idx = start_idx + window_size
            
            # Memory-efficient data processing
            buffer_key = f"{start_idx}_{config['downsample']}"
            
            if buffer_key in st.session_state.data_buffer:
                seg, time_seg = st.session_state.data_buffer[buffer_key]
            else:
                seg = signal[start_idx:end_idx:config["downsample"]]
                time_seg = np.arange(len(seg)) * config["downsample"] / fs
                
                # Real-time R-peak detection
                rpeaks = detect_rpeak_realtime(seg, fs // config["downsample"])
                
                # Beat-per-beat AI inference
                for peak in rpeaks:
                    actual_peak_idx = start_idx + peak * config["downsample"]
                    if actual_peak_idx not in [p[0] for p in st.session_state.rpeak_buffer]:
                        beat = extract_beat_realtime(signal, actual_peak_idx, fs)
                        if len(beat) == 144:
                            label, conf = predict_beat_realtime(beat, session, encoder)
                            st.session_state.rpeak_buffer.append((actual_peak_idx, label, conf))
                            st.session_state.beat_predictions.append({"time": actual_peak_idx/fs, "label": label, "confidence": conf})
                
                # Cache with size limit
                if len(st.session_state.data_buffer) < 3:
                    st.session_state.data_buffer[buffer_key] = (seg, time_seg)
                else:
                    oldest = next(iter(st.session_state.data_buffer))
                    del st.session_state.data_buffer[oldest]
                    st.session_state.data_buffer[buffer_key] = (seg, time_seg)
            
            # Template-based chart optimization
            template_key = f"template_{perf_mode}"
            
            # Create or reuse chart template
            if not st.session_state.chart_initialized or st.session_state.chart_template is None or template_key not in st.session_state.data_buffer:
                # Create new template
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line=dict(width=config["line_width"], color="#00ff41", simplify=True),
                    showlegend=False,
                    hoverinfo='skip' if perf_mode == "fast" else 'x+y'
                ))
                
                # Configure layout once
                layout_config = {
                    "title": {
                        'text': "Real-time ECG Monitor",
                        'x': 0.5,
                        'font': {'size': 16 if perf_mode == "fast" else 20, 'color': '#00ff41'}
                    },
                    "xaxis_title": "Time (s)" if perf_mode == "fast" else "Time (seconds)",
                    "yaxis_title": "mV" if perf_mode == "fast" else "Amplitude (mV)",
                    "height": config["height"],
                    "paper_bgcolor": "#000000",
                    "plot_bgcolor": "#001100",
                    "font": dict(color="#00ff41", size=10 if perf_mode == "fast" else 12),
                    "margin": dict(l=40, r=40, t=50, b=40),
                    "showlegend": False
                }
                
                if config["grid"]:
                    layout_config.update({
                        "xaxis": dict(gridcolor="#003300", range=[0, config["window"]], showgrid=True, zeroline=True, zerolinecolor="#00ff41"),
                        "yaxis": dict(gridcolor="#003300", range=[signal.min() * 1.1, signal.max() * 1.1], showgrid=True, zeroline=True, zerolinecolor="#00ff41")
                    })
                else:
                    layout_config.update({
                        "xaxis": dict(range=[0, config["window"]], showgrid=False, zeroline=False),
                        "yaxis": dict(range=[signal.min() * 1.1, signal.max() * 1.1], showgrid=False, zeroline=False)
                    })
                
                fig.update_layout(**layout_config)
                
                # Cache template
                st.session_state.chart_template = fig
                st.session_state.data_buffer[template_key] = layout_config
                st.session_state.chart_initialized = True
            else:
                # Reuse existing template
                fig = st.session_state.chart_template
            
            # Update ECG data
            fig.data[0].x = time_seg
            fig.data[0].y = seg
            
            # Add R-peaks overlay if detected
            if len(st.session_state.rpeak_buffer) > 0:
                # Get R-peaks in current window
                window_rpeaks = [(idx, label, conf) for idx, label, conf in st.session_state.rpeak_buffer 
                                if start_idx <= idx < end_idx]
                
                if window_rpeaks and len(fig.data) == 1:  # Add R-peak trace
                    fig.add_trace(go.Scatter(
                        x=[], y=[], mode='markers',
                        marker=dict(color='red', size=8, symbol='x'),
                        name='R-peaks', showlegend=False
                    ))
                
                if len(fig.data) > 1 and window_rpeaks:
                    rpeak_times = [(idx - start_idx) * config["downsample"] / fs for idx, _, _ in window_rpeaks]
                    rpeak_values = [seg[int((idx - start_idx) // config["downsample"])] for idx, _, _ in window_rpeaks if int((idx - start_idx) // config["downsample"]) < len(seg)]
                    
                    fig.data[1].x = rpeak_times
                    fig.data[1].y = rpeak_values
            
            # Optimized chart config - disable all animations
            chart_config = {
                'displayModeBar': False,
                'staticPlot': True,
                'responsive': True,
                'doubleClick': False,
                'scrollZoom': False,
                'animation': {'duration': 0}  # No animation for all modes
            }
            
            # Stable key that changes less frequently
            stable_key = f"ecg_{perf_mode}_{st.session_state.rt_index//500}"
            chart_container.plotly_chart(fig, use_container_width=True, config=chart_config, key=stable_key)
            
            # Update frequency tuning
            st.session_state.rt_index += step_size
            
            # Calculate and track frame time
            frame_time = time.time() - frame_start
            st.session_state.frame_times.append(frame_time)
            if len(st.session_state.frame_times) > 10:
                st.session_state.frame_times.pop(0)
            
            # Optimized timing for smoother animation
            target_time = 1.0 / config["target_fps"]
            sleep_time = max(0.02, target_time - frame_time) / speed
            
            time.sleep(sleep_time)
            st.rerun()
        else:
            st.session_state.rt_running = False
            st.session_state.data_buffer.clear()
            st.session_state.chart_template = None
            st.session_state.chart_initialized = False
            with chart_container.container():
                st.success("✅ Monitoring completed!")
    
    # Enhanced progress display with real-time stats - separate containers
    if st.session_state.rt_index > 0:
        progress = min(st.session_state.rt_index / len(signal), 1.0)
        
        # Update stats container independently
        with stats_container.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if len(st.session_state.rpeak_buffer) > 0:
                    current_hr = len(st.session_state.rpeak_buffer) * 60 / (st.session_state.rt_index / fs) if st.session_state.rt_index > 0 else 0
                    st.metric("Current HR", f"{current_hr:.0f} BPM")
                else:
                    st.metric("Current HR", "-- BPM")
            
            with col2:
                if len(st.session_state.beat_predictions) > 0:
                    recent_predictions = list(st.session_state.beat_predictions)[-10:]
                    abnormal_count = sum(1 for p in recent_predictions if p["label"] != "Normal")
                    st.metric("Abnormal (last 10)", f"{abnormal_count}/10")
                else:
                    st.metric("Abnormal (last 10)", "0/0")
            
            with col3:
                if len(st.session_state.beat_predictions) > 0:
                    avg_conf = np.mean([p["confidence"] for p in st.session_state.beat_predictions])
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                else:
                    st.metric("Avg Confidence", "--")
        
        # Update progress container independently
        with progress_container.container():
            if perf_mode == "high" and len(st.session_state.frame_times) > 0:
                avg_time = np.mean(st.session_state.frame_times[-5:]) if len(st.session_state.frame_times) >= 5 else np.mean(st.session_state.frame_times)
                fps = 1 / avg_time if avg_time > 0 else 0
                st.progress(progress, f"Progress: {progress*100:.1f}% | FPS: {fps:.1f} | R-peaks: {len(st.session_state.rpeak_buffer)}")
            else:
                st.progress(progress, f"Progress: {progress*100:.1f}% | R-peaks detected: {len(st.session_state.rpeak_buffer)}")
        
        # Update predictions container independently
        if len(st.session_state.beat_predictions) > 0:
            with predictions_container.container():
                st.subheader("🔬 Real-time AI Predictions")
                recent_df = pd.DataFrame(list(st.session_state.beat_predictions)[-10:])
                recent_df["time"] = recent_df["time"].round(2)
                recent_df["confidence"] = recent_df["confidence"].round(3)
                st.dataframe(recent_df, use_container_width=True)

# ===============================
# ADVANCED ANALYTICS
# ===============================
def analyze_hrv_frequency_domain(rr_intervals, fs=4):
    """HRV frequency domain analysis"""
    if len(rr_intervals) < 50:
        return {}
    
    # Resample RR intervals to uniform time series
    time_rr = np.cumsum(rr_intervals)
    uniform_time = np.arange(0, time_rr[-1], 1/fs)
    rr_uniform = np.interp(uniform_time, time_rr[:-1], rr_intervals[:-1])
    
    # Remove DC component
    rr_uniform = rr_uniform - np.mean(rr_uniform)
    
    # Power spectral density
    freqs, psd = welch(rr_uniform, fs=fs, nperseg=min(256, len(rr_uniform)//4))
    
    # Frequency bands
    vlf_band = (freqs >= 0.003) & (freqs < 0.04)  # Very Low Frequency
    lf_band = (freqs >= 0.04) & (freqs < 0.15)    # Low Frequency
    hf_band = (freqs >= 0.15) & (freqs < 0.4)     # High Frequency
    
    vlf_power = np.trapz(psd[vlf_band], freqs[vlf_band])
    lf_power = np.trapz(psd[lf_band], freqs[lf_band])
    hf_power = np.trapz(psd[hf_band], freqs[hf_band])
    
    total_power = vlf_power + lf_power + hf_power
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    return {
        'vlf_power': vlf_power,
        'lf_power': lf_power, 
        'hf_power': hf_power,
        'total_power': total_power,
        'lf_hf_ratio': lf_hf_ratio,
        'lf_norm': lf_power / (lf_power + hf_power) * 100 if (lf_power + hf_power) > 0 else 0,
        'hf_norm': hf_power / (lf_power + hf_power) * 100 if (lf_power + hf_power) > 0 else 0,
        'freqs': freqs,
        'psd': psd
    }

def detect_qt_intervals(signal, r_peaks, fs):
    """QT interval detection and analysis"""
    qt_intervals = []
    
    for i, r_peak in enumerate(r_peaks[:-1]):
        # Define search window for T wave (after R peak)
        search_start = r_peak + int(0.1 * fs)  # 100ms after R
        search_end = min(r_peaks[i+1] - int(0.05 * fs), r_peak + int(0.5 * fs))  # Before next R or max 500ms
        
        if search_end > search_start:
            segment = signal[search_start:search_end]
            
            # Find T wave end (return to baseline)
            baseline = np.mean(signal[max(0, r_peak-int(0.1*fs)):r_peak])
            
            # Simple T wave end detection
            for j in range(len(segment)-1, 0, -1):
                if abs(segment[j] - baseline) < 0.1 * np.std(segment):
                    t_end = search_start + j
                    qt_interval = (t_end - r_peak) / fs * 1000  # in ms
                    if 200 < qt_interval < 600:  # Valid QT range
                        qt_intervals.append(qt_interval)
                    break
    
    return np.array(qt_intervals)

def analyze_st_segment(signal, r_peaks, fs):
    """ST segment analysis for ischemia detection"""
    st_deviations = []
    
    for r_peak in r_peaks:
        # ST segment: 80ms after R peak
        st_point = r_peak + int(0.08 * fs)
        
        if st_point < len(signal):
            # Baseline: before R peak
            baseline_start = max(0, r_peak - int(0.1 * fs))
            baseline_end = max(0, r_peak - int(0.02 * fs))
            
            if baseline_end > baseline_start:
                baseline = np.mean(signal[baseline_start:baseline_end])
                st_level = signal[st_point]
                st_deviation = (st_level - baseline) * 1000  # in mV
                st_deviations.append(st_deviation)
    
    return np.array(st_deviations)

def trend_analysis(values, times):
    """Trend analysis using linear regression"""
    if len(values) < 3:
        return {'slope': 0, 'r_value': 0, 'trend': 'stable'}
    
    X = times.reshape(-1, 1)
    model = LinearRegression().fit(X, values)
    slope = model.coef_[0]
    r_value, _ = pearsonr(times, values)
    
    if abs(slope) < 0.01:
        trend = 'stable'
    elif slope > 0:
        trend = 'increasing'
    else:
        trend = 'decreasing'
    
    return {'slope': slope, 'r_value': r_value, 'trend': trend}

# ===============================
# CALCULATE HEART RATE
# ===============================
def calculate_heart_rate(r_peaks, fs, signal_duration):
    """Calculate heart rate from R-peaks"""
    if len(r_peaks) < 2:
        return 0, [], []
    
    # Calculate RR intervals (time between consecutive R-peaks)
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    
    # Calculate instantaneous heart rate
    instantaneous_hr = 60 / rr_intervals  # beats per minute
    
    # Calculate average heart rate
    total_beats = len(r_peaks)
    duration_minutes = signal_duration / 60
    avg_heart_rate = total_beats / duration_minutes
    
    # Heart rate variability metrics
    rr_mean = np.mean(rr_intervals) * 1000  # in ms
    rr_std = np.std(rr_intervals) * 1000    # in ms (SDNN)
    
    hr_stats = {
        'avg_hr': avg_heart_rate,
        'min_hr': np.min(instantaneous_hr),
        'max_hr': np.max(instantaneous_hr),
        'rr_mean': rr_mean,
        'rr_std': rr_std,
        'total_beats': total_beats
    }
    
    return hr_stats, rr_intervals, instantaneous_hr

# ===============================
# EXTRACT BEATS 
# ===============================
def extract_beats(signal, r_peaks, fs, window_size=0.4):
    if fs == 128:
        target_length = 58
    else:
        target_length = 144
        
    half_window = int(window_size * fs / 2)
    beats = []

    for peak in r_peaks:
        start = peak - half_window
        end = peak + half_window
        if start >= 0 and end < len(signal):
            beat = signal[start:end]

            if len(beat) != target_length:
                from scipy import signal as scipy_signal
                beat = scipy_signal.resample(beat, target_length)

            beats.append(beat)

    return np.array(beats), target_length

# ===============================
# ANALYZE ECG WITH AI
# ===============================
def analyze_ecg_with_ai():
    if "ecg_data" not in st.session_state:
        st.error("Please load ECG data first")
        return
    
    record = st.session_state.ecg_data["record"]
    annotation = st.session_state.ecg_data["annotation"]
    
    session, encoder = load_onnx_model()
    if session is None:
        st.error("❌ AI Model not available. Please check if onnx_model files are uploaded.")
        return None, None, None

    beats, target_length = extract_beats(record.p_signal[:, 0], annotation.sample, record.fs)

    valid_beats = [b for b in beats if len(b) == target_length]
    if not valid_beats:
        raise Exception("No valid beats found")

    X = np.array(valid_beats)

    if target_length == 58:
        from scipy import signal as scipy_signal
        X = np.array([scipy_signal.resample(x, 144) for x in X])

    X = X.reshape(len(X), 144, 1).astype(np.float32)

    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: X})[0]

    pred_class = np.argmax(preds, axis=1)
    pred_labels = encoder.inverse_transform(pred_class)
    confidence = np.max(preds, axis=1)

    return pred_labels, confidence, valid_beats

# ===============================
# MAIN APP
# ===============================
st.set_page_config(page_title="ECG Monitor", layout="wide")

# CSS optimization - disable transitions
st.markdown("""
<style>
/* Disable all transitions and animations */
* {
    transition: none !important;
    animation: none !important;
}

/* Optimize chart rendering */
.js-plotly-plot {
    transition: none !important;
}

/* Reduce reflow */
.stMetric {
    transition: none !important;
}

.stProgress {
    transition: none !important;
}

/* Stable layout */
.main .block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title(" Real-time ECG Monitor & AI Detector")

# Sidebar
st.sidebar.header("📁 ECG File Selection")
dataset_type = st.sidebar.radio(
    "Dataset:",
    ["MIT-BIH Arrhythmia (100-234)", "MIT-BIH Supraventricular (800-894)"]
)

record_options = (
    ['100','102','103','200','201','203','221','233']
    if dataset_type == "MIT-BIH Arrhythmia (100-234)"
    else
    ['800','801','860','879','880','881','882','883','884','886']
)

selected_record = st.sidebar.selectbox("ECG Record:", record_options)  

# Load ECG
if st.sidebar.button("📂 Load ECG"):
    try:
        record_path = get_record_path(selected_record, dataset_type)
        record, annotation = load_ecg_data(record_path)
        
        # Store in session state
        st.session_state.ecg_data = {
            "record": record,
            "annotation": annotation,
            "signal": record.p_signal[:, 0],
            "fs": record.fs
        }
        
        # Reset real-time state
        st.session_state.rt_running = False
        st.session_state.rt_index = 0
        
        st.sidebar.success(f"✅ Loaded: {selected_record}")
        st.sidebar.info(f"📊 Length: {len(record.p_signal)} samples")
        st.sidebar.info(f"⏱️ Duration: {len(record.p_signal)/record.fs:.1f} seconds")
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

# Main content
if "ecg_data" in st.session_state:
    tab1, tab2, tab3 = st.tabs([" Real-time Monitor", " AI Analysis", " ECG Paper Strip"])
    
    with tab1:
        st.subheader("Real-time ECG Monitor")
        st.info(" real-time ECG monitoring with hospital-grade visualization")
        smooth_realtime_ecg_with_ai()
    
    with tab2:
        st.subheader("AI Arrhythmia Analysis")
        
        if st.button(" Generate ECG Strip"):
            with st.spinner("Generating ECG paper strip..."):
                try:
                    # Get ECG data
                    record = st.session_state.ecg_data["record"]
                    annotation = st.session_state.ecg_data["annotation"]
                    signal = record.p_signal[:, 0]
                    fs = record.fs
                    
                    # Create paper-like ECG strip
                    duration = min(30, len(signal) / fs)  # Max 30 seconds
                    samples = int(duration * fs)
                    time_axis = np.arange(samples) / fs
                    ecg_segment = signal[:samples]
                    
                    # Get AI predictions for this segment
                    session, encoder = load_onnx_model()
                    if session:
                        # Get R-peaks in segment
                        segment_rpeaks = annotation.sample[annotation.sample < samples]
                        
                        # Extract beats and predict
                        beats, _ = extract_beats(ecg_segment, segment_rpeaks, fs)
                        if len(beats) > 0:
                            valid_beats = [b for b in beats if len(b) == 144 or len(b) == 58]
                            if valid_beats:
                                X = np.array(valid_beats)
                                if X.shape[1] == 58:
                                    from scipy import signal as scipy_signal
                                    X = np.array([scipy_signal.resample(x, 144) for x in X])
                                X = X.reshape(len(X), 144, 1).astype(np.float32)
                                
                                input_name = session.get_inputs()[0].name
                                preds = session.run(None, {input_name: X})[0]
                                pred_class = np.argmax(preds, axis=1)
                                pred_labels = encoder.inverse_transform(pred_class)
                                confidence = np.max(preds, axis=1)
                    
                    # Create ECG paper strip figure
                    fig = go.Figure()
                    
                    # ECG signal
                    fig.add_trace(go.Scatter(
                        x=time_axis,
                        y=ecg_segment,
                        mode='lines',
                        line=dict(color='black', width=1.5),
                        name='ECG',
                        showlegend=False
                    ))
                    
                    # Add R-peaks
                    rpeak_times = segment_rpeaks / fs
                    rpeak_values = ecg_segment[segment_rpeaks]
                    
                    fig.add_trace(go.Scatter(
                        x=rpeak_times,
                        y=rpeak_values,
                        mode='markers',
                        marker=dict(color='red', size=6, symbol='circle'),
                        name='R-peaks',
                        showlegend=False
                    ))
                    
                    # Add abnormality annotations
                    if 'pred_labels' in locals():
                        for i, (rpeak_time, label, conf) in enumerate(zip(rpeak_times[:len(pred_labels)], pred_labels, confidence)):
                            if label != "Normal":
                                fig.add_annotation(
                                    x=rpeak_time,
                                    y=max(ecg_segment) * 1.2,
                                    text=f"{label}<br>{conf:.2f}",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor="red",
                                    bgcolor="yellow",
                                    bordercolor="red",
                                    font=dict(size=10, color="red")
                                )
                    
                    # ECG paper grid styling
                    fig.update_layout(
                        title=f"ECG Paper Strip - Record {selected_record} ({duration:.1f}s)",
                        xaxis=dict(
                            title="Time (seconds)",
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="lightcoral",
                            minor=dict(showgrid=True, gridwidth=0.5, gridcolor="pink"),
                            dtick=0.2,  # Major grid every 0.2s (5 small squares)
                            range=[0, duration]
                        ),
                        yaxis=dict(
                            title="Amplitude (mV)",
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="lightcoral",
                            minor=dict(showgrid=True, gridwidth=0.5, gridcolor="pink"),
                            dtick=0.5,  # Major grid every 0.5mV
                            range=[min(ecg_segment)*1.2, max(ecg_segment)*1.4]
                        ),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        width=1200,
                        height=400,
                        font=dict(color="black", size=12)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary of abnormalities
                    if 'pred_labels' in locals():
                        abnormal_beats = [(i, label, conf) for i, (label, conf) in enumerate(zip(pred_labels, confidence)) if label != "Normal"]
                        
                        if abnormal_beats:
                            st.subheader("🚨 Detected Abnormalities")
                            
                            abnormal_df = pd.DataFrame([
                                {
                                    "Beat #": i+1,
                                    "Time (s)": f"{rpeak_times[i]:.2f}",
                                    "Abnormality": label,
                                    "Confidence": f"{conf:.3f}",
                                    "Severity": "High" if conf > 0.8 else "Medium" if conf > 0.6 else "Low"
                                }
                                for i, label, conf in abnormal_beats
                            ])
                            
                            st.dataframe(abnormal_df, use_container_width=True)
                            
                            # Download ECG strip as image
                            img_bytes = fig.to_image(format="png", width=1200, height=400, scale=2)
                            st.download_button(
                                label="📥 Download ECG Strip (PNG)",
                                data=img_bytes,
                                file_name=f"ecg_strip_{selected_record}.png",
                                mime="image/png"
                            )
                        else:
                            st.success("✅ No abnormalities detected in this segment")
                    
                except Exception as e:
                    st.error(f"❌ Error generating ECG strip: {e}")
        
        st.markdown("---")
        
        if st.button(" Analyze with AI", key="ai_analysis_btn"):
            with st.spinner(" Analyzing ECG..."):
                try:
                    pred_labels, confidence, beats = analyze_ecg_with_ai()
                    
                    # Get record and annotation for heart rate calculation
                    record = st.session_state.ecg_data["record"]
                    annotation = st.session_state.ecg_data["annotation"]
                    
                    # Calculate heart rate
                    signal_duration = len(record.p_signal) / record.fs
                    hr_stats, rr_intervals, instantaneous_hr = calculate_heart_rate(
                        annotation.sample, record.fs, signal_duration
                    )
                    
                    # Results with heart rate
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Heart Rate", f"{hr_stats['avg_hr']:.0f} BPM")
                    col2.metric("Total Beats", len(pred_labels))
                    col3.metric("Avg Confidence", f"{np.mean(confidence):.3f}")
                    col4.metric("Abnormal Beats", len([p for p in pred_labels if p != "Normal"]))
                    
                    # Heart rate details
                    st.subheader("Heart Rate Analysis")
                    
                    hr_col1, hr_col2, hr_col3 = st.columns(3)
                    
                    with hr_col1:
                        st.metric("Average HR", f"{hr_stats['avg_hr']:.1f} BPM")
                        st.metric("Min HR", f"{hr_stats['min_hr']:.1f} BPM")
                        st.metric("Max HR", f"{hr_stats['max_hr']:.1f} BPM")
                    
                    with hr_col2:
                        st.metric("RR Mean", f"{hr_stats['rr_mean']:.1f} ms")
                        st.metric("RR Std (SDNN)", f"{hr_stats['rr_std']:.1f} ms")
                        st.metric("Total Beats", hr_stats['total_beats'])
                    
                    with hr_col3:
                        # HR Classification
                        if hr_stats['avg_hr'] < 60:
                            st.error("🔽 Bradycardia (< 60 BPM)")
                        elif hr_stats['avg_hr'] > 100:
                            st.error("🔼 Tachycardia (> 100 BPM)")
                        else:
                            st.success("✅ Normal HR (60-100 BPM)")
                        
                        # HRV Assessment
                        if hr_stats['rr_std'] < 20:
                            st.warning("⚠️ Low HRV")
                        elif hr_stats['rr_std'] > 50:
                            st.info("📈 High HRV")
                        else:
                            st.success("✅ Normal HRV")
                    
                    # Heart rate trend plot
                    if len(instantaneous_hr) > 1:
                        fig_hr = go.Figure()
                        
                        # Time points for HR (between R-peaks)
                        hr_times = annotation.sample[1:] / record.fs
                        
                        fig_hr.add_trace(go.Scatter(
                            x=hr_times,
                            y=instantaneous_hr,
                            mode='lines+markers',
                            name='Heart Rate',
                            line=dict(color='red', width=2),
                            marker=dict(size=4)
                        ))
                        
                        # Add average line
                        fig_hr.add_hline(
                            y=hr_stats['avg_hr'],
                            line_dash="dash",
                            line_color="blue",
                            annotation_text=f"Avg: {hr_stats['avg_hr']:.1f} BPM"
                        )
                        
                        fig_hr.update_layout(
                            title="Heart Rate Trend Over Time",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Heart Rate (BPM)",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_hr, width='stretch')
                    
                    # Classification table
                    unique, counts = np.unique(pred_labels, return_counts=True)
                    df = pd.DataFrame({
                        "Arrhythmia Type": unique,
                        "Count": counts,
                        "Percentage": np.round(counts / len(pred_labels) * 100, 1)
                    })
                    
                    st.subheader(" Detection Results")
                    
                    # Enhanced table with color coding
                    def color_rows(row):
                        if row['Arrhythmia Type'] == 'Normal':
                            return ['background-color:'] * len(row)
                        else:
                            return ['background-color:'] * len(row)
                    
                    styled_df = df.style.apply(color_rows, axis=1)
                    st.dataframe(styled_df, width='stretch')
                    
                    # RR Interval Analysis
                    st.subheader("RR Interval Analysis")
                    
                    rr_col1, rr_col2 = st.columns(2)
                    
                    with rr_col1:
                        # RR interval histogram
                        fig_rr = go.Figure(data=[go.Histogram(
                            x=rr_intervals * 1000,  # Convert to ms
                            nbinsx=30,
                            marker_color='green',
                            opacity=0.7
                        )])
                        fig_rr.update_layout(
                            title="RR Interval Distribution",
                            xaxis_title="RR Interval (ms)",
                            yaxis_title="Frequency",
                            height=350
                        )
                        st.plotly_chart(fig_rr, width='stretch')
                    
                    with rr_col2:
                        # Poincaré plot (RR interval scatter)
                        if len(rr_intervals) > 1:
                            rr1 = rr_intervals[:-1] * 1000  # RR(n)
                            rr2 = rr_intervals[1:] * 1000   # RR(n+1)
                            
                            fig_poincare = go.Figure(data=go.Scatter(
                                x=rr1,
                                y=rr2,
                                mode='markers',
                                marker=dict(color='purple', size=6, opacity=0.6),
                                name='RR Intervals'
                            ))
                            
                            fig_poincare.update_layout(
                                title="Poincare Plot (HRV)",
                                xaxis_title="RR(n) ms",
                                yaxis_title="RR(n+1) ms",
                                height=350
                            )
                            st.plotly_chart(fig_poincare, width='stretch')
                    
                    # RR Intervals Data Table
                    st.subheader("RR Intervals Data")
                    
                    # Create RR intervals dataframe
                    rr_data = []
                    for i, rr in enumerate(rr_intervals):
                        rr_data.append({
                            'Beat #': i + 1,
                            'RR Interval (ms)': f"{rr * 1000:.1f}",
                            'RR Interval (s)': f"{rr:.3f}",
                            'Instant HR (BPM)': f"{60/rr:.1f}" if i < len(instantaneous_hr) else "N/A",
                            'Time (s)': f"{annotation.sample[i+1] / record.fs:.2f}" if i+1 < len(annotation.sample) else "N/A"
                        })
                    
                    rr_df = pd.DataFrame(rr_data)
                    
                    # Show summary stats
                    rr_col1, rr_col2, rr_col3, rr_col4 = st.columns(4)
                    rr_col1.metric("Total RR Intervals", len(rr_intervals))
                    rr_col2.metric("Mean RR", f"{np.mean(rr_intervals)*1000:.1f} ms")
                    rr_col3.metric("Std RR (SDNN)", f"{np.std(rr_intervals)*1000:.1f} ms")
                    rr_col4.metric("RR Range", f"{(np.max(rr_intervals)-np.min(rr_intervals))*1000:.1f} ms")
                    
                    # Display RR intervals table with pagination
                    st.write("**RR Intervals Table:**")
                    
                    # Add download button for RR data
                    csv_rr = rr_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download RR Intervals CSV",
                        data=csv_rr,
                        file_name=f"rr_intervals_{selected_record}.csv",
                        mime="text/csv"
                    )
                    
                    # Show table (limit to first 50 rows for performance)
                    if len(rr_df) > 50:
                        st.dataframe(rr_df.head(50), width='stretch')
                        st.info(f"📊 Showing first 50 of {len(rr_df)} RR intervals. Download CSV for complete data.")
                    else:
                        st.dataframe(rr_df, width='stretch')
                    
                    # RR Intervals Time Series Plot
                    st.subheader("📈 RR Intervals Over Time")
                    
                    fig_rr_time = go.Figure()
                    
                    # Time points for RR intervals
                    rr_times = annotation.sample[1:len(rr_intervals)+1] / record.fs
                    
                    fig_rr_time.add_trace(go.Scatter(
                        x=rr_times,
                        y=rr_intervals * 1000,  # Convert to ms
                        mode='lines+markers',
                        name='RR Intervals',
                        line=dict(color='green', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add mean line
                    fig_rr_time.add_hline(
                        y=np.mean(rr_intervals) * 1000,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text=f"Mean: {np.mean(rr_intervals)*1000:.1f} ms"
                    )
                    
                    fig_rr_time.update_layout(
                        title="RR Intervals Trend Over Time",
                        xaxis_title="Time (seconds)",
                        yaxis_title="RR Interval (ms)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_rr_time, width='stretch')
                    
                    # Arrhythmia visualization plots
                    st.subheader("Arrhythmia Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart for arrhythmia distribution
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=unique,
                            values=counts,
                            hole=0.4,
                            marker_colors=['#28a745' if x == 'Normal' else '#dc3545' for x in unique]
                        )])
                        fig_pie.update_layout(
                            title="Arrhythmia Distribution",
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig_pie, width='stretch')
                    
                    with col2:
                        # Bar chart for counts
                        fig_bar = go.Figure(data=[go.Bar(
                            x=unique,
                            y=counts,
                            marker_color=['#28a745' if x == 'Normal' else '#dc3545' for x in unique],
                            text=counts,
                            textposition='auto'
                        )])
                        fig_bar.update_layout(
                            title="Beat Count by Type",
                            xaxis_title="Arrhythmia Type",
                            yaxis_title="Number of Beats",
                            height=400
                        )
                        st.plotly_chart(fig_bar, width='stretch')
                    
                    # Confidence analysis
                    st.subheader(" AI Confidence Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confidence histogram
                        fig_conf = go.Figure(data=[go.Histogram(
                            x=confidence,
                            nbinsx=20,
                            marker_color='#17a2b8',
                            opacity=0.7
                        )])
                        fig_conf.update_layout(
                            title="AI Confidence Distribution",
                            xaxis_title="Confidence Score",
                            yaxis_title="Frequency",
                            height=350
                        )
                        st.plotly_chart(fig_conf, width='stretch')
                    
                    with col2:
                        # Confidence by class
                        conf_by_class = []
                        for cls in unique:
                            cls_conf = confidence[pred_labels == cls]
                            conf_by_class.append({
                                'Class': cls,
                                'Avg Confidence': np.mean(cls_conf),
                                'Min Confidence': np.min(cls_conf),
                                'Max Confidence': np.max(cls_conf)
                            })
                        
                        conf_df = pd.DataFrame(conf_by_class)
                        st.write("**Confidence by Arrhythmia Type:**")
                        st.dataframe(conf_df.round(3), width='stretch')
                    
                    # Sample beats visualization
                    st.subheader("Sample Beat Patterns")
                    
                    fig_beats = go.Figure()
                    colors = ['#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6f42c1']
                    
                    for i, cls in enumerate(unique[:5]):  # Show max 5 classes
                        idx = np.where(pred_labels == cls)[0][0]
                        beat_sample = beats[idx]
                        
                        fig_beats.add_trace(go.Scatter(
                            x=np.arange(len(beat_sample)),
                            y=beat_sample,
                            mode='lines',
                            name=f'{cls} (Conf: {confidence[idx]:.3f})',
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                    
                    fig_beats.update_layout(
                        title="Representative Beat Patterns by Arrhythmia Type",
                        xaxis_title="Sample Points",
                        yaxis_title="Amplitude",
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_beats, width='stretch')
                    
                    # ECG signal with R-peaks
                    st.subheader("ECG Signal with Detected R-peaks")
                    
                    # Get record and annotation from session state
                    record = st.session_state.ecg_data["record"]
                    annotation = st.session_state.ecg_data["annotation"]
                    
                    # Show first 5000 samples for clarity
                    signal_segment = record.p_signal[:5000, 0]
                    time_segment = np.arange(len(signal_segment)) / record.fs
                    
                    # Find R-peaks in this segment
                    rpeak_mask = annotation.sample < 5000
                    rpeaks_segment = annotation.sample[rpeak_mask]
                    rpeak_times = rpeaks_segment / record.fs
                    rpeak_values = signal_segment[rpeaks_segment]
                    
                    fig_ecg = go.Figure()
                    
                    # ECG signal
                    fig_ecg.add_trace(go.Scatter(
                        x=time_segment,
                        y=signal_segment,
                        mode='lines',
                        name='ECG Signal',
                        line=dict(color='blue', width=1)
                    ))
                    
                    # R-peaks
                    fig_ecg.add_trace(go.Scatter(
                        x=rpeak_times,
                        y=rpeak_values,
                        mode='markers',
                        name='R-peaks',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                    
                    fig_ecg.update_layout(
                        title="ECG Signal with R-peak Detection (First 5000 samples)",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Amplitude (mV)",
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_ecg, width='stretch')
                    
                    # Medical assessment
                    normal_pct = df[df["Arrhythmia Type"] == "Normal"]["Percentage"].values[0] if "Normal" in df["Arrhythmia Type"].values else 0
                    
                    st.subheader("Medical Assessment")
                    
                    # Create assessment card
                    if normal_pct > 90:
                        st.success("✅ **NORMAL RHYTHM** - No immediate concern")
                        st.info("📋 **Recommendation:** Regular monitoring, maintain healthy lifestyle")
                    elif normal_pct > 70:
                        st.warning("⚠️ **MILD ARRHYTHMIA** - Monitoring recommended")
                        st.info("📋 **Recommendation:** Consult cardiologist, avoid excessive caffeine/stress")
                    else:
                        st.error("🚨 **SIGNIFICANT ARRHYTHMIA** - Medical consultation required")
                        st.info("📋 **Recommendation:** Immediate medical attention, detailed cardiac evaluation")
                    
                    # Detailed findings
                    st.subheader("📋 Detailed Findings")
                    
                    findings_data = []
                    for i, cls in enumerate(unique):
                        cls_count = counts[i]
                        cls_pct = df[df["Arrhythmia Type"] == cls]["Percentage"].values[0]
                        cls_conf = np.mean(confidence[pred_labels == cls])
                        
                        findings_data.append({
                            "Finding": cls,
                            "Occurrences": cls_count,
                            "Percentage": f"{cls_pct}%",
                            "Avg Confidence": f"{cls_conf:.3f}",
                            "Severity": "Low" if cls == "Normal" else "High" if cls_pct > 10 else "Medium"
                        })
                    
                    findings_df = pd.DataFrame(findings_data)
                    st.dataframe(findings_df, width='stretch')
                    
                    # Advanced Analytics Dashboard
                    st.subheader("📊 Advanced Analytics Dashboard")
                    
                    # HRV Frequency Domain Analysis
                    st.write("**🔬 HRV Frequency Domain Analysis**")
                    hrv_freq = analyze_hrv_frequency_domain(rr_intervals)
                    
                    if hrv_freq:
                        freq_col1, freq_col2, freq_col3 = st.columns(3)
                        
                        with freq_col1:
                            st.metric("LF Power", f"{hrv_freq['lf_power']:.1f} ms²")
                            st.metric("HF Power", f"{hrv_freq['hf_power']:.1f} ms²")
                        
                        with freq_col2:
                            st.metric("LF/HF Ratio", f"{hrv_freq['lf_hf_ratio']:.2f}")
                            st.metric("Total Power", f"{hrv_freq['total_power']:.1f} ms²")
                        
                        with freq_col3:
                            st.metric("LF Norm", f"{hrv_freq['lf_norm']:.1f}%")
                            st.metric("HF Norm", f"{hrv_freq['hf_norm']:.1f}%")
                        
                        # PSD Plot
                        fig_psd = go.Figure()
                        fig_psd.add_trace(go.Scatter(
                            x=hrv_freq['freqs'],
                            y=hrv_freq['psd'],
                            mode='lines',
                            fill='tozeroy',
                            name='PSD'
                        ))
                        
                        # Add frequency band markers
                        fig_psd.add_vrect(x0=0.04, x1=0.15, fillcolor="red", opacity=0.2, annotation_text="LF")
                        fig_psd.add_vrect(x0=0.15, x1=0.4, fillcolor="blue", opacity=0.2, annotation_text="HF")
                        
                        fig_psd.update_layout(
                            title="HRV Power Spectral Density",
                            xaxis_title="Frequency (Hz)",
                            yaxis_title="Power (ms²/Hz)",
                            height=300
                        )
                        st.plotly_chart(fig_psd, use_container_width=True)
                        
                        # HRV Assessment
                        if hrv_freq['lf_hf_ratio'] > 2.5:
                            st.warning("⚠️ High LF/HF ratio - Possible sympathetic dominance")
                        elif hrv_freq['lf_hf_ratio'] < 0.5:
                            st.info("📈 Low LF/HF ratio - Possible parasympathetic dominance")
                        else:
                            st.success("✅ Balanced autonomic function")
                    
                    # QT Interval Analysis
                    st.write("**⚡ QT Interval Analysis**")
                    qt_intervals = detect_qt_intervals(record.p_signal[:, 0], annotation.sample, record.fs)
                    
                    if len(qt_intervals) > 0:
                        qt_col1, qt_col2, qt_col3 = st.columns(3)
                        
                        qtc_bazett = qt_intervals / np.sqrt(rr_intervals[:len(qt_intervals)] / 1000)  # Bazett correction
                        
                        with qt_col1:
                            st.metric("Avg QT", f"{np.mean(qt_intervals):.0f} ms")
                            st.metric("QT Range", f"{np.max(qt_intervals) - np.min(qt_intervals):.0f} ms")
                        
                        with qt_col2:
                            st.metric("Avg QTc", f"{np.mean(qtc_bazett):.0f} ms")
                            st.metric("QTc Std", f"{np.std(qtc_bazett):.1f} ms")
                        
                        with qt_col3:
                            # QTc Assessment
                            avg_qtc = np.mean(qtc_bazett)
                            if avg_qtc > 450:
                                st.error("🚨 Prolonged QTc (>450ms)")
                            elif avg_qtc < 350:
                                st.warning("⚠️ Short QTc (<350ms)")
                            else:
                                st.success("✅ Normal QTc (350-450ms)")
                        
                        # QT Trend Plot
                        fig_qt = go.Figure()
                        fig_qt.add_trace(go.Scatter(
                            x=np.arange(len(qt_intervals)),
                            y=qt_intervals,
                            mode='lines+markers',
                            name='QT Interval',
                            line=dict(color='purple')
                        ))
                        fig_qt.add_trace(go.Scatter(
                            x=np.arange(len(qtc_bazett)),
                            y=qtc_bazett,
                            mode='lines+markers',
                            name='QTc (Bazett)',
                            line=dict(color='orange')
                        ))
                        
                        fig_qt.update_layout(
                            title="QT/QTc Interval Trend",
                            xaxis_title="Beat Number",
                            yaxis_title="Interval (ms)",
                            height=300
                        )
                        st.plotly_chart(fig_qt, use_container_width=True)
                    
                    # ST Segment Analysis
                    st.write("** ST Segment Analysis**")
                    st_deviations = analyze_st_segment(record.p_signal[:, 0], annotation.sample, record.fs)
                    
                    if len(st_deviations) > 0:
                        st_col1, st_col2, st_col3 = st.columns(3)
                        
                        with st_col1:
                            st.metric("Avg ST Deviation", f"{np.mean(st_deviations):.2f} mV")
                            st.metric("ST Std", f"{np.std(st_deviations):.2f} mV")
                        
                        with st_col2:
                            st.metric("Max ST Elevation", f"{np.max(st_deviations):.2f} mV")
                            st.metric("Max ST Depression", f"{np.min(st_deviations):.2f} mV")
                        
                        with st_col3:
                            # ST Assessment
                            if np.any(st_deviations > 1.0):
                                st.error("🚨 ST Elevation (>1mV)")
                            elif np.any(st_deviations < -1.0):
                                st.error("🚨 ST Depression (<-1mV)")
                            else:
                                st.success("✅ Normal ST Segment")
                        
                        # ST Trend Plot
                        fig_st = go.Figure()
                        fig_st.add_trace(go.Scatter(
                            x=np.arange(len(st_deviations)),
                            y=st_deviations,
                            mode='lines+markers',
                            name='ST Deviation',
                            line=dict(color='red')
                        ))
                        
                        # Add reference lines
                        fig_st.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="ST Elevation Threshold")
                        fig_st.add_hline(y=-1.0, line_dash="dash", line_color="blue", annotation_text="ST Depression Threshold")
                        
                        fig_st.update_layout(
                            title="ST Segment Deviation Trend",
                            xaxis_title="Beat Number",
                            yaxis_title="ST Deviation (mV)",
                            height=300
                        )
                        st.plotly_chart(fig_st, use_container_width=True)
                    
                    # Trend Analysis
                    st.write("**📈 Trend Analysis**")
                    
                    trend_col1, trend_col2 = st.columns(2)
                    
                    with trend_col1:
                        # HR Trend
                        if len(instantaneous_hr) > 3:
                            hr_times = annotation.sample[1:len(instantaneous_hr)+1] / record.fs
                            hr_trend = trend_analysis(instantaneous_hr, hr_times)
                            
                            st.write("**Heart Rate Trend:**")
                            st.write(f"- Slope: {hr_trend['slope']:.3f} BPM/s")
                            st.write(f"- Correlation: {hr_trend['r_value']:.3f}")
                            st.write(f"- Trend: {hr_trend['trend'].title()}")
                            
                            if hr_trend['trend'] == 'increasing' and hr_trend['r_value'] > 0.5:
                                st.warning("⚠️ Significant HR increase detected")
                            elif hr_trend['trend'] == 'decreasing' and hr_trend['r_value'] < -0.5:
                                st.warning("⚠️ Significant HR decrease detected")
                    
                    with trend_col2:
                        # RR Trend
                        if len(rr_intervals) > 3:
                            rr_times = annotation.sample[1:len(rr_intervals)+1] / record.fs
                            rr_trend = trend_analysis(rr_intervals * 1000, rr_times)
                            
                            st.write("**RR Interval Trend:**")
                            st.write(f"- Slope: {rr_trend['slope']:.3f} ms/s")
                            st.write(f"- Correlation: {rr_trend['r_value']:.3f}")
                            st.write(f"- Trend: {rr_trend['trend'].title()}")
                            
                            if abs(rr_trend['r_value']) > 0.7:
                                st.info("Strong trend detected in RR intervals")
                    
                    # Export Section
                    st.subheader("💾 Export Results")
                    
                    export_col1, export_col2, export_col3, export_col4 = st.columns(4)
                    
                    with export_col1:
                        # Export ECG Raw Data
                        ecg_csv, rpeak_csv = export_ecg_data(record, annotation, selected_record)
                        st.download_button(
                            label=" Download ECG Data (CSV)",
                            data=ecg_csv,
                            file_name=f"ecg_data_{selected_record}.csv",
                            mime="text/csv"
                        )
                    
                    with export_col2:
                        # Export R-peaks Data
                        if rpeak_csv:
                            st.download_button(
                                label=" Download R-peaks (CSV)",
                                data=rpeak_csv,
                                file_name=f"rpeaks_{selected_record}.csv",
                                mime="text/csv"
                            )
                    
                    with export_col3:
                        # Export Analysis Results
                        analysis_json = export_analysis_results(hr_stats, rr_intervals, pred_labels, confidence, selected_record)
                        st.download_button(
                            label=" Download Analysis (JSON)",
                            data=analysis_json,
                            file_name=f"analysis_{selected_record}.json",
                            mime="application/json"
                        )
                    
                    with export_col4:
                        # Export Summary Report
                        summary_report = create_summary_report(record, hr_stats, pred_labels, confidence, selected_record)
                        st.download_button(
                            label=" Download Report (MD)",
                            data=summary_report,
                            file_name=f"report_{selected_record}.md",
                            mime="text/markdown"
                        )
                    
                    st.success("✅ Analysis completed! Use the buttons above to download results.")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
    
    with tab3:
        st.subheader(" ECG Paper Strip Visualization")
        st.info(" Generate hospital-grade ECG strip with abnormality detection")
        
        if st.button(" Generate ECG Strip", key="ecg_strip_tab3"):
            with st.spinner("Generating ECG paper strip..."):
                try:
                    record = st.session_state.ecg_data["record"]
                    annotation = st.session_state.ecg_data["annotation"]
                    signal = record.p_signal[:, 0]
                    fs = record.fs
                    
                    duration = min(30, len(signal) / fs)
                    samples = int(duration * fs)
                    time_axis = np.arange(samples) / fs
                    ecg_segment = signal[:samples]
                    
                    session, encoder = load_onnx_model()
                    if session:
                        segment_rpeaks = annotation.sample[annotation.sample < samples]
                        beats, _ = extract_beats(ecg_segment, segment_rpeaks, fs)
                        if len(beats) > 0:
                            valid_beats = [b for b in beats if len(b) == 144 or len(b) == 58]
                            if valid_beats:
                                X = np.array(valid_beats)
                                if X.shape[1] == 58:
                                    from scipy import signal as scipy_signal
                                    X = np.array([scipy_signal.resample(x, 144) for x in X])
                                X = X.reshape(len(X), 144, 1).astype(np.float32)
                                
                                input_name = session.get_inputs()[0].name
                                preds = session.run(None, {input_name: X})[0]
                                pred_class = np.argmax(preds, axis=1)
                                pred_labels = encoder.inverse_transform(pred_class)
                                confidence = np.max(preds, axis=1)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=time_axis, y=ecg_segment, mode='lines',
                        line=dict(color='black', width=1.5), name='ECG', showlegend=False
                    ))
                    
                    rpeak_times = segment_rpeaks / fs
                    rpeak_values = ecg_segment[segment_rpeaks]
                    
                    fig.add_trace(go.Scatter(
                        x=rpeak_times, y=rpeak_values, mode='markers',
                        marker=dict(color='red', size=6, symbol='circle'), name='R-peaks', showlegend=False
                    ))
                    
                    if 'pred_labels' in locals():
                        for i, (rpeak_time, label, conf) in enumerate(zip(rpeak_times[:len(pred_labels)], pred_labels, confidence)):
                            if label != "Normal":
                                fig.add_annotation(
                                    x=rpeak_time, y=max(ecg_segment) * 1.2,
                                    text=f"{label}<br>{conf:.2f}", showarrow=True,
                                    arrowhead=2, arrowcolor="red", bgcolor="yellow",
                                    bordercolor="red", font=dict(size=10, color="red")
                                )
                    
                    fig.update_layout(
                        title=f"ECG Paper Strip - Record {selected_record} ({duration:.1f}s)",
                        xaxis=dict(title="Time (seconds)", showgrid=True, gridwidth=1, gridcolor="lightcoral",
                                  dtick=0.2, range=[0, duration]),
                        yaxis=dict(title="Amplitude (mV)", showgrid=True, gridwidth=1, gridcolor="lightcoral",
                                  dtick=0.5, range=[min(ecg_segment)*1.2, max(ecg_segment)*1.4]),
                        plot_bgcolor="white", paper_bgcolor="white", width=1200, height=400,
                        font=dict(color="black", size=12)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if 'pred_labels' in locals():
                        abnormal_beats = [(i, label, conf) for i, (label, conf) in enumerate(zip(pred_labels, confidence)) if label != "Normal"]
                        
                        if abnormal_beats:
                            st.subheader("🚨 Detected Abnormalities")
                            abnormal_df = pd.DataFrame([{
                                "Beat #": i+1, "Time (s)": f"{rpeak_times[i]:.2f}",
                                "Abnormality": label, "Confidence": f"{conf:.3f}",
                                "Severity": "High" if conf > 0.8 else "Medium" if conf > 0.6 else "Low"
                            } for i, label, conf in abnormal_beats])
                            
                            st.dataframe(abnormal_df, use_container_width=True)
                            
                            img_bytes = fig.to_image(format="png", width=1200, height=400, scale=2)
                            st.download_button(
                                label="📥 Download ECG Strip (PNG)",
                                data=img_bytes, file_name=f"ecg_strip_{selected_record}.png", mime="image/png"
                            )
                        else:
                            st.success("✅ No abnormalities detected in this segment")
                    
                except Exception as e:
                    st.error(f"❌ Error generating ECG strip: {e}")

else:
    st.info(" Please load an ECG file from the sidebar to start monitoring")
    
    # Demo visualization
    st.subheader("📈 Demo ECG Pattern")
    t = np.linspace(0, 3, 1000)
    demo_ecg = 0.8 * np.sin(2*np.pi*1.2*t) + 0.3 * np.sin(2*np.pi*8*t) + 0.1 * np.random.randn(1000)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=demo_ecg, mode="lines", line=dict(color="#00ff41", width=2)))
    fig.update_layout(
        title="Demo ECG Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        paper_bgcolor="#000000",
        plot_bgcolor="#001100",
        font=dict(color="#00ff41"),
        height=300
    )
    st.plotly_chart(fig, width='stretch')

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Instructions")
st.sidebar.info("""
1. Select dataset and record
2. Click 'Load ECG' 
3. Use 'Real-time Monitor' for visualization
4. Use 'AI Analysis' for arrhythmia detection
""")
