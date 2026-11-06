# 🔋 EV Battery Fault Detection Using LSTM Autoencoder

An AI-based system for detecting anomalous behavior in electric vehicle battery sensor data using an LSTM Autoencoder.

## 📌 Project Overview

This project monitors EV battery behavior using:

- Voltage
- Current
- Temperature

An LSTM Autoencoder learns patterns from early battery cycles and detects abnormal behavior using reconstruction error.

## 🧠 System Architecture

Battery Dataset
↓
Data Extraction
↓
Preprocessing
↓
Sequence Creation
↓
LSTM Autoencoder
↓
Reconstruction Error
↓
Anomaly Threshold
↓
Normal / Anomaly
↓
Streamlit Dashboard

## 🚀 Features

- Battery sensor data extraction
- Voltage/current/temperature monitoring
- Data normalization
- Time-series sequence generation
- LSTM Autoencoder
- Reconstruction-error based anomaly detection
- Cycle-level anomaly analysis
- Interactive Streamlit dashboard
- Battery sensor visualization

## 🛠️ Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Streamlit

## 📂 Project Structure

```text
EV-Battery-Fault-Detection/
│
├── app/
│   └── dashboard.py
│
├── data/
│   └── processed/
│
├── models/
│   ├── lstm_autoencoder.keras
│   └── training_history.npy
│
├── results/
│   ├── anomaly_results.csv
│   ├── cycle_anomaly_summary.csv
│   └── *.png
│
├── src/
│   ├── extract_dataset.py
│   ├── visualize_data.py
│   ├── preprocess_data.py
│   ├── create_sequences.py
│   ├── train_lstm_autoencoder.py
│   ├── detect_anomalies.py
│   ├── analyze_anomalies.py
│   └── create_final_plots.py
│
├── .gitignore
├── README.md
└── requirements.txt
