import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf

@st.cache_resource
def load_artifacts():
    """
    Fungsi untuk memuat model dan preprocessing tools.
    """
    try:
        # Load model
        model = load_model("mlp_regression_model.h5")  # Tambahkan custom_objects jika diperlukan
        
        # Load preprocessing tools (LabelEncoders dan Scaler)
        with open("preprocessing_tools.pkl", "rb") as f:
            tools = pickle.load(f)
        
        label_encoders = tools["label_encoders"]
        scaler = tools["scaler"]
        
        return model, label_encoders, scaler
    
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        raise e

import streamlit as st
import numpy as np

# Fungsi Load Artifacts
model, label_encoders, scaler = load_artifacts()

# Safe Encode Function
def safe_label_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return -1

# Halaman Streamlit
st.title("Prediksi Kehadiran Diklat")

# Input dari user
udiklat = st.selectbox("Udiklat:", options=label_encoders["udiklat"].classes_)
kode_judul = st.selectbox("Kode Judul:", options=label_encoders["kode_judul"].classes_)
jnspenyelenggaraandiklat = st.selectbox("Jenis Penyelenggaraan Diklat:", options=label_encoders["jnspenyelenggaraandiklat"].classes_)
bulan = st.number_input("Bulan Diklat:", min_value=1, max_value=12, step=1)

# Tombol Prediksi
if st.button("Prediksi Kehadiran"):
    try:
        # Encode fitur kategorikal
        encoded_udiklat = safe_label_encode(label_encoders["udiklat"], udiklat)
        encoded_kode_judul = safe_label_encode(label_encoders["kode_judul"], kode_judul)
        encoded_jnspenyelenggaraandiklat = safe_label_encode(label_encoders["jnspenyelenggaraandiklat"], jnspenyelenggaraandiklat)
        
        # Buat input fitur
        input_features = np.array([[encoded_udiklat, encoded_kode_judul, encoded_jnspenyelenggaraandiklat, bulan]])
        
        # Scale fitur numerik
        input_scaled = scaler.transform(input_features)
        
        # Prediksi
        prediksi = model.predict(input_scaled)
        
        # Tampilkan hasil prediksi
        st.success(f"Prediksi Kehadiran: {prediksi[0][0]:.2f}%")
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
