import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Fungsi untuk memuat model dan scaler
@st.cache_resource
def load_artifacts():
    # Muat model TensorFlow
    model = load_model("mlp_regression_model.h5", compile=False)

    # Muat scaler (disimpan sebelumnya menggunakan joblib)
    scaler = joblib.load("scaler.pkl")

    return model, scaler

# Fungsi prediksi
def predict_kehadiran(model, scaler, udiklat, kode_judul, jnspenyelenggaraandiklat, bulan):
    # Konversi input fitur ke array
    input_features = np.array([[udiklat, kode_judul, jnspenyelenggaraandiklat, bulan]])
    
    # Standardisasi fitur menggunakan scaler
    input_scaled = scaler.transform(input_features)
    
    # Prediksi menggunakan model
    prediction = model.predict(input_scaled)
    
    return prediction[0][0]  # Mengembalikan nilai prediksi tunggal

# Muat model dan scaler
model, scaler = load_artifacts()

# Judul aplikasi
st.title("Prediksi Persentase Kehadiran Diklat")
st.write("Masukkan informasi di bawah ini untuk memprediksi persentase kehadiran diklat.")

# Input dari pengguna
udiklat = st.number_input("Masukkan UDiklat (dalam angka)", min_value=0, step=1)
kode_judul = st.number_input("Masukkan Kode Judul (dalam angka)", min_value=0, step=1)
jnspenyelenggaraandiklat = st.number_input("Masukkan Jenis Penyelenggara Diklat (dalam angka)", min_value=0, step=1)
bulan = st.slider("Masukkan Bulan", min_value=1, max_value=12, step=1)

# Tombol prediksi
if st.button("Prediksi"):
    # Lakukan prediksi
    try:
        hasil_prediksi = predict_kehadiran(model, scaler, udiklat, kode_judul, jnspenyelenggaraandiklat, bulan)
        st.success(f"Prediksi Persentase Kehadiran: {hasil_prediksi:.2f}%")
    except Exception as e:
        st.error(f"Terjadi error selama prediksi: {str(e)}")
