import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import joblib  # Untuk menyimpan dan memuat scaler dan encoder
import pandas as pd

# Fungsi untuk load model dan encoders
@st.cache(allow_output_mutation=True)
def load_resources():
    # Load model yang telah dilatih
    model = load_model("model_kehadiran.h5")
    # Load scaler
    scaler = joblib.load("scaler.pkl")
    # Load label encoders
    encoders = joblib.load("label_encoders.pkl")
    return model, scaler, encoders

# Fungsi untuk melakukan encoding dengan aman
def safe_label_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return -1  # Nilai default untuk kategori yang tidak dikenal

# Fungsi untuk melakukan prediksi
def predict_kehadiran(model, scaler, encoders, udiklat, kode_judul, jnspenyelenggaraandiklat, bulan):
    # Encode fitur kategorikal
    encoded_udiklat = safe_label_encode(encoders['udiklat'], udiklat)
    encoded_kode_judul = safe_label_encode(encoders['kode_judul'], kode_judul)
    encoded_jnspenyelenggaraandiklat = safe_label_encode(encoders['jnspenyelenggaraandiklat'], jnspenyelenggaraandiklat)

    # Membentuk array input
    input_features = np.array([[encoded_udiklat, encoded_kode_judul, encoded_jnspenyelenggaraandiklat, bulan]])

    # Standardisasi input
    input_scaled = scaler.transform(input_features)

    # Prediksi
    prediksi = model.predict(input_scaled)
    return prediksi[0][0]

# Load model, scaler, dan encoders
model, scaler, encoders = load_resources()

# Antarmuka Streamlit
st.title("Prediksi Persentase Kehadiran Diklat")
st.write("Masukkan informasi berikut untuk memprediksi persentase kehadiran:")

# Input pengguna
udiklat_input = st.text_input("Udiklat", "JAKARTA")
kode_judul_input = st.text_input("Kode Judul", "A.1.1.20.002.2.20R0.IC")
jnspenyelenggaraandiklat_input = st.text_input("Jenis Penyelenggara Diklat", "IHT - In House Training (IHT)")
bulan_input = st.number_input("Bulan (1-12)", min_value=1, max_value=12, value=7)

# Tombol prediksi
if st.button("Prediksi Kehadiran"):
    prediksi_kehadiran = predict_kehadiran(
        model, scaler, encoders, 
        udiklat_input, kode_judul_input, jnspenyelenggaraandiklat_input, bulan_input
    )
    st.success(f"Prediksi persentase kehadiran: {prediksi_kehadiran:.2f}%")
