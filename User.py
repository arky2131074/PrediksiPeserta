import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Fungsi untuk memuat model dan tools preprocessing
@st.cache_resource
def load_artifacts():
    # Load model TensorFlow
    model = load_model("mlp_regression_model.h5")

    # Load preprocessing tools (LabelEncoder dan Scaler)
    with open("preprocessing_tools.pkl", "rb") as f:
        tools = pickle.load(f)
    
    return model, tools["label_encoders"], tools["scaler"]

# Fungsi untuk encode aman
def safe_label_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return -1  # Nilai default jika tidak ditemukan

# Fungsi prediksi
def predict_kehadiran(model, scaler, label_encoders, udiklat, kode_judul, jnspenyelenggaraandiklat, bulan):
    # Encode fitur kategorikal
    encoded_udiklat = safe_label_encode(label_encoders['udiklat'], udiklat)
    encoded_kode_judul = safe_label_encode(label_encoders['kode_judul'], kode_judul)
    encoded_jnspenyelenggaraandiklat = safe_label_encode(label_encoders['jnspenyelenggaraandiklat'], jnspenyelenggaraandiklat)

    # Siapkan fitur input dalam bentuk array
    input_features = np.array([[encoded_udiklat, encoded_kode_judul, encoded_jnspenyelenggaraandiklat, bulan]])

    # Standardisasi fitur menggunakan scaler
    input_scaled = scaler.transform(input_features)

    # Prediksi menggunakan model
    prediction = model.predict(input_scaled)

    return prediction[0][0]  # Mengembalikan nilai prediksi tunggal

# Memuat model dan preprocessing tools
model, label_encoders, scaler = load_artifacts()

# Antarmuka Streamlit
st.title("Prediksi Persentase Kehadiran Diklat")
st.write("Masukkan informasi berikut untuk memprediksi persentase kehadiran.")

# Input dari pengguna
udiklat = st.text_input("Masukkan UDiklat (contoh: nama udiklat)")
kode_judul = st.text_input("Masukkan Kode Judul (contoh: nama kode judul)")
jnspenyelenggaraandiklat = st.text_input("Masukkan Jenis Penyelenggara Diklat (contoh: jenis penyelenggara)")
bulan = st.slider("Masukkan Bulan", min_value=1, max_value=12, step=1)

# Tombol prediksi
if st.button("Prediksi"):
    if udiklat and kode_judul and jnspenyelenggaraandiklat:
        try:
            # Melakukan prediksi
            hasil_prediksi = predict_kehadiran(
                model, scaler, label_encoders, udiklat, kode_judul, jnspenyelenggaraandiklat, bulan
            )
            st.success(f"Prediksi Persentase Kehadiran: {hasil_prediksi:.2f}%")
        except Exception as e:
            st.error(f"Terjadi error selama prediksi: {str(e)}")
    else:
        st.warning("Harap isi semua input dengan benar.")
