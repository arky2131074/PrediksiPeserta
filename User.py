import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Fungsi untuk memuat model dan preprocessing tools
@st.cache_resource
def load_artifacts():
    # Load model
    model = load_model("mlp_regression_model.h5")
    # Load preprocessing tools (LabelEncoders dan Scaler)
    with open("preprocessing_tools.pkl", "rb") as f:
        tools = pickle.load(f)
    label_encoders = tools["label_encoders"]
    scaler = tools["scaler"]
    return model, label_encoders, scaler

# Fungsi aman untuk encode label
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
    
    # Buat array input
    input_features = np.array([[encoded_udiklat, encoded_kode_judul, encoded_jnspenyelenggaraandiklat, bulan]])
    
    # Standardisasi input
    input_scaled = scaler.transform(input_features)
    
    # Prediksi
    prediksi = model.predict(input_scaled)
    return prediksi[0][0]

# Memuat model dan preprocessing tools
model, label_encoders, scaler = load_artifacts()

# Tampilan antarmuka Streamlit
st.title("Prediksi Kehadiran Diklat")
st.write("Masukkan informasi berikut untuk memprediksi kehadiran peserta diklat:")

# Input dari user
udiklat = st.selectbox("Udiklat:", options=label_encoders['udiklat'].classes_)
kode_judul = st.selectbox("Kode Judul:", options=label_encoders['kode_judul'].classes_)
jnspenyelenggaraandiklat = st.selectbox("Jenis Penyelenggaraan Diklat:", options=label_encoders['jnspenyelenggaraandiklat'].classes_)
bulan = st.number_input("Bulan Diklat (1-12):", min_value=1, max_value=12, step=1)

# Tombol prediksi
if st.button("Prediksi Kehadiran"):
    try:
        hasil_prediksi = predict_kehadiran(
            model, scaler, label_encoders, udiklat, kode_judul, jnspenyelenggaraandiklat, bulan
        )
        st.success(f"Prediksi Kehadiran: {hasil_prediksi:.2f}%")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
