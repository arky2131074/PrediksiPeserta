import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Memuat model yang sudah dilatih
model = load_model("mlp_regression_model.h5")

# Memuat preprocessing tools
with open("preprocessing_tools.pkl", "rb") as f:
    preprocessing_tools = pickle.load(f)

label_encoders = preprocessing_tools["label_encoders"]
scaler = preprocessing_tools["scaler"]

# Fungsi untuk encoding aman (mengatasi data yang tidak dikenali)
def safe_label_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1  # Atur nilai default jika tidak dikenali

# Judul aplikasi
st.title("Prediksi Persentase Kehadiran Diklat")

# Input pengguna
udiklat_input = st.text_input("Masukkan UDiklat:")
kode_judul_input = st.text_input("Masukkan Kode Judul:")
jnspenyelenggaraandiklat_input = st.text_input("Masukkan Jenis Penyelenggara Diklat:")
bulan_input = st.slider("Masukkan Bulan (1-12):", min_value=1, max_value=12, step=1)

# Tombol untuk prediksi
if st.button("Prediksi"):
    try:
        # Preprocessing input
        encoded_udiklat = safe_label_encode(label_encoders['udiklat'], udiklat_input)
        encoded_kode_judul = safe_label_encode(label_encoders['kode_judul'], kode_judul_input)
        encoded_jnspenyelenggaraandiklat = safe_label_encode(label_encoders['jnspenyelenggaraandiklat'], jnspenyelenggaraandiklat_input)

        # Menyiapkan fitur input
        input_features = np.array([[encoded_udiklat, encoded_kode_judul, encoded_jnspenyelenggaraandiklat, bulan_input]])

        # Standardisasi fitur
        input_scaled = scaler.transform(input_features)

        # Prediksi
        prediksi = model.predict(input_scaled)

        # Menampilkan hasil prediksi
        st.success(f"Prediksi persentase kehadiran: {prediksi[0][0]:.2f}%")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
