import streamlit as st
from PIL import Image
import pickle
import numpy as np
import pandas as pd

def main():
# Mengatur tampilan CSS untuk latar belakang dan lebar sidebar
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f5f5f5;
        }
        .title {
            text-align: center;
        }
        .center {
            display: flex;
            justify-content: center;
        }
        .input-container {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
        }
        .wide-table {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            width: 350px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
 # Menampilkan judul dan deskripsi utama
    st.markdown('<h1 class="title">Selamat Datang</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="title">Klasifikasi Tanaman Pangan Berdasarkan Kondisi Lahan Pertanian Menggunakan Metode Random Forest</h2>', unsafe_allow_html=True)

    navigation = st.sidebar.radio("Pergi Ke Halaman:", ['Homepage', 'Prediksi Tanaman'])

    if navigation == 'Homepage':
        show_image()
    elif navigation == 'Prediksi Tanaman':
        show_prediction()

def show_image():
 # Menampilkan gambar di tengah halaman
    image = Image.open('utm2.png')

    st.markdown(
        """
        <style>
        .center img {
            width: 300px;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="center">', unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_prediction():
 # Menampilkan header dan deskripsi dataset
    st.header('Dataset')
    st.text('Dataset ini berisi data kondisi lahan pertanian')
    # Baca dataset
    data_pertanian = pd.read_csv('Crop_recommendation.csv')
    st.dataframe(data_pertanian, height=600)

    st.title('Masukkan Data Baru')

    # Muat model random forest
    with open('klasifikasi_rekomendasi_tanaman.sav', 'rb') as file:
        rekomendasi_tanaman = pickle.load(file)

    # Muat MinMaxScaler
    with open('scaler1.sav', 'rb') as f:
        skaler = pickle.load(f)

    # Input data
    kolom1, kolom2 = st.columns(2)
    with kolom1:
        N = st.number_input("Nilai N", min_value=0.0, max_value=100.0, step=0.1)
    with kolom2:
        P = st.number_input("Nilai P", min_value=0.0, max_value=100.0, step=0.1)

    kolom3, kolom4 = st.columns(2)
    with kolom3:
        K = st.number_input("Nilai K", min_value=0.0, max_value=100.0, step=0.1)
    with kolom4:
        suhu = st.number_input("Suhu", min_value=0.0, max_value=50.0, step=0.1)

    kolom5, kolom6 = st.columns(2)
    with kolom5:
        kelembapan = st.number_input("Kelembapan", min_value=0.0, max_value=100.0, step=0.1)
    with kolom6:
        ph = st.number_input("pH", min_value=0.0, max_value=10.0, step=0.1)

    curah_hujan = st.number_input("Curah Hujan", min_value=0.0, max_value=500.0, step=0.1)

    # Tombol prediksi
    if st.button('Prediksi Jenis Tanaman'):
        if N and P and K and suhu and kelembapan and ph and curah_hujan:
            input_data = np.array([[N, P, K, suhu, kelembapan, ph, curah_hujan]])

            # Transformasi data menggunakan skaler yang dimuat
            input_data_scaled = skaler.transform(input_data)
            
            #menampilkan hasil
            hasil = rekomendasi_tanaman.predict(input_data_scaled)
            st.success(f"Tanaman yang Cocok untuk Lahan Tersebut adalah: {hasil[0]}")
        else:
            st.error("Mohon isi semua nilai input.")

if __name__ == '__main__':
    main()
