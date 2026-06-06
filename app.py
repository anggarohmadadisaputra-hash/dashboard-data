import streamlit as st
import pandas as pd
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Data", layout="wide")

st.title("📊 Dashboard Data Interaktif")
st.markdown("Dashboard ini menampilkan visualisasi data secara dinamis berdasarkan dataset yang diunggah.")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    # Menggunakan nama file dataset yang ada di folder Anda
    file_path = "DATASET_KEL_1.xlsx - DATASET.xlsx"
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
            return None
    else:
        return None

df = load_data()

if df is not None:
    st.success("✅ Dataset berhasil dimuat!")
    
    # Menampilkan tabel data
    st.subheader("📋 Tinjauan Data")
    st.dataframe(df.head(10))
    
    st.subheader("📈 Visualisasi Data")
    st.sidebar.header("Pengaturan Visualisasi")
    
    columns = df.columns.tolist()
    
    # Memilih kolom numerik dan kategorikal secara otomatis jika memungkinkan
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Jika tidak ada pemisahan jelas, gunakan semua kolom
    if not numeric_columns:
        numeric_columns = columns
    if not categorical_columns:
        categorical_columns = columns

    if len(columns) >= 2:
        x_axis = st.sidebar.selectbox("Pilih Sumbu X (Kategori/Waktu)", categorical_columns)
        y_axis = st.sidebar.selectbox("Pilih Sumbu Y (Nilai Numerik)", numeric_columns)
        
        st.markdown(f"**Grafik: {y_axis} berdasarkan {x_axis}**")
        
        # Agregasi data jika ada duplikat pada sumbu X
        try:
            chart_data = df.groupby(x_axis)[y_axis].sum().reset_index()
            st.bar_chart(chart_data.set_index(x_axis))
        except Exception as e:
            st.warning("Tidak dapat membuat grafik bar secara otomatis dengan kolom yang dipilih. Menampilkan data aslinya.")
            st.line_chart(df.set_index(x_axis)[y_axis])
    else:
        st.info("Dataset membutuhkan setidaknya 2 kolom untuk divisualisasikan.")
        
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard ini dibuat dengan Streamlit dan dapat diakses dari mana saja.")
else:
    st.error("⚠️ Dataset 'DATASET_KEL_1.xlsx - DATASET.xlsx' tidak ditemukan.")
    st.info("Pastikan file tersebut berada di repositori/folder yang sama dengan file aplikasi ini.")
