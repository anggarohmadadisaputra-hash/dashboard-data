import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Konfigurasi Halaman (Harus diletakkan paling atas)
st.set_page_config(page_title="Dashboard Analisis Data", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Styling Kustom
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Analisis Data Profesional")
st.markdown("Selamat datang di dashboard interaktif. Anda dapat memfilter, menjelajahi, dan menganalisis data Anda menggunakan berbagai visualisasi di bawah ini.")

# Fungsi untuk memuat data
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            st.error(f"Gagal memuat data: {e}")
            return None
    return None

file_path = "DATASET_KEL_1.xlsx - DATASET.xlsx"
df = load_data(file_path)

if df is not None:
    st.sidebar.header("🛠️ Panel Filter")
    st.sidebar.markdown("Gunakan opsi di bawah ini untuk menyaring data.")
    
    # 1. IDENTIFIKASI TIPE KOLOM
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. FILTERING DINAMIS (Berdasarkan Kategori)
    filtered_df = df.copy()
    
    # Otomatis membuat filter untuk maksimal 3 kolom kategorikal agar tidak kepenuhan
    for col in categorical_columns[:3]:
        # Cek jika nilai unik tidak terlalu banyak (agar multiselect tidak lag)
        unique_vals = filtered_df[col].dropna().unique().tolist()
        if 1 < len(unique_vals) <= 50: 
            selected_vals = st.sidebar.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard ini menggunakan Plotly untuk grafik yang interaktif.")
    
    # --- METRIK KPI (Key Performance Indicators) ---
    st.subheader("💡 Ringkasan Eksekutif (Berdasarkan Data yang Difilter)")
    if numeric_columns:
        # Tampilkan maksimal 4 KPI
        kpi_cols = st.columns(min(len(numeric_columns), 4))
        for i, col in enumerate(numeric_columns[:4]):
            with kpi_cols[i]:
                total = filtered_df[col].sum()
                avg = filtered_df[col].mean()
                # Format angka agar lebih rapi
                st.metric(label=f"Total {col}", value=f"{total:,.0f}", delta=f"Rata-rata: {avg:,.2f}")
    else:
        st.info("Dataset ini tidak memiliki kolom numerik yang cukup untuk menampilkan metrik.")
        
    st.markdown("---")
    
    # --- TABS UNTUK ORGANISASI KONTEN ---
    tab1, tab2, tab3 = st.tabs(["📈 Visualisasi Utama", "📊 Analisis Lanjutan", "📋 Tabel Data & Statistik"])
    
    with tab1:
        st.subheader("Eksplorasi Distribusi dan Komposisi")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Grafik Batang (Bar Chart)")
            if categorical_columns and numeric_columns:
                cat_col = st.selectbox("Pilih Kategori untuk Sumbu X (Bar)", categorical_columns, key='bar_cat')
                num_col = st.selectbox("Pilih Nilai untuk Sumbu Y (Bar)", numeric_columns, key='bar_num')
                
                # Agregasi data
                bar_data = filtered_df.groupby(cat_col)[num_col].sum().reset_index().sort_values(by=num_col, ascending=False).head(20)
                fig_bar = px.bar(bar_data, x=cat_col, y=num_col, color=cat_col, 
                                 text_auto='.2s', title=f"Top 20 Total {num_col} berdasarkan {cat_col}")
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Dataset membutuhkan setidaknya 1 kolom kategori dan 1 kolom numerik.")
                
        with col2:
            st.markdown("### 🍩 Grafik Lingkaran (Donut Chart)")
            if categorical_columns and numeric_columns:
                pie_cat = st.selectbox("Pilih Kategori (Donut)", categorical_columns, key='pie_cat')
                pie_num = st.selectbox("Pilih Nilai (Donut)", numeric_columns, key='pie_num')
                
                pie_data = filtered_df.groupby(pie_cat)[pie_num].sum().reset_index()
                # Batasi kategori untuk pie chart agar terlihat rapi
                if len(pie_data) > 10:
                    top_9 = pie_data.sort_values(by=pie_num, ascending=False).head(9)
                    other_sum = pie_data.sort_values(by=pie_num, ascending=False).iloc[9:][pie_num].sum()
                    other_df = pd.DataFrame({pie_cat: ['Lainnya'], pie_num: [other_sum]})
                    pie_data = pd.concat([top_9, other_df])

                fig_pie = px.pie(pie_data, names=pie_cat, values=pie_num, hole=0.4, 
                                 title=f"Persentase {pie_num} berdasarkan {pie_cat}")
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Dataset membutuhkan setidaknya 1 kolom kategori dan 1 kolom numerik.")

    with tab2:
        st.subheader("Analisis Hubungan Antar Variabel")
        if len(numeric_columns) >= 2:
            st.markdown("### 🧮 Grafik Tebar (Scatter Plot)")
            st.markdown("Gunakan grafik ini untuk melihat apakah ada korelasi atau hubungan antara dua nilai numerik.")
            scat_col1, scat_col2, scat_col3 = st.columns(3)
            with scat_col1:
                x_scat = st.selectbox("Sumbu X (Numerik)", numeric_columns, key='scat_x')
            with scat_col2:
                y_scat = st.selectbox("Sumbu Y (Numerik)", numeric_columns, key='scat_y', index=1)
            with scat_col3:
                color_scat = st.selectbox("Pisahkan Warna berdasarkan (Opsional)", ["Tidak ada"] + categorical_columns, key='scat_color')
            
            if color_scat == "Tidak ada":
                fig_scat = px.scatter(filtered_df, x=x_scat, y=y_scat, 
                                      title=f"Korelasi antara {x_scat} dan {y_scat}",
                                      trendline="ols" if len(filtered_df) < 5000 else None)
            else:
                fig_scat = px.scatter(filtered_df, x=x_scat, y=y_scat, color=color_scat, 
                                      title=f"Korelasi antara {x_scat} dan {y_scat} berdasarkan {color_scat}")
            
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("Dibutuhkan minimal 2 kolom angka (numerik) untuk membuat Scatter Plot.")

    with tab3:
        st.subheader("Data Mentah dan Analisis Deskriptif")
        col_tab1, col_tab2 = st.columns(2)
        with col_tab1:
            st.markdown("### 📁 Dataset Tersaring")
            st.write(f"Menampilkan **{len(filtered_df):,}** baris dari total **{len(df):,}** baris.")
            st.dataframe(filtered_df, use_container_width=True)
        with col_tab2:
            st.markdown("### 📈 Statistik Deskriptif")
            st.markdown("Meringkas nilai rata-rata, min, max, dan persentil data numerik.")
            if numeric_columns:
                st.dataframe(filtered_df[numeric_columns].describe(), use_container_width=True)
            else:
                st.info("Tidak ada data numerik untuk ditampilkan statistiknya.")

else:
    st.error(f"⚠️ File dataset tidak ditemukan atau tidak dapat dibaca.")
    st.info("Pastikan file 'DATASET_KEL_1.xlsx - DATASET.xlsx' ada di dalam repositori.")
