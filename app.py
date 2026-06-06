import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import graphviz

# Konfigurasi Halaman (Harus diletakkan paling atas)
st.set_page_config(page_title="Dashboard Analisis & ML", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Styling Kustom
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Analisis & Prediksi Data (Decision Tree)")
st.markdown("Eksplorasi data Anda dan bangun model prediksi cerdas dengan kecerdasan buatan (Machine Learning).")

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
    st.sidebar.header("🛠️ Panel Filter Utama")
    
    # 1. IDENTIFIKASI TIPE KOLOM
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. FILTERING DINAMIS
    filtered_df = df.copy()
    for col in categorical_columns[:3]:
        unique_vals = filtered_df[col].dropna().unique().tolist()
        if 1 < len(unique_vals) <= 50: 
            selected_vals = st.sidebar.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard ini dilengkapi algoritma Decision Tree Scikit-Learn.")
    
    # --- METRIK KPI ---
    st.subheader("💡 Ringkasan Eksekutif")
    if numeric_columns:
        kpi_cols = st.columns(min(len(numeric_columns), 4))
        for i, col in enumerate(numeric_columns[:4]):
            with kpi_cols[i]:
                total = filtered_df[col].sum()
                avg = filtered_df[col].mean()
                st.metric(label=f"Total {col}", value=f"{total:,.0f}", delta=f"Rata-rata: {avg:,.2f}")
    
    st.markdown("---")
    
    # --- TABS UNTUK ORGANISASI KONTEN TERMASUK ML ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Visualisasi Utama", 
        "📊 Analisis Lanjutan", 
        "📋 Data & Statistik", 
        "🤖 Prediksi (Decision Tree)"
    ])
    
    # TAB 1: VISUALISASI UTAMA
    with tab1:
        st.subheader("Eksplorasi Distribusi dan Komposisi")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Grafik Batang (Bar Chart)")
            if categorical_columns and numeric_columns:
                cat_col = st.selectbox("Sumbu X (Bar)", categorical_columns, key='bar_cat')
                num_col = st.selectbox("Sumbu Y (Bar)", numeric_columns, key='bar_num')
                bar_data = filtered_df.groupby(cat_col)[num_col].sum().reset_index().sort_values(by=num_col, ascending=False).head(20)
                fig_bar = px.bar(bar_data, x=cat_col, y=num_col, color=cat_col, title=f"Top 20 {num_col} berdasarkan {cat_col}")
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
                
        with col2:
            st.markdown("### 🍩 Grafik Lingkaran (Donut Chart)")
            if categorical_columns and numeric_columns:
                pie_cat = st.selectbox("Kategori (Donut)", categorical_columns, key='pie_cat')
                pie_num = st.selectbox("Nilai (Donut)", numeric_columns, key='pie_num')
                pie_data = filtered_df.groupby(pie_cat)[pie_num].sum().reset_index()
                if len(pie_data) > 10:
                    top_9 = pie_data.sort_values(by=pie_num, ascending=False).head(9)
                    other_sum = pie_data.sort_values(by=pie_num, ascending=False).iloc[9:][pie_num].sum()
                    other_df = pd.DataFrame({pie_cat: ['Lainnya'], pie_num: [other_sum]})
                    pie_data = pd.concat([top_9, other_df])
                fig_pie = px.pie(pie_data, names=pie_cat, values=pie_num, hole=0.4, title=f"Persentase {pie_num}")
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

    # TAB 2: ANALISIS LANJUTAN
    with tab2:
        st.subheader("Analisis Hubungan Antar Variabel")
        if len(numeric_columns) >= 2:
            scat_col1, scat_col2, scat_col3 = st.columns(3)
            with scat_col1: x_scat = st.selectbox("Sumbu X", numeric_columns, key='scat_x')
            with scat_col2: y_scat = st.selectbox("Sumbu Y", numeric_columns, key='scat_y', index=1)
            with scat_col3: color_scat = st.selectbox("Pisahkan Warna", ["Tidak ada"] + categorical_columns, key='scat_color')
            
            if color_scat == "Tidak ada":
                fig_scat = px.scatter(filtered_df, x=x_scat, y=y_scat, trendline="ols" if len(filtered_df) < 5000 else None)
            else:
                fig_scat = px.scatter(filtered_df, x=x_scat, y=y_scat, color=color_scat)
            st.plotly_chart(fig_scat, use_container_width=True)

    # TAB 3: TABEL DATA
    with tab3:
        st.subheader("Data Mentah dan Analisis Deskriptif")
        col_tab1, col_tab2 = st.columns(2)
        with col_tab1:
            st.markdown("### 📁 Dataset Tersaring")
            st.dataframe(filtered_df, use_container_width=True)
        with col_tab2:
            st.markdown("### 📈 Statistik Deskriptif")
            if numeric_columns: st.dataframe(filtered_df[numeric_columns].describe(), use_container_width=True)

    # TAB 4: MACHINE LEARNING (DECISION TREE)
    with tab4:
        st.subheader("🤖 Machine Learning: Decision Tree")
        st.markdown("Latih model kecerdasan buatan untuk mencari pola dan memprediksi data dari dataset Anda secara otomatis.")
        
        if len(all_columns) >= 2:
            st.markdown("""
            **Langkah-langkah:**
            1. Pilih **Target Prediksi (Y)**: Kolom apa yang ingin ditebak nilainya.
            2. Pilih **Fitur (X)**: Kolom apa saja yang menjadi faktor penentu/tebakan.
            """)
            
            col_ml1, col_ml2 = st.columns([1, 2])
            with col_ml1:
                target_col = st.selectbox("1. Pilih Target (Y)", all_columns, key='ml_target')
            with col_ml2:
                feature_options = [c for c in all_columns if c != target_col]
                # Default pilih beberapa kolom numerik jika ada
                default_features = [c for c in feature_options if c in numeric_columns][:3] 
                if not default_features: default_features = feature_options[:2]
                feature_cols = st.multiselect("2. Pilih Fitur (X)", feature_options, default=default_features, key='ml_features')
            
            if target_col and feature_cols:
                # Menentukan Tipe Model
                # Jika targetnya berupa teks atau angka uniknya sedikit, maka Klasifikasi
                is_classification = (df[target_col].dtype == 'object') or (str(df[target_col].dtype) == 'category') or (df[target_col].nunique() < 15)
                model_type_str = "Klasifikasi (Classification)" if is_classification else "Regresi (Regression)"
                
                st.info(f"Sistem mendeteksi target **{target_col}** cocok diproses menggunakan model **{model_type_str}**.")
                
                max_depth = st.slider("3. Atur Kedalaman Pohon (Max Depth)", min_value=1, max_value=15, value=4, help="Semakin dalam, model semakin rumit (berisiko overfit).")
                
                if st.button("🚀 Latih Model Decision Tree Sekarang!", type="primary"):
                    with st.spinner("Memproses data, melakukan encoding, dan melatih model..."):
                        
                        # 1. Ambil data asli (bukan yang difilter di sidebar agar datanya utuh)
                        ml_df = df[feature_cols + [target_col]].copy()
                        
                        # 2. Buang baris yang memiliki nilai kosong (NaN)
                        ml_df = ml_df.dropna()
                        
                        if len(ml_df) < 20:
                            st.error("Data terlalu sedikit setelah membuang baris yang kosong. Pilih fitur lain atau bersihkan data Anda.")
                        else:
                            # 3. Encoding Fitur Kategorikal (Mengubah teks jadi angka)
                            le_dict = {}
                            for col in feature_cols:
                                if ml_df[col].dtype == 'object' or str(ml_df[col].dtype) == 'category':
                                    le = LabelEncoder()
                                    ml_df[col] = le.fit_transform(ml_df[col].astype(str))
                                    le_dict[col] = le
                                    
                            # 4. Encoding Target (Jika Klasifikasi)
                            target_le = None
                            if ml_df[target_col].dtype == 'object' or str(ml_df[target_col].dtype) == 'category':
                                target_le = LabelEncoder()
                                ml_df[target_col] = target_le.fit_transform(ml_df[target_col].astype(str))
                                
                            X = ml_df[feature_cols]
                            y = ml_df[target_col]
                            
                            # 5. Split Train & Test
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # 6. Train Model
                            if is_classification:
                                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                score = accuracy_score(y_test, y_pred)
                                st.success(f"✅ Model selesai dilatih! **Tingkat Akurasi Prediksi: {score*100:.2f}%**")
                            else:
                                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                score = r2_score(y_test, y_pred)
                                st.success(f"✅ Model selesai dilatih! **R² Score: {score:.4f}**")
                            
                            st.markdown("---")
                            
                            # 7. Visualisasi: Feature Importances
                            st.markdown("### 📊 Faktor Paling Berpengaruh (Feature Importance)")
                            importances = pd.DataFrame({
                                'Fitur': feature_cols,
                                'Tingkat Keparuh': model.feature_importances_
                            }).sort_values(by='Tingkat Keparuh', ascending=True) # Ascending agar di grafik plotly bar tampilnya di atas
                            
                            fig_imp = px.bar(importances, x='Tingkat Keparuh', y='Fitur', orientation='h', 
                                             color='Tingkat Keparuh', color_continuous_scale='Blues')
                            st.plotly_chart(fig_imp, use_container_width=True)
                            
                            # 8. Visualisasi: Decision Tree Graph
                            st.markdown("### 🌳 Visualisasi Pohon Keputusan")
                            st.markdown("Berikut adalah alur logika (JIKA... MAKA...) yang ditemukan komputer berdasarkan data Anda.")
                            try:
                                class_names = target_le.classes_.astype(str).tolist() if target_le else None
                                
                                # Batasi kedalaman gambar agar tidak nge-hang browser
                                viz_depth = min(max_depth, 4) 
                                
                                dot_data = export_graphviz(
                                    model, out_file=None, 
                                    feature_names=feature_cols,  
                                    class_names=class_names,  
                                    filled=True, rounded=True,  
                                    special_characters=True,
                                    max_depth=viz_depth) 
                                
                                st.graphviz_chart(dot_data)
                                
                                if max_depth > viz_depth:
                                    st.info(f"💡 Visualisasi di atas dipotong hanya menampilkan {viz_depth} tingkat pertama agar mudah dibaca, meskipun aslinya memiliki {max_depth} tingkat.")
                                    
                            except Exception as e:
                                st.warning(f"Sistem tidak dapat merender gambar pohon karena keterbatasan server (Graphviz). Anda tetap bisa melihat 'Faktor Paling Berpengaruh' di atas.")
                                st.code(str(e))
else:
    st.error(f"⚠️ File dataset tidak ditemukan.")
