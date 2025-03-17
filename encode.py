import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import zipfile
import tempfile
import os
from io import BytesIO

def load_shapefile_from_zip(zip_file, encoding):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall(temp_dir)
                shp_files = [f for f in os.listdir(temp_dir) if f.endswith(".shp")]
                
                if not shp_files:
                    st.error("Shapefile (.shp) が見つかりません。")
                    return None
                
                gdf = gpd.read_file(os.path.join(temp_dir, shp_files[0]), encoding=encoding)
                return gdf
    except Exception as e:
        st.error(f"エラー: {e}")
        return None

def export_shapefile(gdf, encoding):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_shp = os.path.join(temp_dir, "exported_shapefile.shp")
        gdf.to_file(output_shp, encoding=encoding)
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            for file in os.listdir(temp_dir):
                z.write(os.path.join(temp_dir, file), arcname=file)
        zip_buffer.seek(0)
        return zip_buffer

st.title("シェープファイルビューア＆再エンコード")

# 文字コードの選択
encoding_options = ["utf-8", "shift_jis", "cp932", "latin1", "iso-8859-1"]
encoding = st.selectbox("文字コードを選択", encoding_options)

# ファイルのアップロード
uploaded_zip = st.file_uploader("Shapefile（ZIP）をアップロード", type=["zip"], accept_multiple_files=False)

gdf = None
if uploaded_zip:
    gdf = load_shapefile_from_zip(uploaded_zip, encoding)
    
    if gdf is not None:
        # 地図を作成
        m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=10)
        
        # ポリゴンを追加（圃場名の表示）
        for _, row in gdf.iterrows():
            folium.GeoJson(row.geometry, tooltip=row.get("圃場名", "No Name")).add_to(m)
        
        st_folium(m, width=700, height=500)
        
        # データフレームを表示
        st.write("属性データ:")
        st.dataframe(gdf.drop(columns="geometry"))
        
        # Shapefileのダウンロード
        st.write("### Shapefileをエクスポート")
        export_encoding = st.selectbox("エクスポート時の文字コードを選択", encoding_options, index=0)
        
        if st.button("Shapefileをダウンロード"):
            zip_buffer = export_shapefile(gdf, export_encoding)
            st.download_button(
                label="Shapefileをダウンロード (ZIP)",
                data=zip_buffer,
                file_name="exported_shapefile.zip",
                mime="application/zip"
            )
