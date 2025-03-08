import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from shapely.wkt import loads
from streamlit_folium import folium_static
import zipfile
import os

# タイトル
st.title("圃場データの可視化 & Shapefile出力アプリ")

# ファイルアップロード
uploaded_file = st.file_uploader("Excelファイルをアップロード", type=["xlsx"])

if uploaded_file:
    # データ読み込み
    df = pd.read_excel(uploaded_file, sheet_name='MatchedData')

    # WKTからgeometry変換
    df['geometry'] = df['geometry'].apply(loads)

    # GeoDataFrameを作成
    gdf = gpd.GeoDataFrame(df[['圃場名', 'geometry']], geometry='geometry', crs="EPSG:4326")

    # データ表示
    st.subheader("データプレビュー")
    st.write(gdf.head())

    # 地図の作成
    st.subheader("圃場の地図表示")
    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=14)

    for _, row in gdf.iterrows():
        folium.GeoJson(row.geometry, tooltip=row['圃場名']).add_to(m)

    folium_static(m)

    # Shapefileの出力
    output_dir = "shapefile_output"
    os.makedirs(output_dir, exist_ok=True)
    shapefile_path = os.path.join(output_dir, "houjou_data.shp")
    
    gdf.to_file(shapefile_path, driver="ESRI Shapefile", encoding="utf-8")

    # ShapefileをZIP圧縮
    zip_filename = "shapefile_output.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, file), file)

    # ダウンロードボタン
    with open(zip_filename, "rb") as f:
        st.download_button("Shapefileをダウンロード", f, file_name="shapefile_output.zip")
