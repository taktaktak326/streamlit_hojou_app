import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from shapely.wkt import loads
from shapely.errors import WKTReadingError
from streamlit_folium import folium_static
import zipfile
import os

# タイトル
st.title("圃場データの可視化 & Shapefile出力アプリ")

# ファイルアップロード
uploaded_file = st.file_uploader("Excelファイルをアップロード", type=["xlsx"])

if uploaded_file:
    # Excelデータを読み込む
    df = pd.read_excel(uploaded_file, sheet_name='MatchedData', dtype=str, engine="openpyxl")

    # `圃場名` のカラム名を英語に変更 (Shapefileは日本語不可)
    df = df.rename(columns={"圃場名": "FieldName"})

    # `geometry` 列をWKT形式で変換
    def safe_load_wkt(wkt_str):
        try:
            return loads(wkt_str)
        except (WKTReadingError, UnicodeDecodeError, ValueError) as e:
            print(f"エラー: {wkt_str} - {e}")
            return None

    df['geometry'] = df['geometry'].apply(safe_load_wkt)

    # 無効なデータを削除
    df = df.dropna(subset=['geometry'])

    # GeoDataFrameを作成
    gdf = gpd.GeoDataFrame(df[['FieldName', 'geometry']], geometry='geometry', crs="EPSG:4326")

    # データ表示
    st.subheader("データプレビュー")
    st.write(gdf.head())

    # 地図の作成
    st.subheader("圃場の地図表示")
    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=14)

    for _, row in gdf.iterrows():
        folium.GeoJson(row.geometry, tooltip=row['FieldName']).add_to(m)

    folium_static(m)

    # Shapefileの出力
    output_dir = "shapefile_output"
    os.makedirs(output_dir, exist_ok=True)
    shapefile_path = os.path.join(output_dir, "houjou_data.shp")

    # Shapefile保存 (日本語データを保存するため Shift-JIS を使用)
    gdf.to_file(shapefile_path, driver="ESRI Shapefile", encoding="Shift-JIS")

    # ShapefileをZIP圧縮
    zip_filename = "shapefile_output.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            zipf.write(file_path, os.path.basename(file_path))

    # ダウンロードファイル名の入力欄
    custom_filename = st.text_input("ダウンロードファイル名を入力してください（.zipを含む）", value=zip_filename)

    # ダウンロードボタン
    with open(zip_filename, "rb") as f:
        st.download_button("Shapefileをダウンロード", f, file_name=custom_filename)
