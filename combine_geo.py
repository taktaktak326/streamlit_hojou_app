import streamlit as st
import geopandas as gpd
import pandas as pd
from io import BytesIO

# タイトル
st.title("GeoJSONファイル結合アプリ（重複削除対応）")

# ファイルアップロード
uploaded_files = st.file_uploader("GeoJSONファイルをアップロード（複数可）", accept_multiple_files=True, type=["geojson"])

# 結合ボタン
if st.button("🔄 GeoJSONを結合"):
    if not uploaded_files:
        st.warning("⚠️ ファイルをアップロードしてください！")
    else:
        gdfs = []

        for file in uploaded_files:
            try:
                gdf = gpd.read_file(file)  # GeoJSONを読み込み
                gdf["source_file"] = file.name  # 元のファイル名を記録
                gdfs.append(gdf)
            except Exception as e:
                st.error(f"ファイル {file.name} の読み込みに失敗しました: {e}")

        if gdfs:
            # **GeoDataFrameを結合**
            merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

            # **座標系を統一**
            merged_gdf = merged_gdf.to_crs(epsg=4326)

            # **重複を削除（geometryが同じデータを削除）**
            merged_gdf = merged_gdf.drop_duplicates(subset=["geometry"])

            # **結合結果の表示（テーブル）**
            st.subheader("📊 結合後のデータ（重複削除後 / 上位5件）")
            st.write(merged_gdf)

            # **地図上にプロット（緯度・経度がある場合）**
            if "geometry" in merged_gdf.columns:
                merged_gdf["lon"] = merged_gdf.geometry.centroid.x  # 中心点の経度
                merged_gdf["lat"] = merged_gdf.geometry.centroid.y  # 中心点の緯度
                st.subheader("🗺️ 結合後のデータ（地図表示）")
                st.map(merged_gdf[["lat", "lon"]].dropna())  # NaNを除去して表示

            # **GeoJSONに変換**
            geojson_data = merged_gdf.to_json()
            geojson_bytes = BytesIO(geojson_data.encode())

            # **ダウンロードボタン**
            st.download_button(
                label="📥 結合したGeoJSON（重複削除済）をダウンロード",
                data=geojson_bytes,
                file_name="merged.geojson",
                mime="application/geo+json"
            )
