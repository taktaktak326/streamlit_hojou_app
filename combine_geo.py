import streamlit as st
import geopandas as gpd
import pandas as pd
import time
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
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = len(uploaded_files) + 3  # ファイル読み込み + 結合 + 座標系統一 + 重複削除
        current_step = 0
        
        gdfs = []
        file_names = []
        
        for file in uploaded_files:
            try:
                gdf = gpd.read_file(file)  # GeoJSONを読み込み
                gdf["source_file"] = file.name  # 元のファイル名を記録
                gdfs.append(gdf)
                file_names.append(file.name.split(".")[0])  # 拡張子を除いたファイル名を取得
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                status_text.text(f"{current_step}/{total_steps} ファイル読み込み中: {file.name}")
                time.sleep(0.5)  # 視覚的に進捗をわかりやすくするための遅延
            except Exception as e:
                st.error(f"ファイル {file.name} の読み込みに失敗しました: {e}")
        
        if gdfs:
            status_text.text("データを結合中...")
            merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            time.sleep(0.5)
            
            status_text.text("座標系を統一中...")
            merged_gdf = merged_gdf.to_crs(epsg=4326)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            time.sleep(0.5)
            
            status_text.text("重複を削除中...")
            merged_gdf = merged_gdf.drop_duplicates(subset=["geometry"])
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            time.sleep(0.5)
            
            # **結合結果の表示（テーブル）**
            st.subheader("📊 結合後のデータ（重複削除後 / 上位5件）")
            st.write(merged_gdf.head())
            
            # **地図上にプロット（緯度・経度がある場合）**
            if "geometry" in merged_gdf.columns:
                merged_gdf["lon"] = merged_gdf.geometry.centroid.x  # 中心点の経度
                merged_gdf["lat"] = merged_gdf.geometry.centroid.y  # 中心点の緯度
                st.subheader("🗺️ 結合後のデータ（地図表示）")
                st.map(merged_gdf[["lat", "lon"]].dropna())  # NaNを除去して表示

            
            # **結合後のファイル名を生成**
            merged_file_name = "_".join(file_names)[:100] + ".geojson"  # 長い場合は100文字に制限
            
            # **GeoJSONに変換**
            geojson_data = merged_gdf.to_json()
            geojson_bytes = BytesIO(geojson_data.encode())
            
            # **ダウンロードボタン**
            st.download_button(
                label="📥 結合したGeoJSON（重複削除済）をダウンロード",
                data=geojson_bytes,
                file_name=merged_file_name,
                mime="application/geo+json"
            )
            
            status_text.text("✅ 処理が完了しました！")
            progress_bar.progress(1.0)
            st.success("✅ 処理が完了しました！")
