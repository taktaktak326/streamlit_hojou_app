import streamlit as st
import geopandas as gpd
import pandas as pd
import re
from io import BytesIO
import folium
from streamlit_folium import folium_static
import time


# 全角→半角変換用の関数
def to_half_width(text):
    if isinstance(text, str):
        table = str.maketrans("０１２３４５６７８９", "0123456789")
        return text.translate(table)
    return text

# 住所の正規化
def normalize_address(address):
    if not isinstance(address, str) or pd.isna(address):
        return None, None

    address = to_half_width(address)
    address = re.sub(r"-\d+$", "", address)
    address = re.sub(r"(\d+)\s.*", r"\1", address)
    address_without_pref = re.sub(r"^.+?[都道府県]", "", address)

    return address.strip(), address_without_pref.strip()

# タイトル
st.title("農地ピンと筆ポリゴンの結合 & 圃場登録代行シートの統合アプリ")

# **1️⃣ GeoJSONファイルのアップロード**
st.subheader("農地ピンと筆ポリゴンのGeoJSONをアップロード")
uploaded_pori_files = st.file_uploader("筆ポリゴンファイル（複数可）", accept_multiple_files=True, type=["geojson"])
uploaded_nouchi_files = st.file_uploader("農地ピンファイル（複数可）", accept_multiple_files=True, type=["geojson"])

# **2️⃣ Excelファイルのアップロード**
st.subheader("圃場登録代行シートをアップロード")
uploaded_excel_file = st.file_uploader("Excelファイルを選択", type=["xlsx", "xls"])

# **シート名の選択**
sheet_name = None
if uploaded_excel_file:
    try:
        xls = pd.ExcelFile(uploaded_excel_file)
        sheet_names = xls.sheet_names
        sheet_name = st.selectbox("シート名を選択してください", sheet_names)
    except Exception as e:
        st.error(f"Excelファイルのシート名を取得できませんでした: {e}")

# **ヘッダー行の指定**
header_row = st.number_input("カラム名がある行の番号（0から開始）", min_value=0, value=4, step=1)

# **処理開始**
if st.button("処理を開始"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = 7  # 各処理ステップのカウント
    current_step = 0

    if uploaded_pori_files and uploaded_nouchi_files:
        status_text.text("GeoJSONファイルの読み込み中...")
        gdf_pori_list = [gpd.read_file(file) for file in uploaded_pori_files]
        gdf_nouchi_list = [gpd.read_file(file) for file in uploaded_nouchi_files]

        current_step += 1
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.5)

        status_text.text("ファイルを統合中...")
        df_pori = pd.concat(gdf_pori_list, ignore_index=True)
        df_nouchi = pd.concat(gdf_nouchi_list, ignore_index=True)

        current_step += 1
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.5)

        # **CRS（座標参照系）を統一**
        status_text.text("座標系を統一中...")
        if df_pori.crs != df_nouchi.crs:
            df_nouchi = df_nouchi.to_crs(df_pori.crs)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.5)

        # **空間結合**
        status_text.text("空間結合を実行中...")
        result = gpd.sjoin(df_pori, df_nouchi, predicate='contains')
        result = result.drop_duplicates()

        st.subheader("📌 筆ポリゴンと農地ピンの結合結果（上位5件）")
        st.write(result.head())

        # **住所カラムの確認**
        possible_address_columns = ["住所", "Address", "address", "location", "name"]
        existing_columns = result.columns
        selected_address_column = next((col for col in possible_address_columns if col in existing_columns), None)

        if selected_address_column is None:
            st.error("⚠️ 住所カラムが見つかりませんでした。")
            st.write(existing_columns)
            st.stop()

        current_step += 1
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.5)

        # **3️⃣ Excelファイルの処理**
        status_text.text("圃場代行シートの読み込み中...")
        
        if uploaded_excel_file and sheet_name:
            try:
                df_excel = pd.read_excel(uploaded_excel_file, sheet_name=sheet_name, header=header_row)
                df_excel["住所地番"] = df_excel["住所地番"].astype(str)

                st.subheader("📌 圃場登録代行シートのデータ（上位5件）")
                st.write(df_excel.head())

                current_step += 1
                progress_bar.progress(current_step / total_steps)
                time.sleep(0.5)
                status_text.text("一致する地番の検索中...")
                
                # **一致検索**
                def find_matching_geometry(address):
                    if pd.isna(address):
                        return None

                    normalized_address, normalized_address_without_pref = normalize_address(address)

                    match = result[
                        result[selected_address_column].astype(str).apply(lambda x: normalize_address(x)[0]) == normalized_address
                    ]
                    if match.empty:
                        match = result[
                            result[selected_address_column].astype(str).apply(lambda x: normalize_address(x)[1]) == normalized_address_without_pref
                        ]

                    return match['geometry'].values[0] if not match.empty else None

                df_excel["geometry"] = df_excel["住所地番"].apply(find_matching_geometry)
                df_excel["geometry"] = df_excel["geometry"].fillna("一致なし")

                # **マッチング結果表示**
                st.subheader("📌 一致した & 一致しなかったマッチング結果（上位5件）")
                st.write(df_excel)

                current_step += 1
                progress_bar.progress(current_step / total_steps)
                time.sleep(0.5)
                status_text.text("マップの表示...")

                # **地図プロット（住所地番付き）**
                st.subheader("📍 一致した筆ポリゴンの地図（住所地番付き）")
                matched_gdf = gpd.GeoDataFrame(df_excel[df_excel["geometry"] != "一致なし"], geometry="geometry", crs=df_pori.crs)

                if not matched_gdf.empty:
                    centroid = matched_gdf.geometry.centroid.iloc[0]
                    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14)

                    for _, row in matched_gdf.iterrows():
                        folium.GeoJson(row.geometry, name="筆ポリゴン",
                                       tooltip=row["住所地番"]).add_to(m)

                    folium_static(m)
                else:
                    st.warning("一致する筆ポリゴンが見つかりませんでした。")
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                time.sleep(0.5)
                status_text.text("処理完了")
                
                # **Excelファイルをダウンロード**
                output_buffer = BytesIO()
                with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
                    df_excel.to_excel(writer, sheet_name="MatchedData", index=False)
                output_buffer.seek(0)

                st.download_button(
                    label="📥 更新済みExcelをダウンロード",
                    data=output_buffer,
                    file_name=f"{sheet_name}_圃場地番確認後ファイル.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Excelの処理中にエラーが発生しました: {e}")
