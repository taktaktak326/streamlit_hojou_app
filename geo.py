import streamlit as st
import geopandas as gpd
import pandas as pd
import re
from io import BytesIO
import folium
from streamlit_folium import folium_static

# **全角→半角変換**
def to_half_width(text):
    if isinstance(text, str):
        table = str.maketrans("０１２３４５６７８９", "0123456789")
        return text.translate(table)
    return text

# **住所の正規化**
def normalize_address(address):
    if not isinstance(address, str) or pd.isna(address):
        return None, None
    address = to_half_width(address).strip()
    address = re.sub(r"-\d+$", "", address)
    address = re.sub(r"(\d+)\s.*", r"\1", address)
    address_without_pref = re.sub(r"^.+?[都道府県]", "", address)
    return address, address_without_pref.strip()

# **タイトル**
st.title("農地ピンと筆ポリゴンの結合 & 圃場登録代行シートの統合アプリ")

# **1️⃣ GeoJSONファイルのアップロード**
st.subheader("📂 GeoJSONファイルをアップロード")
uploaded_pori_files = st.file_uploader("筆ポリゴンファイル（複数可）", accept_multiple_files=True, type=["geojson"])
uploaded_nouchi_files = st.file_uploader("農地ピンファイル（複数可）", accept_multiple_files=True, type=["geojson"])

# **2️⃣ Excelファイルのアップロード**
st.subheader("📂 圃場登録代行シートをアップロード")
uploaded_excel_file = st.file_uploader("Excelファイルを選択", type=["xlsx", "xls"])

# **シート名の選択**
if uploaded_excel_file:
    try:
        xls = pd.ExcelFile(uploaded_excel_file)
        sheet_name = st.selectbox("シート名を選択してください", xls.sheet_names)
    except Exception as e:
        st.error(f"Excelファイルのシート名を取得できませんでした: {e}")
        sheet_name = None
else:
    sheet_name = None

# **ヘッダー行の指定**
header_row = st.number_input("カラム名がある行の番号（0から開始）", min_value=0, value=4, step=1)

# **処理開始**
if st.button("🚀 処理を開始"):
    if not uploaded_pori_files or not uploaded_nouchi_files:
        st.error("❌ 筆ポリゴンまたは農地ピンのGeoJSONファイルをアップロードしてください。")
        st.stop()

    # **GeoJSON読み込み**
    gdf_pori = gpd.GeoDataFrame(pd.concat([gpd.read_file(file) for file in uploaded_pori_files], ignore_index=True))
    gdf_nouchi = gpd.GeoDataFrame(pd.concat([gpd.read_file(file) for file in uploaded_nouchi_files], ignore_index=True))

    # **座標系の統一**
    if gdf_pori.crs != gdf_nouchi.crs:
        gdf_nouchi = gdf_nouchi.to_crs(gdf_pori.crs)

    # **空間結合**
    try:
        result = gpd.sjoin(gdf_pori, gdf_nouchi, predicate='contains').drop_duplicates()
    except Exception as e:
        st.error(f"空間結合時にエラーが発生しました: {e}")
        st.stop()

    # **住所カラムの特定**
    possible_address_columns = ["住所", "Address", "address", "location", "name"]
    selected_address_column = next((col for col in possible_address_columns if col in result.columns), None)

    if not selected_address_column:
        st.error("⚠️ 住所カラムが見つかりませんでした。")
        st.write(result.columns)
        st.stop()

    # **Excelファイルの処理**
    if uploaded_excel_file and sheet_name:
        try:
            df_excel = pd.read_excel(uploaded_excel_file, sheet_name=sheet_name, header=header_row)
            df_excel["住所地番"] = df_excel["住所地番"].astype(str)

            # **住所データの正規化**
            result["normalized_address"], result["normalized_address_without_pref"] = zip(
                *result[selected_address_column].astype(str).apply(normalize_address)
            )

            # **一致検索関数**
            def find_matching_geometry(address):
                if pd.isna(address):
                    return None
                normalized_address, normalized_address_without_pref = normalize_address(address)

                match = result[result["normalized_address"] == normalized_address]
                if match.empty:
                    match = result[result["normalized_address_without_pref"] == normalized_address_without_pref]

                return match['geometry'].values[0] if not match.empty else None

            df_excel["geometry"] = df_excel["住所地番"].apply(find_matching_geometry)
            df_excel["geometry"] = df_excel["geometry"].fillna("一致なし")

            # **地図描画**
            matched_gdf = gpd.GeoDataFrame(df_excel[df_excel["geometry"] != "一致なし"], geometry="geometry", crs=gdf_pori.crs)

            if not matched_gdf.empty:
                centroid = matched_gdf.geometry.centroid.iloc[0]
                m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14)

                for _, row in matched_gdf.iterrows():
                    folium.GeoJson(row.geometry, name="筆ポリゴン", tooltip=row["住所地番"]).add_to(m)

                st.subheader("📍 一致した筆ポリゴンの地図")
                folium_static(m)
            else:
                st.warning("⚠️ 一致する筆ポリゴンが見つかりませんでした。")

            # **Excelダウンロード**
            output_buffer = BytesIO()
            with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
                df_excel.to_excel(writer, sheet_name="MatchedData", index=False)
            output_buffer.seek(0)

            st.download_button(
                label="📥 更新済みExcelをダウンロード",
                data=output_buffer,
                file_name="updated_houjou_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Excel処理中にエラーが発生しました: {e}")
