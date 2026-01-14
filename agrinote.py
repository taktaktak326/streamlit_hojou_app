import streamlit as st
st.set_page_config(page_title="AgriNote Shapefile Exporter", layout="wide")

import re
import json
import os
import zipfile
import tempfile
import folium
import geopandas as gpd
from shapely.geometry import Polygon
from streamlit_folium import st_folium
import pandas as pd

st.title("AgriNote 圃場情報取得 & Shapefile エクスポート")

if "fields" not in st.session_state:
    st.session_state.fields = None

st.subheader("データ入力")
st.info("Agrinoteの圃場一覧ページで開発者ツールを開き、`agri-fields` というAPIレスポンスのJSONをコピーして貼り付けるか、ファイルとして保存してアップロードしてください。")

tab1, tab2 = st.tabs(["JSONを貼り付け", "JSONファイルをアップロード"])

with tab1:
    json_text = st.text_area("ここにagri-fieldsのJSONレスポンスを貼り付け", height=250, placeholder="[{\"id\": 1, ...}]")
    if st.button("📝 貼り付けたJSONを読み込む"):
        if json_text:
            try:
                data = json.loads(json_text)
                if isinstance(data, list):
                    st.session_state.fields = data
                    st.success(f"✅ {len(st.session_state.fields)} 件の土地データを読み込みました")
                    st.rerun()
                else:
                    st.error("❌ JSONの形式が正しくありません。リスト（[...]）形式である必要があります。")
            except json.JSONDecodeError:
                st.error("❌ JSONの解析に失敗しました。有効なJSON文字列を貼り付けてください。")
        else:
            st.warning("⚠️ テキストエリアにJSONデータを入力してください。")

with tab2:
    uploaded_file = st.file_uploader("agri-fieldsのJSONファイルをアップロード", type=["json"])
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            if isinstance(data, list):
                st.session_state.fields = data
                st.success(f"✅ {len(st.session_state.fields)} 件の土地データを読み込みました")
                st.rerun()
            else:
                st.error("❌ JSONの形式が正しくありません。リスト（[...]）形式である必要があります。")
        except json.JSONDecodeError:
            st.error("❌ JSONの解析に失敗しました。有効なJSONファイルを選択してください。")
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

# === マップ表示とフィルター ===
if st.session_state.fields:
    st.subheader("絞り込みフィルター")
    
    # --- フィルターUI ---
    filter_cols = st.columns(3)
    with filter_cols[0]:
        # region_color のユニークな値を取得
        all_colors = sorted(list(set(f.get("region_color") for f in st.session_state.fields if f.get("region_color"))))
        selected_colors = st.multiselect(
            "地域カラー",
            options=all_colors,
            default=all_colors
        )

    with filter_cols[1]:
        # calculation_area の範囲を取得
        all_areas = [f.get("calculation_area", 0) for f in st.session_state.fields]
        min_area, max_area = (min(all_areas), max(all_areas)) if all_areas else (0.0, 100.0)
        
        selected_area_range = st.slider(
            "面積 (a)",
            min_value=float(min_area),
            max_value=float(max_area),
            value=(float(min_area), float(max_area))
        )

    with filter_cols[2]:
        # is_deleted のフィルター
        delete_status_options = {"すべて": None, "未削除のみ": False, "削除済みのみ": True}
        selected_delete_status_label = st.radio(
            "削除状態",
            options=delete_status_options.keys(),
            index=1, # デフォルトを「未削除のみ」に
            horizontal=True
        )
        selected_delete_status = delete_status_options[selected_delete_status_label]

    # --- フィルター適用 ---
    filtered_fields = st.session_state.fields
    
    if selected_colors:
        filtered_fields = [f for f in filtered_fields if f.get("region_color") in selected_colors]
        
    min_selected, max_selected = selected_area_range
    filtered_fields = [
        f for f in filtered_fields 
        if min_selected <= f.get("calculation_area", 0) <= max_selected
    ]

    if selected_delete_status is not None:
        filtered_fields = [f for f in filtered_fields if f.get("is_deleted") == selected_delete_status]
    
    st.info(f"フィルター結果: {len(filtered_fields)} / {len(st.session_state.fields)} 件")

    # === マップ表示 ===
    if filtered_fields:
        st.subheader("🖼️ 圃場マップ")
        center = filtered_fields[0]["center_latlng"]
        fmap = folium.Map(location=[center["lat"], center["lng"]], zoom_start=15)

        for f in filtered_fields:
            coords = [(pt['lat'], pt['lng']) for pt in f['region_latlngs']]
            display_name = f["field_name"] or f"ID: {f['id']}"
            
            # region_colorからfoliumで使える色名を取得 (例: green2 -> green)
            raw_color = f.get("region_color", "gray")
            color_match = re.match(r"^[a-zA-Z]+", raw_color)
            folium_color = color_match.group(0) if color_match else "gray"

            folium.Polygon(
                locations=coords,
                popup=display_name,
                tooltip=f"{display_name} ({round(f.get('calculation_area', 0), 2)}a)",
                color=folium_color,
                fill=True,
                fill_opacity=0.5
            ).add_to(fmap)

        st_folium(fmap, use_container_width=True)

        # === 表形式でフィルター・ソート・選択 ===
        st.subheader("📋 圃場一覧と選択")

        st.checkbox("すべて選択", value=True, key="select_all")

        df = pd.DataFrame([
            {
                "ID": f["id"],
                "圃場名": f["field_name"] or f"圃場名なし_ID: {f['id']}",
                "面積 (a)": round(f.get("calculation_area", 0), 2),
                "カラー": f.get("region_color"),
                "削除済": f.get("is_deleted", False),
                "選択": st.session_state.select_all
            } for f in filtered_fields
        ])

        edited_df = st.data_editor(
            df,
            column_config={
                "選択": st.column_config.CheckboxColumn("選択"),
                "削除済": st.column_config.CheckboxColumn("削除済", disabled=True),
                "面積 (a)": st.column_config.NumberColumn(format="%.2f"),
            },
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True
        )

        # CSVダウンロード
        csv_df = edited_df.drop(columns=["選択"]).sort_values(by=["カラー", "圃場名"])
        csv = csv_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 圃場リストをCSVでダウンロード",
            data=csv,
            file_name="agrinote_fields.csv",
            mime="text/csv",
        )

        selected_ids = edited_df[edited_df["選択"] == True]["ID"].tolist()
        selected_fields = [f for f in filtered_fields if f["id"] in selected_ids]

        st.markdown(f"### ✅ 選択された圃場数: {len(selected_fields)} 件")
        st.markdown(f"### 📐 合計面積: {round(sum(f.get('calculation_area', 0) for f in selected_fields), 2)} a")

        if selected_fields:
            # TemporaryDirectoryを使って自動クリーンアップ
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_paths = []
                chunk_size = 300
                chunks = [selected_fields[i:i + chunk_size] for i in range(0, len(selected_fields), chunk_size)]

                for idx, chunk in enumerate(chunks):
                    field_names = []
                    polygons = []
                    for f in chunk:
                        coords = [(pt["lng"], pt["lat"]) for pt in f["region_latlngs"]]
                        if coords and coords[0] != coords[-1]:
                            coords.append(coords[0])
                        field_names.append(f["field_name"] or f"ID: {f['id']}")
                        polygons.append(Polygon(coords))

                    gdf = gpd.GeoDataFrame({
                        "FieldName": field_names,
                        "geometry": polygons
                    }, crs="EPSG:4326")

                    shp_base = os.path.join(temp_dir, f"selected_{idx+1}")
                    gdf.to_file(f"{shp_base}.shp", driver="ESRI Shapefile", encoding="utf-8")

                    zip_path = os.path.join(temp_dir, f"agnote_xarvio_selected_{idx+1}.zip")
                    with zipfile.ZipFile(zip_path, "w") as zipf:
                        for ext in ["shp", "shx", "dbf", "prj", "cpg"]:
                            if os.path.exists(f"{shp_base}.{ext}"):
                                zipf.write(f"{shp_base}.{ext}", arcname=f"selected_{idx+1}.{ext}")

                    # ダウンロードボタン用にメモリに読み込むか、パスを保持してボタン表示
                    # ここではループ内で即座にボタンを表示（withブロック内である必要があるため）
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ ダウンロード Part {idx+1}",
                            data=f.read(),
                            file_name=os.path.basename(zip_path),
                            mime="application/zip",
                            key=f"dl_btn_{idx}"
                        )
        else:
            st.info("🔍 圃場を選択してください")
    else:
        st.warning("フィルター条件に一致する圃場がありません。")
