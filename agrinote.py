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

st.markdown(
    """
<style>
/* ---- App frame ---- */
.stApp { background: #f6f7fb; color: #111827; }
[data-testid="stHeader"] { background: rgba(255,255,255,0.75); border-bottom: 1px solid rgba(17,24,39,0.08); backdrop-filter: blur(10px); }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid rgba(17,24,39,0.08); }
html { color-scheme: light; }

/* ---- Ensure readable text on light ---- */
[data-testid="stAppViewContainer"] { color: #111827; }
[data-testid="stSidebar"] * { color: #111827; }
label, p, li, small, summary { color: #111827; }
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] * { color: rgba(17,24,39,0.70); }
div[data-testid="stMetricValue"] { color: #111827; }
div[data-testid="stMetricLabel"] { color: rgba(17,24,39,0.70); }
button[role="tab"] { color: rgba(17,24,39,0.80) !important; }
button[role="tab"][aria-selected="true"] { color: rgba(17,24,39,0.95) !important; }
code { color: #111827; background: rgba(17,24,39,0.06); padding: 0.10rem 0.30rem; border-radius: 8px; }

/* File uploader (Browse files) */
div[data-testid="stFileUploader"] * { color: #111827 !important; }
div[data-testid="stFileUploader"] section {
  background: rgba(255,255,255,0.98) !important;
  border: 1px dashed rgba(17,24,39,0.22) !important;
  border-radius: 12px !important;
}
div[data-testid="stFileUploader"] button {
  color: #111827 !important;
  border: 1px solid rgba(17,24,39,0.12) !important;
  background: rgba(255,255,255,0.98) !important;
}

/* ---- Typography ---- */
html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "Noto Sans JP", "Hiragino Sans", "Helvetica Neue", Arial; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* ---- Cards ---- */
.agn-card {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(17,24,39,0.08);
  border-radius: 14px;
  padding: 16px 16px 6px 16px;
  box-shadow: 0 12px 28px rgba(17,24,39,0.08);
}
.agn-title {
  font-size: 28px;
  font-weight: 720;
  color: rgba(17,24,39,0.92);
  margin: 0 0 6px 0;
}
.agn-subtitle {
  color: rgba(17,24,39,0.72);
  margin: 0 0 8px 0;
  line-height: 1.45;
}
.agn-muted { color: rgba(17,24,39,0.55); font-size: 12px; }

/* ---- Widgets ---- */
div[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea {
  background: rgba(255,255,255,0.98) !important;
  border: 1px solid rgba(17,24,39,0.12) !important;
  border-radius: 10px !important;
}
.stRadio div[role="radiogroup"] { background: rgba(255,255,255,0.75); border: 1px solid rgba(17,24,39,0.10); border-radius: 12px; padding: 10px 10px 2px 10px; }
.stButton button, .stDownloadButton button {
  border-radius: 12px !important;
  border: 1px solid rgba(17,24,39,0.12) !important;
  background: linear-gradient(180deg, rgba(37,99,235,0.14), rgba(37,99,235,0.06)) !important;
}
.stButton button:hover, .stDownloadButton button:hover { border-color: rgba(17,24,39,0.20) !important; }

/* ---- Data editor ---- */
div[data-testid="stDataEditor"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(17,24,39,0.10); }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="agn-card">
  <div class="agn-title">AgriNote 圃場情報エクスポート</div>
  <div class="agn-subtitle">
    AgriNoteのAPIレスポンスJSONから圃場を可視化し、選択した圃場をShapefile（ZIP）でダウンロードできます。
  </div>
  <div class="agn-muted">左のサイドバーでデータ読込・絞り込みを行い、右側で地図と一覧を確認します。</div>
</div>
""",
    unsafe_allow_html=True,
)

def extract_polygon_latlng(field: dict) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for pt in field.get("region_latlngs") or []:
        try:
            lat = float(pt.get("lat"))
            lng = float(pt.get("lng"))
        except (TypeError, ValueError):
            continue
        coords.append((lat, lng))
    return coords


def extract_polygon_lnglat(field: dict) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for pt in field.get("region_latlngs") or []:
        try:
            lat = float(pt.get("lat"))
            lng = float(pt.get("lng"))
        except (TypeError, ValueError):
            continue
        coords.append((lng, lat))
    return coords


def extract_center_latlng(field: dict) -> tuple[float, float] | None:
    center = field.get("center_latlng") or {}
    try:
        lat = float(center.get("lat"))
        lng = float(center.get("lng"))
    except (TypeError, ValueError):
        return None
    return lat, lng


def build_field_block_indexes(field_blocks: list[dict] | None):
    if not field_blocks:
        return {}, {}

    block_by_id: dict[int, dict] = {}
    blocks_by_field_id: dict[int, list[dict]] = {}

    for blk in field_blocks:
        blk_id = blk.get("id")
        if isinstance(blk_id, int):
            block_by_id[blk_id] = blk

        for field_id in blk.get("agri_field_ids") or []:
            if not isinstance(field_id, int):
                continue
            blocks_by_field_id.setdefault(field_id, []).append(blk)

    return block_by_id, blocks_by_field_id

if "fields" not in st.session_state:
    st.session_state.fields = None
if "field_blocks" not in st.session_state:
    st.session_state.field_blocks = None

with st.sidebar:
    st.subheader("データ入力")
    st.caption("圃場一覧ページの `agri-fields` APIレスポンスJSONを貼り付け/アップロードします。")

    tab1, tab2 = st.tabs(["貼り付け", "ファイル"])

    with tab1:
        json_text = st.text_area("agri-fields JSON", height=200, placeholder="[{\"id\": 1, ...}]")
        if st.button("読み込む（貼り付け）", use_container_width=True):
            if json_text:
                try:
                    data = json.loads(json_text)
                    if isinstance(data, list):
                        st.session_state.fields = data
                        st.success(f"{len(st.session_state.fields)} 件の圃場データを読み込みました")
                        st.rerun()
                    else:
                        st.error("JSONはリスト（[...]）形式である必要があります。")
                except json.JSONDecodeError:
                    st.error("JSONの解析に失敗しました。")
            else:
                st.warning("JSONを入力してください。")

    with tab2:
        uploaded_file = st.file_uploader("agri-fields JSONファイル", type=["json"])
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    st.session_state.fields = data
                    st.success(f"{len(st.session_state.fields)} 件の圃場データを読み込みました")
                    st.rerun()
                else:
                    st.error("JSONはリスト（[...]）形式である必要があります。")
            except json.JSONDecodeError:
                st.error("JSONの解析に失敗しました。")
            except Exception as e:
                st.error(f"読み込み中にエラーが発生しました: {e}")

    # === オプション: field blocks（分類） ===
    st.divider()
    st.subheader("（任意）分類")
    st.caption("`agri-field-blocks` を読み込むと、分類（ブロック名）で絞り込みできます。")

    blk_tab1, blk_tab2 = st.tabs(["貼り付け", "ファイル"])
    with blk_tab1:
        blocks_text = st.text_area(
            "agri-field-blocks JSON（任意）",
            height=140,
            placeholder="[{\"id\": 86611, \"name\": \"...\", \"agri_field_ids\": [ ... ]}]",
        )
        if st.button("読み込む（分類・貼り付け）", use_container_width=True):
            if blocks_text.strip():
                try:
                    blocks_data = json.loads(blocks_text)
                    if isinstance(blocks_data, list):
                        st.session_state.field_blocks = blocks_data
                        st.success(f"{len(st.session_state.field_blocks)} 件の分類データを読み込みました")
                        st.rerun()
                    else:
                        st.error("JSONはリスト（[...]）形式である必要があります。")
                except json.JSONDecodeError:
                    st.error("JSONの解析に失敗しました。")
            else:
                st.info("未入力のためスキップします。")

    with blk_tab2:
        uploaded_blocks_file = st.file_uploader("agri-field-blocks JSONファイル（任意）", type=["json"])
        if uploaded_blocks_file is not None:
            try:
                blocks_data = json.load(uploaded_blocks_file)
                if isinstance(blocks_data, list):
                    st.session_state.field_blocks = blocks_data
                    st.success(f"{len(st.session_state.field_blocks)} 件の分類データを読み込みました")
                    st.rerun()
                else:
                    st.error("JSONはリスト（[...]）形式である必要があります。")
            except json.JSONDecodeError:
                st.error("JSONの解析に失敗しました。")
            except Exception as e:
                st.error(f"読み込み中にエラーが発生しました: {e}")

    if st.session_state.field_blocks:
        with st.expander("読み込み済み分類（概要）", expanded=False):
            block_names = [b.get("name") for b in st.session_state.field_blocks if b.get("name")]
            st.write(f"分類数: {len(st.session_state.field_blocks)}")
            if block_names:
                st.write("例:", ", ".join(block_names[:10]))

# === 空状態 ===
if not st.session_state.fields:
    st.markdown(
        """
<div class="agn-card" style="margin-top: 14px;">
  <div style="font-size:16px; font-weight:650; color: rgba(17,24,39,0.90); margin-bottom: 4px;">
    まずは圃場データを読み込んでください
  </div>
  <div style="color: rgba(17,24,39,0.70); line-height: 1.55;">
    サイドバーの「データ入力」から <code>agri-fields</code> のJSONを貼り付けるか、JSONファイルをアップロードしてください。<br/>
    （任意）<code>agri-field-blocks</code> を読み込むと分類での絞り込みが可能になります。
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# === マップ表示とフィルター ===
if st.session_state.fields:
    block_by_id, blocks_by_field_id = build_field_block_indexes(st.session_state.field_blocks)
    # --- フィルターUI（サイドバー） ---
    with st.sidebar:
        st.divider()
        st.subheader("絞り込み")

        all_colors = sorted(list(set(f.get("region_color") for f in st.session_state.fields if f.get("region_color"))))
        selected_colors = st.multiselect("地域カラー", options=all_colors, default=all_colors)

        search_target = st.radio("検索対象", options=["両方", "圃場名", "住所"], index=0, horizontal=True)
        search_query = st.text_input(
            "文字検索",
            value="",
            placeholder="例: 桐が作 / 金岡大道 / 2148",
        )

        all_areas = [f.get("calculation_area", 0) for f in st.session_state.fields]
        min_area, max_area = (min(all_areas), max(all_areas)) if all_areas else (0.0, 100.0)
        selected_area_range = st.slider(
            "面積 (a)",
            min_value=float(min_area),
            max_value=float(max_area),
            value=(float(min_area), float(max_area)),
        )

        delete_status_options = {"すべて": None, "未削除のみ": False, "削除済みのみ": True}
        selected_delete_status_label = st.radio(
            "削除状態",
            options=delete_status_options.keys(),
            index=1,
            horizontal=True,
        )
        selected_delete_status = delete_status_options[selected_delete_status_label]

        selected_block_ids = None
        if st.session_state.field_blocks:
            all_blocks = sorted(
                [b for b in st.session_state.field_blocks if isinstance(b.get("id"), int)],
                key=lambda b: (
                    b.get("position") if b.get("position") is not None else 10**9,
                    str(b.get("name") or ""),
                ),
            )
            options = [(b.get("id"), b.get("name") or f"ID:{b.get('id')}") for b in all_blocks]
            selected_block_ids = st.multiselect(
                "分類（フィールドブロック）",
                options=[bid for bid, _ in options],
                default=[bid for bid, _ in options],
                format_func=lambda bid: next((name for _bid, name in options if _bid == bid), str(bid)),
            )
        elif st.session_state.field_blocks is None:
            selected_block_ids = None

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

    if selected_block_ids is not None:
        selected_block_ids_set = set(selected_block_ids)

        def _field_block_ids(field: dict) -> set[int]:
            field_id = field.get("id")
            ids: set[int] = set()

            # fields側にfield_block_idが入っている場合も尊重
            blk_id = field.get("field_block_id")
            if isinstance(blk_id, int):
                ids.add(blk_id)

            if isinstance(field_id, int):
                for blk in blocks_by_field_id.get(field_id, []):
                    blk2_id = blk.get("id")
                    if isinstance(blk2_id, int):
                        ids.add(blk2_id)
            return ids

        filtered_fields = [
            f for f in filtered_fields
            if (_field_block_ids(f) & selected_block_ids_set)
        ]

    if search_query.strip():
        tokens = [t for t in re.split(r"[\s　,，]+", search_query.strip()) if t]

        def _normalize(value: str) -> str:
            return re.sub(r"\s+", "", str(value or "")).lower()

        def _matches(field: dict) -> bool:
            if search_target == "圃場名":
                haystack_raw = field.get("field_name", "")
            elif search_target == "住所":
                haystack_raw = field.get("address", "")
            else:
                haystack_raw = f"{field.get('field_name', '')} {field.get('address', '')}"

            haystack = _normalize(haystack_raw)
            return all(_normalize(token) in haystack for token in tokens)

        filtered_fields = [f for f in filtered_fields if _matches(f)]

    m1, m2, m3 = st.columns(3)
    m1.metric("総圃場数", f"{len(st.session_state.fields)}")
    m2.metric("表示中", f"{len(filtered_fields)}")
    m3.metric("分類データ", "あり" if st.session_state.field_blocks else "なし")

    # === マップ表示 ===
    if filtered_fields:
        tab_map, tab_list = st.tabs(["地図", "一覧 / エクスポート"])

        with tab_map:
            st.subheader("圃場マップ")
            center_latlng = None
            for f in filtered_fields:
                center_latlng = extract_center_latlng(f)
                if center_latlng:
                    break
            if not center_latlng:
                # center_latlng が無い場合は、ポリゴンの先頭点を探す
                for f in filtered_fields:
                    coords = extract_polygon_latlng(f)
                    if coords:
                        center_latlng = coords[0]
                        break
            center_latlng = center_latlng or (35.0, 135.0)
            fmap = folium.Map(location=[center_latlng[0], center_latlng[1]], zoom_start=15)

            skipped_empty_polygon = 0
            for f in filtered_fields:
                coords = extract_polygon_latlng(f)
                if len(coords) < 3:
                    skipped_empty_polygon += 1
                    continue
                display_name = f["field_name"] or f"ID: {f['id']}"

                raw_color = f.get("region_color", "gray")
                color_match = re.match(r"^[a-zA-Z]+", raw_color)
                folium_color = color_match.group(0) if color_match else "gray"

                folium.Polygon(
                    locations=coords,
                    popup=display_name,
                    tooltip=f"{display_name} ({round(f.get('calculation_area', 0), 2)}a)",
                    color=folium_color,
                    fill=True,
                    fill_opacity=0.5,
                ).add_to(fmap)

            st_folium(fmap, use_container_width=True)
            if skipped_empty_polygon:
                st.info(f"地図表示できない圃場（ポリゴン未取得/不正）: {skipped_empty_polygon} 件（一覧・CSVには表示されます）")

        with tab_list:
            st.subheader("圃場一覧")
            st.checkbox("すべて選択", value=True, key="select_all")

            df = pd.DataFrame(
                [
                    {
                        "ID": f["id"],
                        "圃場名": f["field_name"] or f"圃場名なし_ID: {f['id']}",
                        "住所": f.get("address") or "",
                        "分類": (
                            (block_by_id.get(f.get("field_block_id"), {}).get("name"))
                            if isinstance(f.get("field_block_id"), int)
                            else (
                                (blocks_by_field_id.get(f.get("id"), [{}])[0].get("name"))
                                if isinstance(f.get("id"), int) and blocks_by_field_id.get(f.get("id"))
                                else ""
                            )
                        ),
                        "面積 (a)": round(f.get("calculation_area", 0), 2),
                        "カラー": f.get("region_color"),
                        "削除済": f.get("is_deleted", False),
                        "選択": st.session_state.select_all,
                    }
                    for f in filtered_fields
                ]
            )

            edited_df = st.data_editor(
                df,
                column_config={
                    "選択": st.column_config.CheckboxColumn("選択"),
                    "削除済": st.column_config.CheckboxColumn("削除済", disabled=True),
                    "面積 (a)": st.column_config.NumberColumn(format="%.2f"),
                },
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
            )

            c1, c2 = st.columns([1, 1])
            with c1:
                csv_df = edited_df.drop(columns=["選択"]).sort_values(by=["カラー", "圃場名"])
                csv = csv_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSVをダウンロード",
                    data=csv,
                    file_name="agrinote_fields.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            selected_ids = edited_df[edited_df["選択"] == True]["ID"].tolist()
            selected_fields = [f for f in filtered_fields if f["id"] in selected_ids]

            s1, s2 = st.columns(2)
            s1.metric("選択数", f"{len(selected_fields)}")
            s2.metric("合計面積 (a)", f"{round(sum(f.get('calculation_area', 0) for f in selected_fields), 2)}")

            if selected_fields:
                st.subheader("Shapefile（ZIP）")
                with tempfile.TemporaryDirectory() as temp_dir:
                    exportable_fields = []
                    skipped_export = 0
                    for f in selected_fields:
                        if len(extract_polygon_lnglat(f)) >= 3:
                            exportable_fields.append(f)
                        else:
                            skipped_export += 1
                    if skipped_export:
                        st.warning(f"ポリゴン座標が無い圃場はエクスポート対象から除外しました: {skipped_export} 件")

                    chunk_size = 300
                    chunks = [exportable_fields[i : i + chunk_size] for i in range(0, len(exportable_fields), chunk_size)]

                    for idx, chunk in enumerate(chunks):
                        field_names = []
                        polygons = []
                        for f in chunk:
                            coords = extract_polygon_lnglat(f)
                            if len(coords) < 3:
                                continue
                            if coords and coords[0] != coords[-1]:
                                coords.append(coords[0])
                            field_names.append(f["field_name"] or f"ID: {f['id']}")
                            polygons.append(Polygon(coords))

                        gdf = gpd.GeoDataFrame({"FieldName": field_names, "geometry": polygons}, crs="EPSG:4326")

                        shp_base = os.path.join(temp_dir, f"selected_{idx+1}")
                        gdf.to_file(f"{shp_base}.shp", driver="ESRI Shapefile", encoding="utf-8")

                        zip_path = os.path.join(temp_dir, f"agnote_xarvio_selected_{idx+1}.zip")
                        with zipfile.ZipFile(zip_path, "w") as zipf:
                            for ext in ["shp", "shx", "dbf", "prj", "cpg"]:
                                if os.path.exists(f"{shp_base}.{ext}"):
                                    zipf.write(f"{shp_base}.{ext}", arcname=f"selected_{idx+1}.{ext}")

                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label=f"ダウンロード Part {idx+1}",
                                data=f.read(),
                                file_name=os.path.basename(zip_path),
                                mime="application/zip",
                                key=f"dl_btn_{idx}",
                                use_container_width=True,
                            )
            else:
                st.info("エクスポートする圃場を選択してください。")
    else:
        st.warning("フィルター条件に一致する圃場がありません。")
