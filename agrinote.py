import streamlit as st
st.set_page_config(page_title="AgriNote Shapefile Exporter", layout="wide")

import re
import json
import html
import colorsys
from datetime import datetime
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
:root {
  --agn-bg: #f3f5f9;
  --agn-panel: #ffffff;
  --agn-border: rgba(17,24,39,0.10);
  --agn-text: #111827;
  --agn-muted: rgba(17,24,39,0.68);
  --agn-accent: #0f766e;
}

/* ---- App frame ---- */
.stApp { background: radial-gradient(1200px 700px at 100% -120px, rgba(15,118,110,0.10), transparent 60%), var(--agn-bg); color: var(--agn-text); }
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
html, body, [class*="css"] { font-family: "Avenir Next", "Noto Sans JP", "Hiragino Sans", "Yu Gothic UI", sans-serif; }
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
.agn-section-title {
  font-size: 18px;
  font-weight: 760;
  color: rgba(17,24,39,0.93);
  margin: 8px 0 2px 0;
}
.agn-section-note {
  color: var(--agn-muted);
  margin: 0 0 8px 0;
  font-size: 13px;
}
.agn-chip-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 6px;
}
.agn-chip {
  background: rgba(15,118,110,0.10);
  border: 1px solid rgba(15,118,110,0.28);
  color: #115e59;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 650;
}

/* ---- Widgets ---- */
div[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea {
  background: rgba(255,255,255,0.98) !important;
  border: 1px solid rgba(17,24,39,0.12) !important;
  border-radius: 10px !important;
}
.stMultiSelect div[data-baseweb="tag"] {
  background: rgba(15,23,42,0.04) !important;
  border: 1px solid rgba(15,23,42,0.10) !important;
  color: rgba(15,23,42,0.92) !important;
  border-radius: 10px !important;
  min-width: 120px !important;
  padding: 2px 8px !important;
  font-size: 12px !important;
  line-height: 1.2 !important;
}
.stMultiSelect div[data-baseweb="tag"] svg { fill: rgba(15,23,42,0.55) !important; }
.stMultiSelect div[data-baseweb="tag"]:hover {
  background: rgba(15,23,42,0.06) !important;
  border-color: rgba(15,23,42,0.14) !important;
}

/* Some Streamlit versions don't apply .stMultiSelect; enforce via testid too */
div[data-testid="stMultiSelect"] [data-baseweb="tag"],
div[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
  background: rgba(15,23,42,0.04) !important;
  background-color: rgba(15,23,42,0.04) !important;
  border-color: rgba(15,23,42,0.10) !important;
  color: rgba(15,23,42,0.92) !important;
}
div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
  min-width: 120px !important;
}
div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
  fill: rgba(15,23,42,0.55) !important;
}
div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
  max-width: none !important;
}
div[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
div[data-testid="stMultiSelect"] [data-baseweb="tag"] div {
  max-width: none !important;
  overflow: visible !important;
  text-overflow: clip !important;
  white-space: nowrap !important;
}
div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
  overflow-x: auto !important;
}
.stRadio div[role="radiogroup"] { background: rgba(255,255,255,0.75); border: 1px solid rgba(17,24,39,0.10); border-radius: 12px; padding: 10px 10px 2px 10px; }
.stButton button, .stDownloadButton button {
  border-radius: 12px !important;
  border: 1px solid rgba(17,24,39,0.12) !important;
  background: linear-gradient(180deg, rgba(15,118,110,0.16), rgba(15,118,110,0.06)) !important;
  font-weight: 650 !important;
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
    AgriNote APIレスポンスJSONから圃場を可視化し、一覧で選択した圃場をShapefile（ZIP）で出力します。
  </div>
  <div class="agn-muted">上部でデータ読込、左サイドバーで絞り込み、中央ワークスペースで地図確認と一覧選択・出力を行います。</div>
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

def build_project_indexes(projects: list[dict] | None):
    if not projects:
        return {}, {}

    project_by_id: dict[int, dict] = {}
    projects_by_field_id: dict[int, list[dict]] = {}

    for proj in projects:
        proj_id = proj.get("id")
        if isinstance(proj_id, int):
            project_by_id[proj_id] = proj

        for field_id in proj.get("agri_field_ids") or []:
            if not isinstance(field_id, int):
                continue
            projects_by_field_id.setdefault(field_id, []).append(proj)

    return project_by_id, projects_by_field_id


def build_address_group_label(address: str | None) -> str:
    if not address:
        return "（住所なし）"

    text = re.sub(r"\s+", "", str(address)).strip()
    if not text:
        return "（住所なし）"

    pref_match = re.match(r"^(東京都|北海道|(?:京都|大阪)府|.{2,3}県)", text)
    prefecture = pref_match.group(1) if pref_match else ""
    rest = text[len(prefecture):] if prefecture else text

    city_match = re.match(r"^([^0-9０-９,，\-－丁目番地号\s]+?[市区町村])", rest)
    if prefecture and city_match:
        return f"{prefecture}{city_match.group(1)}"
    if prefecture:
        return prefecture

    municipal_match = re.match(r"^([^0-9０-９,，\-－丁目番地号\s]+?[市区町村])", text)
    if municipal_match:
        return municipal_match.group(1)

    fallback = re.split(r"[0-9０-９,，\-－]", text, maxsplit=1)[0].strip()
    return fallback or text


def build_category_color_map(categories: list[str]) -> dict[str, str]:
    unique_categories = sorted({c for c in categories if c})
    color_map: dict[str, str] = {}
    used: set[str] = set()
    for idx, cat in enumerate(unique_categories):
        hue = (idx * 0.618033988749895) % 1.0
        sat = 0.70
        val = 0.92
        for _ in range(12):
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            if color not in used:
                used.add(color)
                color_map[cat] = color
                break
            hue = (hue + 0.0833333333) % 1.0
            sat = 0.55 if sat >= 0.70 else 0.78
    return color_map


def load_json_list(source, label: str) -> list | None:
    """JSON をパースしてリストであることを確認。エラー時は st.error を表示。"""
    try:
        data = json.loads(source) if isinstance(source, str) else json.load(source)
        if isinstance(data, list):
            return data
        st.error(f"{label}: JSONはリスト（[...]）形式である必要があります。")
    except json.JSONDecodeError:
        st.error(f"{label}: JSONの解析に失敗しました。")
    except Exception as e:
        st.error(f"{label}: 読み込み中にエラーが発生しました: {e}")
    return None


def _parse_date(value) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def project_years(project: dict) -> set[int]:
    start = _parse_date(project.get("start_date"))
    end = _parse_date(project.get("end_date"))
    if start is None and end is None:
        return set()

    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return set()

    y1, y2 = int(start.year), int(end.year)
    if y2 < y1:
        y1, y2 = y2, y1
    return set(range(y1, y2 + 1))


def project_overlaps_year(project: dict, year: int) -> bool:
    start = _parse_date(project.get("start_date"))
    end = _parse_date(project.get("end_date"))
    if start is None and end is None:
        return True

    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31)

    if start is None:
        start = year_start
    if end is None:
        end = year_end
    if end < start:
        start, end = end, start

    return (start <= year_end) and (end >= year_start)


if "fields" not in st.session_state:
    st.session_state.fields = None
if "field_blocks" not in st.session_state:
    st.session_state.field_blocks = None
if "projects" not in st.session_state:
    st.session_state.projects = None
if "search_tokens" not in st.session_state:
    st.session_state.search_tokens = []
if "search_token_input" not in st.session_state:
    st.session_state.search_token_input = ""
if "active_search_tokens" not in st.session_state:
    st.session_state.active_search_tokens = []
if "flash_message" not in st.session_state:
    st.session_state.flash_message = None

# フラッシュメッセージ表示（rerun後に1回だけ表示）
if st.session_state.flash_message:
    st.success(st.session_state.flash_message)
    st.session_state.flash_message = None

# === 読み込み状態サマリー ===
if st.session_state.fields:
    _status_parts = []
    _status_parts.append(f"圃場: {len(st.session_state.fields)}件")
    if st.session_state.field_blocks:
        _status_parts.append(f"分類: {len(st.session_state.field_blocks)}件")
    if st.session_state.projects:
        _status_parts.append(f"作付: {len(st.session_state.projects)}件")
    st.markdown(
        '<div class="agn-chip-row" style="margin-bottom:8px;">'
        + "".join(f'<span class="agn-chip">{html.escape(p)}</span>' for p in _status_parts)
        + "</div>",
        unsafe_allow_html=True,
    )

with st.expander("データ入力", expanded=(not st.session_state.fields)):
    st.caption("圃場一覧ページの `agri-fields` APIレスポンスJSONを貼り付け/アップロードします。")

    tab1, tab2 = st.tabs(["貼り付け", "ファイル"])

    with tab1:
        json_text = st.text_area("agri-fields JSON", height=200, placeholder="[{\"id\": 1, ...}]")
        if st.button("読み込む（貼り付け）", use_container_width=True):
            if json_text:
                data = load_json_list(json_text, "圃場データ")
                if data is not None:
                    st.session_state.fields = data
                    st.session_state.flash_message = f"{len(data)} 件の圃場データを読み込みました"
                    st.rerun()
            else:
                st.warning("JSONを入力してください。")

    with tab2:
        uploaded_file = st.file_uploader("agri-fields JSONファイル", type=["json"])
        if uploaded_file is not None:
            st.caption(f"選択中: {uploaded_file.name}")
            if st.button("読み込む（ファイル）", use_container_width=True):
                data = load_json_list(uploaded_file, "圃場データ")
                if data is not None:
                    st.session_state.fields = data
                    st.session_state.flash_message = f"{len(data)} 件の圃場データを読み込みました"
                    st.rerun()

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
                data = load_json_list(blocks_text, "分類データ")
                if data is not None:
                    st.session_state.field_blocks = data
                    st.session_state.flash_message = f"{len(data)} 件の分類データを読み込みました"
                    st.rerun()
            else:
                st.info("未入力のためスキップします。")

    with blk_tab2:
        uploaded_blocks_file = st.file_uploader("agri-field-blocks JSONファイル（任意）", type=["json"])
        if uploaded_blocks_file is not None:
            st.caption(f"選択中: {uploaded_blocks_file.name}")
            if st.button("読み込む（分類・ファイル）", use_container_width=True):
                data = load_json_list(uploaded_blocks_file, "分類データ")
                if data is not None:
                    st.session_state.field_blocks = data
                    st.session_state.flash_message = f"{len(data)} 件の分類データを読み込みました"
                    st.rerun()

    if st.session_state.field_blocks:
        with st.expander("読み込み済み分類（概要）", expanded=False):
            block_names = [b.get("name") for b in st.session_state.field_blocks if b.get("name")]
            st.write(f"分類数: {len(st.session_state.field_blocks)}")
            if block_names:
                st.write("例:", ", ".join(block_names[:10]))

    # === オプション: projects（作付） ===
    st.divider()
    st.subheader("（任意）作付（Projects）")
    st.caption("`projects` を読み込むと、作付（年/作付名）で圃場を絞り込みできます。")

    proj_tab1, proj_tab2 = st.tabs(["貼り付け", "ファイル"])
    with proj_tab1:
        projects_text = st.text_area(
            "projects JSON（任意）",
            height=160,
            placeholder="[{\"id\": 151564, \"item\": \"R7 みつひかり\", \"start_date\": \"2025-01-01\", \"end_date\": \"2025-12-31\", \"agri_field_ids\": [ ... ]}]",
        )
        if st.button("読み込む（作付・貼り付け）", use_container_width=True):
            if projects_text.strip():
                data = load_json_list(projects_text, "作付データ")
                if data is not None:
                    st.session_state.projects = data
                    st.session_state.flash_message = f"{len(data)} 件の作付データを読み込みました"
                    st.rerun()
            else:
                st.info("未入力のためスキップします。")

    with proj_tab2:
        uploaded_projects_file = st.file_uploader("projects JSONファイル（任意）", type=["json"])
        if uploaded_projects_file is not None:
            st.caption(f"選択中: {uploaded_projects_file.name}")
            if st.button("読み込む（作付・ファイル）", use_container_width=True):
                data = load_json_list(uploaded_projects_file, "作付データ")
                if data is not None:
                    st.session_state.projects = data
                    st.session_state.flash_message = f"{len(data)} 件の作付データを読み込みました"
                    st.rerun()

    if st.session_state.projects:
        with st.expander("読み込み済み作付（概要）", expanded=False):
            items = [p.get("item") for p in st.session_state.projects if p.get("item")]
            years = sorted({y for p in st.session_state.projects for y in project_years(p)})
            st.write(f"作付数: {len(st.session_state.projects)}")
            if years:
                st.write("作付年:", ", ".join(map(str, years[:20])) + (" ..." if len(years) > 20 else ""))
            if items:
                st.write("例:", ", ".join(items[:10]))

    # === データリセット ===
    if st.session_state.fields or st.session_state.field_blocks or st.session_state.projects:
        st.divider()
        if st.button("すべてのデータをクリア", use_container_width=True):
            st.session_state.fields = None
            st.session_state.field_blocks = None
            st.session_state.projects = None
            st.session_state.search_tokens = []
            st.session_state.active_search_tokens = []
            st.session_state.search_token_input = ""
            for key in ["map_selected_field_ids", "list_selected_field_ids", "list_prev_visible_field_ids",
                        "map_visible_categories", "selected_category_groups"]:
                st.session_state.pop(key, None)
            st.rerun()

# === 空状態 ===
if not st.session_state.fields:
    st.markdown(
        """
<div class="agn-card" style="margin-top: 14px;">
  <div style="font-size:16px; font-weight:650; color: rgba(17,24,39,0.90); margin-bottom: 4px;">
    まずは圃場データを読み込んでください
  </div>
  <div style="color: rgba(17,24,39,0.70); line-height: 1.55;">
    画面上部の「データ入力」から <code>agri-fields</code> のJSONを貼り付けるか、JSONファイルをアップロードしてください。<br/>
    （任意）<code>agri-field-blocks</code> を読み込むと分類での絞り込みが可能になります。<br/>
    （任意）<code>projects</code> を読み込むと作付（年/作付名）で絞り込みできます。
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# === マップ表示とフィルター ===
if st.session_state.fields:
    block_by_id, blocks_by_field_id = build_field_block_indexes(st.session_state.field_blocks)
    project_by_id, projects_by_field_id = build_project_indexes(st.session_state.projects)
    # --- フィルターUI（サイドバー） ---
    with st.sidebar:
        def _add_search_tokens_from_input():
            raw = (st.session_state.search_token_input or "").strip()
            if not raw:
                return

            tokens = [t for t in re.split(r"[\s　,，]+", raw) if t]
            if not tokens:
                return

            existing = set(st.session_state.search_tokens)
            active = set(st.session_state.active_search_tokens)
            for t in tokens:
                if t not in existing:
                    st.session_state.search_tokens.append(t)
                    existing.add(t)
                active.add(t)
            st.session_state.active_search_tokens = [t for t in st.session_state.search_tokens if t in active]
            st.session_state.search_token_input = ""

        st.divider()
        st.subheader("絞り込み")

        all_colors = sorted(list(set(f.get("region_color") for f in st.session_state.fields if f.get("region_color"))))
        selected_colors = st.multiselect("地域カラー", options=all_colors, default=all_colors)

        search_target = st.radio("検索対象", options=["両方", "圃場名", "住所"], index=0, horizontal=True)
        st.caption("キーワードを追加すると、地域カラーと同じようにバッチ（チップ）で複数指定できます。")
        st.text_input(
            "キーワード追加",
            key="search_token_input",
            placeholder="例: 栗山, 2148（Enterで追加 / スペース・カンマ区切りOK）",
            on_change=_add_search_tokens_from_input,
        )
        c_kw1, c_kw2 = st.columns([1, 1])
        with c_kw1:
            st.button("追加", use_container_width=True, on_click=_add_search_tokens_from_input)
        with c_kw2:
            if st.button("クリア", use_container_width=True):
                st.session_state.search_tokens = []
                st.session_state.active_search_tokens = []
                st.session_state.search_token_input = ""

        active_search_tokens = st.multiselect(
            "文字検索（バッチ）",
            options=st.session_state.search_tokens,
            default=st.session_state.active_search_tokens or st.session_state.search_tokens,
            key="active_search_tokens",
            help="複数選択すると、いずれかを含む圃場がヒットします（OR）。",
        )

        all_areas = [f.get("calculation_area", 0) for f in st.session_state.fields]
        min_area, max_area = (min(all_areas), max(all_areas)) if all_areas else (0.0, 100.0)
        if min_area >= max_area:
            max_area = min_area + 1.0
        selected_area_range = st.slider(
            "面積 (a)",
            min_value=float(min_area),
            max_value=float(max_area),
            value=(float(min_area), float(max_area)),
        )

        st.markdown("---")
        st.subheader("削除状態")
        delete_mode = st.radio(
            "削除状態の扱い",
            options=["未削除のみ（推奨）", "すべて（削除済み含む）", "詳細設定"],
            index=0,
            horizontal=True,
            help="AgriNoteの削除フラグは、圃場（agri-fields）と作付（projects）で別々に存在します。通常は両方とも「未削除のみ」でOKです。",
        )

        selected_delete_status: bool | None = False
        selected_delete_status_label = "未削除のみ"
        selected_project_delete_status: bool | None = False
        selected_project_delete_label = "未削除のみ"

        if delete_mode == "すべて（削除済み含む）":
            selected_delete_status = None
            selected_delete_status_label = "すべて"
            selected_project_delete_status = None
            selected_project_delete_label = "すべて"
        elif delete_mode == "詳細設定":
            delete_status_options = {"すべて": None, "未削除のみ": False, "削除済みのみ": True}
            selected_delete_status_label = st.radio(
                "圃場の削除状態（agri-fields.is_deleted）",
                options=delete_status_options.keys(),
                index=1,
                horizontal=True,
            )
            selected_delete_status = delete_status_options[selected_delete_status_label]

        selected_project_ids = None
        if st.session_state.projects:
            st.markdown("---")
            st.subheader("作付（Projects）で絞り込み")

            if delete_mode == "詳細設定":
                project_delete_options = {"未削除のみ": False, "すべて": None, "削除済みのみ": True}
                selected_project_delete_label = st.radio(
                    "作付の削除状態（projects.is_deleted）",
                    options=project_delete_options.keys(),
                    index=0,
                    horizontal=True,
                )
                selected_project_delete_status = project_delete_options[selected_project_delete_label]

            projects_for_filter = st.session_state.projects
            if selected_project_delete_status is not None:
                projects_for_filter = [p for p in projects_for_filter if p.get("is_deleted") == selected_project_delete_status]

            years = sorted({y for p in projects_for_filter for y in project_years(p)})
            year_options = ["すべて"] + [str(y) for y in years]
            selected_year_label = st.selectbox("作付年", options=year_options, index=0)
            selected_year = None if selected_year_label == "すべて" else int(selected_year_label)

            if selected_year is not None:
                projects_for_filter = [p for p in projects_for_filter if project_overlaps_year(p, selected_year)]

            def _proj_label(pid: int) -> str:
                p = project_by_id.get(pid) or {}
                item = p.get("item") or f"ID:{pid}"
                start = p.get("start_date") or ""
                end = p.get("end_date") or ""
                if start or end:
                    return f"{item} ({start}〜{end})"
                return item

            proj_ids = [p.get("id") for p in projects_for_filter if isinstance(p.get("id"), int)]
            selected_project_ids = st.multiselect(
                "作付（Project）",
                options=proj_ids,
                default=proj_ids,
                format_func=_proj_label,
                help="選択した作付に含まれる圃場（agri_field_ids）のみ表示します。",
            )

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

    allowed_field_ids: set[int] | None = None
    if selected_project_ids is not None:
        selected_project_ids_set = set(selected_project_ids)
        allowed_field_ids = set()
        for pid in selected_project_ids_set:
            proj = project_by_id.get(pid)
            if not proj:
                continue
            for fid in proj.get("agri_field_ids") or []:
                if isinstance(fid, int):
                    allowed_field_ids.add(fid)
        filtered_fields = [
            f for f in filtered_fields
            if isinstance(f.get("id"), int) and f.get("id") in allowed_field_ids
        ]
    
    selected_colors_set = set(selected_colors) if selected_colors else set()
    filtered_fields = [f for f in filtered_fields if f.get("region_color") in selected_colors_set]
        
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

    if active_search_tokens:
        tokens = active_search_tokens

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
            return any((_normalize(token) in haystack) for token in tokens)

        filtered_fields = [f for f in filtered_fields if _matches(f)]

    m1, m2, m3 = st.columns(3)
    m1.metric("総圃場数", f"{len(st.session_state.fields)}")
    m2.metric("表示中", f"{len(filtered_fields)}")
    m3.metric("分類データ", "あり" if st.session_state.field_blocks else "なし")
    st.markdown(
        f"""
<div class="agn-chip-row">
  <span class="agn-chip">地図表示: {len(filtered_fields)}件</span>
  <span class="agn-chip">一覧選択: {len(st.session_state.get('map_selected_field_ids', []))}件</span>
  <span class="agn-chip">作付データ: {"あり" if st.session_state.projects else "なし"}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    download_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if allowed_field_ids is not None:
        all_field_ids = {f.get("id") for f in st.session_state.fields if isinstance(f.get("id"), int)}
        present_ids = allowed_field_ids & all_field_ids
        missing_ids = allowed_field_ids - all_field_ids
        deleted_in_present = [
            f for f in st.session_state.fields
            if isinstance(f.get("id"), int) and f.get("id") in present_ids and f.get("is_deleted") is True
        ]

        with st.expander("作付絞り込みの内訳（Projects）", expanded=False):
            st.write(f"作付に含まれる圃場ID数（ユニーク）: {len(allowed_field_ids)}")
            st.write(f"agri-fieldsに存在する圃場数: {len(present_ids)}")
            st.write(f"agri-fieldsに存在しない圃場ID数: {len(missing_ids)}")
            st.write(f"存在する圃場のうち削除済み: {len(deleted_in_present)}（圃場の削除状態: {selected_delete_status_label}）")
            st.write(f"作付の削除状態: {selected_project_delete_label}")
            if missing_ids:
                st.caption("不足ID（先頭20件）: " + ", ".join(map(str, sorted(list(missing_ids))[:20])))

    # タブ分割せず、同一画面に連続表示する
    tab_map = st.container()
    tab_list = st.container()
    tab_projects = st.container() if st.session_state.projects else None

    def _field_category_name(field: dict) -> str:
        blk_id = field.get("field_block_id")
        if isinstance(blk_id, int):
            name = (block_by_id.get(blk_id) or {}).get("name")
            if name:
                return str(name)
        field_id = field.get("id")
        if isinstance(field_id, int):
            blocks = blocks_by_field_id.get(field_id) or []
            if blocks:
                name = blocks[0].get("name")
                if name:
                    return str(name)
        return "未分類"

    visible_field_ids = [f.get("id") for f in filtered_fields if isinstance(f.get("id"), int)]
    map_selected_key = "map_selected_field_ids"
    list_selected_ids_key = "list_selected_field_ids"
    current_map_selected = st.session_state.get(map_selected_key, [])
    if not isinstance(current_map_selected, list):
        current_map_selected = []
    # 一覧での選択を地図ハイライトの一次ソースにする
    list_selected_snapshot = st.session_state.get(list_selected_ids_key, [])
    if isinstance(list_selected_snapshot, list) and list_selected_snapshot:
        current_map_selected = [fid for fid in list_selected_snapshot if isinstance(fid, int)]
    visible_set = set(visible_field_ids)
    current_map_selected = [fid for fid in current_map_selected if fid in visible_set]
    st.session_state[map_selected_key] = current_map_selected
    map_selected_ids_set = set(current_map_selected)

    with tab_map:
        st.markdown('<div class="agn-section-title">地図表示</div>', unsafe_allow_html=True)
        st.markdown('<div class="agn-section-note">地図は確認用です。選択は下の一覧テーブルで行います。</div>', unsafe_allow_html=True)
        if not filtered_fields:
            st.warning("フィルター条件に一致する圃場がありません。")
        else:
            all_map_categories = sorted({_field_category_name(f) for f in filtered_fields}, key=lambda x: (x == "未分類", x))
            map_category_key = "map_visible_categories"
            if map_category_key not in st.session_state:
                st.session_state[map_category_key] = all_map_categories.copy()
            else:
                st.session_state[map_category_key] = [c for c in st.session_state.get(map_category_key, []) if c in all_map_categories]

            selected_map_categories = st.multiselect(
                "地図に表示する分類",
                options=all_map_categories,
                format_func=lambda category_name: f"({sum(1 for f in filtered_fields if _field_category_name(f) == category_name)}件) {category_name}",
                key=map_category_key,
                help="選んだ分類だけ地図に表示します（分類名ラベル付き）。",
            )
            map_mode = st.radio(
                "地図モード",
                options=["通常", "分類"],
                index=0,
                horizontal=True,
                help="通常: 圃場色。分類: 分類色。",
                key="map_mode",
            )
            map_render_fields = [f for f in filtered_fields if _field_category_name(f) in set(selected_map_categories)]
            st.caption(f"地図に表示中の圃場: {len(map_render_fields)} 件")

            if not map_render_fields:
                st.warning("表示対象の分類がありません。分類を1つ以上選択してください。")
            
            center_latlng = None
            for f in map_render_fields:
                center_latlng = extract_center_latlng(f)
                if center_latlng:
                    break
            if not center_latlng:
                # center_latlng が無い場合は、ポリゴンの先頭点を探す
                for f in map_render_fields:
                    coords = extract_polygon_latlng(f)
                    if coords:
                        center_latlng = coords[0]
                        break
            center_latlng = center_latlng or (35.0, 135.0)
            fmap = folium.Map(location=[center_latlng[0], center_latlng[1]], zoom_start=15)

            skipped_empty_polygon = 0
            category_color_map = build_category_color_map([_field_category_name(f) for f in map_render_fields])
            for f in map_render_fields:
                coords = extract_polygon_latlng(f)
                if len(coords) < 3:
                    skipped_empty_polygon += 1
                    continue
                display_name = f["field_name"] or f"ID: {f['id']}"
                field_id = f.get("id")
                category_name = _field_category_name(f)
                area_a = round(f.get("calculation_area", 0), 2)
                safe_display_name = html.escape(display_name)
                safe_category_name = html.escape(category_name)

                if map_mode == "通常":
                    raw_color = f.get("region_color", "gray")
                    color_match = re.match(r"^[a-zA-Z]+", str(raw_color))
                    folium_color = color_match.group(0).lower() if color_match else "gray"
                else:
                    folium_color = category_color_map.get(category_name, "#7f7f7f")
                is_selected_on_map = isinstance(field_id, int) and field_id in map_selected_ids_set
                border_color = "#111111" if is_selected_on_map else folium_color
                border_weight = 4 if is_selected_on_map else 2
                fill_color = "#ffd54f" if is_selected_on_map else folium_color
                fill_opacity = 0.75 if is_selected_on_map else 0.5

                folium.Polygon(
                    locations=coords,
                    popup=folium.Popup(
                        (
                            "<div style='min-width:280px;'>"
                            f"ID:{field_id}<br>{safe_display_name}<br>分類: {safe_category_name}<br>面積: {area_a}a"
                            "</div>"
                        ),
                        max_width=320,
                    ),
                    tooltip=(
                        f"ID:{field_id} | {safe_display_name} | 分類:{safe_category_name} | {area_a}a"
                        f"{' | 選択中' if is_selected_on_map else ''}"
                    ),
                    color=border_color,
                    weight=border_weight,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                ).add_to(fmap)

                if map_mode == "通常":
                    center = extract_center_latlng(f)
                    if center:
                        folium.Marker(
                            location=[center[0], center[1]],
                            icon=folium.DivIcon(
                                html=(
                                    f"<div style=\"background:{folium_color};color:white;font-size:10px;"
                                    "font-weight:700;padding:2px 6px;border-radius:8px;white-space:nowrap;"
                                    "box-shadow:0 1px 2px rgba(0,0,0,0.25);"
                                    "text-shadow:-1px -1px 0 rgba(0,0,0,0.9),1px -1px 0 rgba(0,0,0,0.9),"
                                    "-1px 1px 0 rgba(0,0,0,0.9),1px 1px 0 rgba(0,0,0,0.9);\">"
                                    f"{safe_display_name}</div>"
                                )
                            ),
                        ).add_to(fmap)

            if map_mode == "分類":
                category_fields: dict[str, list[dict]] = {}
                for f in map_render_fields:
                    category_fields.setdefault(_field_category_name(f), []).append(f)
                for category_name, fields_in_category in category_fields.items():
                    label_latlng = None
                    centers = []
                    for f in fields_in_category:
                        center = extract_center_latlng(f)
                        if center:
                            centers.append(center)
                    if centers:
                        label_latlng = (
                            sum(lat for lat, _ in centers) / len(centers),
                            sum(lng for _, lng in centers) / len(centers),
                        )
                    else:
                        for f in fields_in_category:
                            coords = extract_polygon_latlng(f)
                            if coords:
                                label_latlng = coords[0]
                                break
                    if label_latlng:
                        label_color = category_color_map.get(category_name, "#7f7f7f")
                        folium.Marker(
                            location=[label_latlng[0], label_latlng[1]],
                            icon=folium.DivIcon(
                                html=(
                                    f"<div style=\"background:{label_color};color:white;font-size:11px;"
                                    "font-weight:800;padding:3px 8px;border-radius:10px;white-space:nowrap;"
                                    "box-shadow:0 1px 3px rgba(0,0,0,0.30);"
                                    "text-shadow:-1px -1px 0 rgba(0,0,0,0.9),1px -1px 0 rgba(0,0,0,0.9),"
                                    "-1px 1px 0 rgba(0,0,0,0.9),1px 1px 0 rgba(0,0,0,0.9);\">"
                                    f"{html.escape(category_name)}</div>"
                                )
                            ),
                        ).add_to(fmap)

            st_folium(
                fmap,
                use_container_width=True,
                height=560,
                key="field_map",
            )
            if skipped_empty_polygon:
                st.info(f"地図表示できない圃場（ポリゴン未取得/不正）: {skipped_empty_polygon} 件（一覧・CSVには表示されます）")
            st.caption("選択と出力は下の一覧・エクスポートで行います。")
            st.write(f"地図で選択中: {len(st.session_state.get(map_selected_key, []))} 件")

    with tab_list:
        st.markdown('<div class="agn-section-title">一覧・エクスポート</div>', unsafe_allow_html=True)
        st.markdown('<div class="agn-section-note">チェックボックスで選択した圃場が、地図ハイライトと出力対象に反映されます。</div>', unsafe_allow_html=True)
        if not filtered_fields:
            st.warning("フィルター条件に一致する圃場がありません。")
        else:
            map_selected_ids_set = set(st.session_state.get(map_selected_key, []))
            list_selected_ids_key = "list_selected_field_ids"
            list_prev_visible_ids_key = "list_prev_visible_field_ids"

            def _project_items_for_field(field_id: int) -> str:
                projs = projects_by_field_id.get(field_id, [])
                items = [p.get("item") for p in projs if p.get("item") and not p.get("is_deleted", False)]
                seen = set()
                uniq = []
                for it in items:
                    if it in seen:
                        continue
                    seen.add(it)
                    uniq.append(it)
                return ", ".join(uniq)
            
            # 共通のデータ構築関数
            def _build_df(fields, default_selected: bool = True, selected_id_set: set[int] | None = None):
                return pd.DataFrame(
                    [
                        {
                            "ID": f["id"],
                            "圃場名": f["field_name"] or f"圃場名なし_ID: {f['id']}",
                            "住所": f.get("address") or "",
                            "分類": _field_category_name(f),
                            "作付": _project_items_for_field(f["id"]) if isinstance(f.get("id"), int) else "",
                            "面積 (a)": round(f.get("calculation_area", 0), 2),
                            "ポリゴン状態": ("🔴不足" if len(extract_polygon_lnglat(f)) < 3 else "🟢OK"),
                            "カラー": f.get("region_color"),
                            "削除済": f.get("is_deleted", False),
                            "選択": (
                                (f.get("id") in selected_id_set)
                                if selected_id_set is not None
                                else default_selected
                            ),
                        }
                        for f in fields
                    ]
                )

            selected_ids = []

            grouped = {}
            for f in filtered_fields:
                group_label = _field_category_name(f)
                grouped.setdefault(group_label, []).append(f)
            sorted_group_labels = sorted(grouped.keys(), key=lambda x: (x == "未分類", x))
            group_selector_key = "selected_category_groups"
            current_selected_groups = st.session_state.get(group_selector_key, [])
            current_selected_groups = [g for g in current_selected_groups if g in sorted_group_labels]
            if group_selector_key not in st.session_state:
                st.session_state[group_selector_key] = sorted_group_labels.copy()
            else:
                st.session_state[group_selector_key] = current_selected_groups

            selected_group_labels = st.multiselect(
                "テーブルに表示する分類",
                options=sorted_group_labels,
                format_func=lambda group_name: f"({len(grouped.get(group_name, []))}件) {group_name}",
                key=group_selector_key,
                help="選んだ分類の圃場だけテーブルに表示します。エクスポート対象はテーブル内のチェックボックスで選びます。",
            )
            selected_group_set = set(selected_group_labels)
            apply_selected_groups_only = st.button("表示中の圃場をすべてチェック", use_container_width=True)

            table_fields = [f for f in filtered_fields if _field_category_name(f) in selected_group_set]
            table_fields.sort(key=lambda f: (_field_category_name(f), str(f.get("address") or ""), str(f.get("field_name") or "")))
            visible_id_set = {f.get("id") for f in table_fields if isinstance(f.get("id"), int)}
            c_sel_all, c_sel_none = st.columns(2)
            with c_sel_all:
                apply_select_all_visible = st.button("表示中を全選択", use_container_width=True)
            with c_sel_none:
                apply_clear_all_visible = st.button("表示中を全解除", use_container_width=True)
            selected_id_set = set()
            if list_selected_ids_key not in st.session_state:
                selected_id_set = (map_selected_ids_set & visible_id_set) if map_selected_ids_set else set(visible_id_set)
            else:
                selected_id_set = set(st.session_state.get(list_selected_ids_key, [])) & visible_id_set
                prev_visible_ids = set(st.session_state.get(list_prev_visible_ids_key, []))
                new_visible_ids = visible_id_set - prev_visible_ids
                selected_id_set.update(new_visible_ids)

            if apply_selected_groups_only:
                selected_id_set = {
                    f.get("id") for f in table_fields if isinstance(f.get("id"), int)
                }
            if apply_select_all_visible:
                selected_id_set = set(visible_id_set)
            if apply_clear_all_visible:
                selected_id_set = set()

            st.session_state[list_selected_ids_key] = sorted(selected_id_set)
            st.session_state[list_prev_visible_ids_key] = sorted(visible_id_set)

            if not table_fields:
                st.info("一括選択する分類グループから1つ以上選ぶと、圃場一覧を表示します。")
                edited_df = pd.DataFrame(columns=["ID", "選択", "カラー", "圃場名"])
            else:
                df = _build_df(
                    table_fields,
                    default_selected=True,
                    selected_id_set=selected_id_set,
                )

                edited_df = st.data_editor(
                    df,
                    column_config={
                        "選択": st.column_config.CheckboxColumn("選択", default=True),
                        "削除済": st.column_config.CheckboxColumn("削除済", disabled=True),
                        "面積 (a)": st.column_config.NumberColumn(format="%.2f"),
                        "ポリゴン状態": st.column_config.TextColumn("ポリゴン状態", disabled=True),
                    },
                    use_container_width=True,
                    num_rows="fixed",
                    hide_index=True,
                )
                st.caption("凡例: 🔴不足 = ポリゴン座標不足（Shapefile出力対象外）")

            selected_ids = edited_df[edited_df["選択"] == True]["ID"].tolist()
            st.session_state[list_selected_ids_key] = sorted([int(fid) for fid in selected_ids if pd.notna(fid)])
            st.session_state[map_selected_key] = st.session_state[list_selected_ids_key].copy()

            selected_fields = [f for f in table_fields if f["id"] in selected_ids]

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
                    split_mode = st.radio(
                        "出力単位",
                        options=["一括", "住所グループ", "分類グループ"],
                        index=0,
                        horizontal=True,
                        help="地域ごとにShapefileを分割して出力できます。",
                    )

                    grouped_fields: dict[str, list[dict]] = {}
                    if split_mode == "一括":
                        grouped_fields = {"全体": exportable_fields}
                    elif split_mode == "住所グループ":
                        for f in exportable_fields:
                            grouped_fields.setdefault(build_address_group_label(f.get("address")), []).append(f)
                    else:
                        for f in exportable_fields:
                            grouped_fields.setdefault(_field_category_name(f), []).append(f)

                    st.caption("出力グループ: " + ", ".join(f"{k} ({len(v)}件)" for k, v in grouped_fields.items() if v))

                    chunk_size = 300
                    btn_idx = 0
                    for group_label, group_fields in grouped_fields.items():
                        if not group_fields:
                            continue
                        chunks = [group_fields[i : i + chunk_size] for i in range(0, len(group_fields), chunk_size)]
                        safe_group_label = re.sub(r'[\\/:*?"<>|\s]+', "_", str(group_label)).strip("_") or "group"
                        for part_idx, chunk in enumerate(chunks, start=1):
                            shp_field_ids = []
                            field_names = []
                            field_areas = []
                            field_addresses = []
                            field_categories = []
                            polygons = []
                            for f in chunk:
                                coords = extract_polygon_lnglat(f)
                                if len(coords) < 3:
                                    continue
                                if coords and coords[0] != coords[-1]:
                                    coords.append(coords[0])
                                shp_field_ids.append(f.get("id"))
                                field_names.append(f["field_name"] or f"ID: {f['id']}")
                                field_areas.append(round(f.get("calculation_area", 0), 2))
                                field_addresses.append(f.get("address") or "")
                                field_categories.append(_field_category_name(f))
                                polygons.append(Polygon(coords))

                            gdf = gpd.GeoDataFrame({
                                "FieldID": shp_field_ids,
                                "FieldName": field_names,
                                "Area_a": field_areas,
                                "Address": field_addresses,
                                "Category": field_categories,
                                "geometry": polygons,
                            }, crs="EPSG:4326")

                            shp_base = os.path.join(temp_dir, f"selected_{safe_group_label}_{part_idx}")
                            gdf.to_file(f"{shp_base}.shp", driver="ESRI Shapefile", encoding="utf-8")

                            zip_path = os.path.join(
                                temp_dir,
                                f"agnote_xarvio_selected_{safe_group_label}_{part_idx}_{download_timestamp}.zip",
                            )
                            with zipfile.ZipFile(zip_path, "w") as zipf:
                                for ext in ["shp", "shx", "dbf", "prj", "cpg"]:
                                    if os.path.exists(f"{shp_base}.{ext}"):
                                        zipf.write(
                                            f"{shp_base}.{ext}",
                                            arcname=f"selected_{safe_group_label}_{part_idx}.{ext}",
                                        )

                            with open(zip_path, "rb") as f:
                                st.download_button(
                                    label=f"ダウンロード [{group_label}] Part {part_idx} / {len(group_fields)}件",
                                    data=f.read(),
                                    file_name=os.path.basename(zip_path),
                                    mime="application/zip",
                                    key=f"dl_btn_{btn_idx}",
                                    use_container_width=True,
                                )
                                btn_idx += 1
            else:
                st.info("エクスポートする圃場を選択してください。")

    if tab_projects is not None:
        with tab_projects:
            st.subheader("作付一覧（Projects）")
            fields_by_id = {f.get("id"): f for f in st.session_state.fields if isinstance(f.get("id"), int)}

            rows = []
            for p in st.session_state.projects or []:
                pid = p.get("id")
                if not isinstance(pid, int):
                    continue

                field_ids = [fid for fid in (p.get("agri_field_ids") or []) if isinstance(fid, int)]
                total_area = float(sum((fields_by_id.get(fid, {}).get("calculation_area", 0) or 0) for fid in field_ids))

                start = p.get("start_date")
                end = p.get("end_date")
                period = f"{start or ''}〜{end or ''}" if (start or end) else ""

                rows.append(
                    {
                        "作付名": p.get("item") or "",
                        "期間": period,
                        "作目(crop_id)": p.get("crop_id"),
                        "圃場数": len(field_ids),
                        "面積 (a)": round(total_area, 2),
                        "メモ": p.get("other") or "",
                        "削除済": p.get("is_deleted", False),
                        "ID": pid,
                    }
                )

            proj_df = pd.DataFrame(rows)
            if proj_df.empty:
                st.info("作付データがありません。サイドバーから `projects` を読み込んでください。")
            else:
                st.dataframe(
                    proj_df.sort_values(by=["削除済", "作付名", "ID"], ascending=[True, True, True]),
                    use_container_width=True,
                    hide_index=True,
                )
                csv = proj_df.drop(columns=["ID"]).to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="作付一覧CSVをダウンロード",
                    data=csv,
                    file_name=f"agrinote_projects_{download_timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
