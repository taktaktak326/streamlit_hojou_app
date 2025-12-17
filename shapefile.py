# streamlit_app.py
# pip install streamlit geopandas shapely folium streamlit-folium rtree openpyxl

import re
import unicodedata
import zipfile
import tempfile
import html
from io import BytesIO
from typing import List, Optional, Tuple

import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from shapely.geometry.base import BaseGeometry
from shapely.wkt import loads as wkt_loads
from shapely.errors import WKTReadingError
from streamlit_folium import folium_static


# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="筆ポリゴン×ピン：住所照合→地図→出力", layout="wide")

ADDRESS_COL_CANONICAL = "Address"   # 空間結合後に使う住所列は Address 固定
LABEL_FONT_SIZE = 16               # ラベル文字サイズ固定（スライダー無し）


# =========================================================
# Session State
# =========================================================
STATE_DEFAULTS = {
    "merged_pori": None,
    "merged_pins": None,
    "joined": None,
    "matched": None,
    "excel_hash": None,
    "sheet_name": None,
    "header_row": None,
    "excel_addr_col": None,
    "strip_last_num": True,     # 末尾の -数字 を無視（. , 、枝番は常に削除）
    "show_map": False,
    "uploader_nonce": 0,        # uploader強制リセット用
    "upload_error": None,       # 直近のアップロード制約違反メッセージ
}
for k, v in STATE_DEFAULTS.items():
    st.session_state.setdefault(k, v)


# =========================================================
# Styles
# =========================================================
CSS = """
<style>
:root{
  --muted: rgba(130,130,130,.9);
  --card: rgba(255,255,255,.04);
  --card2: rgba(255,255,255,.06);
  --border: rgba(255,255,255,.10);
  --ok: rgba(0, 200, 83, .18);
  --ng: rgba(255, 82, 82, .18);
}
.block-container{padding-top: 1.1rem;}
h1,h2,h3{letter-spacing: .2px;}
.hr{height:1px; background: var(--border); margin: 1.1rem 0;}
.step{
  padding: .75rem .95rem; border: 1px solid var(--border); border-radius: 16px;
  background: var(--card); margin-bottom: .85rem;
}
.step-head{display:flex; align-items:center; justify-content:space-between; gap: .6rem;}
.step-title{font-size:1.06rem; font-weight:800; margin: 0;}
.step-desc{color:var(--muted); font-size:.92rem; margin:.35rem 0 0;}
.badge{
  padding: .18rem .55rem; border-radius: 999px; font-size:.78rem; font-weight:700;
  border: 1px solid var(--border); background: rgba(255,255,255,.04);
  white-space: nowrap;
}
.badge-ok{background: var(--ok);}
.badge-ng{background: var(--ng);}
.kpi{
  padding:.65rem .8rem; border:1px solid var(--border); border-radius: 14px;
  background: var(--card2);
}
.sidebar-title{font-weight:900; margin-bottom:.35rem;}
.sidebar-item{margin:.35rem 0; color: rgba(220,220,220,.95);}
.small{color: var(--muted); font-size:.88rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# Utilities
# =========================================================
HEADER_HINTS = ["住所地番", "住所", "地番", "筆", "地目", "面積", "圃場", "農地", "字", "番地"]
HYPHENS = r"[‐-‒–—―ー－-]"


def reset_all():
    for k in list(STATE_DEFAULTS.keys()):
        st.session_state[k] = STATE_DEFAULTS[k]


def step_card_render(slot, title: str, desc: str, done: bool):
    badge = "<span class='badge badge-ok'>✅ 完了</span>" if done else "<span class='badge badge-ng'>⏳ 未完</span>"
    slot.markdown(
        f"""
        <div class="step">
          <div class="step-head">
            <div class="step-title">{html.escape(title)}</div>
            {badge}
          </div>
          <div class="step-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def norm_filename(s: str) -> str:
    """ファイル名比較用：NFKC + 空白除去"""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("　", " ")
    s = re.sub(r"\s+", "", s)
    return s


def fail_upload(offending_name: str, label: str, allow_words: List[str]):
    allow = " / ".join(allow_words)
    st.session_state.upload_error = (
        f"❌ {label} はファイル名に「{allow}」を含む GeoJSON のみアップロードできます。\n\n"
        f"- 選択されたファイル: {offending_name}\n"
        f"- 対応: アップロード欄をリセットしました。正しいファイルを選び直してください。"
    )
    st.session_state.uploader_nonce += 1  # keyを変えてアップローダーを強制クリア
    st.rerun()


def validate_filename_or_reset(files, must_include_any: List[str], label: str):
    """選択後に厳格チェックして違反なら即リセット"""
    if not files:
        return
    musts = [norm_filename(x) for x in must_include_any]
    for f in files:
        name_norm = norm_filename(f.name)
        if not any(m in name_norm for m in musts):
            fail_upload(f.name, label, must_include_any)


def is_ready(obj) -> bool:
    return obj is not None and not (hasattr(obj, "empty") and obj.empty)


def to_half(s):
    return s.translate(str.maketrans("０１２３４５６７８９", "0123456789")) if isinstance(s, str) else s


def score_header_row(vals) -> int:
    score = 0
    for v in vals:
        s = str(v)
        score += 2 * sum(h in s for h in HEADER_HINTS)
        if 2 <= len(s) <= 12 and re.fullmatch(r"[^\d\s]{2,}", s or ""):
            score += 1
    return score


def suggest_header_rows(pre: pd.DataFrame, topk=6):
    n = min(40, len(pre))
    cand = sorted([(i, score_header_row(pre.iloc[i].values)) for i in range(n)],
                  key=lambda x: x[1], reverse=True)
    return [i for i, sc in cand[:topk] if sc > 0]


def is_good_header_choice(pre: pd.DataFrame, hdr_row: int, tmp_cols, cand_rows: List[int]) -> bool:
    if hdr_row in (cand_rows or []):
        return True
    try:
        row_score = score_header_row(pre.iloc[hdr_row].values)
    except Exception:
        row_score = 0
    has_addr_col = any(any(h in str(c) for h in ["住所地番", "住所", "地番"]) for c in tmp_cols)
    return (row_score >= 8) or has_addr_col


def style_header_preview(df: pd.DataFrame, good: bool):
    ok_bg = "rgba(0, 200, 83, 0.12)"
    ng_bg = "rgba(255, 82, 82, 0.12)"
    bg = ok_bg if good else ng_bg
    return (
        df.style.set_table_styles([
            {"selector": "thead th", "props": [("background-color", bg), ("font-weight", "800")]},
            {"selector": "tbody td", "props": [("background-color", bg)]},
        ])
    )


def slim_gdf_preview(gdf: gpd.GeoDataFrame, n: int = 5, max_cols: int = 12) -> pd.DataFrame:
    """st.table用（ツールバー無し）"""
    if gdf is None or getattr(gdf, "empty", True):
        return pd.DataFrame()
    df = gdf.head(n).copy()
    if "geometry" in df.columns:
        df["geometry"] = df["geometry"].apply(lambda g: g.geom_type if isinstance(g, BaseGeometry) else "")
    cols = list(df.columns)[:max_cols]
    return pd.DataFrame(df[cols])


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def dedupe_by_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ジオメトリ重複を削除（WKBで比較）"""
    if gdf is None or gdf.empty or "geometry" not in gdf.columns:
        return gdf
    tmp = gdf.copy()
    tmp["__wkb"] = tmp.geometry.apply(lambda g: g.wkb_hex if isinstance(g, BaseGeometry) else None)
    tmp = tmp.drop_duplicates(subset=["__wkb"]).drop(columns=["__wkb"])
    return tmp


def read_geojson(files) -> gpd.GeoDataFrame:
    gdfs = []
    for f in files:
        g = gpd.read_file(f)
        g["source_file"] = f.name
        g = ensure_wgs84(g)
        gdfs.append(g)
    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    return ensure_wgs84(merged)


def gdf_signature(gdf: gpd.GeoDataFrame, col_for_hash: Optional[str] = None) -> tuple:
    bounds = tuple(map(float, gdf.total_bounds)) if gdf is not None and not gdf.empty else (0, 0, 0, 0)
    n = int(len(gdf)) if gdf is not None else 0
    h = 0
    if gdf is not None and col_for_hash and col_for_hash in gdf.columns:
        try:
            h = int(pd.util.hash_pandas_object(gdf[col_for_hash].astype(str), index=False).sum())
        except Exception:
            h = 0
    return (n, bounds, h)


@st.cache_data(show_spinner=False)
def sjoin_pori_pin(_g_pori: gpd.GeoDataFrame, _g_pin: gpd.GeoDataFrame, pori_sig: tuple, pin_sig: tuple):
    try:
        j = gpd.sjoin(_g_pori, _g_pin, predicate="covers", how="left")
    except Exception:
        j = gpd.sjoin(_g_pori, _g_pin, predicate="intersects", how="left")
    return j.drop_duplicates()


def ensure_address_column(joined: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Address / Address_left / Address_right を Address に正規化"""
    if joined is None or joined.empty:
        return joined
    cols = set(joined.columns)
    if ADDRESS_COL_CANONICAL in cols:
        return joined
    if f"{ADDRESS_COL_CANONICAL}_left" in cols:
        joined[ADDRESS_COL_CANONICAL] = joined[f"{ADDRESS_COL_CANONICAL}_left"]
        return joined
    if f"{ADDRESS_COL_CANONICAL}_right" in cols:
        joined[ADDRESS_COL_CANONICAL] = joined[f"{ADDRESS_COL_CANONICAL}_right"]
        return joined
    return joined


def norm_addr_key(s: str, strip_last_num: bool = True) -> str:
    """住所の照合キー化：
    - 全角→半角数字
    - ハイフン類を統一
    - 空白除去
    - 末尾の . , 、 の枝番を削除（常に）
    - 末尾の -数字 を削除（オプション）
    """
    if not isinstance(s, str) or pd.isna(s):
        return ""
    s = to_half(s.strip()).lower().replace("　", " ")
    s = re.sub(HYPHENS, "-", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("丁目", "-").replace("番地", "-").replace("番", "-").replace("号", "")

    # 末尾の枝番（. , 、）を削除（連続もOK）
    s = re.sub(r"(?:[\.．,，、]\d{1,4})+$", "", s)

    # 末尾の -数字 を無視（任意）
    if strip_last_num:
        s = re.sub(r"-\d{1,4}$", "", s)

    return s


def addr_key_loose(k: str) -> str:
    s = re.sub(r"(東京都|北海道|京都府|大阪府|..県|..都|..道|..府)", "", k)
    return re.sub(r".{1,6}(市|区|町|村)", "", s)


@st.cache_data(show_spinner=False)
def build_addr_dict(_gdf: gpd.GeoDataFrame, col: str, gdf_sig: tuple, strip_last_num: bool):
    t = _gdf[[col, "geometry"]].dropna(subset=[col, "geometry"]).copy()
    t["k1"] = t[col].astype(str).map(lambda s: norm_addr_key(s, strip_last_num=strip_last_num))
    t["k2"] = t["k1"].map(addr_key_loose)
    d1 = t.groupby("k1")["geometry"].first().to_dict()
    d2 = t.groupby("k2")["geometry"].first().to_dict()
    return d1, d2


def apply_match(df: pd.DataFrame, excel_addr_col: str, d1: dict, d2: dict, strip_last_num: bool) -> pd.DataFrame:
    out = df.copy()
    out["__k"] = out[excel_addr_col].astype(str).map(lambda s: norm_addr_key(s, strip_last_num=strip_last_num))

    # 一旦Shapelyを作る（最後に必ず消す）
    out["geom"] = out["__k"].map(d1)
    miss = out["geom"].isna()
    out.loc[miss, "geom"] = out.loc[miss, "__k"].map(addr_key_loose).map(d2)

    out["match_status"] = out["geom"].apply(lambda g: "一致" if isinstance(g, BaseGeometry) else "一致なし")
    out["geometry_wkt"] = out["geom"].apply(lambda g: g.wkt if isinstance(g, BaseGeometry) else "")

    # JSON/SHAPEで落ちる原因（shapely列）を除去
    out = out.drop(columns=["__k", "geom"], errors="ignore")
    return out


def safe_load_wkt(wkt_str):
    if not isinstance(wkt_str, str) or not wkt_str.strip():
        return None
    try:
        return wkt_loads(wkt_str)
    except (WKTReadingError, UnicodeDecodeError, ValueError):
        return None


def safe_centroid_lonlat(gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    """地図中心の計算（centroidの警告回避のためEPSG:3857で算出→4326へ）"""
    if gdf is None or gdf.empty:
        return (35.681236, 139.767125)
    try:
        g3857 = gdf.to_crs(epsg=3857)
        cent = g3857.geometry.centroid
        cent = gpd.GeoSeries(cent, crs="EPSG:3857").to_crs(epsg=4326)
        return (float(cent.y.mean()), float(cent.x.mean()))
    except Exception:
        b = gdf.total_bounds
        return (float((b[1] + b[3]) / 2), float((b[0] + b[2]) / 2))


def format_label(v) -> str:
    """
    表示用の圃場名を整形：
    - 408.0 → 408
    - "0000484" → "484"（数字だけの文字列は先頭ゼロ除去）
    """
    if v is None:
        return ""

    # 数値
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        if v.is_integer():
            return str(int(v))
        s = str(v)
        return s.rstrip("0").rstrip(".")
    if isinstance(v, int):
        return str(v)

    # 文字列
    s = str(v).strip()
    if not s:
        return ""

    # "408.0" みたいな文字列
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]

    # ✅ 数字だけなら先頭ゼロを落とす（"0000484"→"484"）
    if re.fullmatch(r"\d+", s):
        s2 = s.lstrip("0")
        return s2 if s2 != "" else "0"

    return s


def gdf_to_shapefile_zip_bytes(gdf: gpd.GeoDataFrame, filename_prefix: str = "houjou_data") -> bytes:
    """Shapefile ZIP を bytes で返す"""
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = f"{tmpdir}/{filename_prefix}.shp"
        gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="UTF-8")

        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                p = f"{tmpdir}/{filename_prefix}{ext}"
                try:
                    with open(p, "rb") as f:
                        zf.writestr(f"{filename_prefix}{ext}", f.read())
                except FileNotFoundError:
                    pass
        bio.seek(0)
        return bio.read()


# =========================================================
# Sidebar (状態 + リセットのみ)
# =========================================================
def sidebar_status(label: str, done: bool):
    st.sidebar.markdown(
        f"<div class='sidebar-item'>{html.escape(label)}：{'✅' if done else '—'}</div>",
        unsafe_allow_html=True,
    )


st.sidebar.markdown("<div class='sidebar-title'>現在の状態</div>", unsafe_allow_html=True)

done_join = is_ready(st.session_state.joined) and (ADDRESS_COL_CANONICAL in st.session_state.joined.columns)
done_match = is_ready(st.session_state.matched)

sidebar_status("空間結合", done_join)
sidebar_status("住所照合", done_match)

if done_match:
    mm = st.session_state.matched
    ok = int((mm["match_status"] == "一致").sum())
    tot = int(len(mm))
    st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-title'>結果</div>", unsafe_allow_html=True)
    st.sidebar.write(f"一致: {ok:,} / {tot:,}（{(ok/tot if tot else 0):.1%}）")

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
if st.sidebar.button("🔁 すべてリセット", use_container_width=True):
    reset_all()
    st.rerun()


# =========================================================
# Header
# =========================================================
st.title("筆ポリゴン×ピン：住所照合→地図→出力（1ページ）")
st.caption("出力Shapefileの属性は FieldName（圃場名）のみ。表示の先頭ゼロ（0000484→484）も自動修正します。")

progress_steps = 0
progress_steps += 1 if done_join else 0
progress_steps += 1 if done_match else 0
st.progress(progress_steps / 2)

if st.session_state.upload_error:
    st.error(st.session_state.upload_error)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# =========================================================
# Step 1｜Upload（状態でカード更新）
# =========================================================
step1_slot = st.empty()
step_card_render(
    step1_slot,
    "Step 1｜GeoJSONをアップロード",
    "ファイル名条件に合わない場合は、理由を表示してアップロード欄をリセットします。",
    done=False,
)

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("**筆ポリゴン GeoJSON（複数可）**")
    pori_files = st.file_uploader(
        "GeoJSON を選択",
        type=["geojson"],
        accept_multiple_files=True,
        key=f"pori_files_{st.session_state.uploader_nonce}",
        help="ファイル名に「筆ポリゴン」を含む必要があります。",
    )
    validate_filename_or_reset(pori_files, ["筆ポリゴン"], "筆ポリゴン GeoJSON")

with c2:
    st.markdown("**ピン GeoJSON（複数可）**")
    pin_files = st.file_uploader(
        "GeoJSON を選択",
        type=["geojson"],
        accept_multiple_files=True,
        key=f"pin_files_{st.session_state.uploader_nonce}",
        help="ファイル名に「農地ピン」または「農場ピン」を含む必要があります。",
    )
    validate_filename_or_reset(pin_files, ["農地ピン", "農場ピン"], "ピン GeoJSON")

if st.session_state.upload_error and pori_files and pin_files:
    st.session_state.upload_error = None

done_step1 = bool(pori_files) and bool(pin_files) and (not st.session_state.upload_error)
step_card_render(
    step1_slot,
    "Step 1｜GeoJSONをアップロード",
    "ファイル名条件に合わない場合は、理由を表示してアップロード欄をリセットします。",
    done=done_step1,
)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# =========================================================
# Step 2｜Merge + Spatial Join（一括）
# =========================================================
step2_slot = st.empty()
step_card_render(
    step2_slot,
    "Step 2｜結合 → 空間結合",
    f"CRS統一→重複削除→空間結合をまとめて実行します。住所列は「{ADDRESS_COL_CANONICAL}」を使用します。",
    done=done_join,
)

can_run_step2 = bool(pori_files) and bool(pin_files)
run_clicked = st.button(
    "🚀 Step 2 を実行（結合→空間結合まで一括）",
    use_container_width=True,
    disabled=not can_run_step2,
)

if run_clicked:
    prog = st.progress(0)
    info = st.empty()
    try:
        info.text("1/3 読み込み・結合（筆ポリゴン）…")
        g_pori = dedupe_by_geometry(read_geojson(pori_files))
        prog.progress(0.33)

        info.text("2/3 読み込み・結合（ピン）…")
        g_pin = dedupe_by_geometry(read_geojson(pin_files))
        prog.progress(0.66)

        st.session_state.merged_pori = g_pori
        st.session_state.merged_pins = g_pin
        st.session_state.joined = None
        st.session_state.matched = None

        info.text("3/3 空間結合…")
        pori_sig = gdf_signature(st.session_state.merged_pori)
        pin_sig = gdf_signature(st.session_state.merged_pins)
        joined = sjoin_pori_pin(st.session_state.merged_pori, st.session_state.merged_pins, pori_sig, pin_sig)
        joined = ensure_address_column(joined)
        st.session_state.joined = joined

        prog.progress(1.0)
        if ADDRESS_COL_CANONICAL not in joined.columns:
            st.error(
                "空間結合結果に Address 列が見つかりません。\n\n"
                "対応: GeoJSONのプロパティに「Address」（または Address_left / Address_right）があるか確認してください。"
            )
        else:
            st.success(f"✅ Step2 完了：筆ポリゴン {len(g_pori):,}件 / ピン {len(g_pin):,}件（住所列：Address）")
            st.rerun()

    except Exception as e:
        st.error(f"Step2でエラーが発生しました。\n\nエラー: {e}")

if is_ready(st.session_state.merged_pori):
    with st.expander("統合プレビュー（筆ポリゴン・先頭5件）", expanded=False):
        st.table(slim_gdf_preview(st.session_state.merged_pori, n=5))

if is_ready(st.session_state.merged_pins):
    with st.expander("統合プレビュー（ピン・先頭5件）", expanded=False):
        st.table(slim_gdf_preview(st.session_state.merged_pins, n=5))

if is_ready(st.session_state.joined):
    with st.expander("空間結合プレビュー（先頭5件）", expanded=False):
        st.table(slim_gdf_preview(st.session_state.joined, n=5))

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# =========================================================
# Step 3｜Excel settings（状態でカード更新）
# =========================================================
step3_slot = st.empty()
step_card_render(
    step3_slot,
    "Step 3｜Excelを読み込み（ヘッダー行・住所列を設定）",
    "ヘッダー行が合っていそうならプレビューが緑になります（色＋テキストで判定）。",
    done=False,
)

f_xlsx = st.file_uploader("圃場登録代行シート（Excel）", type=["xlsx", "xls"], key="xlsx")

st.session_state.strip_last_num = st.checkbox(
    "末尾の枝番（-数字）を無視して照合（. / , / 、 の枝番は常に削除）",
    value=bool(st.session_state.strip_last_num),
)

excel_ready = False
cand = []
pre = None

if f_xlsx:
    h = hash(f_xlsx.getvalue())
    if st.session_state.excel_hash != h:
        st.session_state.update({"sheet_name": None, "header_row": None, "excel_hash": h, "excel_addr_col": None})
        st.session_state.matched = None

    try:
        xls = pd.ExcelFile(f_xlsx)
        sheets = xls.sheet_names
    except Exception as e:
        st.error(f"Excelのシート取得に失敗: {e}")
        sheets = []

    if sheets:
        st.session_state.sheet_name = st.selectbox(
            "シート名",
            sheets,
            index=0 if st.session_state.sheet_name not in sheets else sheets.index(st.session_state.sheet_name),
        )

        pre = pd.read_excel(f_xlsx, sheet_name=st.session_state.sheet_name, header=None, nrows=40)
        st.caption("Excelプレビュー（最初の40行）")
        st.dataframe(pre, use_container_width=True, height=260)

        cand = suggest_header_rows(pre)
        default_header = cand[0] if cand else 0

        c1, c2 = st.columns([1, 1.6], gap="large")
        with c1:
            hdr = st.number_input(
                "ヘッダー行（0始まり）",
                min_value=0, max_value=len(pre) - 1,
                value=st.session_state.header_row if st.session_state.header_row is not None else int(default_header),
                step=1,
            )
            if cand:
                pick = st.radio("候補（おすすめ）", options=cand, index=0, format_func=lambda i: f"行 {i}")
                hdr = pick
            st.session_state.header_row = int(hdr)

        with c2:
            try:
                tmp = pd.read_excel(
                    f_xlsx,
                    sheet_name=st.session_state.sheet_name,
                    header=st.session_state.header_row,
                    nrows=10,
                    dtype=str,
                )
                good_hdr = is_good_header_choice(pre, st.session_state.header_row, list(tmp.columns), cand)

                st.caption("ヘッダー適用後プレビュー")
                st.dataframe(style_header_preview(tmp, good_hdr), use_container_width=True, height=220)
                st.write("判定:", "✅ ヘッダー適合の可能性が高い" if good_hdr else "⚠️ ヘッダーが合っていない可能性")

                excel_candidates = [c for c in tmp.columns if any(h in str(c) for h in ["住所地番", "住所", "地番"])]
                if st.session_state.excel_addr_col not in list(tmp.columns):
                    st.session_state.excel_addr_col = excel_candidates[0] if excel_candidates else list(tmp.columns)[0]

                st.session_state.excel_addr_col = st.selectbox(
                    "住所列（Excel側）",
                    options=list(tmp.columns),
                    index=list(tmp.columns).index(st.session_state.excel_addr_col)
                    if st.session_state.excel_addr_col in list(tmp.columns) else 0,
                )
                excel_ready = True

            except Exception as e:
                st.error(f"ヘッダー適用プレビューでエラー: {e}")

step_card_render(
    step3_slot,
    "Step 3｜Excelを読み込み（ヘッダー行・住所列を設定）",
    "ヘッダー行が合っていそうならプレビューが緑になります（色＋テキストで判定）。",
    done=excel_ready,
)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# =========================================================
# Step 4｜Matching
# =========================================================
done_match = is_ready(st.session_state.matched)
can_match = done_join and excel_ready and (st.session_state.header_row is not None) and (st.session_state.excel_addr_col is not None)

step4_slot = st.empty()
step_card_render(
    step4_slot,
    "Step 4｜住所照合（Excel → 筆ポリゴン）",
    f"空間結合結果の「{ADDRESS_COL_CANONICAL}」を辞書化して、Excelの住所にポリゴンを付与します。",
    done=done_match,
)

match_clicked = st.button("🚀 住所照合を実行", use_container_width=True, disabled=not can_match)

if match_clicked:
    excel_addr = st.session_state.excel_addr_col
    with st.spinner("照合中…"):
        df = pd.read_excel(
            f_xlsx,
            sheet_name=st.session_state.sheet_name,
            header=st.session_state.header_row,
            dtype=str,  # 408.0 / 0000484 を壊さない（ここで整形する）
        )

        if excel_addr not in df.columns:
            st.error("選択した住所列がExcelに存在しません。住所列の選択を見直してください。")
            st.stop()

        before = len(df)
        df = df.dropna(subset=[excel_addr]).copy()
        dropped = before - len(df)

        sig = gdf_signature(st.session_state.joined, ADDRESS_COL_CANONICAL)
        d1, d2 = build_addr_dict(
            st.session_state.joined,
            ADDRESS_COL_CANONICAL,
            sig,
            bool(st.session_state.strip_last_num),
        )

        matched = apply_match(df, excel_addr, d1, d2, bool(st.session_state.strip_last_num))
        st.session_state.matched = matched

    if dropped > 0:
        st.info(f"住所が空の {dropped:,} 件を除外しました。")
    st.success("✅ 住所照合が完了しました。")
    st.rerun()

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# =========================================================
# Step 5｜Map + Export (Shapefile only, FieldName only)
# =========================================================
step5_slot = st.empty()
step_card_render(
    step5_slot,
    "Step 5｜地図表示 & 出力（Shapefileのみ）",
    "一致データのみを対象にします。Shapefile属性は FieldName（圃場名）だけを書き込みます。",
    done=False,
)

if is_ready(st.session_state.matched):
    m = st.session_state.matched
    ok = int((m["match_status"] == "一致").sum())
    tot = int(len(m))
    ng = tot - ok
    rate = ok / tot if tot else 0

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"<div class='kpi'><b>一致</b><br>{ok:,}</div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi'><b>未一致</b><br>{ng:,}</div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi'><b>一致率</b><br>{rate:.1%}</div>", unsafe_allow_html=True)

    st.markdown("### 結果プレビュー（先頭20行）")
    st.dataframe(m.head(20), use_container_width=True)

    st.session_state.show_map = st.checkbox("🗺️ 地図を表示する（重い場合はOFF）", value=bool(st.session_state.show_map))

    mg = m[m["match_status"] == "一致"].copy()
    mg["geometry"] = mg["geometry_wkt"].apply(safe_load_wkt)
    mg = mg.dropna(subset=["geometry"])
    gdf = gpd.GeoDataFrame(mg, geometry="geometry", crs="EPSG:4326")

    if gdf.empty:
        st.warning("一致データが無いため、地図表示・出力はできません。")
        step_card_render(
            step5_slot,
            "Step 5｜地図表示 & 出力（Shapefileのみ）",
            "一致データのみを対象にします。Shapefile属性は FieldName（圃場名）だけを書き込みます。",
            done=False,
        )
    else:
        # 圃場名の元列（優先順）
        label_candidates = ["圃場名", "FieldName", "field_name", "name", "圃場", "圃場ID", "FieldID"]
        src_name_col = next((c for c in label_candidates if c in gdf.columns), None)
        if src_name_col is None and st.session_state.excel_addr_col in gdf.columns:
            src_name_col = st.session_state.excel_addr_col  # 最低限のフォールバック

        # 地図
        if st.session_state.show_map:
            lat, lon = safe_centroid_lonlat(gdf)
            mp = folium.Map(location=[lat, lon], zoom_start=14)

            gdf_map = gdf.drop(columns=["geometry_wkt"], errors="ignore")

            # propertiesがShapelyを含まないよう文字列化（安全）
            for c in [c for c in gdf_map.columns if c != "geometry"]:
                gdf_map[c] = gdf_map[c].astype(str).fillna("")

            excel_addr = st.session_state.excel_addr_col
            tooltip_fields = [excel_addr] if excel_addr in gdf_map.columns else []

            folium.GeoJson(
                gdf_map.__geo_interface__,
                tooltip=folium.features.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
            ).add_to(mp)

            # ラベル（大量件数は事故防止）
            if src_name_col:
                if len(gdf) > 1000:
                    st.info("データ件数が多いため、ラベル表示は無効化しています（性能保護）。")
                else:
                    default_on = len(gdf) <= 200
                    show_labels = st.checkbox(f"🏷️ 圃場名ラベルを表示（元列: {src_name_col}）", value=default_on)
                    if show_labels:
                        for _, row in gdf.iterrows():
                            label = format_label(row.get(src_name_col, ""))
                            if not label:
                                continue
                            p = row.geometry.representative_point()
                            folium.Marker(
                                location=[p.y, p.x],
                                icon=folium.DivIcon(
                                    html=(
                                        f"<div style="
                                        f"'font-size:{LABEL_FONT_SIZE}px;"
                                        f"font-weight:800;"
                                        f"color:#111;"
                                        f"background:rgba(255,255,255,0.75);"
                                        f"padding:1px 4px;"
                                        f"border-radius:6px;"
                                        f"border:1px solid rgba(0,0,0,0.15);"
                                        f"white-space:nowrap;'>"
                                        f"{html.escape(label)}"
                                        f"</div>"
                                    )
                                ),
                            ).add_to(mp)

            minx, miny, maxx, maxy = gdf.total_bounds
            mp.fit_bounds([[miny, minx], [maxy, maxx]])
            folium_static(mp, width=1100, height=650)

        # ------------- 出力（FieldNameのみ）-------------
        st.markdown("### 出力（Shapefile ZIP）")
        out_prefix = st.text_input("出力ファイル名（拡張子なし）", value="houjou_data")

        if src_name_col:
            names = gdf[src_name_col].apply(format_label).fillna("").astype(str)
        else:
            names = pd.Series([""] * len(gdf))

        # DBFは1フィールド最大254bytes目安 → 念のため短めに切る
        names = names.str.slice(0, 200)

        gdf_export = gpd.GeoDataFrame(
            {"FieldName": names, "geometry": gdf.geometry},
            geometry="geometry",
            crs="EPSG:4326",
        )

        shp_bytes = gdf_to_shapefile_zip_bytes(gdf_export, filename_prefix=out_prefix)

        st.download_button(
            "📥 Shapefile（ZIP）をダウンロード",
            data=shp_bytes,
            file_name=f"{out_prefix}.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.caption("※ Shapefileの属性は FieldName（圃場名）だけです（先頭ゼロは自動除去）。")

        # Step5 完了（一致データが存在する時点で完了扱い）
        step_card_render(
            step5_slot,
            "Step 5｜地図表示 & 出力（Shapefileのみ）",
            "一致データのみを対象にします。Shapefile属性は FieldName（圃場名）だけを書き込みます。",
            done=True,
        )
else:
    st.info("Step 4 の「住所照合を実行」を押すと、ここに地図と Shapefile 出力が表示されます。")
