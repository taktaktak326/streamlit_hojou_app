# streamlit_app.py
# 必要: pip install streamlit geopandas shapely folium streamlit-folium xlsxwriter rtree

import re, math
from io import BytesIO

import geopandas as gpd
import pandas as pd
import streamlit as st
import folium
from shapely.geometry.base import BaseGeometry
from streamlit_folium import folium_static

# ========== Page / Session ==========
st.set_page_config(page_title="農地ピン×筆ポリゴン 結合", layout="wide")
for k, v in {
    "sheet_name": None, "header_row": None, "addr_col": None, "excel_hash": None,
    "joined": None  # ← 空間結合の結果を保持
}.items():
    st.session_state.setdefault(k, v)

# ========== Utils ==========
HYPHENS = r"[‐-‒–—―ー－-]"
HEADER_HINTS = ["住所地番","住所","地番","筆","地目","面積","圃場","農地","字","番地"]

def to_half(s): return s.translate(str.maketrans("０１２３４５６７８９","0123456789")) if isinstance(s,str) else s

def norm_addr_key(s: str) -> str:
    if not isinstance(s, str) or pd.isna(s): return ""
    s = to_half(s.strip()).lower().replace("　"," ")
    s = re.sub(HYPHENS,"-", s); s = re.sub(r"\s+","", s)
    s = s.replace("丁目","-").replace("番地","-").replace("番","-").replace("号","")
    return re.sub(r"-\d{1,4}$","", s)

def addr_key_loose(k: str) -> str:
    s = re.sub(r"(東京都|北海道|京都府|大阪府|..県|..都|..道|..府)","", k)
    return re.sub(r".{1,6}(市|区|町|村)","", s)

def score_header_row(vals) -> int:
    score = 0
    for v in vals:
        s = str(v)
        score += 2*sum(h in s for h in HEADER_HINTS)
        if 2 <= len(s) <= 12 and re.fullmatch(r"[^\d\s]{2,}", s or ""): score += 1
    return score

def suggest_header_rows(pre: pd.DataFrame, topk=6):
    n = min(40, len(pre))
    cand = sorted([(i, score_header_row(pre.iloc[i].values)) for i in range(n)],
                  key=lambda x: x[1], reverse=True)
    return [i for i,sc in cand[:topk] if sc>0]

@st.cache_data(show_spinner=False)
def read_geojson(files):
    gdfs = []
    for f in files:
        gdf = gpd.read_file(f)
        gdf = (gdf.set_crs(epsg=4326, allow_override=True) if gdf.crs is None else gdf.to_crs(4326))
        gdfs.append(gdf)
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")

@st.cache_data(show_spinner=False)
def sjoin_pori_pin(_g_pori: gpd.GeoDataFrame, _g_pin: gpd.GeoDataFrame):
    """GeoJSONの空間結合。GeoDataFrameはunhashableなのでキャッシュ鍵から除外"""
    try:
        j = gpd.sjoin(_g_pori, _g_pin, predicate="covers", how="left")
    except Exception:
        j = gpd.sjoin(_g_pori, _g_pin, predicate="intersects", how="left")
    return j.drop_duplicates()


def gdf_signature(gdf: gpd.GeoDataFrame, addr_col: str) -> tuple:
    bounds = tuple(map(float, gdf.total_bounds)); n = int(len(gdf))
    h = int(pd.util.hash_pandas_object(gdf[addr_col].astype(str), index=False).sum()) if addr_col in gdf.columns else 0
    return (n, bounds, h)

# GeoDataFrameはunhashable → 第1引数を _gdf にして鍵から除外、代わりに gdf_sig を使う
@st.cache_data(show_spinner=False)
def build_addr_dict(_gdf: gpd.GeoDataFrame, col: str, gdf_sig: tuple):
    t = _gdf[[col,"geometry"]].dropna(subset=[col,"geometry"]).copy()  # 住所NaN/geomNaN除外
    t["k1"] = t[col].astype(str).map(norm_addr_key)
    t["k2"] = t["k1"].map(addr_key_loose)
    return (t.groupby("k1")["geometry"].first().to_dict(),
            t.groupby("k2")["geometry"].first().to_dict())

def apply_match(df: pd.DataFrame, col: str, d1: dict, d2: dict):
    out = df.copy()
    out["__k"]  = out[col].astype(str).map(norm_addr_key)
    out["geom"] = out["__k"].map(d1)
    miss        = out["geom"].isna()
    out.loc[miss,"geom"] = out.loc[miss,"__k"].map(addr_key_loose).map(d2)
    out["geom"] = out["geom"].apply(lambda g: g if isinstance(g, BaseGeometry) else "一致なし")
    return out.drop(columns="__k")

def to_excel_bytes(df: pd.DataFrame) -> BytesIO:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        df.to_excel(w, sheet_name="MatchedData", index=False)
    bio.seek(0); return bio

def to_wkt_safe(g): return g.wkt if isinstance(g, BaseGeometry) else ""

def paginate(df: pd.DataFrame, page: int, page_size: int) -> pd.DataFrame:
    start = (page-1) * page_size
    return df.iloc[start:start+page_size]

# ========== UI ==========
st.title("農地ピンと筆ポリゴンの結合 & 圃場登録代行シートの統合")

# ---- A. GeoJSONをアップ → 空間結合を自動実行 → 住所カラムを選択 ----
left, right = st.columns(2)
with left:
    st.subheader("1️⃣ GeoJSONアップロード")
    f_pori = st.file_uploader("筆ポリゴン（複数可）", type=["geojson"], accept_multiple_files=True, key="poris")
    f_pins = st.file_uploader("農地ピン（複数可）", type=["geojson"], accept_multiple_files=True, key="pins")

with right:
    st.subheader("2️⃣ 圃場登録代行シート（Excel）")
    f_xlsx = st.file_uploader("Excelを選択", type=["xlsx","xls"], key="xlsx")

# GeoJSONがそろったら空間結合を実施（“処理開始”前に準備する）
if f_pori and f_pins:
    try:
        g_pori = read_geojson(f_pori); g_pin = read_geojson(f_pins)
        st.session_state.joined = sjoin_pori_pin(g_pori, g_pin)
        st.success("✅ 空間結合を実行しました。下で『住所に使うカラム』を選択してください。")
        st.caption("空間結合（先頭5件）")
        st.dataframe(st.session_state.joined.head(), use_container_width=True)
    except Exception as e:
        st.error(f"空間結合でエラー: {e}")
        st.session_state.joined = None

# 住所カラム選択（空間結合が済んでいる前提で、処理前に選ぶ）
if st.session_state.joined is not None and not st.session_state.joined.empty:
    jcols = list(st.session_state.joined.columns)
    candidates = ["住所","住所地番","Address","address","location","name"]
    auto = next((c for c in candidates if c in jcols), jcols[0])
    if st.session_state.addr_col not in jcols: st.session_state.addr_col = auto
    st.subheader("3️⃣ 住所に使うカラムを選択")
    st.session_state.addr_col = st.selectbox(
        "住所カラム", jcols, index=jcols.index(st.session_state.addr_col), help="住所・住所地番・location 等"
    )

st.divider()

# ---- B. Excelヘッダー行の選択（プレビュー＋候補） ----
if f_xlsx:
    h = hash(f_xlsx.getvalue())
    if st.session_state.excel_hash != h:
        st.session_state.update({"sheet_name": None, "header_row": None, "excel_hash": h})
    try:
        xls = pd.ExcelFile(f_xlsx); sheets = xls.sheet_names
    except Exception as e:
        st.error(f"Excelのシート取得に失敗: {e}")
        sheets = []
    if sheets:
        idx = sheets.index(st.session_state.sheet_name) if st.session_state.sheet_name in sheets else 0
        st.session_state.sheet_name = st.selectbox("シート名", sheets, index=idx)
        pre = pd.read_excel(f_xlsx, sheet_name=st.session_state.sheet_name, header=None, nrows=40)
        st.caption("Excelプレビュー（最初の40行）")
        st.dataframe(pre, use_container_width=True, height=280)

        st.markdown("**🧠 ヘッダー候補（住所/地番/筆などを優先）**")
        cand = suggest_header_rows(pre); default_header = cand[0] if cand else 0
        c1, c2 = st.columns([2,3])
        with c1:
            hdr = st.number_input("ヘッダー行（0始まり）", 0, len(pre)-1,
                                  value=st.session_state.header_row if st.session_state.header_row is not None else default_header, step=1)
            if cand:
                pick = st.radio("候補", options=cand, index=0, format_func=lambda i: f"行 {i}（候補）")
                hdr = pick
            st.session_state.header_row = int(hdr)
        with c2:
            try:
                tmp = pd.read_excel(f_xlsx, sheet_name=st.session_state.sheet_name,
                                    header=st.session_state.header_row, nrows=10)
                st.caption("ヘッダー適用後プレビュー（上位10行）")
                st.dataframe(tmp, use_container_width=True, height=240)
                likely = [c for c in tmp.columns if any(h in str(c) for h in ["住所地番","住所","地番"])]
                if likely: st.info("住所っぽい列候補: " + ", ".join(map(str, likely)))
            except Exception as e:
                st.error(f"ヘッダー適用プレビューでエラー: {e}")

# ---- C. 「マッチング処理を開始」ボタン（空間結合＆住所カラム選択の後） ----
with st.form("match_form"):
    submitted = st.form_submit_button("🚀 マッチング処理を開始", use_container_width=True)

if submitted:
    prog = st.progress(0); msg = st.empty()
    TOTAL_STEPS = 5
    def tick(i): prog.progress(int(i*100/TOTAL_STEPS))

    # 前提チェック：空間結合と住所カラム
    if st.session_state.joined is None or st.session_state.addr_col is None:
        st.error("先にGeoJSONをアップして空間結合を完了し、『住所カラム』を選択してください。"); st.stop()
    if not f_xlsx or not st.session_state.sheet_name or st.session_state.header_row is None:
        st.error("Excelのシートとヘッダー行を指定してください。"); st.stop()

    # 1) Excel読込 & 住所NaN除外
    msg.text("Excel読み込み…")
    try:
        df = pd.read_excel(f_xlsx, sheet_name=st.session_state.sheet_name, header=st.session_state.header_row)
    except Exception as e:
        st.error(f"Excelの読み込みに失敗: {e}"); st.stop()

    excel_addr = "住所地番" if "住所地番" in df.columns else next(
        (c for c in df.columns if any(h in str(c) for h in ["住所","地番"])), None
    )
    if not excel_addr:
        st.error("Excelに『住所地番/住所/地番』に相当する列が見つかりません。")
        st.write("検出列:", list(df.columns)); st.stop()

    before = len(df)
    df = df.dropna(subset=[excel_addr]).copy()
    dropped = before - len(df)
    if dropped > 0: st.info(f"住所がNaNの {dropped:,} 件を除外しました。")
    df[excel_addr] = df[excel_addr].astype(str)
    tick(1)

    # 2) 住所辞書（joined + 選択した住所カラム）
    msg.text("住所辞書を構築中…")
    sig = gdf_signature(st.session_state.joined, st.session_state.addr_col)
    d1, d2 = build_addr_dict(st.session_state.joined, st.session_state.addr_col, sig)
    tick(2)

    # 3) 照合＋WKT
    msg.text("住所照合…")
    matched = apply_match(df, excel_addr, d1, d2)
    matched["geometry_wkt"] = matched["geom"].apply(lambda g: g.wkt if isinstance(g, BaseGeometry) else "")
    tick(3)

    # 4) 集計＋ページネーション一覧
    msg.text("集計・一覧表示…")
    ok = (matched["geom"]!="一致なし").sum()
    tot = len(matched); ng = tot-ok; rate = (ok/tot if tot else 0)
    a,b,c = st.columns(3)
    a.metric("一致件数", f"{ok:,}"); b.metric("未一致件数", f"{ng:,}"); c.metric("一致率", f"{rate:.1%}")

    st.subheader("🔎 照合結果プレビュー（先頭20行）")
    st.dataframe(matched.head(20), use_container_width=True)

    if ok > 0:
        st.subheader("✅ 一致データ（全件・ページネーション）")
        matched_only = matched[matched["geom"]!="一致なし"].copy()
        lcol, rcol, _ = st.columns([1,1,4])
        with lcol:
            page_size = st.selectbox("ページサイズ", [25, 50, 100, 200], index=1, key="pg_size")
        total_pages = max(1, math.ceil(len(matched_only)/page_size))
        with rcol:
            page = st.number_input("ページ", min_value=1, max_value=total_pages,
                                   value=min(st.session_state.get("pg_no", 1), total_pages), step=1, key="pg_no")
        page_df = paginate(matched_only, int(page), int(page_size))
        start_idx = (int(page)-1)*int(page_size)
        show_df = page_df.drop(columns=["geom"]).copy()
        show_df.insert(0, "No.", range(start_idx+1, start_idx+1+len(show_df)))
        st.dataframe(show_df, use_container_width=True, height=420)
        st.caption(f"全 {len(matched_only):,} 件中、{start_idx+1:,}–{start_idx+len(show_df):,} 件を表示 / {total_pages} ページ")

        st.download_button("📥 このページをCSVで保存",
            show_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"一致一覧_p{page}_n{page_size}.csv", mime="text/csv"
        )

    # 5) ダウンロード＋地図
    msg.text("出力生成…")
    st.download_button("📥 全件Excel", to_excel_bytes(matched),
        file_name=f"{st.session_state.sheet_name}_照合_全件.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    um = matched[matched["geom"]=="一致なし"]
    if not um.empty:
        st.download_button("📥 未一致のみ", to_excel_bytes(um),
            file_name=f"{st.session_state.sheet_name}_未一致のみ.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    mgdf = matched[matched["geom"]!="一致なし"]
    if not mgdf.empty:
        mgdf = gpd.GeoDataFrame(mgdf, geometry="geom", crs="EPSG:4326")
        m = folium.Map(zoom_start=14)
        folium.GeoJson(mgdf.__geo_interface__,
                       tooltip=folium.features.GeoJsonTooltip(fields=[excel_addr], aliases=["住所地番"])
                      ).add_to(m)
        minx,miny,maxx,maxy = mgdf.total_bounds
        m.fit_bounds([[miny, minx],[maxy, maxx]])
        st.subheader("🗺️ 一致した筆ポリゴン"); folium_static(m, width=1100, height=680)
    else:
        st.warning("一致する筆ポリゴンがありませんでした。")

    # 100% 完了表示
    prog.progress(100); msg.text("処理完了 🎉")
