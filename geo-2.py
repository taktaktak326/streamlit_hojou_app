# streamlit_app.py
# 必要: pip install streamlit geopandas shapely folium streamlit-folium xlsxwriter rtree

import re, math
from io import BytesIO

import geopandas as gpd
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
from shapely.geometry.base import BaseGeometry

# ========== Page / Session ==========
st.set_page_config(page_title="農地ピン×筆ポリゴン 結合", layout="wide")
for k, v in {"sheet_name": None, "header_row": None, "addr_col": None, "excel_hash": None}.items():
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

def gdf_signature(gdf: gpd.GeoDataFrame, addr_col: str) -> tuple:
    bounds = tuple(map(float, gdf.total_bounds)); n = int(len(gdf))
    h = int(pd.util.hash_pandas_object(gdf[addr_col].astype(str), index=False).sum()) if addr_col in gdf.columns else 0
    return (n, bounds, h)

@st.cache_data(show_spinner=False)
def build_addr_dict(_gdf: gpd.GeoDataFrame, col: str, gdf_sig: tuple):
    t = _gdf[[col,"geometry"]].dropna(subset=[col,"geometry"]).copy()
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

c1, c2 = st.columns(2)
with c1:
    st.subheader("1️⃣ GeoJSONアップロード")
    f_pori = st.file_uploader("筆ポリゴン（複数可）", type=["geojson"], accept_multiple_files=True, key="poris")
    f_pins = st.file_uploader("農地ピン（複数可）", type=["geojson"], accept_multiple_files=True, key="pins")

with c2:
    st.subheader("2️⃣ 圃場登録代行シート（Excel）")
    f_xlsx = st.file_uploader("Excelを選択", type=["xlsx","xls"], key="xlsx")
    if f_xlsx:
        h = hash(f_xlsx.getvalue())
        if st.session_state.excel_hash != h:
            st.session_state.update({"sheet_name": None, "header_row": None, "addr_col": None, "excel_hash": h})

        try:
            xls = pd.ExcelFile(f_xlsx); sheets = xls.sheet_names
        except Exception as e:
            st.error(f"Excelのシート取得に失敗: {e}"); st.stop()

        idx = sheets.index(st.session_state.sheet_name) if st.session_state.sheet_name in sheets else 0
        st.session_state.sheet_name = st.selectbox("シート名", sheets, index=idx)

        pre = pd.read_excel(f_xlsx, sheet_name=st.session_state.sheet_name, header=None, nrows=40)
        st.caption("プレビュー（最初の40行）"); st.dataframe(pre, use_container_width=True, height=280)

        st.markdown("**🧠 ヘッダー候補（住所/地番/筆などを優先）**")
        cand = suggest_header_rows(pre); default_header = cand[0] if cand else 0

        cols = st.columns([2,3])
        with cols[0]:
            hdr_num = st.number_input("ヘッダー行（0始まり）", 0, len(pre)-1,
                                      value=st.session_state.header_row if st.session_state.header_row is not None else default_header, step=1)
            if cand:
                pick = st.radio("候補", options=cand, index=0, format_func=lambda i: f"行 {i}（候補）")
                hdr_num = pick
            st.session_state.header_row = int(hdr_num)

        with cols[1]:
            try:
                tmp = pd.read_excel(f_xlsx, sheet_name=st.session_state.sheet_name,
                                    header=st.session_state.header_row, nrows=10)
                st.caption("ヘッダー適用後プレビュー（上位10行）")
                st.dataframe(tmp, use_container_width=True, height=240)
                likely = [c for c in tmp.columns if any(h in str(c) for h in ["住所地番","住所","地番"])]
                if likely: st.info("住所っぽい列候補: " + ", ".join(map(str, likely)))
            except Exception as e:
                st.error(f"ヘッダー適用プレビューでエラー: {e}")

st.divider()

# ========== Run ==========
with st.form("run"):
    submitted = st.form_submit_button("処理を開始", use_container_width=True)

if submitted:
    prog = st.progress(0); msg = st.empty()
    TOTAL_STEPS = 7
    def tick(i): prog.progress(int(i * 100 / TOTAL_STEPS))  # 常に%で更新

    # 入力チェック
    if not f_pori or not f_pins:
        st.error("筆ポリゴンと農地ピンのGeoJSONを両方アップロードしてください。"); st.stop()
    if not f_xlsx or not st.session_state.sheet_name or st.session_state.header_row is None:
        st.error("Excelのシートとヘッダー行を指定してください。"); st.stop()

    # 1) GeoJSON
    msg.text("GeoJSON読み込み…")
    try:
        g_pori = read_geojson(f_pori); g_pin = read_geojson(f_pins)
        assert not g_pori.empty and not g_pin.empty
    except Exception as e:
        st.error(f"GeoJSONの読み込みに失敗: {e}"); st.stop()
    tick(1)

    # 2) 空間結合
    msg.text("筆×ピンの空間結合…")
    try:
        joined = gpd.sjoin(g_pori, g_pin, predicate="covers", how="left")
    except Exception:
        joined = gpd.sjoin(g_pori, g_pin, predicate="intersects", how="left")
    joined = joined.drop_duplicates()
    st.subheader("📌 空間結合（先頭5件）"); st.write(joined.head())
    tick(2)

    # 3) 住所カラム選択（保持）
    msg.text("住所カラムの選択…")
    candidates = ["住所","住所地番","Address","address","location","name"]
    jcols = list(joined.columns)
    auto = next((c for c in candidates if c in jcols), jcols[0])
    if st.session_state.addr_col not in jcols: st.session_state.addr_col = auto
    idx = jcols.index(st.session_state.addr_col)
    addr_col = st.selectbox("住所に使うカラム", jcols, index=idx, key="addr_sel")
    st.session_state.addr_col = addr_col
    tick(3)

    # 4) Excel（住所NaNはここで除外）
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
    df = df.dropna(subset=[excel_addr]).copy()  # 住所NaNを除外
    dropped = before - len(df)
    if dropped > 0: st.info(f"住所がNaNの {dropped:,} 件を除外しました。")

    df[excel_addr] = df[excel_addr].astype(str)
    st.subheader("📄 Excel（先頭5件）"); st.write(df.head())
    tick(4)

    # 5) 住所辞書（unhashable回避＆住所NaN/geometryNaNは除外）
    msg.text("住所辞書構築…")
    sig = gdf_signature(joined, addr_col)
    d1, d2 = build_addr_dict(joined, addr_col, sig)
    tick(5)

    # 6) 照合＋WKT
    msg.text("住所照合…")
    matched = apply_match(df, excel_addr, d1, d2)
    matched["geometry_wkt"] = matched["geom"].map(to_wkt_safe)
    tick(6)

    # 7) 集計・DL・地図・ページネーション
    msg.text("集計・地図描画…")
    ok = (matched["geom"]!="一致なし").sum()
    tot = len(matched); ng = tot-ok; rate = (ok/tot if tot else 0)
    a,b,c = st.columns(3)
    a.metric("一致件数", f"{ok:,}"); b.metric("未一致件数", f"{ng:,}"); c.metric("一致率", f"{rate:.1%}")

    # 一致のみページネーション表示
    if ok > 0:
        st.subheader("✅ 一致データ（全件・ページネーション）")
        matched_only = matched[matched["geom"]!="一致なし"].copy()

        lcol, rcol, _ = st.columns([1,1,4])
        with lcol:
            page_size = st.selectbox("ページサイズ", [25, 50, 100, 200], index=1, key="pg_size")
        total_pages = max(1, math.ceil(len(matched_only)/page_size))
        with rcol:
            default_page = min(st.session_state.get("pg_no", 1), total_pages)
            page = st.number_input("ページ", min_value=1, max_value=total_pages, value=default_page, step=1, key="pg_no")

        page_df = paginate(matched_only, int(page), int(page_size))
        start_idx = (int(page)-1)*int(page_size)
        page_df = page_df.drop(columns=["geom"]).copy()
        page_df.insert(0, "No.", range(start_idx+1, start_idx+1+len(page_df)))
        st.dataframe(page_df, use_container_width=True, height=420)
        st.caption(f"全 {len(matched_only):,} 件中、{start_idx+1:,}–{start_idx+len(page_df):,} 件を表示 / {total_pages} ページ")

        st.download_button("📥 このページをCSVで保存",
            page_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"一致一覧_p{page}_n{page_size}.csv", mime="text/csv"
        )

    # 全件プレビュー
    st.subheader("🔎 照合結果プレビュー（先頭20行）")
    st.dataframe(matched.head(20), use_container_width=True)

    # ダウンロード
    st.download_button("📥 全件Excel", to_excel_bytes(matched),
        file_name=f"{st.session_state.sheet_name}_照合_全件.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    um = matched[matched["geom"]=="一致なし"]
    if not um.empty:
        st.download_button("📥 未一致のみ", to_excel_bytes(um),
            file_name=f"{st.session_state.sheet_name}_未一致のみ.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # 地図
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

    # ★最後に必ず 100% にする
    prog.progress(100)
    msg.text("処理完了 🎉")
