import streamlit as st
import browser_cookie3
import urllib.parse, json
import requests
import folium
import shapefile
import tempfile
import os
import zipfile
from streamlit_folium import st_folium

st.set_page_config(page_title="AgriNote 土地取得 (Cookie版)", layout="wide")
st.title("AgriNote 土地情報取得 & Shapefile エクスポート (Cookie利用)")

# --- 1️⃣ ブラウザ選択 ---
st.subheader("1️⃣ ご使用中のブラウザを選んでください")
st.caption("※ Safariには対応していません。ChromeやFirefoxをご利用ください")
browser = st.radio("使用ブラウザ", ["Chrome", "Firefox", "Edge"])
if st.button("✅ 決定"):
    try:
        with st.spinner("Cookieを取得中..."):
            if browser == "Chrome":
                cookies = browser_cookie3.chrome(domain_name="agri-note.jp")
            elif browser == "Firefox":
                cookies = browser_cookie3.firefox(domain_name="agri-note.jp")
            elif browser == "Edge":
                cookies = browser_cookie3.edge(domain_name="agri-note.jp")
            else:
                st.error("未対応のブラウザです")
                st.stop()
            st.session_state.cookie_dict = {c.name: c.value for c in cookies}
            st.success("Cookieの取得に成功しました！")
    except Exception as e:
        st.error(f"❌ クッキー取得エラー: {e}")
        st.stop()

# --- 2️⃣ Cookie取得後の処理 ---
if "cookie_dict" in st.session_state:
    cookie_dict = st.session_state.cookie_dict
    st.subheader("Cookie情報")
    st.json(cookie_dict)

    # --- クッキーから必要情報を抽出 ---
    an_api_token = urllib.parse.unquote(cookie_dict.get("an_api_token", "")).split(":")[0]
    an_login_status = json.loads(urllib.parse.unquote(cookie_dict.get("an_login_status", "{}")))
    csrf = an_login_status.get("csrf")
    uuid = cookie_dict.get("tracking_user_uuid")

    if not (an_api_token and csrf and uuid):
        st.error("必要なクッキーが不足しています。AgriNoteにログイン済みか確認してください。")
        st.stop()

    headers = {
        "x-an-csrf-token": csrf,
        "x-user-uuid": uuid,
        "x-agri-note-api-client": "v2.97.0",
        "x-requested-with": "XMLHttpRequest",
        "referer": "https://agri-note.jp/b/",
        "user-agent": "Mozilla/5.0"
    }
    cookies_req = {
        "an_api_token": cookie_dict["an_api_token"],
        "an_login_status": cookie_dict["an_login_status"],
        "tracking_user_uuid": cookie_dict["tracking_user_uuid"]
    }

    # --- 3️⃣ API呼び出し ---
    st.subheader("2️⃣ 土地データの取得")
    if st.button("🔄 土地データを取得"):
        res = requests.get("https://agri-note.jp/an-api/v1/agri_fields", headers=headers, cookies=cookies_req)
        if res.status_code == 200:
            fields = res.json()
            st.session_state.fields = fields
            st.success(f"{len(fields)}件の圃場を取得しました！")
        else:
            st.error(f"取得失敗: {res.status_code}")

# --- 4️⃣ マップ表示 & Shapefile作成 ---
if st.session_state.get("fields"):
    st.subheader("3️⃣ 土地マップとShapefile出力")
    fields = st.session_state.fields
    center = fields[0]["center_latlng"]
    fmap = folium.Map(location=[center["lat"], center["lng"]], zoom_start=15)

    field_map = {}
    options = []
    for f in fields:
        name = f['field_name'] or f"ID: {f['id']}"
        area = round(f.get("calculation_area", 0), 2)
        label = f"{name} ({area}a)"
        field_map[label] = f
        options.append(label)
        coords = [(pt['lat'], pt['lng']) for pt in f['region_latlngs']]
        folium.Polygon(
            locations=coords,
            popup=name,
            tooltip=label,
            color='red', fill=True, fill_opacity=0.5
        ).add_to(fmap)

    st_folium(fmap, width=700, height=500)

    selected = st.multiselect("Shapefileに出力する圃場を選択", options, default=options)
    selected_fields = [field_map[label] for label in selected]

    if selected_fields:
        temp_dir = tempfile.mkdtemp()
        shp_path = os.path.join(temp_dir, "selected_fields")

        with shapefile.Writer(shp_path, shapeType=shapefile.POLYGON) as w:
            w.field("id", "N")
            w.field("name", "C")
            w.field("area", "F", decimal=3)
            for field in selected_fields:
                coords = [(pt["lng"], pt["lat"]) for pt in field["region_latlngs"]]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                w.poly([coords])
                w.record(field["id"], field["field_name"], field["calculation_area"])

        zip_path = os.path.join(temp_dir, "agnote_selected_shapefile.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for ext in ["shp", "shx", "dbf"]:
                zipf.write(f"{shp_path}.{ext}", arcname=f"selected_fields.{ext}")

        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Shapefileをダウンロード",
                data=f,
                file_name="agnote_selected_shapefile.zip",
                mime="application/zip"
            )
