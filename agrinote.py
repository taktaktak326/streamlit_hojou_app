import streamlit as st
st.set_page_config(page_title="AgriNote Shapefile Exporter", layout="wide")

import time
import requests
import json
import urllib.parse
import geopandas as gpd
import os
import zipfile
import tempfile
import folium
from shapely.geometry import Polygon
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from streamlit_folium import st_folium

st.title("AgriNote 土地情報取得 & Shapefile エクスポート")

if "fields" not in st.session_state:
    st.session_state.fields = None

EMAIL = st.text_input("📧 メールアドレス")
PASSWORD = st.text_input("🔑 パスワード", type="password")

if st.button("🔐 ログイン & データ取得"):
    try:
        with st.spinner("ログイン中..."):
            chrome_options = Options()
            chrome_options.binary_location = "/usr/bin/chromium"
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

            driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"), options=chrome_options)
            driver.get("https://agri-note.jp/b/login/")
            time.sleep(2)

            inputs = driver.find_elements(By.CLASS_NAME, "_1g2kt34")
            if len(inputs) < 2:
                st.error("❌ ログインフォームが見つかりません")
                driver.quit()
                st.stop()

            inputs[0].send_keys(EMAIL)
            inputs[1].send_keys(PASSWORD)
            inputs[1].send_keys(Keys.RETURN)

            time.sleep(5)

            cookies_list = driver.get_cookies()
            cookie_dict = {cookie['name']: cookie['value'] for cookie in cookies_list}
            required = ['an_api_token', 'an_login_status', 'tracking_user_uuid']

            if not all(k in cookie_dict for k in required):
                st.error("❌ 必要なCookieが見つかりません")
                st.write("Cookie debug", cookie_dict)
                driver.quit()
                st.stop()

            csrf_token = json.loads(urllib.parse.unquote(cookie_dict['an_login_status']))["csrf"]

            cookies = {
                "an_api_token": cookie_dict['an_api_token'],
                "an_login_status": cookie_dict['an_login_status'],
                "tracking_user_uuid": cookie_dict['tracking_user_uuid'],
            }

            headers = {
                "x-an-csrf-token": csrf_token,
                "x-user-uuid": cookie_dict['tracking_user_uuid'],
                "x-agri-note-api-client": "v2.97.0",
                "x-requested-with": "XMLHttpRequest",
                "referer": "https://agri-note.jp/b/",
                "user-agent": "Mozilla/5.0"
            }

            response = requests.get("https://agri-note.jp/an-api/v1/agri_fields", headers=headers, cookies=cookies)
            driver.quit()

            if response.status_code != 200:
                st.error(f"API取得失敗: {response.status_code}")
                st.stop()

            st.session_state.fields = response.json()
            st.success(f"✅ {len(st.session_state.fields)} 件の土地データを取得しました")

    except Exception as e:
        st.error(f"予期せぬエラー: {e}")

# === マップ & 表形式選択 ===
if st.session_state.fields:
    st.subheader("🖼️ 土地マップ")

    center = st.session_state.fields[0]["center_latlng"]
    fmap = folium.Map(location=[center["lat"], center["lng"]], zoom_start=15)

    field_map = {}
    table_data = []

    for f in st.session_state.fields:
        name = f['field_name'] or f"ID: {f['id']}"
        area = round(f.get("calculation_area", 0), 2)
        coords = [(pt['lat'], pt['lng']) for pt in f['region_latlngs']]
        folium.Polygon(
            locations=coords,
            popup=name,
            tooltip=f"{name} ({area}a)",
            color='red',
            fill=True,
            fill_opacity=0.5
        ).add_to(fmap)
        key = f"{name} ({area}a)"
        field_map[key] = f
        table_data.append((key, area))

    st_folium(fmap, width=700, height=500)

    st.subheader("✅ 表形式で圃場選択")

    all_selected = st.checkbox("すべて選択", value=True)
    selections = {}
    total_area = 0.0

    for label, area in table_data:
        selected = st.checkbox(label, value=all_selected, key=label)
        selections[label] = selected
        if selected:
            total_area += area

    selected_fields = [field_map[label] for label, selected in selections.items() if selected]

    st.markdown(f"**🧮 選択した圃場数: {len(selected_fields)} / 合計面積: {round(total_area, 2)}a**")

    if selected_fields:
        temp_dir = tempfile.mkdtemp()
        chunk_size = 300
        chunks = [selected_fields[i:i + chunk_size] for i in range(0, len(selected_fields), chunk_size)]

        for idx, chunk in enumerate(chunks):
            polygons = []
            names = []
            for field in chunk:
                coords = [(pt["lng"], pt["lat"]) for pt in field["region_latlngs"]]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                polygon = Polygon(coords)
                name = field.get("field_name") or f"ID: {field['id']}"
                names.append(name)
                polygons.append(polygon)

            gdf = gpd.GeoDataFrame({"FieldName": names, "geometry": polygons}, crs="EPSG:4326")
            shp_dir = os.path.join(temp_dir, f"shp_{idx+1}")
            os.makedirs(shp_dir, exist_ok=True)
            shp_path = os.path.join(shp_dir, "selected_fields.shp")
            gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")

            zip_path = os.path.join(temp_dir, f"agnote_xarvio_selected_part_{idx+1}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for file in os.listdir(shp_dir):
                    zipf.write(os.path.join(shp_dir, file), arcname=file)

            with open(zip_path, "rb") as f:
                st.download_button(
                    label=f"⬇️ Part {idx+1} - Shapefile ダウンロード",
                    data=f,
                    file_name=f"agnote_xarvio_selected_part_{idx+1}.zip",
                    mime="application/zip"
                )
    else:
        st.info("🔍 土地を選択してください")
