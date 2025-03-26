import streamlit as st
st.set_page_config(page_title="AgriNote Shapefile Exporter", layout="wide")

import time
import requests
import json
import urllib.parse
import os
import zipfile
import tempfile
import folium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd

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

# === マップ & 選択 ===
if st.session_state.fields:
    st.subheader("🖼️ 土地マップ")

    center = st.session_state.fields[0]["center_latlng"]
    fmap = folium.Map(location=[center["lat"], center["lng"]], zoom_start=15)

    field_map = {}
    options = []
    for f in st.session_state.fields:
        name = f['field_name'] or f"ID: {f['id']}"
        area = round(f.get("calculation_area", 0), 2)
        display_name = f"{name} ({area}a)"
        options.append(display_name)
        field_map[display_name] = f
        coords = [(pt['lat'], pt['lng']) for pt in f['region_latlngs']]
        folium.Polygon(
            locations=coords,
            popup=name,
            tooltip=display_name,
            color='red',
            fill=True,
            fill_opacity=0.5
        ).add_to(fmap)

    st_folium(fmap, width=700, height=500)

    st.subheader("💪 土地選択 & ダウンロード")

    all_selected = st.checkbox("すべて選択", value=True)
    selected = st.multiselect("選択した土地のShapefileをダウンロード", options=options, default=options if all_selected else [])

    selected_fields = [field_map[label] for label in selected]

    if selected_fields:
        temp_dir = tempfile.mkdtemp()
        shp_dir = os.path.join(temp_dir, "shapefile")
        os.makedirs(shp_dir, exist_ok=True)
        shp_path = os.path.join(shp_dir, "selected_fields.shp")

        # GeoDataFrame作成
        names = []
        areas = []
        geometries = []

        for field in selected_fields:
            coords = [(pt["lng"], pt["lat"]) for pt in field["region_latlngs"]]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            polygon = Polygon(coords)
            geometries.append(polygon)
            names.append(field["field_name"] or f"ID: {field['id']}")
            areas.append(field["calculation_area"])

        df = pd.DataFrame({"name": names, "area": areas})
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries(geometries, crs="EPSG:4326"))

        gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")

        zip_path = os.path.join(temp_dir, "agnote_xarvio_selected_shapefile.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(shp_dir):
                file_path = os.path.join(shp_dir, file)
                zipf.write(file_path, arcname=file)

        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ 選択した圃場を Shapefile (ZIP) としてダウンロード",
                data=f,
                file_name="agnote_xarvio_selected_shapefile.zip",
                mime="application/zip"
            )
    else:
        st.info("🔍 土地を選択してください")
