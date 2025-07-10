import os
import time
import json
import zipfile
import urllib.parse
import tempfile
import requests
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# --- ページ設定 ---
st.set_page_config(page_title="AgriNote Shapefile Exporter", layout="wide")
st.title("AgriNote 圃場情報取得 & Shapefile エクスポート")

# --- セッション初期化 ---
if "fields" not in st.session_state:
    st.session_state.fields = None

# --- ログイン & 圃場取得 ---
def fetch_fields(email, password):
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
        driver.quit()
        raise Exception("ログインフォームが見つかりません")

    inputs[0].send_keys(email)
    inputs[1].send_keys(password)
    inputs[1].send_keys(Keys.RETURN)
    time.sleep(5)

    cookies_list = driver.get_cookies()
    cookie_dict = {c['name']: c['value'] for c in cookies_list}
    required = ['an_api_token', 'an_login_status', 'tracking_user_uuid']
    if not all(k in cookie_dict for k in required):
        driver.quit()
        raise Exception("必要なCookieが見つかりません")

    csrf_token = json.loads(urllib.parse.unquote(cookie_dict['an_login_status']))["csrf"]

    headers = {
        "x-an-csrf-token": csrf_token,
        "x-user-uuid": cookie_dict['tracking_user_uuid'],
        "x-agri-note-api-client": "v2.97.0",
        "x-requested-with": "XMLHttpRequest",
        "referer": "https://agri-note.jp/b/",
        "user-agent": "Mozilla/5.0"
    }

    cookies = {
        "an_api_token": cookie_dict['an_api_token'],
        "an_login_status": cookie_dict['an_login_status'],
        "tracking_user_uuid": cookie_dict['tracking_user_uuid']
    }

    response = requests.get("https://agri-note.jp/an-api/v1/agri_fields", headers=headers, cookies=cookies)
    driver.quit()

    if response.status_code != 200:
        raise Exception(f"API取得失敗: {response.status_code}")

    return response.json()

# --- Shapefile 出力 ---
def create_shapefiles(fields):
    temp_dir = tempfile.mkdtemp()
    zip_paths = []
    chunk_size = 300
    chunks = [fields[i:i + chunk_size] for i in range(0, len(fields), chunk_size)]

    for idx, chunk in enumerate(chunks):
        field_names, polygons = [], []
        for f in chunk:
            coords = [(pt["lng"], pt["lat"]) for pt in f["region_latlngs"]]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            field_names.append(f["field_name"] or f"ID: {f['id']}")
            polygons.append(Polygon(coords))

        gdf = gpd.GeoDataFrame({
            "FieldName": field_names,
            "geometry": polygons
        }, crs="EPSG:4326")

        shp_base = os.path.join(temp_dir, f"selected_{idx + 1}")
        gdf.to_file(f"{shp_base}.shp", driver="ESRI Shapefile", encoding="utf-8")

        zip_path = os.path.join(temp_dir, f"agnote_selected_{idx + 1}.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for ext in ["shp", "shx", "dbf", "prj"]:
                zipf.write(f"{shp_base}.{ext}", arcname=f"selected_{idx + 1}.{ext}")
        zip_paths.append(zip_path)

    return zip_paths

# --- UI 入力 ---
EMAIL = st.text_input("📧 メールアドレス", placeholder="your@email.com")
PASSWORD = st.text_input("🔑 パスワード", type="password", placeholder="パスワードを入力")

# --- 実行ボタン ---
if st.button("🔐 ログイン & データ取得"):
    try:
        with st.spinner("ログイン中..."):
            fields = fetch_fields(EMAIL, PASSWORD)
            st.session_state.fields = fields
            st.success(f"✅ {len(fields)} 件の圃場データを取得しました")
    except Exception as e:
        st.error(f"❌ エラー: {e}")

# --- ダウンロード処理 ---
if st.session_state.fields:
    st.subheader("📦 ダウンロード")
    zip_files = create_shapefiles(st.session_state.fields)
    for idx, zip_path in enumerate(zip_files):
        with open(zip_path, "rb") as f:
            st.download_button(
                label=f"⬇️ ダウンロード Part {idx + 1}",
                data=f,
                file_name=os.path.basename(zip_path),
                mime="application/zip"
            )
