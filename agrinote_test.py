import streamlit as st
import time, json, urllib.parse, requests, tempfile, zipfile, os
import geopandas as gpd
from shapely.geometry import Polygon
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import subprocess 

st.set_page_config(page_title="AgriNote Shapefile Exporter", layout="wide")
# 🔍 Chromeとchromedriverのパスを確認（Render用）
def debug_paths():
    try:
        chrome_path = subprocess.run(["which", "chromium-browser"], stdout=subprocess.PIPE).stdout.decode().strip()
    except:
        chrome_path = "Not Found"

    try:
        driver_path = subprocess.run(["which", "chromedriver"], stdout=subprocess.PIPE).stdout.decode().strip()
    except:
        driver_path = "Not Found"

    st.write("🔍 chromium path:", chrome_path if chrome_path else "Not Found")
    st.write("🔍 chromedriver path:", driver_path if driver_path else "Not Found")

debug_paths()


# 入力
email = st.text_input("ログインメールアドレス")
password = st.text_input("パスワード", type="password")

def create_driver():
    # Chromeバイナリの場所
    chrome_bin = "/usr/bin/chromium"

    # Chromedriverの候補パス（順に確認）
    possible_driver_paths = [
        "/usr/lib/chromium/chromedriver",
        "/usr/lib/chromium-browser/chromedriver",
        "/usr/bin/chromedriver"
    ]

    driver_bin = None
    for path in possible_driver_paths:
        if os.path.exists(path):
            driver_bin = path
            break

    if driver_bin is None:
        raise FileNotFoundError("chromedriver が見つかりません。PATHを確認してください。")

    # オプション設定
    chrome_options = Options()
    chrome_options.binary_location = chrome_bin
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # ドライバ作成
    return webdriver.Chrome(service=Service(driver_bin), options=chrome_options)


# ログイン & データ取得
def fetch_field_data(email, password):
    driver = create_driver()  # ← ここを書き換える

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
    driver.quit()
    cookie_dict = {c['name']: c['value'] for c in cookies_list}

    if not all(k in cookie_dict for k in ['an_api_token', 'an_login_status', 'tracking_user_uuid']):
        raise Exception("入力情報が違うか、ご利用の営農ツールは対応していないので、情報が取得できませんでした( ;∀;)")

    csrf = json.loads(urllib.parse.unquote(cookie_dict['an_login_status']))["csrf"]

    headers = {
        "x-an-csrf-token": csrf,
        "x-user-uuid": cookie_dict['tracking_user_uuid'],
        "x-agri-note-api-client": "v2.97.0",
        "x-requested-with": "XMLHttpRequest",
        "referer": "https://agri-note.jp/b/",
        "user-agent": "Mozilla/5.0"
    }
    cookies = {
        "an_api_token": cookie_dict['an_api_token'],
        "an_login_status": cookie_dict['an_login_status'],
        "tracking_user_uuid": cookie_dict['tracking_user_uuid'],
    }

    res = requests.get("https://agri-note.jp/an-api/v1/agri_fields", headers=headers, cookies=cookies)
    if res.status_code != 200:
        raise Exception(f"圃場データ取得失敗: {res.status_code}")
    return res.json()


# Shapefile作成
def make_shapefile(fields):
    temp_dir = tempfile.mkdtemp()
    shp_path = os.path.join(temp_dir, "fields")
    polygons, names = [], []

    for f in fields:
        coords = [(pt["lng"], pt["lat"]) for pt in f["region_latlngs"]]
        if coords[0] != coords[-1]: coords.append(coords[0])
        polygons.append(Polygon(coords))
        names.append(f["field_name"] or f"ID: {f['id']}")

    gdf = gpd.GeoDataFrame({"FieldName": names, "geometry": polygons}, crs="EPSG:4326")
    gdf.to_file(f"{shp_path}.shp", driver="ESRI Shapefile", encoding="utf-8")

    zip_path = os.path.join(temp_dir, "fields.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for ext in ["shp", "shx", "dbf", "prj"]:
            zipf.write(f"{shp_path}.{ext}", arcname=f"fields.{ext}")
    return zip_path

# 実行
if st.button("🔐 ログイン & 出力"):
    try:
        with st.spinner("ログイン・取得中..."):
            fields = fetch_field_data(email, password)
            zip_file = make_shapefile(fields)
        with open(zip_file, "rb") as f:
            st.download_button("⬇️ 圃場Shapefileをダウンロード", data=f, file_name="fields.zip", mime="application/zip")
            st.success("✅ 圃場データ取得 & Shapefile作成成功")
    except Exception as e:
        st.error(f"❌ エラー: {e}")
