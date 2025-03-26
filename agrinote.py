import streamlit as st
import requests
import json
import shapefile
import os
import zipfile
import tempfile
import folium
from streamlit_folium import st_folium
import math
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from pyvirtualdisplay import Display

st.set_page_config(page_title="AgriNote Shapefile Exporter", layout="wide")
st.title("AgriNote 圃場データ取得＆Shapefileエクスポーター（Selenium対応版）２")

if "fields" not in st.session_state:
    st.session_state.fields = None
if "zip_paths" not in st.session_state:
    st.session_state.zip_paths = []

EMAIL = st.text_input("メールアドレス")
PASSWORD = st.text_input("パスワード", type="password")
login_clicked = st.button("ログインしてデータ取得")

if login_clicked:
    try:
        with st.spinner("ログイン中..."):
            # 仮想ディスプレイの開始（クラウド用）
            display = Display(visible=0, size=(1024, 768), use_xauth=False)
            display.start()

            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.binary_location = "/usr/bin/chromium-browser"

            driver = webdriver.Chrome(
                executable_path="/usr/bin/chromedriver",
                options=chrome_options
            )

            driver.get("https://agri-note.jp/b/login/")
            st.info("ログインページにアクセス中...")
            driver.implicitly_wait(10)

            inputs = driver.find_elements(By.CLASS_NAME, "_1g2kt34")
            if len(inputs) < 2:
                st.error("❌ ログインフォームが見つかりません")
                driver.quit()
                st.stop()

            inputs[0].send_keys(EMAIL)
            inputs[1].send_keys(PASSWORD)
            inputs[1].send_keys(Keys.RETURN)

            driver.implicitly_wait(10)
            st.info("クッキー取得中...")
            cookies_list = driver.get_cookies()
            cookie_dict = {cookie['name']: cookie['value'] for cookie in cookies_list}

            required = ['an_api_token', 'an_login_status', 'tracking_user_uuid']
            if not all(k in cookie_dict for k in required):
                st.error("❌ 必要なクッキーが揃っていません")
                st.write(cookie_dict)
                driver.quit()
                st.stop()

            csrf_token = json.loads(requests.utils.unquote(cookie_dict['an_login_status']))["csrf"]

            cookies = {
                "an_api_token": cookie_dict["an_api_token"],
                "an_login_status": cookie_dict["an_login_status"],
                "tracking_user_uuid": cookie_dict["tracking_user_uuid"],
            }

            headers = {
                "x-an-csrf-token": csrf_token,
                "x-user-uuid": cookie_dict['tracking_user_uuid'],
                "x-agri-note-api-client": "v2.97.0",
                "x-requested-with": "XMLHttpRequest",
                "referer": "https://agri-note.jp/b/",
                "user-agent": "Mozilla/5.0"
            }

            driver.quit()

        response = requests.get("https://agri-note.jp/an-api/v1/agri_fields", headers=headers, cookies=cookies)

        if response.status_code != 200:
            st.error(f"API取得失敗: {response.status_code}")
            st.stop()

        st.session_state.fields = response.json()
        st.success(f"✅ {len(st.session_state.fields)}件の圃場データを取得しました")

        center = st.session_state.fields[0]["center_latlng"]
        fmap = folium.Map(location=[center["lat"], center["lng"]], zoom_start=15)
        for field in st.session_state.fields:
            coords = [(pt['lat'], pt['lng']) for pt in field['region_latlngs']]
            folium.Polygon(
                locations=coords,
                popup=field['field_name'] or f"ID: {field['id']}",
                tooltip=field['field_name'] or f"ID: {field['id']}",
                color='red',
                fill=True,
                fill_opacity=0.5
            ).add_to(fmap)
        st_folium(fmap, width=700, height=500)

        temp_dir = tempfile.mkdtemp()
        st.session_state.zip_paths = []

        chunk_size = 300
        chunks = [st.session_state.fields[i:i + chunk_size] for i in range(0, len(st.session_state.fields), chunk_size)]

        for idx, chunk in enumerate(chunks):
            shp_path = os.path.join(temp_dir, f"fields_{idx+1}")
            with shapefile.Writer(shp_path, shapeType=shapefile.POLYGON) as w:
                w.field("id", "N")
                w.field("name", "C")
                w.field("area", "F", decimal=3)

                for field in chunk:
                    coords = [(pt["lng"], pt["lat"]) for pt in field["region_latlngs"]]
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    w.poly([coords])
                    w.record(field["id"], field["field_name"], field["calculation_area"])

            zip_path = os.path.join(temp_dir, f"agnote_xarvio_shapefile_{idx+1}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for ext in ["shp", "shx", "dbf"]:
                    zipf.write(f"{shp_path}.{ext}", arcname=f"fields_{idx+1}.{ext}")
            st.session_state.zip_paths.append(zip_path)

    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}")

if st.session_state.fields:
    st.subheader("圃場マップ")
    center = st.session_state.fields[0]["center_latlng"]
    fmap = folium.Map(location=[center["lat"], center["lng"]], zoom_start=15)
    for field in st.session_state.fields:
        coords = [(pt['lat'], pt['lng']) for pt in field['region_latlngs']]
        folium.Polygon(
            locations=coords,
            popup=field['field_name'] or f"ID: {field['id']}",
            tooltip=field['field_name'] or f"ID: {field['id']}",
            color='red',
            fill=True,
            fill_opacity=0.5
        ).add_to(fmap)
    st_folium(fmap, width=700, height=500)

    for idx, zip_path in enumerate(st.session_state.zip_paths):
        with open(zip_path, "rb") as f:
            st.download_button(
                label=f"Shapefileをダウンロード (ZIP形式) - Part {idx+1}",
                data=f,
                file_name=f"agnote_xarvio_shapefile_{idx+1}.zip",
                mime="application/zip"
            )
