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
import geopandas as gpd
from shapely.geometry import Polygon
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from streamlit_folium import st_folium
import pandas as pd

st.title("AgriNote 圃場情報取得 & Shapefile エクスポート")

if "fields" not in st.session_state:
    st.session_state.fields = None

col1, col2 = st.columns([3, 3])
with col1:
    EMAIL = st.text_input("📧 メールアドレス", placeholder="your@email.com")
with col2:
    PASSWORD = st.text_input("🔑 パスワード", type="password", placeholder="パスワードを入力")

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

# === マップ表示 ===
if st.session_state.fields:
    st.subheader("🖼️ 圃場マップ")
    center = st.session_state.fields[0]["center_latlng"]
    fmap = folium.Map(location=[center["lat"], center["lng"]], zoom_start=15)

    for f in st.session_state.fields:
        coords = [(pt['lat'], pt['lng']) for pt in f['region_latlngs']]
        display_name = f["field_name"] or f"ID: {f['id']}"
        folium.Polygon(
            locations=coords,
            popup=display_name,
            tooltip=f"{display_name} ({round(f.get('calculation_area', 0), 2)}a)",
            color='red',
            fill=True,
            fill_opacity=0.5
        ).add_to(fmap)

    st_folium(fmap, use_container_width=True)

    # === 表形式でフィルター・ソート・選択 ===
    st.subheader("📋 圃場一覧と選択")

    st.checkbox("すべて選択", value=True, key="select_all")

    df = pd.DataFrame([
        {
            "ID": f["id"],
            "圃場名": f["field_name"] or f"圃場名なし_ID: {f['id']}",
            "面積 (a)": round(f.get("calculation_area", 0), 2),
            "選択": st.session_state.select_all
        } for f in st.session_state.fields
    ])

    edited_df = st.data_editor(
        df,
        column_config={"選択": st.column_config.CheckboxColumn("選択")},
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True
    )

    selected_ids = edited_df[edited_df["選択"] == True]["ID"].tolist()
    selected_fields = [f for f in st.session_state.fields if f["id"] in selected_ids]

    st.markdown(f"### ✅ 選択された圃場数: {len(selected_fields)} 件")
    st.markdown(f"### 📐 合計面積: {round(sum(f.get('calculation_area', 0) for f in selected_fields), 2)} a")

    if selected_fields:
        temp_dir = tempfile.mkdtemp()
        zip_paths = []
        chunk_size = 300
        chunks = [selected_fields[i:i + chunk_size] for i in range(0, len(selected_fields), chunk_size)]

        for idx, chunk in enumerate(chunks):
            field_names = []
            polygons = []
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

            shp_base = os.path.join(temp_dir, f"selected_{idx+1}")
            gdf.to_file(f"{shp_base}.shp", driver="ESRI Shapefile", encoding="utf-8")

            zip_path = os.path.join(temp_dir, f"agnote_xarvio_selected_{idx+1}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for ext in ["shp", "shx", "dbf", "prj"]:
                    zipf.write(f"{shp_base}.{ext}", arcname=f"selected_{idx+1}.{ext}")

            zip_paths.append(zip_path)

        for idx, zip_path in enumerate(zip_paths):
            with open(zip_path, "rb") as f:
                st.download_button(
                    label=f"⬇️ ダウンロード Part {idx+1}",
                    data=f,
                    file_name=os.path.basename(zip_path),
                    mime="application/zip"
                )
    else:
        st.info("🔍 圃場を選択してください")
