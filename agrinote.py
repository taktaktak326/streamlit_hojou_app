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

st.set_page_config(page_title="AgriNote Shapefile Exporter", layout="wide")
st.title("AgriNote 圃場データ取得＆Shapefileエクスポーター（Seleniumなし版）")

if "fields" not in st.session_state:
    st.session_state.fields = None
if "zip_paths" not in st.session_state:
    st.session_state.zip_paths = []

st.markdown("""
以下の手順で `an_api_token` を取得してください：
1. ブラウザの開発者ツール（F12）を開く
2. AgriNoteにログインし、`Network` タブで `an-api/v1/agri_fields` を探す
3. リクエストヘッダー内の `cookie` をコピーし、以下に貼り付け
""")

cookie_input = st.text_area("Cookieヘッダー（an_api_token を含む）")
fetch_clicked = st.button("圃場データを取得")

if fetch_clicked:
    try:
        if "an_api_token=" not in cookie_input:
            st.error("❌ an_api_token が Cookie に含まれていません")
            st.stop()

        # Cookieを辞書に変換
        cookie_dict = {}
        for pair in cookie_input.split(";"):
            if "=" in pair:
                k, v = pair.strip().split("=", 1)
                cookie_dict[k] = v

        if "tracking_user_uuid" not in cookie_dict or "an_login_status" not in cookie_dict:
            st.error("❌ 必要なクッキー（tracking_user_uuid, an_login_status）が不足しています")
            st.stop()

        csrf_token = json.loads(requests.utils.unquote(cookie_dict["an_login_status"]))["csrf"]

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
