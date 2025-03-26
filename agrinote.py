import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import shapefile
import tempfile
import os
import zipfile

st.set_page_config(page_title="AgriNote 圃場マップ＆Shapefile出力", layout="wide")
st.title("📍 AgriNote 圃場マップ（API連携）")

# あなたのCloud RunのURLに変更！
API_URL = "https://agrinote-api-xxxxx.a.run.app/fetch-fields"  # ← ← 替えてください

email = st.text_input("メールアドレス")
password = st.text_input("パスワード", type="password")

if st.button("✅ ログインして取得"):
    with st.spinner("圃場データ取得中..."):
        try:
            res = requests.post(API_URL, json={"email": email, "password": password})
            if res.status_code != 200:
                st.error(f"APIエラー: {res.status_code}")
                st.stop()

            fields = res.json()
            st.success(f"{len(fields)} 件の圃場を取得しました")

            # 地図を表示
            center = fields[0]["center_latlng"]
            fmap = folium.Map(location=[center["lat"], center["lng"]], zoom_start=15)
            for field in fields:
                coords = [(pt["lat"], pt["lng"]) for pt in field["region_latlngs"]]
                folium.Polygon(
                    locations=coords,
                    tooltip=field["field_name"] or f"ID: {field['id']}",
                    color="red",
                    fill=True,
                    fill_opacity=0.5
                ).add_to(fmap)
            st.subheader("🗺 圃場マップ")
            st_folium(fmap, width=700, height=500)

            # Shapefile保存（temp dirに保存してzipにする）
            temp_dir = tempfile.mkdtemp()
            shp_path = os.path.join(temp_dir, "fields")
            with shapefile.Writer(shp_path, shapeType=shapefile.POLYGON) as w:
                w.field("id", "N")
                w.field("name", "C")
                w.field("area", "F", decimal=3)
                for field in fields:
                    coords = [(pt["lng"], pt["lat"]) for pt in field["region_latlngs"]]
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    w.poly([coords])
                    w.record(field["id"], field["field_name"], field["calculation_area"])

            # ZIP作成
            zip_path = os.path.join(temp_dir, "agnote_xarvio_shapefile.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for ext in ["shp", "shx", "dbf"]:
                    file = f"{shp_path}.{ext}"
                    zipf.write(file, arcname=os.path.basename(file))

            # ダウンロードボタン
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="📦 Shapefileをダウンロード",
                    data=f,
                    file_name="agnote_xarvio_shapefile.zip",
                    mime="application/zip"
                )

        except Exception as e:
            st.error(f"❌ 通信または処理中にエラー: {e}")
