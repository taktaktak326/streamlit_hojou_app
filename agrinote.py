import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import shapefile
import tempfile
import os
import zipfile
import pandas as pd

st.set_page_config(page_title="AgriNote 圃場マップ＆Shapefile出力", layout="wide")
st.title("📍 AgriNote 圃場マップ（API連携）")

# Cloud Run にデプロイしたAPIのURLに置き換えてください
API_URL = "https://agrinote-api-908507328312.asia-northeast1.run.app"

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

            # 圃場一覧をDataFrameで表示
            df = pd.DataFrame([
                {"ID": f["id"], "圃場名": f["field_name"], "面積": f["calculation_area"]} for f in fields
            ])

            # フィルター（圃場名）
            search = st.text_input("🔍 圃場名で検索")
            if search:
                fields = [f for f in fields if search in f["field_name"]]
                df = df[df["圃場名"].str.contains(search)]

            st.dataframe(df, use_container_width=True)

            if not fields:
                st.warning("🔍 条件に一致する圃場がありません")
                st.stop()

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

            # 圃場を300件ごとに分割してShapefileを作成
            chunk_size = 300
            chunks = [fields[i:i + chunk_size] for i in range(0, len(fields), chunk_size)]

            for idx, chunk in enumerate(chunks):
                temp_dir = tempfile.mkdtemp()
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

                # ZIP作成
                zip_path = os.path.join(temp_dir, f"agnote_xarvio_shapefile_part{idx+1}.zip")
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    for ext in ["shp", "shx", "dbf"]:
                        file = f"{shp_path}.{ext}"
                        zipf.write(file, arcname=os.path.basename(file))

                # ダウンロードボタン
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label=f"📦 Shapefileをダウンロード（Part {idx+1}）",
                        data=f,
                        file_name=f"agnote_xarvio_shapefile_part{idx+1}.zip",
                        mime="application/zip"
                    )

        except Exception as e:
            st.error(f"❌ 通信または処理中にエラー: {e}")
