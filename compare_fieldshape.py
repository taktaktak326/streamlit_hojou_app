import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from shapely.geometry import Polygon, shape
from streamlit_folium import st_folium
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import os
from fastkml import kml

st.set_page_config(page_title="ポリゴン比較アプリ", layout="wide")

st.title("\U0001F5FA️ ポリゴン形状比較アプリ")
st.write("2つのファイルをアップロードして、ポリゴン形状の違いを地図上に表示します。")

# --- ファイルアップロード ---
file1 = st.file_uploader("ファイル1をアップロード (Shapefile ZIP / KML / ISOXML)", type=["zip", "kml", "xml"])
file2 = st.file_uploader("ファイル2をアップロード (Shapefile ZIP / KML / ISOXML)", type=["zip", "kml", "xml"])

# --- ユーティリティ関数 ---
def load_shapefile(zip_file):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if shp_files:
                return gpd.read_file(shp_files[0])
            else:
                st.error("ZIPファイル内に .shp ファイルが見つかりませんでした。")
                return None
    except Exception as e:
        st.error(f"Shapefile読み込みエラー: {e}")
        return None

def load_kml(file):
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon

    doc = file.read()
    root = ET.fromstring(doc)

    ns = {'kml': 'http://www.opengis.net/kml/2.2'}  # 名前空間無いなら空でもOK

    geoms = []
    
    for placemark in root.findall('.//Placemark'):
        name = placemark.find('name').text if placemark.find('name') is not None else 'Unnamed'

        for polygon in placemark.findall('.//Polygon'):
            coords_text = polygon.find('.//coordinates').text.strip()
            coords_pairs = coords_text.split()
            coords = []
            for pair in coords_pairs:
                lon, lat = map(float, pair.split(',')[:2])
                coords.append((lon, lat))
            if coords:
                poly = Polygon(coords)
                geoms.append({'name': name, 'geometry': poly})

    if geoms:
        return gpd.GeoDataFrame(geoms, crs="EPSG:4326")
    else:
        st.warning("KML内にポリゴンが見つかりませんでした。")
        return gpd.GeoDataFrame(columns=['name', 'geometry'], geometry='geometry', crs="EPSG:4326")


def load_isoxml(file):
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon
    import geopandas as gpd

    try:
        tree = ET.parse(file)
        root = tree.getroot()

        polygons = []
        # PFD (フィールド定義) を探す
        for pfd in root.findall('.//PFD'):
            field_name = pfd.attrib.get('C', 'Unnamed Field')
            for pln in pfd.findall('.//PLN'):
                for lsg in pln.findall('.//LSG'):
                    points = []
                    for pnt in lsg.findall('.//PNT'):
                        lat = float(pnt.attrib['C'])  # 緯度
                        lon = float(pnt.attrib['D'])  # 経度
                        points.append((lon, lat))
                    if len(points) >= 3:  # ポリゴン作成には最低3点必要
                        polygon = Polygon(points)
                        polygons.append({'name': field_name, 'geometry': polygon})

        if polygons:
            return gpd.GeoDataFrame(polygons, crs="EPSG:4326")
        else:
            st.warning("ISOXML内にポリゴンデータが見つかりませんでした。")
            return gpd.GeoDataFrame(columns=['name', 'geometry'], geometry='geometry', crs="EPSG:4326")

    except ET.ParseError as e:
        st.error(f"XMLパースエラー: {e}")
        return None
    except Exception as e:
        st.error(f"ISOXML読み込みエラー: {e}")
        return None



def load_file(file):
    st.write(f"デバッグ: アップロードされたファイル名 → {file.name}")

    if file.name.lower().endswith('.zip'):
        st.info("Shapefile ZIPとして読み込みます")
        return load_shapefile(file)
    elif file.name.lower().endswith('.kml'):
        st.info("KMLファイルとして読み込みます")
        return load_kml(file)
    elif file.name.lower().endswith('.xml') or file.name.lower().endswith('.isoxml'):
        st.info("ISOXMLファイルとして読み込みます")
        return load_isoxml(file)
    else:
        st.error(f"未対応ファイル形式: {file.name}")
        return None
    
def highlight_mismatch(row):
    gdf1 = row['gdf1_coordinates']
    gdf2 = row['gdf2_coordinates']
    
    style_gdf1 = ''
    style_gdf2 = ''
    
    # 比較できるかチェック
    if isinstance(gdf1, list) and isinstance(gdf2, list):
        if gdf1 != gdf2:
            style_gdf1 = 'background-color: #ffcccc;'  # 薄い赤
            style_gdf2 = 'background-color: #ffcccc;'
    
    return [ '', '', style_gdf1, style_gdf2 ]


# --- メイン処理 ---
if file1 and file2:
    gdf1 = load_file(file1)
    gdf2 = load_file(file2)

    if gdf1 is not None and not gdf1.empty and gdf2 is not None and not gdf2.empty:
        st.success("ファイルの読み込みに成功しました！")

        # --- 比較処理（アップロードした順で座標一覧を持つ版） ---
        results = []

        # 比較は、gdf1の行数だけループ
        for idx in range(len(gdf1)):
            geom1 = gdf1.geometry.iloc[idx]
            gdf1_coords = list(geom1.exterior.coords) if geom1.geom_type == 'Polygon' else None

            # gdf2にも同じインデックスがあれば取得、なければNone
            if idx < len(gdf2):
                geom2 = gdf2.geometry.iloc[idx]
                gdf2_coords = list(geom2.exterior.coords) if geom2.geom_type == 'Polygon' else None
                matched = geom1.equals(geom2)
            else:
                gdf2_coords = None
                matched = False

            results.append({
                'index': idx,
                'matched': matched,
                'gdf1_coordinates': gdf1_coords,
                'gdf2_coordinates': gdf2_coords,
                'geometry': geom1  # マップ描画用にgdf1側の形状
            })

        # 正しくDataFrameにする
        result_df = pd.DataFrame(results)


        # Foliumマップ作成
        m = folium.Map(location=[gdf1.geometry.centroid.y.mean(), gdf1.geometry.centroid.x.mean()], zoom_start=20)

        # ファイル1のポリゴン（gdf1）を緑で表示
        for idx, row in gdf1.iterrows():
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x: {"fillColor": "green", "color": "green", "weight": 2, "fillOpacity": 0.4},
                tooltip=f"File1 - Index: {idx}"
            ).add_to(m)

        # ファイル2のポリゴン（gdf2）を青で表示
        for idx, row in gdf2.iterrows():
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 2, "fillOpacity": 0.4},
                tooltip=f"File2 - Index: {idx}"
            ).add_to(m)

        # 地図を表示
        st.subheader("地図で比較結果を表示（ファイルごとに色分け）")
        st_folium(m, width=1000, height=600)


        # テーブル表示
        st.subheader("ポリゴン比較結果一覧（緯度経度付き）")

        # カラム幅を調整して表示する
        st.data_editor(
            result_df[['index', 'matched', 'gdf1_coordinates', 'gdf2_coordinates']],
            column_config={
                "gdf1_coordinates": st.column_config.Column(width="large"),
                "gdf2_coordinates": st.column_config.Column(width="large"),
            },
            hide_index=True,
            use_container_width=True
        )
        


    else:
        st.error("ファイルの読み込みに失敗しました。ファイル形式やポリゴンデータの有無を確認してください。")
