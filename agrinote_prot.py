import streamlit as st
import folium
from streamlit_folium import folium_static
from shapely.geometry import Polygon
import geopandas as gpd
import json
import os
import zipfile
import math
import pandas as pd

def get_center_coordinates(data):
    """JSONデータの圃場の中心座標を計算"""
    latitudes = []
    longitudes = []

    for field in data:
        region_latlngs = field.get("region_latlngs", [])
        for point in region_latlngs:
            latitudes.append(point["lat"])
            longitudes.append(point["lng"])

    if latitudes and longitudes:
        center_lat = sum(latitudes) / len(latitudes)
        center_lng = sum(longitudes) / len(longitudes)
        return [center_lat, center_lng]
    
    # デフォルト位置 (データがない場合)
    return [35.3967687262, 139.1971242012]

def create_map(data):
    """圃場データを地図にプロット"""
    center_coords = get_center_coordinates(data)  # 圃場の中心を取得
    m = folium.Map(location=center_coords, zoom_start=15)  # 中心位置を設定

    for field in data:
        field_name = field.get("field_name", "（圃場名なし）").strip()
        region_latlngs = field.get("region_latlngs", [])
        
        if region_latlngs:
            coords = [(point["lat"], point["lng"]) for point in region_latlngs]
            
            folium.Polygon(
                locations=coords,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.4,
                tooltip=field_name
            ).add_to(m)
    
    return m

def create_shapefile(data):
    """JSONデータをShapefileに変換してZIPファイルとして保存"""
    output_dir = "shapefile_from_agrinote"
    os.makedirs(output_dir, exist_ok=True)
    shapefile_path = os.path.join(output_dir, "houjou_data.shp")
    
    field_names = []
    polygons = []
    
    for field in data:
        field_name = field.get("field_name", "（圃場名なし）").strip()
        region_latlngs = field.get("region_latlngs", [])
        
        if region_latlngs:
            coords = [(point["lng"], point["lat"]) for point in region_latlngs]  # GeoJSON uses (lng, lat)
            polygon = Polygon(coords)
            field_names.append(field_name)
            polygons.append(polygon)
    
    gdf = gpd.GeoDataFrame({"FieldName": field_names, "geometry": polygons}, geometry="geometry", crs="EPSG:4326")
    gdf.to_file(shapefile_path, driver="ESRI Shapefile", encoding="utf-8")
    
    zip_filename = "shapefile.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            zipf.write(file_path, os.path.basename(file_path))
    
    return zip_filename

def get_field_dataframe(data):
    """圃場データから面積を計算し、Pandas DataFrameを返す"""
    if not data:
        return pd.DataFrame()

    field_names = []
    polygons = []
    
    for field in data:
        field_name = field.get("field_name", "（圃場名なし）").strip()
        region_latlngs = field.get("region_latlngs", [])
        
        if region_latlngs:
            coords = [(point["lng"], point["lat"]) for point in region_latlngs]
            polygon = Polygon(coords)
            field_names.append(field_name)
            polygons.append(polygon)

    gdf = gpd.GeoDataFrame({"FieldName": field_names, "geometry": polygons}, geometry="geometry", crs="EPSG:4326")

    # 面積計算のために適切なUTMゾーンに変換
    center_lng = gdf.unary_union.centroid.x
    utm_zone = math.floor((center_lng + 180) / 6) + 1
    utm_crs = f"EPSG:326{utm_zone}"
    gdf_proj = gdf.to_crs(utm_crs)
    
    df = pd.DataFrame({"圃場名": gdf["FieldName"], "面積(ha)": gdf_proj.geometry.area / 10000})
    return df

# Streamlit UI
st.title("Agrinote圃場形状JSONからファイル作成アプリ")

uploaded_file = st.file_uploader("JSONファイルをアップロード", type=["json"])
json_input = st.text_area("↓　開発者ツールから取得したJSONデータを貼り付け")

data = None
if uploaded_file is not None:
    data = json.load(uploaded_file)
elif json_input:
    try:
        data = json.loads(json_input)
    except json.JSONDecodeError:
        st.error("JSONの形式が正しくありません。修正してください。")

if data:
    st.subheader("マップ表示")
    field_map = create_map(data)
    folium_static(field_map)
    
    st.subheader("圃場情報")
    df_fields = get_field_dataframe(data)
    
    total_fields = len(df_fields)
    total_area = df_fields["面積(ha)"].sum()
    st.metric(label="合計圃場数", value=f"{total_fields} 筆")
    st.dataframe(df_fields)
    
    st.subheader("シェープファイルの作成")
    zip_filepath = create_shapefile(data)
    
    custom_filename = st.text_input("ダウンロードファイル名を入力してください（.zipを含む）", value=zip_filepath)
    
    with open(zip_filepath, "rb") as f:
        st.download_button("Shapefileをダウンロード", f, file_name=custom_filename)
