import streamlit as st
import folium
from streamlit_folium import folium_static
from shapely.geometry import Polygon
import geopandas as gpd
import json
import os
import zipfile
import pandas as pd

def create_map(data):
    m = folium.Map(location=[35.3967687262, 139.1971242012], zoom_start=15)
    
    for field in data:
        field_name = field.get("field_name")
        if not field_name or field_name.strip() == "":
            field_name = "（圃場名なし）"
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
    output_dir = "shapefile_from_agrinote"
    os.makedirs(output_dir, exist_ok=True)
    shapefile_path = os.path.join(output_dir, "houjou_data.shp")
    
    field_names = []
    polygons = []
    
    for field in data:
        field_name = field.get("field_name")
        if not field_name or field_name.strip() == "":
            field_name = "（圃場名なし）"
        region_latlngs = field.get("region_latlngs", [])
        
        if region_latlngs:
            coords = [(point["lng"], point["lat"]) for point in region_latlngs]  # GeoJSON uses (lng, lat)
            polygon = Polygon(coords)
            field_names.append(field_name)
            polygons.append(polygon)
    
    gdf = gpd.GeoDataFrame({"FieldName": field_names, "geometry": polygons}, geometry="geometry", crs="EPSG:4326")
    gdf.to_file(shapefile_path, driver="ESRI Shapefile", encoding="utf-8")
    
    zip_filename = "shapefile_output.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            zipf.write(file_path, os.path.basename(file_path))
    
    return zip_filename

# Streamlit UI
st.title("アグリノートから圃場移行アプリ")

uploaded_file = st.file_uploader("JSONファイルをアップロード", type=["json"])
json_input = st.text_area("またはJSONデータを直接入力")

data = None
if uploaded_file is not None:
    data = json.load(uploaded_file)
elif json_input:
    try:
        data = json.loads(json_input)
    except json.JSONDecodeError:
        st.error("JSONの形式が正しくありません。修正してください。")
else:
    sample_data = [
        {
            "id": 1517291,
            "field_name": "あ",
            "region_latlngs": [
                {"lat": 35.396732128, "lng": 139.197647329},
                {"lat": 35.396702992, "lng": 139.197627988},
                {"lat": 35.39668799, "lng": 139.197597037}
            ]
        }
    ]
    data = sample_data

if data:
    st.subheader("マップ表示")
    field_map = create_map(data)
    folium_static(field_map)
    
    st.subheader("圃場名一覧")
    field_names = [field.get("field_name", "（圃場名なし）") for field in data]
    st.write(field_names)
    
    st.subheader("シェープファイルの作成")
    zip_filepath = create_shapefile(data)
    
    custom_filename = st.text_input("ダウンロードファイル名を入力してください（.zipを含む）", value=zip_filepath)
    
    with open(zip_filepath, "rb") as f:
        st.download_button("Shapefileをダウンロード", f, file_name=custom_filename)
