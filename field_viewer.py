import streamlit as st
st.set_page_config(page_title="xarvio BBCH Viewer", layout="wide")
import plotly.graph_objects as go
import tempfile
import base64
import requests
import urllib.parse
import pandas as pd
import json
from shapely.geometry import shape, MultiPolygon, Polygon
from geopy.geocoders import Nominatim
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
from datetime import datetime, timezone, timedelta
import plotly.express as px
import time 
from geopy.distance import geodesic

# カラーマッピング用
date_color_map = {}


def extract_lat_lon(coord_str):
    try:
        lon, lat = map(float, coord_str.split(","))
        return lat, lon
    except:
        return None, None

def create_efficient_route(bbch_df, bbch_code):
    filtered_df = bbch_df[bbch_df["BBCHコード"] == bbch_code].dropna(subset=["中心座標"])
    
    # 圃場ごとの座標を抽出
    points = []
    for _, row in filtered_df.iterrows():
        lat, lon = extract_lat_lon(row["中心座標"])
        if lat and lon:
            points.append({
                "name": row["圃場名"],
                "lat": lat,
                "lon": lon
            })

    if not points:
        return None, []

    # Greedy法：現在位置に最も近い順に巡回
    start = points[0]
    route = [start]
    remaining = points[1:]

    while remaining:
        last = route[-1]
        next_point = min(remaining, key=lambda p: geodesic((last["lat"], last["lon"]), (p["lat"], p["lon"])).km)
        route.append(next_point)
        remaining.remove(next_point)

    return route[0], route  # 最初の圃場, 巡回順

def generate_google_maps_route(route):
    if len(route) < 2:
        return None
    max_waypoints = 23
    trimmed_route = route[:max_waypoints]

    origin = "My+Location"
    destination = "My+Location"
    waypoints = "|".join([f'{pt["lat"]},{pt["lon"]}' for pt in trimmed_route])

    return f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}&waypoints={waypoints}"


def plot_bbch_stacked_bar(df):
    """BBCH開始日の積立棒グラフ（x軸はカテゴリ型で日別に明示的に分離）"""
    required_columns = ["BBCH開始日", "市区町村", "BBCHステージ", "BBCHコード", "作物", "品種", "圃場名", "農場名"]
    if not all(col in df.columns for col in required_columns):
        st.warning("必要なカラム（BBCH開始日、BBCHステージ、作物など）が不足しています。")
        return

    # ① 日付をUTC→日本時間へ変換して日付文字列に変換（カテゴリ軸対応）
    df["BBCH開始日"] = pd.to_datetime(df["BBCH開始日"], utc=True, errors='coerce')
    df["BBCH開始日"] = df["BBCH開始日"].dt.tz_convert("Asia/Tokyo").dt.date.astype(str)


    # ③ 色分け方法を選択
    color_by_option = st.radio(
        "色分けの基準を選択",
        ["圃場名", "作物", "品種","市区町村" ],
        horizontal=True
    )

        
    # ✅ 🌾 表示する作物のラジオボタンを追加
    crop_options = sorted(df["作物"].dropna().unique(), reverse=True)
    selected_crop = st.radio("🌾 表示する作物を選択", options=crop_options, horizontal=True)
    unique_stages = df[df["作物"] == selected_crop][["BBCHコード", "BBCH名称"]].dropna().drop_duplicates()

    # ✅ BBCHステージのラジオボタン（元の df_filtered で取得）
    unique_stages["BBCHコードソート"] = unique_stages["BBCHコード"].astype(int)
    unique_stages = unique_stages.sort_values("BBCHコードソート")

    # 表示用に整形（例: "13 (3葉期)"）
    unique_stages["ラベル"] = unique_stages["BBCHコード"].astype(str) + " (" + unique_stages["BBCH名称"] + ")"

    # ラジオボタンに渡す
    selected_stage = st.radio("表示するBBCHステージを選んでください", unique_stages["ラベル"].tolist(), horizontal=True)


    filtered_df = df[(df["作物"] == selected_crop) & (df["BBCHステージ"] == selected_stage)].copy()

    # 圃場名（農場名）というラベル列を追加
    filtered_df["圃場ラベル"] = filtered_df["圃場名"] + "（" + filtered_df["農場名"] + "）"



    if color_by_option == "市区町村":
        group_cols = ["BBCH開始日", "市区町村"]
        color_column = "市区町村"
    elif color_by_option == "作物":
        group_cols = ["BBCH開始日", "作物"]
        color_column = "作物"

    elif color_by_option == "品種":
        group_cols = ["BBCH開始日", "品種"]
        color_column = "品種"

    elif color_by_option == "圃場名":
        group_cols = ["BBCH開始日", "圃場ラベル"]  # ← 変更
        color_column = "圃場ラベル"               # ← 変更

        
    # ④ 集計
    date_counts = filtered_df.groupby(group_cols).size().reset_index(name="カウント")
    
    # ✅ 日付順にソートしてカテゴリ型に変換（順番を固定）
    sorted_dates = sorted(date_counts["BBCH開始日"].unique())
    date_counts["BBCH開始日"] = pd.Categorical(
        date_counts["BBCH開始日"],
        categories=sorted_dates,
        ordered=True
    )


    # ⑤ グラフ作成
    fig = px.bar(
        date_counts,
        x="BBCH開始日",
        y="カウント",
        color=color_column,
        title=f"BBCHステージ {selected_stage} の開始日分布",
        hover_data={"BBCH開始日": True},
        labels={
            "BBCH開始日": "BBCH開始日",
            "カウント": "圃場数",
            "市区町村": "市区町村",
            "市区町村_BBCH": "市区町村 + BBCHステージ",
        },
    )

    # レイアウト調整（カテゴリ軸としてxを扱う！）
    fig.update_layout(
        xaxis_title="BBCH開始日",
        yaxis_title="圃場数",
        barmode="stack",
        bargap=0.1
    )
    fig.update_xaxes(
    type="category",  # ← 明示的にカテゴリ扱い
    categoryorder="array",
    categoryarray=sorted_dates,  # ← 並び順指定
    tickangle=45
    )

    # グラフ表示
    st.plotly_chart(fig, use_container_width=True)

    st.write("🔍 グラフのデータ", filtered_df)
    
@st.cache_data(show_spinner=False)
def reverse_geocode(lat, lon):
    #st.write(f"📍 reverse_geocode called: {lat}, {lon}")
    geolocator = Nominatim(user_agent="xarvio-app")
    location = geolocator.reverse((lat, lon), language="ja")
    return location.raw.get("address", {})

def show_debug_info():
    st.markdown("## 🧪 テスト用デバッグ情報")
    st.write(f"🔁 GraphQL API呼び出し回数: {st.session_state['graphql_api_call_count']} 回")
    st.write(f"🧠 キャッシュヒット: {st.session_state['reverse_geocode_cache_hits']} 回")
    st.write(f"❌ キャッシュミス: {st.session_state['reverse_geocode_cache_misses']} 回")



    
def get_color_for_date(date):
    if date not in date_color_map:
        color = "hsl({}, 70%, 60%)".format((len(date_color_map) * 40) % 360)
        date_color_map[date] = color
    return date_color_map[date]

def get_user_inputs(field_data):
    """地図スタイル・BBCH・タイトルの選択をUIで取得"""
    map_style_label_to_value = {
        "標準": "open-street-map",
        "シンプル": "carto-positron"
    }

    with st.expander("⚙️ 表示設定", expanded=False):
        title_prefix = st.text_input("圃場マップのタイトル（任意）", placeholder="例: 新潟西部")

        selected_style_label = st.radio("地図スタイル", options=list(map_style_label_to_value.keys()), horizontal=True)
        selected_map_style = map_style_label_to_value[selected_style_label]


        # 文字列 → 数値 → ソート → 文字列に戻す
        all_bbch = sorted(
            {int(f["BBCHコード"]) for f in field_data if "BBCHコード" in f and str(f["BBCHコード"]).isdigit()}
        )
        all_bbch = [str(code) for code in all_bbch]

        selected_bbch = st.radio("BBCHステージを選択", options=all_bbch, index=0, horizontal=True)

        if selected_bbch:
            st.caption(f"📘 {selected_bbch}：{bbch_df[bbch_df['BBCHコード'] == selected_bbch]['BBCH名称'].iloc[0]}")
    
        # ラベル表示項目の選択
        label_options = {
            "圃場名": "name",
            "品種": "variety",
            "作付日": "date"
        }
        selected_label_key = st.radio("圃場ラベルに表示する情報", list(label_options.keys()), horizontal=True)
        selected_label = label_options[selected_label_key]

    return selected_map_style, selected_bbch, title_prefix, selected_label

def generate_map_title(prefix, bbch):
    if prefix.strip():
        return f"【{prefix.strip()}】圃場マップ BBCH{bbch}"
    else:
        return f"圃場マップ BBCH{bbch}"

def create_field_map(field_data, selected_bbch, map_style, map_title, label_key, center_override=None, zoom_override=None):
    """Plotly地図の生成"""
    filtered_data = [f for f in field_data if f.get("BBCHコード") == selected_bbch]
    fig = go.Figure()

    legend_dates_added = set()

    for field in filtered_data:
        poly = Polygon(field["coords"])
        lons, lats = poly.exterior.xy
        date = field["date"]
        color = get_color_for_date(date)

        if date not in legend_dates_added:
            fig.add_trace(go.Scattermapbox(
                lat=[None], lon=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=date,
                legendgroup=date,
                showlegend=True
            ))
            legend_dates_added.add(date)

        fig.add_trace(go.Scattermapbox(
            lat=list(lats), lon=list(lons),
            mode="lines", fill="toself",
            name=field["name"],
            line=dict(width=2, color="grey"),  
            fillcolor=color,
            hoverinfo="skip", showlegend=False,
            legendgroup=date
        ))

        centroid = poly.centroid
        lat, lon = centroid.y, centroid.x
        # 🔴 赤いピンマークを追加（圃場の中心に）
        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon],
            mode="markers",
            marker=dict(size=10, color="red", symbol="circle"),  # ← ここが目立つポイント
            name=field["name"],
            hoverinfo="skip",
            showlegend=False
        ))

        gmap_url = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"
        hover_html = (
            f"<b>{field['name']}</b><br>"
            f"農場名: {field.get('農場名', '不明')}<br>"
            f"作物: {field.get('作物', '不明')}<br>"
            f"品種: {field['variety']}<br>"
            f"作付方法: {field.get('作付方法', '')}<br>"
            f"<a href='{gmap_url}' target='_blank'>📍Googleマップ</a><br>"
            f"面積: {field.get('面積 (a)', '')} a<br>"
            f"作付日: {field['date']}<br>"
            f"BBCH: {field.get('BBCHコード', '')}（{field.get('BBCH名称', '')}）<br>"
            
        )

        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode="markers",
            marker=dict(
                size=30,                # ← 大きくすることで hover しやすくなる
                color="rgba(0,0,0,0)"   # ← 完全に透明
            ),
            hoverinfo="text",
            hovertext=hover_html,
            showlegend=False
        ))
        label_text = str(field.get(label_key, ""))
        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon],
            mode="text",
            marker=dict(size=3, color="rgba(255,255,255,0.1)"),
            text=[label_text],
            textposition="middle center",
            textfont=dict(size=14, color="black"),
            hoverinfo="skip",
            showlegend=False
        ))
        

    # 圃場の重心（中心）を元に地図の中心座標を動的に設定
    centroids = []
    for field in filtered_data:
        try:
            poly = Polygon(field["coords"])
            centroids.append(poly.centroid)
        except:
            continue

    if centroids:
        lats = [pt.y for pt in centroids]
        lons = [pt.x for pt in centroids]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        avg_lat = sum(lats) / len(lats)
        avg_lon = sum(lons) / len(lons)

        # 地理的な広がりの距離を計算（緯度・経度の差から簡易推定）
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        max_range = max(lat_range, lon_range)

        # 地図ズームレベルの目安を決定（日本全体ならzoom=5、狭い範囲ならzoom=10以上）
        if max_range < 0.01:
            map_zoom = 15
        elif max_range < 0.05:
            map_zoom = 13
        elif max_range < 0.1:
            map_zoom = 11
        elif max_range < 0.5:
            map_zoom = 9
        elif max_range < 1.5:
            map_zoom = 7
        else:
            map_zoom = 5

        map_center = {"lat": avg_lat, "lon": avg_lon}
    else:
        map_center = {"lat": 36.2048, "lon": 138.2529}
        map_zoom = 5

    # 地図のレイアウトに反映
    fig.update_layout(
        title={"text": map_title, "x": 0.5, "xanchor": "center", "font": dict(size=20, color="black")},
        mapbox_style=map_style,
        mapbox_zoom=zoom_override if zoom_override else map_zoom,
        mapbox_center=center_override if center_override else map_center,
        height=800, 
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        legend=dict(orientation="v", x=1.02, y=1.0, xanchor="left", yanchor="top", bordercolor="gray", borderwidth=1)
    )

    return fig

def download_map_html(fig):
    """地図をHTMLとして保存し、ダウンロードリンク表示"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        fig.write_html(
            tmpfile.name,
            include_plotlyjs="cdn",
            
            config={
                    "scrollZoom": True,
                    "displayModeBar": True,  # モードバー表示を有効に
                    "modeBarButtonsToRemove": [
                        "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                        "autoScale2d", "resetScale2d", "hoverClosestCartesian", "hoverCompareCartesian",
                        "toggleSpikelines", "toImage"
                    ],
                    "modeBarButtonsToAdd": ["toggleFullscreen"]  # ← 全画面ボタンのみ有効
                }
        )
        tmpfile.seek(0)
        html_data = tmpfile.read()

    b64 = base64.b64encode(html_data).decode("utf-8")
    href = f'''
        <a href="data:text/html;base64,{b64}" download="field_map.html">
            <button style="
                background-color: #007BFF;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                cursor: pointer;
            ">📥 HTMLで地図をダウンロード</button>
        </a>
    '''
    st.markdown(f"<div style='text-align:center'>{href}</div>", unsafe_allow_html=True)

def to_jst_ymd(date_str):
    if not date_str:
        return ""
    try:
        dt_utc = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        dt_jst = dt_utc.astimezone(timezone(timedelta(hours=9)))
        return dt_jst.strftime("%Y-%m-%d")  # ← 日付のみ（時刻なし）
    except Exception:
        return date_str  # 変換失敗時はそのまま

def create_kml_from_bbch_df(bbch_df):
    kml = Element('kml')
    document = SubElement(kml, 'Document', {'id': 'featureCollection'})

    # 圃場名・作付UUIDでまとめる
    grouped = bbch_df.groupby(["圃場名", "作物", "作付UUID"])

    for (field_name, crop_name, cs_uuid), group in grouped:
        first = group.iloc[0]
        placemark = SubElement(document, 'Placemark', {'id': cs_uuid})

        # <name>
        name = SubElement(placemark, 'name')
        name.text = f"{field_name} - {crop_name}"

        # Geometry
        multi_geometry = SubElement(placemark, 'MultiGeometry')
        polygon = SubElement(multi_geometry, 'Polygon')
        outer = SubElement(polygon, 'outerBoundaryIs')
        ring = SubElement(outer, 'LinearRing')
        coords = SubElement(ring, 'coordinates')

        # ポリゴン座標
        try:
            poly_json = json.loads(first.get("ポリゴン情報", ""))
            coordinates = []
            if poly_json["type"] == "Polygon":
                rings = poly_json["coordinates"]
            elif poly_json["type"] == "MultiPolygon":
                rings = poly_json["coordinates"][0]
            else:
                continue

            for lon, lat in rings[0]:
                coordinates.append(f"{lon},{lat}")
            coords.text = " ".join(coordinates)
        except Exception:
            continue

        # <ExtendedData>
        ext_data = SubElement(placemark, "ExtendedData")

        # 通常のフィールドを出力
        info_keys = [
            "品種", "作付方法", "作付時のBBCH", "作付日",
            "面積 (a)", "都道府県", "市区町村", "中心座標"
        ]
        for key in info_keys:
            val = first.get(key)
            if pd.notna(val):
                data = SubElement(ext_data, "Data", {"name": key})
                value = SubElement(data, "value")
                value.text = str(val)

        # ▼ BBCHステージを個別に書き出す
        for _, row in group.iterrows():
            code = row.get("BBCHコード")
            name = row.get("BBCH名称")
            start = row.get("BBCH開始日")

            if pd.notna(code) and pd.notna(start):
                # 例: <Data name="BBCH25"><value>2025-05-19</value></Data>
                data_date = SubElement(ext_data, "Data", {"name": f"BBCH{code}"})
                value_date = SubElement(data_date, "value")
                value_date.text = str(start)

                # 例: <Data name="BBCH25_説明"><value>分げつ期（主茎と分げつ5本）</value></Data>
                if pd.notna(name):
                    data_desc = SubElement(ext_data, "Data", {"name": f"BBCH{code}_説明"})
                    value_desc = SubElement(data_desc, "value")
                    value_desc.text = str(name)
    # XML整形
    rough_string = tostring(kml, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


# --- 定数定義 ---
BASE_LOGIN_URL = "https://accounts.eu1.gigya.com/accounts.login"
TOKEN_API_URL = "https://fm-api.xarvio.com/api/users/tokens"
GRAPHQL_END_POINT = "https://fm-api.xarvio.com/api/graphql/data"
API_KEY = "3_W-AXsoj7TvX-9gi7S-IGxXfLWVkEbnGSl57M7t49GN538umaKs2EID8hyipAux2y"

# --- ISOコードから都道府県名への変換辞書 ---
ISO_TO_PREF_NAME = {
    "JP-01": "北海道", "JP-02": "青森県", "JP-03": "岩手県", "JP-04": "宮城県",
    "JP-05": "秋田県", "JP-06": "山形県", "JP-07": "福島県", "JP-08": "茨城県",
    "JP-09": "栃木県", "JP-10": "群馬県", "JP-11": "埼玉県", "JP-12": "千葉県",
    "JP-13": "東京都", "JP-14": "神奈川県", "JP-15": "新潟県", "JP-16": "富山県",
    "JP-17": "石川県", "JP-18": "福井県", "JP-19": "山梨県", "JP-20": "長野県",
    "JP-21": "岐阜県", "JP-22": "静岡県", "JP-23": "愛知県", "JP-24": "三重県",
    "JP-25": "滋賀県", "JP-26": "京都府", "JP-27": "大阪府", "JP-28": "兵庫県",
    "JP-29": "奈良県", "JP-30": "和歌山県", "JP-31": "鳥取県", "JP-32": "島根県",
    "JP-33": "岡山県", "JP-34": "広島県", "JP-35": "山口県", "JP-36": "徳島県",
    "JP-37": "香川県", "JP-38": "愛媛県", "JP-39": "高知県", "JP-40": "福岡県",
    "JP-41": "佐賀県", "JP-42": "長崎県", "JP-43": "熊本県", "JP-44": "大分県",
    "JP-45": "宮崎県", "JP-46": "鹿児島県", "JP-47": "沖縄県"
}

def login_to_xarvio(email, password):
    try:
        login_url = f"{BASE_LOGIN_URL}?include=emails,profile,data,sessionInfo&loginID={urllib.parse.quote(email)}&password={urllib.parse.quote(password)}&apiKey={API_KEY}"
        login_res = requests.get(login_url)
        login_res.raise_for_status()
        login_data = login_res.json()

        login_token = login_data["sessionInfo"]["cookieValue"]
        gigya_uuid = login_data["UID"]
        gigya_uuid_signature = login_data["UIDSignature"]
        gigya_signature_timestamp = login_data["signatureTimestamp"]

        token_res = requests.post(TOKEN_API_URL, json={
            "gigyaUuid": gigya_uuid,
            "gigyaUuidSignature": gigya_uuid_signature,
            "gigyaSignatureTimestamp": gigya_signature_timestamp
        }, headers={"Cookie": f"LOGIN_TOKEN={login_token}"})
        token_res.raise_for_status()
        api_token = token_res.json()["token"]
        return login_token, api_token
    except Exception as e:
        st.error(f"ログインエラー: {e}")
        return None, None

def get_farms(api_token, login_token):
    st.session_state["graphql_api_call_count"] += 1
    farms_query = {
        "operationName": "FarmsOverview",
        "variables": {},
        "query": """
            query FarmsOverview {
                farms: farmsV2(uuids: []) {
                    uuid
                    name
                }
            }
        """
    }
    headers = {
        "Content-Type": "application/json",
        "Cookie": f"LOGIN_TOKEN={login_token}; DF_TOKEN={api_token}"
    }
    farms_res = requests.post(GRAPHQL_END_POINT, json=farms_query, headers=headers)
    farms_res.raise_for_status()
    return farms_res.json()["data"]["farms"]


def initialize_session_state():
    keys_defaults = {
        "is_logged_in": False,
        "login_token": None,
        "api_token": None,
        "farms_data": None,
        "show_map": False,
        "selected_stage": None,
        "show_labels": True,
        "graphql_api_call_count": 0,
        "reverse_geocode_api_call_count": 0,
        "reverse_geocode_cache_hits": 0,
        "reverse_geocode_cache_misses": 0,
    }
    for key in ["is_logged_in", "login_token", "api_token", "farms_data"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "is_logged_in" else False

    if "show_map" not in st.session_state:
        st.session_state["show_map"] = False

    if "selected_stage" not in st.session_state:
        st.session_state["selected_stage"] = None

    if "show_labels" not in st.session_state:
        st.session_state["show_labels"] = True

    # 👇 ここを追加
    if "reverse_geocode_cache_hits" not in st.session_state:
        st.session_state["reverse_geocode_cache_hits"] = 0
    if "reverse_geocode_cache_misses" not in st.session_state:
        st.session_state["reverse_geocode_cache_misses"] = 0
    if "reverse_geocode_api_call_count" not in st.session_state:
        st.session_state["reverse_geocode_api_call_count"] = 0
    if "graphql_api_call_count" not in st.session_state:
        st.session_state["graphql_api_call_count"] = 0


def build_field_dataframe(fields, geolocator):
    field_data = []
    for field in fields:
        field_uuid = field.get("uuid")
        field_name = field.get("name")
        area = round(field.get("area", 0) * 0.01, 2)
        boundary = field.get("boundary", {})
        crop_seasons = field.get("cropSeasonsV2", [])

        prefecture = city = ""
        centroid_lat = centroid_lon = None
        try:
            polygon = shape(boundary)
            if isinstance(polygon, (Polygon, MultiPolygon)):
                centroid = polygon.centroid
                centroid_lat, centroid_lon = round(centroid.y, 6), round(centroid.x, 6)

                if st.session_state.get("use_reverse_geocode", True):
                    address = reverse_geocode(centroid_lat, centroid_lon)
                    iso = address.get("ISO3166-2-lvl4") or address.get("ISO3166-2-lvl3")
                    prefecture = ISO_TO_PREF_NAME.get(iso, "")
                    city = address.get("city", address.get("town", address.get("village", "")))

        except:
            pass

        for cs in crop_seasons or [{}]:
            crop = cs.get("crop", {}).get("name", "未登録")
            cs_uuid = cs.get("uuid")
            start_date = to_jst_ymd(cs.get("startDate"))
            variety = cs.get("variety", {}).get("name", "未登録")
            cropEstablishmentMethodCode = cs.get("cropEstablishmentMethodCode")
            cropEstablishmentGrowthStageIndex = cs.get("cropEstablishmentGrowthStageIndex")

            field_data.append({
                "Field UUID": field_uuid,
                "農場名": field.get("farmName", "不明な農場"),
                "圃場名": field_name,
                "作物": crop,
                "品種": variety,
                "作付方法": cropEstablishmentMethodCode,
                "作付時のBBCH": cropEstablishmentGrowthStageIndex,
                "cropseason_uuid": cs_uuid,
                "作付日": start_date,
                "面積 (a)": area,
                "都道府県": prefecture,
                "市区町村": city,
                "中心座標": f"{centroid_lon}, {centroid_lat}" if centroid_lat and centroid_lon else "",
                "ポリゴン情報": json.dumps(boundary, ensure_ascii=False)
            })
    return field_data

def fetch_fields_for_multiple_farms(farm_uuids, login_token, api_token):
    st.session_state["graphql_api_call_count"] += 1

    all_fields = []
    for farm_uuid in farm_uuids:
        query = {
            "operationName": "CombinedFieldData",
            "variables": {
                "farmUuids": [farm_uuid],
                "languageCode": "ja",
                "cropSeasonLifeCycleStates": ["ACTIVE", "PLANNED"],
                "withBoundarySvg": True
            },
            "query": """
                query CombinedFieldData(
                  $farmUuids: [UUID!]!, 
                  $languageCode: String!, 
                  $cropSeasonLifeCycleStates: [LifecycleState]!, 
                  $withBoundarySvg: Boolean!
                ) {
                  farms: farmsV2(uuids: $farmUuids) {
                    uuid
                    name
                  }
                  fieldsV2(farmUuids: $farmUuids) {
                    uuid
                    name
                    area
                    boundary
                    boundarySvg @include(if: $withBoundarySvg)
                    cropSeasonsV2(lifecycleState: $cropSeasonLifeCycleStates) {
                      uuid
                      startDate
                      crop(languageCode: $languageCode) {
                        name
                      }
                      variety(languageCode: $languageCode) {
                        name
                      }
                      cropEstablishmentGrowthStageIndex
                      cropEstablishmentMethodCode
                      countryCropGrowthStagePredictions {
                        index
                        startDate
                        endDate
                        scale
                        gsOrder
                        cropGrowthStageV2(languageCode: $languageCode) {
                          uuid
                          name
                          code
                        }
                      }
                    }
                  }
                }
            """
        }

        headers = {
            "Content-Type": "application/json",
            "Cookie": f"LOGIN_TOKEN={login_token}; DF_TOKEN={api_token}"
        }

        response = requests.post(GRAPHQL_END_POINT, json=query, headers=headers)
        response.raise_for_status()
        data = response.json()["data"]

        farm_name = data["farms"][0]["name"] if data["farms"] else "不明な農場"
        fields = data["fieldsV2"]

        # 各圃場に農場名を付与
        for field in fields:
            field["farmName"] = farm_name

        all_fields.extend(fields)

    return all_fields


def extract_bbch_data(fields, selected_field_uuids, geolocator):
    bbch_data = []
    for field in fields:
        if field.get("uuid") not in selected_field_uuids:
            continue

        farm_name = field.get("farmName", "不明な農場")
        field_name = field.get("name", "不明な圃場名")
        area = round(field.get("area", 0) * 0.01, 2)
        boundary = field.get("boundary", {})
        crop_seasons = field.get("cropSeasonsV2") or []

        prefecture = city = ""
        centroid_lat = centroid_lon = None
        try:
            polygon = shape(boundary)
            if isinstance(polygon, (Polygon, MultiPolygon)):
                centroid = polygon.centroid
                centroid_lat, centroid_lon = round(centroid.y, 6), round(centroid.x, 6)

                if use_reverse_geocode:
                    address = reverse_geocode(centroid_lat, centroid_lon)
                    iso = address.get("ISO3166-2-lvl4") or address.get("ISO3166-2-lvl3")
                    prefecture = ISO_TO_PREF_NAME.get(iso, "")
                    city = address.get("city", address.get("town", address.get("village", "")))
        except:
            pass

        for cs in crop_seasons:
            crop_name = cs.get("crop", {}).get("name", "不明な作物")
            variety = cs.get("variety", {}).get("name", "未登録")
            cs_uuid = cs.get("uuid", "UUID不明")
            cropEstablishmentMethodCode = cs.get("cropEstablishmentMethodCode")
            cropEstablishmentGrowthStageIndex = cs.get("cropEstablishmentGrowthStageIndex")
            start_date = to_jst_ymd(cs.get("startDate"))

            predictions = cs.get("countryCropGrowthStagePredictions")
            if not predictions:
                continue

            for pred in predictions:
                gs = pred.get("cropGrowthStageV2")
                if not gs:
                    continue
                coords = []
                try:
                    poly_json = boundary
                    if poly_json["type"] == "Polygon":
                        coords = poly_json["coordinates"][0]
                    elif poly_json["type"] == "MultiPolygon":
                        coords = poly_json["coordinates"][0][0]
                except Exception:
                    pass
                
                bbch_data.append({
                    "農場名": farm_name,
                    "圃場名": field_name,
                    "作物": crop_name,
                    "作付UUID": cs_uuid,
                    "BBCHコード": gs.get("code", "不明"),
                    "BBCH名称": gs.get("name", "不明"),
                    "品種": variety,
                    "作付方法": cropEstablishmentMethodCode,
                    "作付時のBBCH": cropEstablishmentGrowthStageIndex,
                    "作付日": start_date,
                    "面積 (a)": area,
                    "都道府県": prefecture,
                    "市区町村": city,
                    "中心座標": f"{centroid_lon}, {centroid_lat}" if centroid_lat and centroid_lon else "",
                    "ポリゴン情報": json.dumps(boundary, ensure_ascii=False),
                    "BBCH開始日": to_jst_ymd(pred.get("startDate", "不明")),
                    "coords": coords,
                    "date": to_jst_ymd(pred.get("startDate", "")),
                    "name": field_name,
                    "variety": variety
                })
    return bbch_data

# ----------------------------------
# アプリ実行部分
# ----------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["🌾 xarvio","📊 グラフ", "📋 BBCH一覧（PIVOT）", "🗺 地図"])

# --- Streamlit ページ設定 ---
with tab1:
    
    initialize_session_state()
    st.title("🌾 xarvio 圃場マップビューア")

    # --- ログインフォーム ---
    if not st.session_state.is_logged_in:
        with st.form("login_form"):
            email = st.text_input("メールアドレス")
            password = st.text_input("パスワード", type="password")
            submitted = st.form_submit_button("ログイン")

        if submitted:
            if not email or not password:
                st.warning("⚠️ メールアドレスとパスワードの両方を入力してください。")
            else:
                login_token, api_token = login_to_xarvio(email, password)
                if login_token and api_token:
                    farms = get_farms(api_token, login_token)
                    st.session_state.login_token = login_token
                    st.session_state.api_token = api_token
                    st.session_state.farms_data = farms
                    st.session_state.is_logged_in = True
                    st.rerun()
                else:
                    st.warning("⚠️ メールアドレスかパスワードが正しくありません。")

    # --- ログイン後処理 ---
    if st.session_state.is_logged_in:
        st.success("ログイン済み")

        farms = st.session_state.farms_data
        farm_name_to_uuid = {f["name"]: f["uuid"] for f in farms}

        selected_farm_names = st.multiselect("🚜 複数の農場を選択", list(farm_name_to_uuid.keys()))
        selected_farm_uuids = [farm_name_to_uuid[name] for name in selected_farm_names]

        if "geolocator" not in st.session_state:
            st.session_state.geolocator = Nominatim(user_agent="xarvio-app")

        # 地域情報を取得するかどうかのオプション
        use_reverse_geocode = st.toggle("📍 地域情報（都道府県・市区町村）を取得する（処理に時間がかかります。）", value=False)
        st.session_state.use_reverse_geocode = use_reverse_geocode

        if st.button("📥 圃場情報を取得"):
            if not selected_farm_uuids:
                st.warning("⚠️ 取得する農場を1つ以上選んでください。")
                st.stop()
            total_start = time.time()
            status = st.empty()
            status.info("🔄 圃場データを取得中...")
            # Step 1: API呼び出し
            t1 = time.time()
            geolocator = Nominatim(user_agent="xarvio-app")
            fields = fetch_fields_for_multiple_farms(
                selected_farm_uuids,
                st.session_state.login_token,
                st.session_state.api_token
            )
            t2 = time.time()
            st.markdown(f"✅ **API取得時間**: `{t2 - t1:.2f}秒`　｜　**圃場数**: `{len(fields)}`件")
            st.success(f"✅ APIレスポンス取得完了（{t2 - t1:.2f}秒）")
            # Step 2: データ整形
            t3 = time.time()
            field_data = build_field_dataframe(fields, geolocator)
            t4 = time.time()
            st.success(f"✅ 圃場データ整形完了（{t4 - t3:.2f}秒）")
            # Step 3: DataFrame生成
            t5 = time.time()
            df = pd.DataFrame(field_data)
            st.session_state.fields = fields
            st.session_state.field_data = field_data
            st.session_state.df = df
            t6 = time.time()
            st.success(f"✅ DataFrame構築完了（{t6 - t5:.2f}秒）")
            total_end = time.time()
            st.info(f"⏱️ 全処理時間: {total_end - total_start:.2f} 秒")
            status.empty()

        if "df" in st.session_state:
            df = st.session_state.df

            st.subheader("📋 圃場一覧（BBCH選択可能）")
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
            gb.configure_column("Field UUID", headerCheckboxSelection=True, checkboxSelection=True)
            grid_options = gb.build()

            with st.form("select_fields"):
#                grid_response = AgGrid(df, gridOptions=grid_options, update_mode=GridUpdateMode.SELECTION_CHANGED)
                grid_response = AgGrid(df, gridOptions=grid_options, update_mode=GridUpdateMode.MODEL_CHANGED)

                submit = st.form_submit_button("🎯 BBCH取得")

            if submit:
                selected_rows = grid_response.selected_rows
                if selected_rows is None or selected_rows.empty:
                    st.warning("⚠ 圃場を1つ以上選択してください。")
                    st.stop()
                else:
                    if isinstance(selected_rows, pd.DataFrame):
                        selected_rows = selected_rows.to_dict(orient="records")

                    if not isinstance(selected_rows[0], dict):
                        st.error("⚠️ 選択されたデータの形式が想定と異なります。")
                        st.stop()

                    selected_field_uuids = [r["Field UUID"] for r in selected_rows]
                    status = st.empty()
                    status.info("⏳ BBCHデータを抽出中...")
                    

                    total_start = time.time()

                    t1 = time.time()

                    geolocator = Nominatim(user_agent="xarvio-app")
                    bbch_data = extract_bbch_data(st.session_state.fields, selected_field_uuids, geolocator)
                    t2 = time.time()
                    st.success(f"✅ データを取得しました。タブを切り替えて確認してください。")
                    st.success(f"✅ データ抽出完了（{t2 - t1:.2f} 秒）")

                    t3 = time.time()
                    bbch_df = pd.DataFrame(bbch_data)

                    if not bbch_df.empty:
                        bbch_df["BBCHステージ"] = bbch_df["BBCHコード"].astype(str) + " (" + bbch_df["BBCH名称"] + ")"
                        st.session_state.bbch_df = bbch_df  # 🎯 保存
                    t4 = time.time()
                    st.success(f"✅ DataFrame整形完了（{t4 - t3:.2f} 秒）")
                    # キャッシュ使用状況の表示
                    #st.markdown(f"🧠 **キャッシュ使用**: `reverse_geocode` → {'使用済み（@st.cache_data）' if use_reverse_geocode else '未使用（チェックオフ）'}`")
                    total_end = time.time()
                    st.info(f"⏱️ BBCH取得 全処理時間: **{total_end - total_start:.2f} 秒**")
                    status.empty()


                    if bbch_df.empty:
                        selected_field_names = [r["圃場名"] for r in selected_rows if "圃場名" in r]
                        st.warning("⚠️ 選択された圃場はBBCHの情報がありません。予測機能が有効になっていない可能性があります。")
                        st.markdown("#### 📋 BBCHデータが存在しない圃場")
                        st.write(selected_field_names)

            if "bbch_df" in st.session_state:
                bbch_df = st.session_state.bbch_df

                pivot_index_cols = [
                    "農場名", "圃場名", "作物", "作付UUID", "品種", "作付方法",
                    "作付時のBBCH", "作付日", "面積 (a)",
                    "都道府県", "市区町村", "中心座標" #, "ポリゴン情報"
                ]

                pivot_df = bbch_df.pivot_table(
                    index=pivot_index_cols,
                    columns="BBCHステージ",
                    values="BBCH開始日",
                    aggfunc="first"
                ).reset_index()
                pivot_df.columns.name = None
                st.session_state.pivot_df = pivot_df
            with tab2:
                st.subheader("📊 選択圃場のBBCHステージ一覧")
                if "pivot_df" in st.session_state:
                    pivot_df = st.session_state.pivot_df
                    gb = GridOptionsBuilder.from_dataframe(pivot_df)
                    gb.configure_default_column(resizable=True)
                    gb.configure_grid_options(domLayout='normal', enableCharts=True, enableRangeSelection=True)
                    grid_options = gb.build()
                    plot_bbch_stacked_bar(bbch_df)
                else:
                    st.warning("⚠ PIVOTデータがまだ生成されていません。先に圃場を選択してBBCHを取得してください。")
                
            # 積立棒グラフの表示
            #plot_bbch_stacked_bar(bbch_df)
            with tab3:
                st.subheader("📋 PIVOTデータ")
                
                if "pivot_df" in st.session_state:
                    pivot_df = st.session_state.pivot_df
                    AgGrid(
                        pivot_df,
                        gridOptions=grid_options,
                        update_mode=GridUpdateMode.NO_UPDATE,
                        fit_columns_on_grid_load=False,  # 横スクロール用にカラム幅を自動で詰めない
                        allow_unsafe_jscode=True,
                        height=500
                    )

                    csv = pivot_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("📥 BBCHステージ一覧（CSV）をダウンロード", data=csv, file_name="BBCHステージ一覧.csv", mime="text/csv")
                else:
                    st.warning("⚠ PIVOTデータがまだ生成されていません。先に圃場を選択してBBCHを取得してください。")

            with tab4:                    
                # 地図の生成と表示
                if "bbch_df" in st.session_state:
                    bbch_df = st.session_state.bbch_df
                    bbch_records = bbch_df.to_dict(orient="records")
                    selected_map_style, selected_bbch, title_prefix, selected_label = get_user_inputs(bbch_records)


                    # タイトルの生成と表示
                    map_title = generate_map_title(title_prefix, selected_bbch)

                    st.markdown(f"### 📌 現在の表示: {map_title}")

                    # 圃場名でソートして選択肢を作る
                    field_options = {
                        f'{row["圃場名"]}（{row.get("農場名", "不明な農場")}）': row["中心座標"]
                        for row in sorted(
                            bbch_df.dropna(subset=["中心座標"]).to_dict(orient="records"),
                            key=lambda x: x["圃場名"]
                        )
                    }


                    # UIの選択ボックス
                    selected_jump_field = st.selectbox("📍 地図をズーム表示したい圃場を選んでください", options=list(field_options.keys()))

                    # 選択された圃場の中心座標を取得
                    jump_lat, jump_lon = extract_lat_lon(field_options[selected_jump_field])


                    # 地図生成・表示
                    fig = create_field_map(
                        field_data=bbch_records,
                        selected_bbch=selected_bbch,
                        map_style=selected_map_style,
                        map_title=map_title,
                        label_key=selected_label,
                        center_override={"lat": jump_lat, "lon": jump_lon},
                        zoom_override=14  # 適度にズームイン
                    )
                    st.plotly_chart(fig, use_container_width=True, 
                            #    config={"scrollZoom": True, "displayModeBar": False})
                                config={
                                "scrollZoom": True,
                                "displayModeBar": True,  # モードバー表示を有効に
                                "modeBarButtonsToRemove": [
                                    "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                                    "autoScale2d", "resetScale2d", "hoverClosestCartesian", "hoverCompareCartesian",
                                    "toggleSpikelines"
                                ],
                                "modeBarButtonsToAdd": ["toggleFullscreen", "toImage"]  # ← 全画面ボタンのみ有効
                                })
                # ダウンロードボタン
                    download_map_html(fig)


                # === 🗂 BBCHごとにKMLダウンロード ===
                if "bbch_df" in st.session_state:
                    bbch_df = st.session_state.bbch_df

                    st.markdown("### 📦 BBCHステージ別にKMLをダウンロード")
                    selected_bbch_codes = st.multiselect(
                        "📌 ダウンロードしたいBBCHコードを選んでください",
                        sorted(bbch_df["BBCHコード"].unique()),
                        default=[],
                    )

                    for code in selected_bbch_codes:
                        filtered_df = bbch_df[bbch_df["BBCHコード"] == code]
                        if not filtered_df.empty:
                            kml_content = create_kml_from_bbch_df(filtered_df).encode("utf-8")
                            bbch_name = filtered_df.iloc[0]["BBCH名称"]
                            file_name = f"bbch_{code}_{bbch_name}.kml".replace(" ", "_").replace("（", "_").replace("）", "_")

                            st.download_button(
                                label=f"📥 KMLダウンロード - BBCH{code}（{bbch_name}）",
                                data=kml_content,
                                file_name=file_name,
                                mime="application/vnd.google-earth.kml+xml",
                                key=f"kml_download_{code}"
                            )
                with st.expander("🚗 BBCH圃場のおすすめ巡回ルートを表示", expanded=False):
                    if "bbch_df" in st.session_state:
                        bbch_df = st.session_state.bbch_df

                        # ① BBCHコードを選択
                        bbch_codes = sorted(
                            bbch_df["BBCHコード"].dropna().unique(),
                            key=lambda x: int(x) if str(x).isdigit() else x
                        )
                        selected_bbch_code = st.selectbox("① 対象のBBCHコードを選んでください", bbch_codes)

                        # ② 選択されたBBCHコードに対応する開始日を「複数選択」
                        bbch_dates = sorted(
                            bbch_df[bbch_df["BBCHコード"] == selected_bbch_code]["BBCH開始日"].dropna().unique(),
                            key=lambda x: pd.to_datetime(x)
                        )
                        selected_dates = st.multiselect("② 該当BBCHステージの開始日を選んでください（複数選択可）", bbch_dates, default=bbch_dates)

                        # ③ BBCH + 選択された日付に該当する圃場のみ抽出
                        filtered_df = bbch_df[
                            (bbch_df["BBCHコード"] == selected_bbch_code) &
                            (bbch_df["BBCH開始日"].isin(selected_dates))
                        ].dropna(subset=["中心座標"])

                        field_names = sorted(filtered_df["圃場名"].dropna().unique())
                        selected_fields = st.multiselect("③ 巡回対象とする圃場を選んでください", options=field_names, default=field_names)

                        # ④ Googleマップの上限（23件）制限
                        if len(selected_fields) > 23:
                            st.warning("⚠️ Googleマップの仕様により、選択できる圃場は最大23個までです。")
                            selected_fields = selected_fields[:23]

                        # ⑤ 巡回ルート生成
                        if selected_fields:
                            selected_df = filtered_df[filtered_df["圃場名"].isin(selected_fields)]
                            route_input = []
                            for _, row in selected_df.iterrows():
                                lat, lon = extract_lat_lon(row["中心座標"])
                                if lat and lon:
                                    route_input.append({
                                        "name": row["圃場名"],
                                        "lat": lat,
                                        "lon": lon
                                    })

                            if len(route_input) >= 2:
                                # Greedyなルート作成
                                start = route_input[0]
                                route = [start]
                                unvisited = route_input[1:]

                                while unvisited:
                                    last = route[-1]
                                    next_point = min(unvisited, key=lambda p: geodesic((last["lat"], last["lon"]), (p["lat"], p["lon"])).km)
                                    route.append(next_point)
                                    unvisited.remove(next_point)

                                # Googleマップ用のURL（現在地スタート・戻る）
                                gmap_url = generate_google_maps_route(route)

                                st.markdown("### 🧭 巡回ルート（Googleマップ）")
                                st.markdown(f"[📍 道順を表示する]({gmap_url})", unsafe_allow_html=True)

                                st.markdown("#### 🔍 巡回順の圃場一覧")
                                for i, pt in enumerate(route, start=1):
                                    st.markdown(f"{i}. **{pt['name']}**（{pt['lat']:.5f}, {pt['lon']:.5f}）")
                            else:
                                st.warning("⚠️ 2つ以上の圃場を選択してください。")
