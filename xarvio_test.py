import streamlit as st
import requests
import pandas as pd
import urllib.parse  # URLエンコード用

# APIエンドポイント
BASE_LOGIN_URL = "https://accounts.eu1.gigya.com/accounts.login"
TOKEN_API_URL = "https://fm-api.xarvio.com/api/users/tokens"
FARM_API_BASE_URL = "https://fm-api.xarvio.com/api/farms/v2/farms"

# APIキー
API_KEY = "3_W-AXsoj7TvX-9gi7S-IGxXfLWVkEbnGSl57M7t49GN538umaKs2EID8hyipAux2y"

# Streamlit アプリのタイトル
st.title("Xarvio Farm Data Viewer (Debug Mode)")

# セッション状態の初期化
if "login_token" not in st.session_state:
    st.session_state.login_token = None
if "api_token" not in st.session_state:
    st.session_state.api_token = None
if "gigya_uuid" not in st.session_state:
    st.session_state.gigya_uuid = None
if "gigya_uuid_signature" not in st.session_state:
    st.session_state.gigya_uuid_signature = None
if "gigya_signature_timestamp" not in st.session_state:
    st.session_state.gigya_signature_timestamp = None

# ユーザー入力
st.subheader("ログイン")
email = st.text_input("メールアドレス", type="default")
password = st.text_input("パスワード", type="password", help="ログイン用のパスワードを入力してください。")

if st.button("ログイン"):
    if email and password:
        encoded_email = urllib.parse.quote(email)
        encoded_password = urllib.parse.quote(password)

        login_url = f"{BASE_LOGIN_URL}?include=emails,profile,data,sessionInfo&loginID={encoded_email}&password={encoded_password}&apiKey={API_KEY}"

        try:
            response = requests.get(login_url)
            response.raise_for_status()
            data = response.json()

            st.write("✅ **ログイン API レスポンス (生データ)**")
            st.json(data)

            # 必要なパラメータを取得
            if "sessionInfo" in data:
                session_info = data["sessionInfo"]

                # LOGIN_TOKEN の取得
                st.session_state.login_token = session_info.get("cookieValue", None)

                # Gigya認証に必要な値の取得
                st.session_state.gigya_uuid = data.get("UID", None)
                st.session_state.gigya_uuid_signature = data.get("UIDSignature", None)
                st.session_state.gigya_signature_timestamp = data.get("signatureTimestamp", None)

                if st.session_state.login_token and st.session_state.gigya_uuid:
                    st.success("✅ ログイン成功！")
                    st.write(f"🔑 **取得した LOGIN_TOKEN:** {st.session_state.login_token}")
                    st.write(f"🔑 **Gigya UUID:** {st.session_state.gigya_uuid}")
                    st.write(f"🔑 **Gigya UUID Signature:** {st.session_state.gigya_uuid_signature}")
                    st.write(f"🔑 **Gigya Signature Timestamp:** {st.session_state.gigya_signature_timestamp}")

                else:
                    st.error("⚠️ 必要な認証情報が取得できませんでした。")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ ログイン API エラー: {e}")
            st.text(response.text)
            st.text(response.status_code)

# `LOGIN_TOKEN` を取得したら `tokens` API に `POST` でリクエスト
if st.session_state.login_token and st.session_state.gigya_uuid:
    st.subheader("API トークン取得")

    if st.button("API トークンを取得"):
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Cookie": f"LOGIN_TOKEN={st.session_state.login_token}",
            "Origin": "https://fm.xarvio.com",
            "Referer": "https://fm.xarvio.com/"
        }

        payload = {
            "gigyaUuid": st.session_state.gigya_uuid,
            "gigyaUuidSignature": st.session_state.gigya_uuid_signature,
            "gigyaSignatureTimestamp": st.session_state.gigya_signature_timestamp
        }

        try:
            response = requests.post(TOKEN_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            token_data = response.json()

            st.write("✅ **API トークン取得 API レスポンス (生データ)**")
            st.json(token_data)

            if "token" in token_data:
                st.session_state.api_token = token_data["token"]
                st.success("✅ API トークン取得成功！")
                st.write(f"🔑 **取得した API トークン:** {st.session_state.api_token}")
            else:
                st.error("⚠️ `token` がレスポンスに含まれていません。")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ API トークン取得エラー: {e}")
            st.text(response.text)
            st.text(response.status_code)

# `API_TOKEN` を使って `farm` データを取得
if st.session_state.api_token:
    st.subheader("農場データの取得")

    farm_uuid = st.text_input("Farm UUID を入力してください")

    if st.button("農場データを取得"):
        if not farm_uuid:
            st.warning("⚠️ Farm UUID を入力してください。")
        else:
            farm_api_url = f"{FARM_API_BASE_URL}?farmuuid={farm_uuid}"

            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
                "Cookie": f"LOGIN_TOKEN={st.session_state.login_token}; DF_TOKEN={st.session_state.api_token}",
                "Origin": "https://fm.xarvio.com",
                "Referer": "https://fm.xarvio.com/"
            }

            # 🔍 **デバッグ情報を表示**
            st.write("📡 **送信するヘッダー:**")
            st.json(headers)

            try:
                response = requests.get(farm_api_url, headers=headers)
                response.raise_for_status()
                farm_data = response.json()

                st.write("✅ **農場データ API レスポンス (生データ)**")
                st.json(farm_data)

                # 🔍 **レスポンスの長さを確認**
                st.write(f"🔢 **レスポンスのデータ数: {len(farm_data)}**")

                # **リスト形式のデータを DataFrame に変換**
                if isinstance(farm_data, list) and len(farm_data) > 0:
                    df = pd.DataFrame(farm_data)
                    st.success("✅ 農場データ取得成功！")
                    st.dataframe(df)
                else:
                    st.error("⚠️ 農場データが空です。レスポンスを確認してください。")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ 農場データ取得エラー: {e}")
                st.write("📡 **HTTPレスポンス (テキスト)**")
                st.text(response.text)
                st.write("📡 **HTTPステータスコード:**")
                st.text(response.status_code)



