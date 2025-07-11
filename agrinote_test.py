import streamlit as st
import os
import subprocess

st.set_page_config(page_title="Chromium / Chromedriver Debugger", layout="wide")
st.title("🛠️ Render 環境の Chromium / Chromedriver デバッグ")

def get_binary_path(binary_name):
    try:
        result = subprocess.run(["which", binary_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode().strip()
    except Exception as e:
        return f"エラー: {e}"

def get_version(path, args=["--version"]):
    try:
        result = subprocess.run([path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode().strip() or result.stderr.decode().strip()
    except Exception as e:
        return f"エラー: {e}"

def list_dir(path):
    try:
        return subprocess.run(["ls", "-l", path], stdout=subprocess.PIPE).stdout.decode()
    except Exception as e:
        return f"❌ エラー: {e}"

# 候補パス
chrome_candidates = [
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/usr/lib/chromium/chromium",
    "/usr/lib/chromium-browser/chromium",
]
driver_candidates = [
    "/usr/bin/chromedriver",
    "/usr/lib/chromium/chromedriver",
    "/usr/lib/chromium-browser/chromedriver",
]

# 検出
chrome_path = next((p for p in chrome_candidates if os.path.exists(p)), None)
driver_path = next((p for p in driver_candidates if os.path.exists(p)), None)

st.subheader("🔍 パス検出結果")
st.write("✅ Chromium path:", chrome_path or "❌ Not Found")
st.write("✅ Chromedriver path:", driver_path or "❌ Not Found")

st.subheader("🔐 実行権限")
if chrome_path:
    st.write("Chromium 実行可能:", "✅ Yes" if os.access(chrome_path, os.X_OK) else "❌ No")
if driver_path:
    st.write("Chromedriver 実行可能:", "✅ Yes" if os.access(driver_path, os.X_OK) else "❌ No")

st.subheader("🧭 バージョン確認")
if chrome_path:
    st.write("Chromium バージョン:", get_version(chrome_path))
else:
    st.warning("Chromium が見つかりませんでした。")

if driver_path:
    st.write("Chromedriver バージョン:", get_version(driver_path))
else:
    st.warning("Chromedriver が見つかりませんでした。")

st.subheader("📦 PATH環境変数")
st.code(os.environ.get("PATH", "❌ Not Found"))

st.subheader("📂 ディレクトリの中身")

st.text("📁 /usr/bin:")
st.code(list_dir("/usr/bin"))

st.text("📁 /usr/lib/chromium:")
st.code(list_dir("/usr/lib/chromium"))

st.text("📁 /usr/lib/chromium-browser:")
st.code(list_dir("/usr/lib/chromium-browser"))
