import streamlit as st
import os
import subprocess

st.set_page_config(page_title="Chromium Debugger", layout="wide")
st.title("🛠️ Render 環境の Chromium / Chromedriver デバッグ")

# --- 1. パス確認 ---
def find_binaries():
    chrome_candidates = [
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/lib/chromium/chromium",
        "/usr/lib/chromium-browser/chromium"
    ]
    driver_candidates = [
        "/usr/bin/chromedriver",
        "/usr/lib/chromium/chromedriver",
        "/usr/lib/chromium-browser/chromedriver"
    ]

    chrome_bin = next((p for p in chrome_candidates if os.path.exists(p)), None)
    driver_bin = next((p for p in driver_candidates if os.path.exists(p)), None)

    return chrome_bin, driver_bin

chrome_bin, driver_bin = find_binaries()

st.subheader("🔍 パス確認")
st.write("✅ Chromium:", chrome_bin or "❌ Not Found")
st.write("✅ Chromedriver:", driver_bin or "❌ Not Found")

# --- 2. 実行権限確認 ---
st.subheader("🔐 実行権限確認")
if chrome_bin:
    st.write(f"Chromium 実行可能: {'✅ Yes' if os.access(chrome_bin, os.X_OK) else '❌ No'}")
if driver_bin:
    st.write(f"Chromedriver 実行可能: {'✅ Yes' if os.access(driver_bin, os.X_OK) else '❌ No'}")

# --- 3. バージョン確認 ---
def get_version(cmd):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode().strip() or result.stderr.decode().strip()
    except Exception as e:
        return f"エラー: {e}"

st.subheader("🧭 バージョン確認")
if chrome_bin:
    st.write("Chromium バージョン:", get_version([chrome_bin, "--version"]))
if driver_bin:
    st.write("Chromedriver バージョン:", get_version([driver_bin, "--version"]))

# --- 4. PATH環境変数表示 ---
st.subheader("📦 環境変数 PATH")
st.code(os.environ.get("PATH", "Not Found"))

# --- 5. ls -l で確認 ---
st.subheader("📁 /usr/bin と /usr/lib/chromium の中身")

def list_dir(path):
    try:
        return subprocess.run(["ls", "-l", path], stdout=subprocess.PIPE).stdout.decode()
    except Exception as e:
        return f"❌ エラー: {e}"

st.text("📂 /usr/bin:")
st.code(list_dir("/usr/bin"))

st.text("📂 /usr/lib/chromium:")
st.code(list_dir("/usr/lib/chromium"))
