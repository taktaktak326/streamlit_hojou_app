import os
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# ========== デバッグログ ==========
st.markdown("### ✅ Chrome 実行ファイル & Driver パスチェック")

binary_path = "/usr/bin/chromium"
driver_path = "/usr/lib/chromium/chromedriver"

st.write(f"🔍 Chrome binary path: `{binary_path}`")
st.write(f"🔍 ChromeDriver path: `{driver_path}`")

# パスの存在確認
binary_exists = os.path.exists(binary_path)
driver_exists = os.path.exists(driver_path)

st.write(f"✅ Chrome Binary Exists: {binary_exists}")
st.write(f"✅ ChromeDriver Exists: {driver_exists}")

if not binary_exists or not driver_exists:
    st.error("❌ 必要な実行ファイルが見つかりません。packages.txt を確認してください。")
    st.stop()

# ========== Selenium セットアップ ==========
chrome_options = Options()
chrome_options.binary_location = binary_path
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

try:
    driver = webdriver.Chrome(
        service=Service(driver_path),
        options=chrome_options
    )
    st.success("✅ Selenium driver 初期化成功！")
    driver.quit()
except Exception as e:
    st.error(f"❌ Selenium 初期化エラー: {e}")
