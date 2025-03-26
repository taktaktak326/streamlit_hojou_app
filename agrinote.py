import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import traceback

st.markdown("### 🧪 Selenium 初期化デバッグ")

try:
    st.write("🔧 Chrome Options セットアップ中...")
    chrome_options = Options()
    chrome_options.binary_location = "/usr/bin/chromium"
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    st.write("🚀 WebDriver 起動を試みます...")

    driver = webdriver.Chrome(
        service=Service("/usr/bin/chromedriver"),
        options=chrome_options
    )

    st.success("✅ Selenium 初期化成功！")
    driver.quit()

except Exception as e:
    st.error("❌ Selenium 初期化でエラーが発生しました")
    st.code(traceback.format_exc())
