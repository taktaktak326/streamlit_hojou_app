import subprocess
import streamlit as st

st.markdown("### 🔍 chromedriver 実体パス探索中...")

# `chromedriver` の実体を探す
chromedriver_path = subprocess.getoutput("which chromedriver")
find_path = subprocess.getoutput("find / -name chromedriver 2>/dev/null")

st.write("✅ which chromedriver の結果:")
st.code(chromedriver_path)

st.write("✅ find / -name chromedriver の結果（フルスキャン）:")
st.code(find_path)
