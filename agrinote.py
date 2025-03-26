from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import json
import urllib.parse
import requests

app = FastAPI()

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/fetch-fields")
def fetch_fields(req: LoginRequest):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    chrome_options.binary_location = "/usr/bin/google-chrome"
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get("https://agri-note.jp/b/login/")
        time.sleep(2)

        inputs = driver.find_elements(By.CLASS_NAME, "_1g2kt34")
        if len(inputs) < 2:
            raise HTTPException(status_code=500, detail="Login form not found")

        inputs[0].send_keys(req.email)
        inputs[1].send_keys(req.password)
        inputs[1].send_keys(Keys.RETURN)
        time.sleep(5)

        cookies_list = driver.get_cookies()
        cookie_dict = {cookie['name']: cookie['value'] for cookie in cookies_list}

        required = ['an_api_token', 'an_login_status', 'tracking_user_uuid']
        if not all(k in cookie_dict for k in required):
            raise HTTPException(status_code=403, detail="Required cookies missing")

        csrf_token = json.loads(urllib.parse.unquote(cookie_dict['an_login_status']))["csrf"]

        cookies = {
            "an_api_token": cookie_dict["an_api_token"],
            "an_login_status": cookie_dict["an_login_status"],
            "tracking_user_uuid": cookie_dict["tracking_user_uuid"],
        }

        headers = {
            "x-an-csrf-token": csrf_token,
            "x-user-uuid": cookie_dict['tracking_user_uuid'],
            "x-agri-note-api-client": "v2.97.0",
            "x-requested-with": "XMLHttpRequest",
            "referer": "https://agri-note.jp/b/",
            "user-agent": "Mozilla/5.0"
        }

        driver.quit()

        response = requests.get("https://agri-note.jp/an-api/v1/agri_fields", headers=headers, cookies=cookies)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch data")

        return response.json()

    except Exception as e:
        driver.quit()
        raise HTTPException(status_code=500, detail=str(e))
