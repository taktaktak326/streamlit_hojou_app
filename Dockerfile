FROM python:3.10-slim

# Install Chrome and dependencies
RUN apt-get update && apt-get install -y \
    wget unzip gnupg curl \
    fonts-liberation libappindicator3-1 libasound2 libatk-bridge2.0-0 \
    libatk1.0-0 libcups2 libdbus-1-3 libgdk-pixbuf2.0-0 libnspr4 \
    libnss3 libx11-xcb1 libxcomposite1 libxdamage1 libxrandr2 \
    xdg-utils --no-install-recommends

# Chrome
RUN curl -sSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/google.gpg
RUN echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable agrinote" > /etc/apt/sources.list.d/google.list
RUN apt-get update && apt-get install -y google-chrome-stable

# ChromeDriver
RUN CHROME_VERSION=$(google-chrome --version | grep -oP '\\d+\\.\\d+\\.\\d+') && \\
    CHROMEDRIVER_VERSION=$(curl -s \"https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROME_VERSION\") && \\
    wget -O /tmp/chromedriver.zip \"https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip\" && \\
    unzip /tmp/chromedriver.zip -d /usr/bin && \\
    chmod +x /usr/bin/chromedriver

# Set display env
ENV DISPLAY=:99

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY agrinote.py .

CMD [\"uvicorn\", \"agrinote:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\"]
