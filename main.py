import os, time, pandas as pd
from kalshi_python import Configuration, KalshiClient
from telegram import Bot
from fastapi import FastAPI
import uvicorn, logging
logging.basicConfig(level=logging.INFO)

# Env
kalshi_key_id = os.getenv('KALSHI_KEY_ID')
kalshi_priv_key = os.getenv('KALSHI_PRIVATE_KEY')
telegram_token = os.getenv('TELEGRAM_TOKEN')
chat_id = os.getenv('CHAT_ID')

app = FastAPI()
client = None
bot = Bot(token=telegram_token) if telegram_token else None

@app.get("/")
async def root():
    return {"status": "KMDW Kalshi Bot live", "kalshi_ready": client is not None}

def init_kalshi():
    global client
    config = Configuration()
    config.api_key_id = kalshi_key_id
    config.private_key_pem = kalshi_priv_key
    client = KalshiClient(config)
    logging.info("Kalshi init OK")

def get_kmdw_cli():
    try:
        # Lokālais CSV vai NOAA URL
        url = 'https://.../ptype_meteonetwork_IL_ASOS.csv'  # Jūsu CSV
        df = pd.read_csv(url)
        cli = df['CLI'].max()  # Jūsu metrika
        return cli
    except:
        return None

async def main_loop():
    init_kalshi()
    while True:
        cli = get_kmdw_cli()
        if client and cli:
            balance = client.get_balance()
            msg = f"KMDW CLI: {cli}°F, Balance: {balance['balance']}"
            await bot.send_message(chat_id=chat_id, text=msg)
        time.sleep(300)  # 5min
if __name__ == "__main__":
    import asyncio
    port = int(os.getenv("PORT", 10000))
    # Start server fonā
    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
    # Bot loop paralēli (vai threading)
import requests
import pandas as pd
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
model = None
def fetch_kmdw_data():
    url = "https://mesonet.agron.iastate.edu/request/asos/1min.phtml?station=KMDW"
    df = pd.read_csv(url)
    return df
def train_model():
    global model
    df = fetch_kmdw_data()
    df['CLI_proxy'] = df['tmpf'].rolling(24).max().shift(-1)
    df = df.dropna()
    X = df[['tmpf', 'sknt']]
    y = df['CLI_proxy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    logging.info(f"Model accuracy: {model.score(X_test, y_test):.2f}")
def predict_cli(metar):
    global model
    if model is None:
        model = joblib.load('model.pkl')
    pred = model.predict([[metar['tmpf'], metar['sknt']]])[0]
    return pred
metar = {'tmpf': 25, 'sknt': 10}  # No METAR API
cli_pred = predict_cli(metar)
msg = f"CLI Prognoze: {cli_pred:.1f}°F"
