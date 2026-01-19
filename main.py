import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
import asyncio
import uvicorn

from fastapi import FastAPI
from kalshi_python import Configuration, KalshiClient
from telegram import Bot
from telegram.error import TelegramError

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Elite imports (requirements.txt vajadzīgs)
try:
    from xgboost import XGBRegressor
    import cachetools
    from ecmwf.opendata import Client as OpendataClient
    ELITE_MODE = True
except ImportError:
    ELITE_MODE = False
    logging.warning("Elite deps missing - RF only")

# Logging format fix
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

client = None
bot_obj = None
chat_id = os.getenv("CHAT_ID")

# per stacija: dict[icao: model]
model_rf = {}
model_xgb = {}

# Multi-stacijas Kalshi weather (ICAO, city, lat, lon, Kalshi suffix)
STATIONS = {
    "KMDW": {"icao": "KMDW", "city": "chi", "lat": 41.785, "lon": -87.752, "suffix": "chi"},
    "KJFK": {"icao": "KJFK", "city": "nyc", "lat": 40.6413, "lon": -73.7781, "suffix": "nyc"},
    "KNYC": {"icao": "KNYC", "city": "nyc", "lat": 40.7812, "lon": -73.9665, "suffix": "nyc"},  # NYC Central Park CLI
    "KMIA": {"icao": "KMIA", "city": "mia", "lat": 25.7932, "lon": -80.2906, "suffix": "mia"},
    "KAUS": {"icao": "KAUS", "city": "aus", "lat": 30.1945, "lon": -97.6699, "suffix": "aus"},
    "KDEN": {"icao": "KDEN", "city": "den", "lat": 39.8617, "lon": -104.6732, "suffix": "den"},
    "KHOU": {"icao": "KHOU", "city": "hou", "lat": 29.7846, "lon": -95.3414, "suffix": "hou"},
    "KPHL": {"icao": "KPHL", "city": "phl", "lat": 39.8719, "lon": -75.2411, "suffix": "phl"},
    "KLAX": {"icao": "KLAX", "city": "lax", "lat": 33.9425, "lon": -118.4081, "suffix": "lax"},
    "KBOS": {"icao": "KBOS", "city": "bos", "lat": 42.3643, "lon": -71.0062, "suffix": "bos"},
}


@app.get("/")
async def root():
    return {
        "status": "Elite Multi-Station Kalshi Bot live (9 stations)",
        "elite": ELITE_MODE,
        "kalshi_ready": client is not None,
        "stations": len(STATIONS),
    }


def init_kalshi():
    global client, bot_obj

    kalshi_key_id = os.getenv("KALSHI_KEY_ID")
    kalshi_priv_key = os.getenv("KALSHI_PRIVATE_KEY")
    telegram_token = os.getenv("TELEGRAM_TOKEN")

    if telegram_token:
        bot_obj = Bot(token=telegram_token)
    else:
        bot_obj = None
        logger.warning("TELEGRAM_TOKEN missing - Telegram disabled")

    if kalshi_key_id and kalshi_priv_key:
        config = Configuration()
        config.api_key_id = kalshi_key_id
        config.private_key_pem = kalshi_priv_key
        client = KalshiClient(config)
        try:
            balance_resp = client.get_balance()
            logger.info(f"Elite Kalshi init OK. Balance: ${balance_resp.balance / 100:.2f}")
        except Exception as e:
            logger.error(f"Kalshi balance check failed: {e}")
    else:
        logger.error("Kalshi credentials missing - client not initialized")
        client = None


# Telegram paziņojums par restartu
async def _notify_restart():
    if bot_obj and chat_id:
        try:
            await bot_obj.send_message(
                chat_id=chat_id,
                text="Elite Multi-Station Bot restarted - 9 stations + GraphCast ready!",
            )
        except TelegramError as e:
            logger.error(f"Telegram restart message error: {e}")


try:
    loop = asyncio.get_running_loop()
    loop.create_task(_notify_restart())
except RuntimeError:
    # Nav event loop – ok, kad FastAPI/uvicorn pats startēs
    pass


async def fetch_metar(icao: str):
    """Reāls METAR fetch no AviationWeather.gov API"""
    url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=decoded"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("METAR", {}).get("data", []):
            metar_data = data["METAR"]["data"][0]
            # Parse decoded fields
            temp_f = float(metar_data.get("temp", 45.0))
            wind_kt = float(metar_data.get("wind_kt", 10.0))
            return {"tmpf": temp_f, "sknt": wind_kt}
        else:
            logger.warning(f"No METAR data for {icao}")
            return {"tmpf": 45.0, "sknt": 10.0}
    except Exception as e:
        logger.error(f"METAR fetch error {icao}: {e}")
        return {"tmpf": 45.0, "sknt": 10.0}


def get_cli(icao: str):
    """CLI no Iowa State ASOS per stacija (24h max T kā proxy)"""
    networks = {
        "KMDW": ("IL_ASOS", "MDW"),
        "KJFK": ("NY_ASOS", "JFK"),
        "KNYC": ("NY_ASOS", "NYC"),
        "KMIA": ("FL_ASOS", "MIA"),
        "KAUS": ("TX_ASOS", "AUS"),
        "KDEN": ("CO_ASOS", "DEN"),
        "KHOU": ("TX_ASOS", "HOU"),
        "KPHL": ("PA_ASOS", "PHL"),
        "KLAX": ("CA_ASOS", "LAX"),
        "KBOS": ("MA_ASOS", "BOS"),
    }

    net, st = networks.get(icao, ("US_ASOS", icao[:3]))

    # start/end šobrīd placeholder – pielāgo, ja vajag precīzu datumu
    url = (
        "https://mesonet.agron.iastate.edu/request/download.phtml?"
        f"network={net}&station={st}&data=all&start=20260101&end=today&format=csv"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), skiprows=1)
        df["tmpf"] = pd.to_numeric(df.get("tmpf"), errors="coerce")
        cli = df["tmpf"].tail(24).max()
        return float(cli) if pd.notna(cli) else None
    except Exception as e:
        logger.error(f"CLI fetch error {icao}: {e}")
        return None


def fetch_asos_data(icao: str) -> pd.DataFrame:
    """ASOS data per stacija no Iowa State"""
    url = f"https://mesonet.agron.iastate.edu/request/asos/1min.phtml?station={icao}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        return df.dropna()
    except Exception as e:
        logger.error(f"ASOS fetch error {icao}: {e}")
        return pd.DataFrame()


def train_models():
    """Train per stacija RF (+XGB ja pieejams)"""
    global model_rf, model_xgb

    for icao, info in STATIONS.items():
        df = fetch_asos_data(icao)
        if len(df) <= 100:
            logger.warning(f"Not enough data for {icao}")
            continue

        for col in ["tmpf", "sknt"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")

        df = df.dropna(subset=["tmpf", "sknt"])
        # CLI proxy: rolling 24h max (shift -1 lai prognozētu nākamo 24h)
        df["CLI_proxy"] = df["tmpf"].rolling(24).max().shift(-1)
        df = df.dropna(subset=["CLI_proxy"])

        if df.empty:
            continue

        X = df[["tmpf", "sknt"]]
        y = df["CLI_proxy"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )

        # RF
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        acc_rf = rf.score(X_test, y_test)
        model_rf[icao] = rf
        joblib.dump(model_rf[icao], f"model_rf_{icao}.pkl")

        if ELITE_MODE:
            try:
                xgb = XGBRegressor(
                    n_estimators=200,
                    random_state=42,
                )
                xgb.fit(X_train, y_train)
                acc_xgb = xgb.score(X_test, y_test)
                model_xgb[icao] = xgb
                joblib.dump(model_xgb[icao], f"model_xgb_{icao}.pkl")
                logger.info(f"{icao}: RF {acc_rf:.2f}, XGB {acc_xgb:.2f}")
            except Exception as e:
                logger.error(f"XGB {icao} error: {e}")
        else:
            logger.info(f"{icao}: RF {acc_rf:.2f}")

    logger.info(f"Trained models for {len(model_rf)} stations")


def predict_rf(metar: dict, icao: str) -> float:
    global model_rf

    if icao not in model_rf:
        try:
            model_rf[icao] = joblib.load(f"model_rf_{icao}.pkl")
        except Exception:
            model_rf[icao] = None

    if not model_rf.get(icao):
        return 45.0

    try:
        tmpf = float(metar["tmpf"])
        sknt = float(metar["sknt"])
        pred = model_rf[icao].predict([[tmpf, sknt]])[0]
        return float(pred)
    except Exception as e:
        logger.error(f"RF predict error {icao}: {e}")
        return 45.0


if ELITE_MODE:

    @cachetools.ttl_cache(maxsize=10, ttl=1800)
    def fetch_graphcast(lat: float, lon: float) -> float:
        try:
            od_client = OpendataClient(source="ecmwf")
            # 2m T, 24h step, šeit minimāls piemērs – vajadzības gadījumā pielāgot
            od_client.retrieve(param="2t", step=[24], target="graphcast.grib2")

            import cfgrib  # dynamic import

            ds = cfgrib.open_dataset("graphcast.grib2")
            max_temp = float(
                ds.t2m.sel(latitude=lat, longitude=lon, method="nearest").max()
            )
            # Konvertē no K uz F, ja ECMWF ir Kelvin
            max_temp_f = (max_temp - 273.15) * 9 / 5 + 32
            logger.info(f"GraphCast max T {lat},{lon}: {max_temp_f:.1f}°F")
            return max_temp_f
        except Exception as e:
            logger.error(f"GraphCast parse error: {e}")
            return 52.3  # fallback
else:

    def fetch_graphcast(lat: float, lon: float) -> float:
        # Simple fallback, ja ELITE_MODE nav
        return 50.0


def nws_alerts(lat: float, lon: float) -> float:
    """Heat alert bonuss, ja NWS active alerts satur heat headline"""
    url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"
    try:
        resp = requests.get(url, timeout=10).json()
        features = resp.get("features", [])
        if any(
            "heat"
            in a.get("properties", {})
            .get("headline", "")
            .lower()
            for a in features
        ):
            return 0.1
        return 0.0
    except Exception:
        return 0.0


def elite_ensemble(metar: dict, info: dict) -> float:
    icao, lat, lon = info["icao"], info["lat"], info["lon"]

    rf_temp = predict_rf(metar, icao)
    graph_temp = fetch_graphcast(lat, lon)
    alert_boost = nws_alerts(lat, lon)

    # Normalize to p
    rf_p = rf_temp / 100.0
    graph_p = min(1.0, graph_temp / 55.0)

    if ELITE_MODE and icao in model_xgb and model_xgb[icao]:
        try:
            tmpf = float(metar["tmpf"])
            sknt = float(metar["sknt"])
            xgb_temp = model_xgb[icao].predict([[tmpf, sknt]])[0]
            xgb_p = max(0.0, min(1.0, xgb_temp / 100.0))
        except Exception:
            xgb_p = 0.78
    else:
        xgb_p = 0.78

    p_final = 0.2 * rf_p + 0.4 * xgb_p + 0.4 * graph_p + alert_boost
    return min(1.0, max(0.0, p_final))


def predict_outcome(ticker: str, metar: dict, info: dict) -> float:
    ticker_l = ticker.lower()
    city = info["city"]

    if city in ticker_l:
        if ELITE_MODE:
            return elite_ensemble(metar, info)
        temp_pred = predict_rf(metar, info["icao"])
        return max(0.0, min(1.0, temp_pred / 100.0))

    # ja nav skaidras match ar city suffix, default
    return 0.6


def kelly_size(p: float, b: float) -> float:
    """Simple Kelly, atgriež pozīcijas lielumu (konvertēts uz 'kontrakta skaitu' % formā)"""
    try:
        if b <= 1:
            return 0.0
        f = (p * b - 1) / (b - 1)
        # limitē 0–10 kontrakti, lai neriskētu pārāk daudz
        return max(0.0, min(10.0, f * 100.0))
    except Exception:
        return 0.0


async def main_loop():
    init_kalshi()
    train_models()

    if client is None:
        logger.error("Kalshi client not initialized - aborting")
        return

    station_list = list(STATIONS.items())

    while True:
        try:
            # Parallel METAR fetch visām stacijām
            tasks = [fetch_metar(info["icao"]) for _, info in station_list]
            metars_list = await asyncio.gather(*tasks, return_exceptions=True)
            metars = {}
            for (icao, _), res in zip(station_list, metars_list):
                metars[icao] = res

            balance_resp = client.get_balance()

            # Portfolio management (visas pozīcijas)
            try:
                portfolio = client.get_positions()
            except Exception as e:
                logger.error(f"Get positions error: {e}")
                portfolio = None

            if portfolio is not None:
                for pos in portfolio.positions:
                    current_bid = pos.yes_bid
                    entry_price = pos.avg_price
                    if current_bid is None or entry_price is None:
                        continue
                    try:
                        # Take profit
                        if current_bid > entry_price * 1.10:
                            client.sell_order(
                                pos.ticker,
                                side="yes",
                                count=pos.count,
                                type="limit",
                                price=current_bid,
                            )
                            logger.info(f"Take profit {pos.ticker}")
                        # Stop loss
                        elif current_bid < entry_price * 0.80:
                            client.sell_order(
                                pos.ticker,
                                side="yes",
                                count=pos.count,
                                type="market",
                            )
                            logger.info(f"Stop loss {pos.ticker}")
                    except Exception as e:
                        logger.error(f"Sell error {pos.ticker}: {e}")

            # Market scan + per-stacija trades
            try:
                markets_resp = client.list_markets(
                    {"category": "climate", "status": "open"}
                )
            except Exception as e:
                logger.error(f"List markets error: {e}")
                markets_resp = None

            if markets_resp is not None:
                for market in markets_resp.markets[:20]:  # vairāk tirgu
                    ticker = market.ticker
                    ticker_l = ticker.lower()

                    if not any(
                        word in ticker_l
                        for word in ["temperature", "rain", "hurricane", "high"]
                    ):
                        continue

                    yes_bid = market.yes_bid
                    if yes_bid is None:
                        continue

                    # Match tirgu ar staciju
                    for icao, info in STATIONS.items():
                        if info["suffix"] in ticker_l:
                            metar = metars.get(icao)
                            if metar is None or isinstance(metar, Exception):
                                continue

                            pred_prob = predict_outcome(ticker, metar, info)
                            ev = pred_prob - yes_bid / 100.0

                            if ev > 0.05:
                                price = yes_bid / 100.0
                                b = price / max(0.01, 1 - price)
                                size = kelly_size(pred_prob, b)

                                if size > 0 and balance_resp.balance > 0:
                                    try:
                                        order = client.buy_order(
                                            ticker,
                                            side="yes",
                                            count=int(size),
                                            type="market",
                                        )
                                        trade_msg = (
                                            f"Elite BUY {ticker} {int(size)} "
                                            f"EV:{ev:.1%} ({icao})"
                                        )
                                        if bot_obj and chat_id:
                                            await bot_obj.send_message(
                                                chat_id=chat_id, text=trade_msg
                                            )
                                        logger.info(trade_msg)
                                    except Exception as e:
                                        logger.error(f"Buy error {ticker}: {e}")

            await asyncio.sleep(300)  # 5min cycle

        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
