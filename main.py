import os
import time
import requests
import pandas as pd
import numpy as np
from kalshi_python import Kalshi
from telegram import Bot

# Auth
exchange = Kalshi()
exchange.login(key_id=os.getenv('KALSHI_KEY_ID'), 
               private_key=os.getenv('KALSHI_PRIVATE_KEY'))
bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
chat_id = int(os.getenv('CHAT_ID'))

def get_kmdw_cli():
  # NOAA METAR KMDW 1h data
  url = 'https://api.aviationweather.gov/metar?ids=K
