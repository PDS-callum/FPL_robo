import requests
import os
import pandas as pd
import json

def get_data(url):
    response = requests.get(url)
    data = response.json()
    return data