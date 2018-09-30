#importing libraries and loading dataset

import numpy as np
import pandas as pd
#while displaying we want to see all the columns
pd.set_option('display.max_columns', 100)

DATA_PATH = r'./data/nycflights13/nycflights13_weather.csv'
HEADERS = ["origin", "year", "month", "day", "hour", "temp", "dewp", "humid", "wind_dir",
           "wind_speed", "wind_gust", "precip", "pressure", "visib", "time_hour"]
def get_dataframe():
    return pd.read_csv(DATA_PATH, sep=",", names=HEADERS, skiprows=43)

#converting Kelvins to Celcius
def tempConverter(temperature):
    temperature = (temperature - 32) * (5.0/9)
    return float(temperature)

def temperature_conversion(data):
    data["temp"] = data["temp"].apply(tempConverter)

#EX1 use
def pandas_excersise1():
    #loading the data
    weatherData = get_dataframe()
    print(weatherData.head())
    #displaying the loaded data
    temperature_conversion(weatherData)
    print(weatherData.head(40))
    #finding the daily mean temperature and interpolating the missing data
    weatherdataJFK = weatherData[weatherData["origin"] == "JFK"]
    weatherdataJFK["temp"].interpolate().mean()
    print(weatherdataJFK.head(40))
    warmDays = weatherdataJFK[(weatherdataJFK.shift(periods=1)["temp"] - weatherdataJFK["temp"]) > 0]
    best_days = weatherdataJFK.sort_values(by=["temp"], ascending=False).head(5)
    print(best_days)
    print(warmDays.head())
    print(weatherData.shape)
    print(weatherdataJFK.shape)
    print(warmDays.shape)
    print(best_days.shape)

pandas_excersise1()

