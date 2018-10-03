
#importing libraries and loading dataset
import pandas as pd
import matplotlib as plt

#while displaying we want to see all the columns
pd.set_option('display.max_columns', 100)


def get_dataframe(path, header, skipRows):
    return pd.read_csv(path, sep=",", names=header, skiprows=skipRows)


#converting Kelvin to Celcius
def tempConverter(temperature):
    temperature = (temperature - 32) * (5.0/9)
    return float(temperature)


def temperature_conversion(data):
    data["temp"] = data["temp"].apply(tempConverter)
    return data["temp"]


def interpolate_and_mean(data, column):
    return data[column].interpolate().mean()


#Exercise 1
def pandas_exercise1():
    #we specify the path, headers and number of lines to be ommitted in the file
    dataPath = r'./data/nycflights13/nycflights13_weather.csv'
    header = ["origin", "year", "month", "day", "hour", "temp", "dewp", "humid", "wind_dir",
               "wind_speed", "wind_gust", "precip", "pressure", "visib", "time_hour"]
    skipR = 43

    #loading the data
    weatherData = get_dataframe(dataPath, header, skipR)
    print(weatherData.head(5))

    #displaying the loaded data
    temperature_conversion(weatherData)
    print(weatherData.head(5))

    #finding the daily mean temperature and interpolating the missing data
    weatherdataJFK = weatherData[weatherData["origin"] == "JFK"]
    
    #weatherdataJFK["temp"].interpolate().mean()
    interpolate_and_mean(weatherdataJFK, "temp")
    print("mean temperature of JFK")
    print(weatherdataJFK.head(5))
    
    #plotting the mean temperatures



    #generating the days with greater mean temperature than the day before
    warmDays = weatherdataJFK[(weatherdataJFK.shift(periods=1)["temp"] - weatherdataJFK["temp"]) > 0]

    #generating the warmest days
    bestDays = weatherdataJFK.sort_values(by=["temp"], ascending = False).head(5)
    return warmDays, bestDays, weatherdataJFK.head(5)

def between_year_and_day(dataset):
    list_of_columns = dataset.columns.tolist()
    
    if list_of_columns.index("day") > list_of_columns.index("year"):
        dataset_chosen = dataset.loc[:, "year":"day"]
    else:
        dataset_chosen = dataset.loc[:, "day":"year"]

    return dataset_chosen


def outside_year_and_day(dataset):
    list_of_columns = dataset.columns.tolist()
    
    if list_of_columns.index("day") > list_of_columns.index("year"):
        dataset_left = dataset.loc[:,:"year"]
        dataset_right = dataset.loc[:,"day":]
        dataset_outside_y_and_d = pd.concat([dataset_left, dataset_right], axis=1)
        return dataset_outside_y_and_d
        
    else:
        dataset_left = dataset.loc[:,:"day"]
        dataset_right = dataset.loc[:,"year":]
        dataset_outside_y_and_d = pd.concat([dataset_left, dataset_right], axis=1)
        return dataset_outside_y_and_d


#Exercise 2
def pandas_exercise2():
    #we specify the path, headers and number of lines to be ommitted in the file
    dataPath = r'./data/nycflights13/nycflights13_flights.csv'
    header = ["year", "month", "day", "dep_time", "sched_dep_time", "dep_delay", "arr_time", "sched_arr_time",
              "arr_delay", "carrier", "flight", "tailnum", "origin", "dest", "air_time", "distance", "hour",
              "minute", "time_hour"]
    skipR = 55

    #we are loading the data into python
    flight_data = get_dataframe(dataPath, header, skipR)
    
    #We are selecting only columns between Year and Day columns
    flightDataInRange = between_year_and_day(flight_data)
    print(flightDataInRange)

    #We are selecting only columns that do not lay between Year and Day columns
    flightDataOutsideRange = outside_year_and_day(flight_data)
    print(flightDataOutsideRange)
    return flightDataInRange, flightDataOutsideRange


#Exercise 3
def pandas_exercise3():

    A = pd.read_csv(r'./data/some_birth_dates1.csv')
    B = pd.read_csv(r'./data/some_birth_dates2.csv')
    C = pd.read_csv(r'./data/some_birth_dates3.csv')

    #UNION OF A B
    AuB = pd.merge(A, B, on="Name", how="outer")
    print(AuB)

    #UNION OF A B C
    AuBuC = pd.merge(AuB, C, on="Name", how="outer")
    print(AuBuC)

    #INTERSECTION OF A B
    AnB = pd.merge(A, B, on="Name", how="inner")
    print(AnB)

    #INTERSECTION OF A C
    AnC = pd.merge(A, C, on="Name", how="inner")
    print(AnC)

    #A SUBSTRACTED B
    AsubB = A[~A["Name"].isin(B["Name"])]
    print(AsubB)
    
    return AnB, AuB, AuBuC, AnC, AsubB

