#importing libraries and loading dataset
import pandas as pd
import matplotlib as plt
#while displaying we want to see all the columns
pd.set_option('display.max_columns', 100)

def get_dataframe(path,header,skipRows):
    return pd.read_csv(path, sep=",", names=header, skiprows=skipRows)


#converting Kelvins to Celcius
def tempConverter(temperature):
    temperature = (temperature - 32) * (5.0/9)
    return float(temperature)


def temperature_conversion(data):
    data["temp"] = data["temp"].apply(tempConverter)


def interpolate_and_mean(data, column):
    data[column].interpolate().mean()

#Excersise 1
def pandas_excersise1():
    #we specify the path, headers and number of lines to be ommited in the file
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
    print(weatherdataJFK.head(5))

    #ploting the mean temperatures



    #getting the days with greater mean temperature than the day before
    warmDays = weatherdataJFK[(weatherdataJFK.shift(periods=1)["temp"] - weatherdataJFK["temp"]) > 0]

    #getting the warmest days
    bestDays = weatherdataJFK.sort_values(by=["temp"], ascending=False).head(5)

def pandas_excersise2():
    #we specify the path, headers and number of lines to be ommited in the file
    dataPath = r'C:\Users\Michal\Documents\GitHub\StatisticsProjectGroupH\Project1\data\nycflights13_flights.csv'
    header = ["year","month","day","dep_time","sched_dep_time","dep_delay","arr_time","sched_arr_time",
              "arr_delay","carrier","flight","tailnum","origin","dest","air_time","distance","hour",
              "minute","time_hour"]
    skipR = 54

    #we are loading the data into python
    flight_data = get_dataframe(dataPath, header, skipR)
    
    #We are selecting only columns between Year and Day columns
    list_of_columns = flight_data.columns.tolist()
    
    if list_of_columns.index("day") > list_of_columns.index("year"):
        flight_data_chosen = flight_data.loc[:, "year":"day"]
    else:
        flight_data_chosen = flight_data.loc[:, "day":"year"]
        
    #We are selecting only columns that do not lay between Year and Day columns
    if list_of_columns.index("day") > list_of_columns.index("year"):
        flight_data_left = flight_data.loc[:,:"year"]
        flight_data_right = flight_data.loc[:,"day":]
        flight_data_between_y_and_d = pd.concat([flight_data_left, flight_data_right], axis=1)
        
        print(flight_data_between_y_and_d)
        
    else:
        flight_data_left = flight_data.loc[:,:"day"]
        flight_data_right = flight_data.loc[:,"year":]
        flight_data_outside_y_and_d = pd.concat([flight_data_left, flight_data_right], axis=1)
        
        print(flight_data_outside_y_and_d)
    
def pandas_excersise3():

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

