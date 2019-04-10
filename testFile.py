import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sbn

import time

# sns.set(color_codes=True)
#sf = pd.read_csv('data/toateSeriileCurate.csv')
#sf = pd.read_csv('data/toateSeriileCurate.csv')
#sf = pd.read_csv('data/toateSeriileCurate.csv')
sf = pd.read_csv('data/seria7Curata.csv')


def createModelColum(X):
    model = X.split()
    if (model[1].startswith('3')):
        return 1
    elif (model[1].startswith('5')):
        return 2
    elif (model[1].startswith('7')):
        return 3


def createYearColumn(X):

    if (X <= 2010):
        return 1
    elif (X > 2010 and X <= 2013):
        return 2
    elif (X > 2013 and X <= 2015):
        return 3
    elif (X > 2015 and X <= 2017):
        return 4
    elif (X > 2017):
        return 5


def createHorsePowerColumn(X):
    horsePowerString = X.split('(')
    horsePower = int(horsePowerString[1].replace(')', '').replace(' ', '').replace('CP', ''))

    if (horsePower < 180):
        return 1
    elif (horsePower >= 180 and horsePower < 200):
        return 2
    elif (horsePower >= 200 and horsePower < 250):
        return 3
    elif (horsePower >= 250 and horsePower < 300):
        return 4
    elif (horsePower >= 300 and horsePower < 350):
        return 5
    else:
        return 6


def createFuelColumn(X):
    if (X == 'Diesel'):
        return 1
    elif (X == 'Benzina'):
        return 2


def createTransmisionColumn(X):
    if (X == 'automata'):
        return 1
    elif (X == 'manuala'):
        return 2


def createMilageColumn(X):
    milage = int(X.replace('.', '').replace('km', ''))

    if (milage <= 10000):
        return 1
    elif (milage > 10000 and milage <= 30000):
        return 2
    elif (milage > 30000 and milage <= 50000):
        return 3
    elif (milage > 50000 and milage <= 70000):
        return 4
    elif (milage > 70000 and milage <= 100000):
        return 5
    elif (milage > 100000 and milage <= 125000):
        return 6
    elif (milage > 125000 and milage <= 150000):
        return 7
    elif (milage > 150000 and milage <= 200000):
        return 8
    elif (milage > 200000):
        return 9


def createPriceColumn(Y):
    return int(Y.split()[0].replace('.', ''))
    # if (price <= 10000):
    #     return 1
    # elif (price > 10000 and price <= 15000):
    #     return 2
    # elif (price > 15000 and price <= 20000):
    #     return 3
    # elif (price > 20000 and price <= 30000):
    #     return 4
    # elif (price > 30000 and price <= 40000):
    #     return 5
    # elif (price > 40000):
    #     return 6


print(sf.columns)

sf['ModelCategory'] = sf.Model.apply(createModelColum)
sf['YearCategory'] = sf.Year.apply(createYearColumn)
sf['HorsePowerCategory'] = sf.HorsePower.apply(createHorsePowerColumn)
sf['FuelCategory'] = sf.Fuel.apply(createFuelColumn)
sf['Km'] = sf.Mileage.apply(createMilageColumn)
sf['TransmissionCategory'] = sf.Transmission.apply(createTransmisionColumn)
sf['PriceCategory'] = sf.Price.apply(createPriceColumn)

sf = sf.drop(['Model', 'Year', 'HorsePower', 'Fuel', 'Mileage', 'Transmission', 'Price'], axis=1)

sf = sf.rename(index=str, columns={'ModelCategory': 'Model', "YearCategory": "Year",
                                   "HorsePowerCategory": "HorsePower", "FuelCategory": "Fuel", 'Km': 'Mileage',
                                   'TransmissionCategory': 'Transmission', 'PriceCategory': 'Price'})

print(sf.describe(include=[np.number]))
#
# plt.figure(figsize=(15, 10))
# plt.tight_layout()
# sbn.distplot(sf['Price'])

# print(sf.head())

X = sf[['Year', 'HorsePower', 'Fuel', 'Transmission', 'Mileage']]
Y = sf['Price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=int(time.time()))

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("------------------------Regression coefficient------------------------")
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

y_pred = regressor.predict(X_test)

print("------------------------Actual vs Predicted values--------------------------")
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.astype(int)})
df1 = df.head(25)
print(df1)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#
# sf.plot(x='Mileage', y='Price', style='o')
# plt.title('Dependentele vs Pretul')
# plt.xlabel('Dependente')
# plt.ylabel('Pretul')
# plt.show()

# df_X = pd.DataFrame(X)
# df_Y = pd.DataFrame(Y)
#
# sf.describe()
#
# reg = linear_model.LinearRegression()
# x_train, x_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=4)
# reg.fit(x_train, y_train)
#
# print ("------------------- DATELE Prezise DE TEST ------------------------")
# a = reg.predict(x_test)
# print(a.astype(int))
#
# print("------------------- DATELE ASTEPTATE DE TEST ------------------------")
# print(y_test)
# sns.distplot(sf.Price, kde=True)
# plt.ylim(0, 1)
# plt.show()


# print(sf.Price)
# print(sf.Model)
# print(sf.Year)
# print(sf.Km)
# print(sf.HorsePower)
# print(sf.Transmission)
