import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

#sns.set(color_codes=True)
sf = pd.read_csv('data/cars_good_fara_x.csv')


def createModelColum(X):
    model = X.split()
    if (model[1] == 'Seria'):
        return int(model[2])
    # elif model[1].startswith('X'):
    # return int(model[1].split('X')[1]) + 8


def createYearColumn(X):
    if (X <= 2005):
        return 1
    elif (X >= 2006 and X <= 2008):
        return 2
    elif (X >= 2009 and X <= 2011):
        return 3
    elif (X >= 2012 and X <= 2014):
        return 4
    elif (X >= 2015 and X <= 2017):
        return 5
    elif (X > 2017):
        return 6


def createEngineColumn(X):
    engineCapacityString = X.split()
    engineCapacity = int(engineCapacityString[0] + engineCapacityString[1])
    if (engineCapacity < 1600):
        return 1
    elif (engineCapacity >= 1601 and engineCapacity <= 1900):
        return 2
    elif (engineCapacity >= 1901 and engineCapacity <= 2200):
        return 3
    elif (engineCapacity >= 2201 and engineCapacity <= 2500):
        return 4
    elif (engineCapacity >= 2501 and engineCapacity <= 3000):
        return 5
    else:
        return 6


def createFuelColumn(X):
    if (X == 'Benzina'):
        return 1
    elif (X == 'Diesel'):
        return 2


def createMilageColumn(X):
    milage = int(X.replace(' ', '').replace('km', ''))
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
    price = int(Y.replace(' ', '').replace('EUR', ''))

    if (price <= 10000):
        return 1
    elif (price > 10000 and price <= 15000):
        return 2
    elif (price > 15000 and price <= 20000):
        return 3
    elif (price > 20000 and price <= 30000):
        return 4
    elif (price > 30000 and price <= 40000):
        return 5
    elif (price > 40000):
        return 6


# print(int('X6'.split('X')[1]) + 7 )
sf['Model'] = sf.CarModel.apply(createModelColum)
sf['Year'] = sf.Year.apply(createYearColumn)
sf['EngineCapacity'] = sf.Engine.apply(createEngineColumn)
sf['Fuel'] = sf.Fuel.apply(createFuelColumn)
sf['km'] = sf.Milage.apply(createMilageColumn)
sf['Price'] = sf.Price.apply(createPriceColumn)

#sns.distplot(sf.Price, kde=True)
#plt.ylim(0, 1)
#plt.show()

print(sf.Price)
