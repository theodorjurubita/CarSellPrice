################ Data preparation ################

import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

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


sf['Model'] = sf.CarModel.apply(createModelColum)
sf['YearCategory'] = sf.Year.apply(createYearColumn)
sf['EngineCapacity'] = sf.Engine.apply(createEngineColumn)
sf['FuelCategory'] = sf.Fuel.apply(createFuelColumn)
sf['Km'] = sf.Milage.apply(createMilageColumn)
sf['PriceCategory'] = sf.Price.apply(createPriceColumn)

sf = sf.drop(['CarModel', 'Year', 'Engine', 'Fuel', 'Milage', 'Price'], axis=1)

sf = sf.rename(index=str, columns={"YearCategory": "Year", "EngineCapacity": "Engine",
                                   "FuelCategory": "Fuel", 'Km': 'Mileage', 'PriceCategory': 'Price'})

# print(sf.head())

X = sf[['Model', 'Year', 'Engine', 'Fuel', 'Mileage']]
Y = sf['Price']

################# Training Classifier ################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

import time

X_train, X_test = train_test_split(sf, test_size=0.1, random_state=int(time.time()))
used_features = ['Model', 'Year', 'Engine', 'Fuel', 'Mileage']

# gnb = GaussianNB()
gnb = GradientBoostingClassifier()
# gnb = GradientBoostingRegressor()
# gnb = LinearRegression()

gnb.fit(X_train[used_features].values, X_train['Price'])

y_pred = gnb.predict(X_test[used_features])
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
    .format(
    X_test.shape[0],
    (X_test["Price"] != y_pred).sum(),
    100 * (1 - (X_test["Price"] != y_pred).sum() / X_test.shape[0])
))

################# API ################
from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)


class PricePrediction(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('model')
        parser.add_argument('year')
        parser.add_argument('engine')
        parser.add_argument('fuel')
        parser.add_argument('mileage')

        args = parser.parse_args()
        print(args)

        result_proba = gnb.predict_proba([[
            createModelColum(args['model']),
            createYearColumn(int(args['year'])),
            createEngineColumn(args['engine']),
            createFuelColumn(args['fuel']),
            createMilageColumn(args['mileage'])
        ]]
        )

        print("------------------------Classes probabilities------------------------")
        print(result_proba)

        result = gnb.predict(
            [[
                createModelColum(args['model']),
                createYearColumn(int(args['year'])),
                createEngineColumn(args['engine']),
                createFuelColumn(args['fuel']),
                createMilageColumn(args['mileage'])
            ]]
        )

        print("-----------------------RESULT-------------------------")
        print(result)

        responseBody = {
            "score": "pulh"
        }

        return responseBody, 200


api.add_resource(PricePrediction, "/price")

app.run(debug=True, port=5000)
