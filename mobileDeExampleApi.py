################ Data preparation ################

import pandas as pd

# import seaborn as sns
# import matplotlib.pyplot as plt

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
    if (X <= 2009):
        return 1
    elif (X > 2009 and X <= 2011):
        return 2
    elif (X > 2011 and X <= 2013):
        return 3
    elif (X > 2013 and X <= 2015):
        return 4
    elif (X > 2015 and X <= 2017):
        return 5
    elif (X > 2017):
        return 6


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
    if (X == 'Benzina'):
        return 1
    elif (X == 'Diesel'):
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
    price = int(Y.split()[0].replace('.', ''))

    if (price <= 9000):
        return 1
    elif (price > 9000 and price <= 14000):
        return 2
    elif (price > 14000 and price <= 20000):
        return 3
    elif (price > 20000 and price <= 27000):
        return 4
    elif (price > 27000 and price <= 36000):
        return 5
    elif (price > 36000):
        return 6


sf['ModelCategory'] = sf['Model'].apply(createModelColum)
sf['YearCategory'] = sf['Year'].apply(createYearColumn)
sf['HorsePowerCategory'] = sf['HorsePower'].apply(createHorsePowerColumn)
sf['FuelCategory'] = sf['Fuel'].apply(createFuelColumn)
sf['Km'] = sf['Mileage'].apply(createMilageColumn)
sf['TransmissionCategory'] = sf['Transmission'].apply(createTransmisionColumn)
sf['PriceCategory'] = sf['Price'].apply(createPriceColumn)

sf = sf.drop(['Model', 'Year', 'HorsePower', 'Fuel', 'Mileage', 'Transmission', 'Price'], axis=1)

sf = sf.rename(index=str, columns={'ModelCategory': 'Model', "YearCategory": "Year",
                                   "HorsePowerCategory": "HorsePower", "FuelCategory": "Fuel", 'Km': 'Mileage',
                                   'TransmissionCategory': 'Transmission', 'PriceCategory': 'Price'})

# print(sf.head())

X = sf[['Year', 'HorsePower', 'Fuel', 'Transmission', 'Mileage']]
Y = sf['Price']

################# Training Classifier ################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

import time

X_train, X_test = train_test_split(sf, test_size=0.2, random_state=int(time.time()))
used_features = ['Year', 'HorsePower', 'Fuel', 'Transmission', 'Mileage']

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
        parser.add_argument('horsePower')
        parser.add_argument('fuel')
        parser.add_argument('mileage')
        parser.add_argument('transmission')

        args = parser.parse_args()
        print(args)

        result_proba = gnb.predict_proba([[
            createModelColum(args['model']),
            createYearColumn(int(args['year'])),
            createHorsePowerColumn(args['horsePower']),
            createFuelColumn(args['fuel']),
            createMilageColumn(args['mileage']),
            createTransmisionColumn(args['transmission'])
        ]]
        )

        print("------------------------Classes probabilities------------------------")
        print(result_proba)

        result = gnb.predict(
            [[
                createModelColum(args['model']),
                createYearColumn(int(args['year'])),
                createHorsePowerColumn(args['horsePower']),
                createFuelColumn(args['fuel']),
                createMilageColumn(args['mileage']),
                createTransmisionColumn(args['transmission'])
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
