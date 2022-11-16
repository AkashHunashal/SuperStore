from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer, OneHotEncoder
import xgboost
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from scipy.special import inv_boxcox
from scipy.stats import boxcox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def readdata():
    data = pd.read_excel('Sample - Superstore.xls')
    data['delivery_days'] = (data['Ship Date']-data['Order Date']).dt.days
    data.drop(['Row ID', 'Order ID', 'Order Date', 'Product ID',
              'Ship Date', 'Customer ID', 'Customer Name'], axis=1, inplace=True)
    data.drop('Country', axis=1, inplace=True)
    global categorical_columns
    global numerical_columns
    categorical_columns = data.select_dtypes(include='object').columns
    numerical_columns = data.select_dtypes(exclude='object').columns
    return data


ohe = OneHotEncoder(sparse=False, drop='first')


def preprocessing(data):
    onehotencoded = ohe.fit_transform(data[categorical_columns.drop(
        ['City', 'State', 'Sub-Category', 'Product Name'])])
    columns = ohe.get_feature_names_out()
    data[columns] = onehotencoded
    data.drop(categorical_columns.drop(
        ['City', 'State', 'Sub-Category', 'Product Name']), axis=1, inplace=True)
    global encoded
    encoded = []
    for column in ['City', 'State', 'Sub-Category', 'Product Name', 'Postal Code']:
        col_encoded = data.groupby(column)['Sales'].agg(
            'median').sort_values(ascending=True).to_dict()
        data[column] = data[column].map(col_encoded)
        cat_encoded = {column: col_encoded}
        encoded.append(cat_encoded)
    return data


def modeltraining(data):
    X = data.drop(['Sales', 'Profit'], axis=1)
    y = data[['Sales']]
    standard = StandardScaler()
    robust = RobustScaler()
    minmax = MinMaxScaler()
    normalize = Normalizer()
    for columns in ['City', 'Postal Code']:
        X[columns] = np.log1p(X[columns])
    for columns in ['Sub-Category', 'Product Name', 'Discount']:
        X[columns] = boxcox(X[columns]+.00001)[0]
    X['State'] = 1/X['State']
    upper_boundary = y.mean()+6*y.std()
    y['Sales'] = np.where(y > upper_boundary, upper_boundary, y)
    return X, y


Selector = SelectKBest(score_func=f_regression, k=14)


def featureselection(X, y):
    Selector.fit(X, y)
    X_new = Selector.fit_transform(X, y)
    X_new = pd.DataFrame(Selector.fit_transform(
        X, y), columns=X.columns[Selector.get_support()])
    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def modelprediction(X_test):
    xgboost_model = pickle.load(open('xgboost.pkl', 'rb'))
    xgboost_pred = xgboost_model.predict(X_test)
    return xgboost_pred


def model():
    df = readdata()
    df = preprocessing(df)
    X, y = modeltraining(df)
    X_train, X_test, y_train, y_test = featureselection(X, y)
    return X_train, X_test, y_train, y_test


def predict():
    X_train, X_test, y_train, y_test = model()
    predictions = modelprediction(X_test)
    return predictions


Y_pred = predict()


def testmodel(data):
    onehotencoded = ohe.transform(
        data[['Ship Mode', 'Segment', 'Region', 'Category']])
    columns = ohe.get_feature_names_out()
    data[columns] = onehotencoded
    data.drop(['Ship Mode', 'Segment', 'Region',
              'Category'], axis=1, inplace=True)
    for category in encoded:
        for column, values in category.items():
            data[column] = data[column].map(values)
    X = data.copy()
    for columns in ['City', 'Postal Code']:
        X[columns] = np.log1p(X[columns])
    X['State'] = 1/X['State']
    df = readdata()
    X2 = preprocessing(df)
    for columns in ['Sub-Category', 'Product Name', 'Discount']:
        X2[columns], lmbda = boxcox(X2[columns]+.00001)
        if lmbda == 0:
            X[columns] = log(X[columns]+0.00001)
        else:
            X[columns] = ((X[columns]+0.00001)**lmbda-1)/lmbda
    X_new = pd.DataFrame(X, columns=X.columns[Selector.get_support()])
    Prediction = modelprediction(X_new)
    Prediction = round(Prediction[0], 2)
    return Prediction
