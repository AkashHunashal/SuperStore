from main import *
import pickle
from flask import Flask, render_template, request, redirect
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

app = Flask(__name__)


@app.route('/', methods=['GET'])
def Home():
    Y_pred = model()
    Ship_Mode = []
    Segment = []
    City = []
    State = []
    Postal_Code = []
    Region = []
    Category = []
    Sub_Category = []
    Product_Name = []
    category_dict = dict(zip(ohe.feature_names_in_, ohe.categories_))
    for category, value in category_dict.items():
        if category == 'Ship Mode':
            Ship_Mode.extend(value)
        elif category == 'Segment':
            Segment.extend(value)
        elif category == 'Region':
            Region.extend(value)
        else:
            Category.extend(value)
    for column in encoded:
        for key, values in column.items():
            for category in sorted(values.keys()):
                if key == 'City':
                    City.append(category)
                elif key == 'State':
                    State.append(category)
                elif key == 'Postal Code':
                    Postal_Code.append(category)
                elif key == 'Sub-Category':
                    Sub_Category.append(category)
                else:
                    Product_Name.append(category)
    context = {
        'Product Name': Product_Name,
        'Category': Category,
        'Sub-Category': Sub_Category,
        'Segment': Segment,
        'Ship Mode': Ship_Mode,
        'City': City,
        'State': State,
        'Region': Region,
        'Postal Code': Postal_Code,
    }
    columns = ['Shipping Days', 'Quantity', 'Discount']
    return render_template('index.html', context=context, column=columns)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Ship_Mode = request.form['Ship Mode']
        Segment = request.form['Segment']
        City = request.form['City']
        State = request.form['State']
        Postal_Code = float(request.form['Postal Code'])
        Region = request.form['Region']
        Category = request.form['Category']
        Sub_Category = request.form['Sub-Category']
        Product_Name = request.form['Product Name']
        Quantity = int(request.form['Quantity'])
        Discount = float(request.form['Discount'])
        Shipping_Days = int(request.form['Shipping Days'])
        data = pd.DataFrame([[Ship_Mode, Segment, City, State, Postal_Code, Region,
                              Category, Sub_Category, Product_Name, Quantity,
                              Discount, Shipping_Days]], columns=['Ship Mode', 'Segment', 'City', 'State', 'Postal Code', 'Region',
                                                                  'Category', 'Sub-Category', 'Product Name', 'Quantity',
                                                                  'Discount', 'Shipping_days'])
        predictions = testmodel(data)
        print(predictions)
        if predictions > 0:
            return render_template('predict.html', prediction_text=f'The Total Sales generated on {Product_Name} is ${round(predictions,2)}')
        else:
            return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
 
