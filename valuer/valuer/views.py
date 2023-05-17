from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def home(request):
    return render(request, "home.html")


def predict(request):
    return render(request, "predict.html")


def result(request):
    data = pd.read_csv('./static/simulated_data.csv')
    X = data.drop('price', axis=1)
    Y = data['price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    var1 = float(request.GET['bedrooms'])
    var2 = float(request.GET['bathrooms'])
    var3 = float(request.GET['sqft_living'])
    var4 = float(request.GET['sqft_lot'])
    var5 = float(request.GET['floors'])
    var6 = float(request.GET['waterfront'])
    var7 = float(request.GET['view'])
    var8 = float(request.GET['condition'])
    var9 = float(request.GET['grade'])
    var10 = float(request.GET['sqft_above'])
    var11 = float(request.GET['sqft_basement'])
    var12 = float(request.GET['yr_built'])
    var13 = float(request.GET['yr_renovated'])
    var14 = float(request.GET['zipcode'])
    var15 = float(request.GET['sqft_living15'])
    var16 = float(request.GET['sqft_lot15'])

    pred = model.predict(np.array([var1, var2, var3, var4, var5, var6, var7, var8,
                         var9, var10, var11, var12, var13, var14, var15, var16]).reshape(1, -1))

    pred = round(pred[0])

    price = "The Predicted Price is $"+str(pred)

    return render(request, "predict.html", {"result2": price})
