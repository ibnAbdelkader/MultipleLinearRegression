
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import warnings
import warnings
warnings.filterwarnings("ignore")

# We will use some methods from the sklearn module
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
def Pearson_correlation(X,Y):
    if len(X)==len(Y):
        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
        Sum_x_squared = sum((X-X.mean())**2)
        Sum_y_squared = sum((Y-Y.mean())**2)       
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr

def main():
    # observations / data
    ds = pd.read_csv("RealEstate.csv")
    ds.head()
    ds.shape
    #print(ds.corr())
    #print(ds.describe())
    x = ds[['X1TransactionDate','X2HouseAge','X3DistanceToTheNearestMRTStation','X4NumberOfConvenienceStores','X5Latitude','X6Longitude']]
    y = ds['YHousePriceOfUnitArea']
    sns.distplot(ds['YHousePriceOfUnitArea']);
    sns.pairplot(ds, x_vars=['X1TransactionDate','X2HouseAge','X3DistanceToTheNearestMRTStation','X4NumberOfConvenienceStores','X5Latitude','X6Longitude'], y_vars='YHousePriceOfUnitArea', height=4, aspect=1, kind='scatter')
    plt.show()
    sns.heatmap(ds.corr(), annot = True, cmap = 'coolwarm')
    plt.show()
    X_train,X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
    reg_model = linear_model.LinearRegression()
    reg_model = LinearRegression().fit(X_train, y_train)
    #Printing the model coefficients
    print('Intercept: ',reg_model.intercept_)
    # pair the feature names with the coefficients
    list(zip(x, reg_model.coef_))
    #Predicting the Test and Train set result 
    y_pred= reg_model.predict(X_test)  
    x_pred= reg_model.predict(X_train) 
    print("Prediction for test set: {}".format(y_pred))
    #Actual value and the predicted value
    reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
    reg_model_diff    

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    r2 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print('Mean Absolute Error:', mae)
    print('Mean Square Error:', mse)
    print('Root Mean Square Error:', r2)
    
    
    
    #Check the correlation between predictor and response
    # print("Correlation: " ,Pearson_correlation(x,y))
    
    # estimating coefficients
    '''b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
    '''
    # plotting regression line
    # plot_regression_line(x, y)
   # plt.scatter(x,y)
if __name__ == "__main__":
    main()