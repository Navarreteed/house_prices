from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def x_and_y(df):
    y=df['SalePrice']
    x=df.drop(['SalePrice'],axis=1)
    return x,y

def RF(x_train,y_train,x_test):
    rf=RandomForestRegressor()
    rf.fit(x_train,y_train)
    y_hat=rf.predict(x_test)
    return y_hat
