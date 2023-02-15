import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def load_data(df, path):
    '''
    Lee el csv de entrenamiento o de prueba dependiendo el parametro proporcionado.
    
    Params:
        df (str): train o test
        Path (str): Path relativo

    Returns:
        Un dataframe con los datos de entrenamiento o prueba.
    '''
    if df == "train":
        data = pd.read_csv(path+"/"+"data/raw/train.csv")
    elif df == "test":
        data = pd.read_csv(path+"/"+"data/raw/test.csv")
    return data

def clean_df(df):
    '''
    Sustituye na's con la media para columanas numericas y con la moda para cualquier otro tipo de variable.
    
    Params:
        df (DataFrame): DataFrame

    Returns:
        Un dataframe sin na's y sin columnas indeseadas
    '''
    for col in df.columns:
        if((df[col].dtype == 'float64') or (df[col].dtype == 'int64')):
            df[col].fillna(df[col].mean(),inplace = True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    drop_col = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MoSold', 'YrSold', 'MSSubClass',
            'GarageType', 'GarageArea', 'GarageYrBlt', 'GarageFinish', 'YearRemodAdd', 'LandSlope',
            'BsmtUnfSF', 'BsmtExposure', '2ndFlrSF', 'LowQualFinSF', 'Condition1', 'Condition2', 'Heating',
             'Exterior1st', 'Exterior2nd', 'HouseStyle', 'LotShape', 'LandContour', 'LotConfig', 'Functional',
             'BsmtFinSF1', 'BsmtFinSF2', 'FireplaceQu', 'WoodDeckSF', 'GarageQual', 'GarageCond', 'OverallCond'
           ]
    df.drop(drop_col,axis=1,inplace=True)
    return df

def OE(df):
    '''
    Transforma las columnas que no son tipo float ni integer a ordinal encoder; es decir una etiqueta única a un valor entero.
        
    Params:
        df (DataFrame): DataFrame

    Returns:
        Un dataframe 
    '''
    for col in df.columns:
        if((df[col].dtype == 'float64') or (df[col].dtype == 'int64')):
            pass
        else:    
            OE = OrdinalEncoder()
            df[col] = OE.fit_transform(df[[col]]) 
    return df

def output(df, path, destination, name):
    df.to_csv(path+"/"+"data/output/"+name+".csv")