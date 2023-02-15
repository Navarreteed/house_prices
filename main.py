# se importan las librerias necesarias
import src.tools_etl as tools
import pandas as pd
import src.model as model
import os 
import logging

import logging
logging.basicConfig(
    filename='./logs/results.log',
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

#Se identifica el path
CURRENT = os.path.dirname(os.path.abspath(__file__))

#Se leen los datos

try:
    train = tools.load_data("train",CURRENT)
    test = tools.load_data("test",CURRENT)
    logging.info("Se cargaron exitosamente los datos")
    logging.info(f"la forma de los datos de entrenamiento es: {train.shape}")
    logging.info(f"la forma de los datos de test es: {test.shape}")
except:
    logging.error("No se cargaron los datos de manera adecuada")



#Se limpian los datos
try:
    train = tools.clean_df(train)
    test = tools.clean_df(test)
    logging.info("Los datos fueron limpiados exitosamente")
except:
    logging.error("No se cargaron los datos de manera adecuada")
    
#Se convierten cateogorias a ordinal encoder
train_clean = tools.OE(train)
test_clean = tools.OE(test)

#Se tiran columnas innecesarias paraa el modelo
cols_to_drop = ['OverallQual', 
            'ExterCond',
            'ExterQual',
            'BsmtCond',
            'BsmtQual',
            'BsmtFinType1',
            'BsmtFinType2',
            'HeatingQC',
            'OpenPorchSF',
            'EnclosedPorch',
            '3SsnPorch',
            'ScreenPorch',
            'BsmtFullBath',
            'BsmtHalfBath',
            'FullBath',
            'HalfBath',]

train_clean.drop(cols_to_drop, axis=1, inplace=True)
test_clean.drop(cols_to_drop, axis=1, inplace=True)

#Se guardan los DF limpios en la carpeta data/clean
train_clean.to_csv(CURRENT+"/data/clean/train.csv")
test_clean.to_csv(CURRENT+"/data/clean/test.csv")

#Se identifica X y Y para entrenar en el modelo
x_train,y_train=model.x_and_y(train_clean)

#Se entrena el modelo y se generan predicciones
try:
    y_hat=model.RF(x_train,y_train, test_clean)
    logging.info("se entreno el modelo exitosamente")
    logging.info(f"las predicciones tienen una forma de:{y_hat.shape}")
except:
    logging.error("No se cargaron los datos de manera adecuada")
    
#Se crea un DF con las pedicciones
ids = test_clean['Id']
submission = pd.DataFrame({
    'Id': ids,
    'SalePrice':y_hat
})

#Se guardan el DF resultado en la carpeta data/output
submission.to_csv(CURRENT+"/data/output/submission.csv")


print(y_hat)