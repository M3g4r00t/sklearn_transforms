import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import numpy as np


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return data.drop(labels=self.columns, axis='columns')

    
# All sklearn Transforms must have the `transform` and `fit` methods
class SumColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns, sum_column):
        self.columns = columns
        self.sum_column = sum_column

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        data[self.sum_column] = data[self.columns].sum(axis=1)
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return data
    
    
class NaNSearch(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()   
        res = pd.DataFrame()
        for c in data.columns[data.isna().any()].tolist():
            res[c+'isnan'] = np.where(data[c].isna(), -1, 1)
        return res

    
class Dummy(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        return data
    
    
class ModelTransformer(TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        if len(args) > 1:
            X_train, X_test, y_train, y_test = train_test_split(args[0], args[1], test_size=0.2, stratify=args[1].values, random_state=42)
            self.model.fit(X_train, y_train, **kwargs)
        else:            
            X_train, X_test = train_test_split(args[0], test_size=0.2, random_state=42)
            self.model.fit(X_train, **kwargs)        
        return self

    def transform(self, X, **transform_params):        
        df = pd.DataFrame()
        if callable(getattr(self.model, 'predict', None)):
            df = pd.DataFrame(self.model.predict(X, **transform_params))
        elif callable(getattr(self.model, 'transform', None)):
            df = pd.DataFrame(self.model.transform(X, **transform_params))
        elif callable(getattr(self.model, 'fit_transform', None)):
            df = pd.DataFrame(self.model.fit_transform(X, **transform_params))
                
        for i in df.dtypes:
            if str(i) == "object":
                enc = OrdinalEncoder()
                return enc.fit_transform(df)
        return df  

