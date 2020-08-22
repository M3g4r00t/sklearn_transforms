import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

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
            X_train, X_test, y_train, y_test = train_test_split(args[0], args[1], test_size=0.3, stratify=args[1].values, random_state=42)
            self.model.fit(X_train, y_train, **kwargs)
        else:            
            X_train, X_test = train_test_split(args[0], test_size=0.3, random_state=42)
            self.model.fit(X_train, **kwargs)        
        return self

    def transform(self, X, **transform_params):        
        df = pd.DataFrame(self.model.predict(X))
        for i in df.dtypes:
            if str(i) == "object":
                enc = OrdinalEncoder()
                return enc.fit_transform(df)
        return df  
    
    