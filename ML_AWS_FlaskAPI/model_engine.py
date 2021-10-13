import pickle
import pandas as pd
import numpy as numpy 
from sklearn import datasets, ensemble

from app.features import FEATURES

def model_pipeline(FEATURES):
    data = datasets.load_boston()
    df = pd.DataFrame(data.data,columns=data.feature_names)
    df['Target'] = data.target
    df = df.loc[df['Target'] != 50].copy()
    X, Y = df[FEATURES].values, df['Target'].values
    model = ensemble.RandomForestRegressor(n_estimators=50)
    model.fit(X,Y)
    pickle.dump(model, open('app/model.pkl', 'wb'))
if __name__ == "__main__":
    model_pipeline(FEATURES)