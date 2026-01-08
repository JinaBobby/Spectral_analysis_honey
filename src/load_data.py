import pandas as pd

def load_spectral_data(path):
    path='data/data.xlsx'
    path='data/data.xlsx'
    df = pd.read_excel(path)
    y = df.iloc[:, 0].astype(str).str.strip()   
    X = df.iloc[:, 1:1022].values                   

    return X, y