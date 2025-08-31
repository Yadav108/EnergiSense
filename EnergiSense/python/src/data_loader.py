import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    data = pd.read_csv(path)
    X = data[['AT', 'V', 'AP', 'RH']]
    y = data['PE']
    return train_test_split(X, y, test_size=0.2, random_state=42)
