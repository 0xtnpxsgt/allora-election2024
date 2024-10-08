import pandas as pd
import numpy as np
import os
import requests
import joblib
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import data_base_path, model_file_path, TOKEN, MODEL


def get_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['history']
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='s')
        df.columns = ['Timestamp', 'Predict']
        df['Predict'] = df['Predict'].apply(lambda x: x * 100)
        print(df.head())
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
    return df

def download_data(token):
    os.makedirs(data_base_path, exist_ok=True)
    if token == 'R':
        url = "https://clob.polymarket.com/prices-history?interval=all&market=21742633143463906290569050155826241533067272736897614950488156847949938836455&fidelity=1000"
        data = get_data(url)
        save_path = os.path.join(data_base_path, 'polymarket_R.csv')
        data.to_csv(save_path)
    elif token == 'D':
        url = "https://clob.polymarket.com/prices-history?interval=all&market=69236923620077691027083946871148646972011131466059644796654161903044970987404&fidelity=1000"
        data = get_data(url)
        save_path = os.path.join(data_base_path, 'polymarket_D.csv')
        data.to_csv(save_path)

def train_model(token):
    training_price_data_path = os.path.join(data_base_path, f"polymarket_{token}.csv")
    df = pd.read_csv(training_price_data_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour

    X = df[['year', 'month', 'day', 'hour']]
    y = df['Predict']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model and hyperparameter grid
    if MODEL == "SVR":
        model = SVR()
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf']
        }
    elif MODEL == "RandomForest":
        model = RandomForestRegressor()
        param_grid = {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [10, 20, None]
        }
    elif MODEL == "GradientBoosting":
        model = GradientBoostingRegressor()
        param_grid = {
            'model__n_estimators': [100, 200, 500],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 10]
        }
    elif MODEL == "LinearRegression":
        model = LinearRegression()
        param_grid = {}  # No hyperparameters for LinearRegression
    elif MODEL == "Ridge":
        model = Ridge()
        param_grid = {
            'model__alpha': [0.1, 1.0, 10.0]
        }
    elif MODEL == "DecisionTree":
        model = DecisionTreeRegressor()
        param_grid = {
            'model__max_depth': [5, 10, 20, None]
        }
    elif MODEL == "KNeighbors":
        model = KNeighborsRegressor()
        param_grid = {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance']
        }
    elif MODEL == "MLP":
        model = MLPRegressor(max_iter=500, early_stopping=True, random_state=42)
        param_grid = {
        'model__hidden_layer_sizes': [(100,), (100, 100), (50, 50, 50)],
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__learning_rate_init': [0.001, 0.01]
        }
    elif MODEL == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor()
        param_grid = {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [10, 20, None]
        }
    elif MODEL == "AdaBoost":
        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor()
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1.0]
        }
    else:
        raise ValueError("Unsupported model")

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaling the features
        ('model', model)               # The model from the list
    ])

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Best model after hyperparameter tuning
    best_model = grid_search.best_estimator_
    
    # Making predictions
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Best Parameters: {grid_search.best_params_}")

    # Save the best model
    os.makedirs(model_file_path, exist_ok=True)
    save_path_model = os.path.join(model_file_path, f'{MODEL}_model_{token}.pkl')
    joblib.dump(best_model, save_path_model)
    
    print(f"Trained {MODEL} model saved to {save_path_model}")



def get_inference(token):
    save_path_model = os.path.join(model_file_path, f'{MODEL}_model_{token}.pkl')
    loaded_model = joblib.load(save_path_model)
    print("Loaded model successfully")

    single_input = pd.DataFrame({
        'year': [2024],
        'month': [10],
        'day': [10],
        'hour': [12]
    })

    # Making a prediction
    predicted_price = loaded_model.predict(single_input)
    return predicted_price[0]
