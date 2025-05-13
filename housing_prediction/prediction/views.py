# from django.shortcuts import render
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.tree import export_graphviz
# import graphviz
# from sklearn.tree import DecisionTreeRegressor     

# def my_r2_score(y_true, y_pred):
#     y_true = np.array(y_true)  
#     y_pred = np.array(y_pred)

#     y_mean = np.mean(y_true)
#     sst = np.sum((y_true - y_mean) ** 2)

#     sse = np.sum((y_true - y_pred) ** 2)

#     if sst == 0: # Обработка случая, когда все значения y_true одинаковы
#         return 1.0 if np.all(y_true == y_pred) else 0.0 # Если y_true и y_pred тоже одинаковы, R2 = 1, иначе 0
#     r2 = 1 - (sse / sst)

#     return r2

# try:
#     data = pd.read_csv('krasnodar_apartments_cleaned.csv')
#     print("Data loaded successfully. Shape:", data.shape)
# except FileNotFoundError:
#     print("Error: updated_flat_prices_dataset.csv not found.")
#     data = pd.DataFrame()
# except Exception as e:
#     print(f"Error loading CSV: {e}")
#     data = pd.DataFrame()

# # Preprocess the data
# if not data.empty: # Добавлено условие
#     data = pd.get_dummies(data, columns=['location', 'jkh'], drop_first=True)
#     X = data.drop(['price', 'id', 'title'], axis=1, errors='ignore')  # Исключаем 'id' и 'title'
#     y = data['price']

#     try: # Добавлено для обработки ошибки, если X или y пусты
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)
#         r2 = my_r2_score(y_test, y_pred) * 100
#     except ValueError as e:
#         print(f"Error during model training: {e}")
#         X_train, X_test, y_train, y_test, model, r2 = None, None, None, None, None, 0 
# else:
#     print("No data loaded, skipping model training.")
#     X_train, X_test, y_train, y_test, model, r2 = None, None, None, None, None, 0 

# def predict_price(input_data):
#     global model, feature_names
#     if model is None: # Проверка, что модель обучена
#         print("Error: Model not trained.")
#         return None 

#     input_df = pd.DataFrame([input_data])
#     input_df = pd.get_dummies(input_df, drop_first=True)
#     input_df = input_df.reindex(columns=X.columns, fill_value=0)

#     predicted_price_rub = model.predict(input_df)[0]

#     return predicted_price_rub

# def home(request):

#     # Визуализация дерева
#     if model is not None: # Проверка, что модель обучена
#         try:
#             tree = model.estimators_[0]  # Берем первое дерево из леса

#             feature_names = list(X.columns) 
#             dot_data = export_graphviz(tree,
#                                         feature_names=feature_names,  # Передаем имена признаков
#                                         filled=True,
#                                         rounded=True,
#                                         special_characters=True)
#             graph = graphviz.Source(dot_data)
#             graph.render("flat_price_tree", view=False)  
#         except Exception as e:
#             print(f"Error generating tree visualization: {e}")
#     else:
#         print("Model not trained, skipping tree visualization.")
    

#     if request.method == 'POST':
#         input_data = {
#                 'rooms': int(request.POST.get('rooms')),
#                 'square': float(request.POST.get('square')),
#                 'location': request.POST.get('location'),
#                 'jkh': request.POST.get('jkh', ''),
#                 'floor': int(request.POST.get('floor')),
#                 'home_floor': int(request.POST.get('home_floor')),
#             }
#         if model is not None: # Проверка, что модель обучена
#             predicted_price= predict_price(input_data) 
#         else:
#             predicted_price = None


#         return render(request, 'prediction/result.html', {
#             'predicted_price': round(predicted_price, 2) if predicted_price is not None else "Ошибка: Модель не обучена", 
#             'r2': round(r2, 2) if model is not None else 0, 
#         })
    
#     return render(request, 'prediction/home.html',{'r2': round(r2, 2) if model is not None else 0}) 

from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
from sklearn.tree import DecisionTreeRegressor
import random

def my_r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_mean = np.mean(y_true)
    sst = np.sum((y_true - y_mean) ** 2)

    sse = np.sum((y_true - y_pred) ** 2)

    if sst == 0:  # Handle the case where all y_true values are the same
        return 1.0 if np.all(y_true == y_pred) else 0.0  # If y_true and y_pred are also the same, R2 = 1, otherwise 0
    r2 = 1 - (sse / sst)

    return r2

# Custom Random Forest Regressor Implementation
class MyRandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        if self.random_state is not None:
            random.seed(self.random_state)  # Seed the random number generator

    def _bootstrap_sample(self, X, y):
        """Creates a bootstrap sample of X and y."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        """Fits the Random Forest Regressor to the training data."""
        self.feature_names = list(X.columns)  # Store feature names
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=random.randint(0, 1000))  # Each tree gets a unique random_state
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """Predicts target values for X."""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)  # Average the predictions of all trees


try:
    data = pd.read_csv('krasnodar_apartments_cleaned.csv')
    print("Data loaded successfully. Shape:", data.shape)
except FileNotFoundError:
    print("Error: krasnodar_apartments_cleaned.csv not found.")
    data = pd.DataFrame()
except Exception as e:
    print(f"Error loading CSV: {e}")
    data = pd.DataFrame()

# Preprocess the data
if not data.empty:  # Added a condition
    # Handle missing values using a suitable strategy like imputation
    for col in ['rooms', 'square', 'floor', 'home_floor']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce') # convert to numeric, invalid values to NaN
            data[col] = data[col].fillna(data[col].median()) # Impute missing values with the median

    data = pd.get_dummies(data, columns=['location', 'jkh'], drop_first=True)
    X = data.drop(['price', 'id', 'title'], axis=1, errors='ignore')  # Excluding 'id' and 'title'
    y = data['price']

    # Ensure X has all the necessary columns, fill with 0 if missing (necessary after dummies)
    missing_cols = set(X.columns) - set(['rooms', 'square', 'floor', 'home_floor'])  # More precise
    for c in missing_cols:
        X[c] = 0

    try:  # Added to handle the error if X or y are empty
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MyRandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)  # Use our custom implementation and set max_depth
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = my_r2_score(y_test, y_pred) * 100
        feature_names = list(X.columns) # Store feature names for later use
    except ValueError as e:
        print(f"Error during model training: {e}")
        X_train, X_test, y_train, y_test, model, r2, feature_names = None, None, None, None, None, 0, []
    except Exception as e:  # Catch other potential errors
        print(f"An unexpected error occurred during model training: {e}")
        X_train, X_test, y_train, y_test, model, r2, feature_names = None, None, None, None, None, 0, []
else:
    print("No data loaded, skipping model training.")
    X_train, X_test, y_train, y_test, model, r2, feature_names = None, None, None, None, None, 0, []


def predict_price(input_data):
    global model, feature_names
    if model is None:  # Check that the model is trained
        print("Error: Model not trained.")
        return None

    input_df = pd.DataFrame([input_data])

    # Ensure that the input_df has the same columns as the training data X.
    input_df = pd.get_dummies(input_df, columns=['location', 'jkh'], drop_first=True)

    # Reindex to match the training data columns and fill missing values with 0
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Handle cases where input_df might have different columns
    if len(input_df.columns) != len(X.columns):
        print("Error: Input data columns do not match training data columns.")
        return None

    predicted_price_rub = model.predict(input_df)[0]

    return predicted_price_rub


def home(request):
    global model, r2, feature_names
    # Tree visualization
    if model is not None:  # Check that the model is trained
        try:
            tree = model.trees[0]  # Access the first tree in the forest

            dot_data = export_graphviz(tree,
                                        feature_names=feature_names,  # Pass feature names
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render("flat_price_tree", view=False)
        except Exception as e:
            print(f"Error generating tree visualization: {e}")
    else:
        print("Model not trained, skipping tree visualization.")

    if request.method == 'POST':
        try:
            input_data = {
                'rooms': int(request.POST.get('rooms')),
                'square': float(request.POST.get('square')),
                'location': request.POST.get('location'),
                'jkh': request.POST.get('jkh', ''),
                'floor': int(request.POST.get('floor')),
                'home_floor': int(request.POST.get('home_floor')),
            }
        except ValueError:  # Handle potential errors from user input
            return render(request, 'prediction/result.html', {
                'predicted_price': "Ошибка: Неверный ввод данных",
                'r2': round(r2, 2) if model is not None else 0,
            })

        if model is not None:  # Check that the model is trained
            predicted_price = predict_price(input_data)
        else:
            predicted_price = None

        return render(request, 'prediction/result.html', {
            'predicted_price': round(predicted_price, 2) if predicted_price is not None else "Ошибка: Модель не обучена",
            'r2': round(r2, 2) if model is not None else 0,
        })

    return render(request, 'prediction/home.html', {'r2': round(r2, 2) if model is not None else 0})