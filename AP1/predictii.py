import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

#citirea datelor din fișier
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name=0, names=['Data', 'Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 
                              'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]',
                              'Biomasa[MW]', 'Sold[MW]'], parse_dates=['Data'])
    return df

#preprocesarea datelor
def preprocess_data(df):
    #am extras ziua, luna si anul din coloana Data
    df['Day'] = df['Data'].dt.day
    df['Month'] = df['Data'].dt.month
    df['Year'] = df['Data'].dt.year

    df_train = df[df['Data'] < '2024-12-01']  #luam datele pana in decembrie 2024 pentru antrenare
    df_test = df[df['Data'] >= '2024-12-01']  #luam datele din luna decembrie 2024 pentru validare
    
    return df_train, df_test

#bucketiung pentru coloana Sold
def bucketize_sold(df, n_bins=10):
    scaler = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    df['Sold_binned'] = scaler.fit_transform(df[['Sold[MW]']])
    return df, scaler

#implementarea ID3
def train_decision_tree(df_train):
    features = ['Day','Month', 'Consum[MW]', 'Productie[MW]']
    X_train = df_train[features]
    y_train = df_train['Sold_binned']
    
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    return model

#implementarea Bayes Naiv
def train_naive_bayes(df_train):
    features = ['Day', 'Month', 'Consum[MW]', 'Productie[MW]']
    X_train = df_train[features]
    y_train = df_train['Sold_binned']
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    return model

#functia de predictie
def predict(df_test, model, scaler):
    features = ['Day', 'Month', 'Consum[MW]', 'Productie[MW]']
    X_test = df_test[features]
    
    y_pred_binned = model.predict(X_test)
    
    # Inversam bucketizarea
    y_pred = scaler.inverse_transform(y_pred_binned.reshape(-1, 1))
    
    return y_pred

#functia de calcul a soldului total real si prezis din decembrie 2024
def calculate_total_sold(df_test, y_pred_id3, y_pred_bayes):
    tot_sold_real = df_test['Sold[MW]'].sum()
    tot_sold_id3 = y_pred_id3.sum()
    tot_sold_bayes = y_pred_bayes.sum()
    
    return tot_sold_real, tot_sold_id3, tot_sold_bayes

#functia de calcul a acurateței
def calculate_accuracy(sold_real, sold_pred):
    return (1 - abs(sold_pred - sold_real) / sold_real) * 100

#evaluarea performantei cu RMSE, MAE și MSE
def evaluate_model(df_test, y_pred):
    y_true = df_test['Sold[MW]'].values
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def main(file_path):
    df = load_data(file_path)
    df_train, df_test = preprocess_data(df)
    
    df_train, scaler = bucketize_sold(df_train)
    
    #antrenarea modelele
    id3_model = train_decision_tree(df_train)
    nb_model = train_naive_bayes(df_train)
    
    #predictiile pentru fiecare model
    y_pred_id3 = predict(df_test, id3_model, scaler)
    y_pred_bayes = predict(df_test, nb_model, scaler)
    
    #evaluarea predictiilor
    mae_id3, mse_id3, rmse_id3 = evaluate_model(df_test, y_pred_id3)
    mae_bayes, mse_bayes, rmse_bayes = evaluate_model(df_test, y_pred_bayes)
    
    tot_sold_real, tot_sold_id3, tot_sold_bayes = calculate_total_sold(df_test, y_pred_id3, y_pred_bayes)
    
    #se calculeaza acuratetea fiecarui model
    accuracy_id3 = calculate_accuracy(tot_sold_real, tot_sold_id3)
    accuracy_bayes = calculate_accuracy(tot_sold_real, tot_sold_bayes)
    
    print(f"Algoritmul ID3 - MAE: {mae_id3:.2f}, MSE: {mse_id3:.2f}, RMSE: {rmse_id3:.2f}, Acuratețe: {accuracy_id3:.2f}%")
    print(f"Algoritmul Bayes Naiv - MAE: {mae_bayes:.2f}, MSE: {mse_bayes:.2f}, RMSE: {rmse_bayes:.2f}, Acuratețe: {accuracy_bayes:.2f}%")
    print(f"Sold total real in decembrie 2024: {tot_sold_real:.2f} MW")
    print(f"Sold total prezis de ID3: {tot_sold_id3:.2f} MW")
    print(f"Sold total prezis de Bayes Naiv: {tot_sold_bayes:.2f} MW")
    
    #grafic cu predictiile vs soldul real
    plt.figure(figsize=(10, 5))
    plt.plot(df_test['Data'], df_test['Sold[MW]'], label='Sold real', color='blue')
    plt.plot(df_test['Data'], y_pred_id3, label='Predictie ID3', color='green', linestyle='--')
    plt.plot(df_test['Data'], y_pred_bayes, label='Predictie Bayes Naiv', color='red', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Sold [MW]')
    plt.title('Predictiile vs Soldul real pentru luna decembrie 2024')
    plt.legend()
    plt.show()

file_path = 'date_sen.xlsx'
main(file_path)
