import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

#citirea datelor
def get_data(file_path):
    df = pd.read_excel(file_path, sheet_name=0, names=['Data', 'Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 
                                                      'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]', 
                                                      'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]', 'Sold[MW]'], parse_dates=['Data'])
    return df

#preprocesarea datelor
def preprocess_data(df):
    #extragerea zilei si lunii din coloana Data
    df['Day'] = df['Data'].dt.day
    df['Month'] = df['Data'].dt.month
    df_train = df[df['Data'] < '2024-12-01']  #luam datele pana in decembrie 2024 pentru antrenare
    df_test = df[df['Data'] >= '2024-12-01']  #luam datele din luna decembrie 2024 pentru validare
    
    return df_train, df_test

#bucketing pentru coloana Sold
def bucketize_sold(df, n_bins=10):
    s = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    df['Sold_binned'] = s.fit_transform(df[['Sold[MW]']])
    return df, s

#functia pentru entropie
def entropy(values):
    total = len(values)
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / total
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

#functia pentru castigul de informatie
def info_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / len(data)) * entropy(data[data[feature] == values[i]][target])
        for i in range(len(values))
    )
    return total_entropy - weighted_entropy

#implementarea ID3
class ID3:
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.bin_midpoints = None

    def fit(self, X, y, bin_midpoints):
        data = X.copy()
        data['target'] = y
        self.bin_midpoints = bin_midpoints
        self.tree = self.build_tree(data, depth=0)

    def build_tree(self, data, depth):
        if len(data) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(data['target'])

        best_attr = max(data.columns[:-1], key=lambda feature: info_gain(data, feature, 'target'))
        tree = {best_attr: {}}

        for value in np.unique(data[best_attr]):
            subset = data[data[best_attr] == value]
            tree[best_attr][value] = self.build_tree(subset, depth + 1)

        return tree

    def predict_instance(self, instance, tree):
        if not isinstance(tree, dict):
            return self.bin_midpoints[int(tree)]

        feature = next(iter(tree))
        value = instance[feature]

        if value in tree[feature]:
            return self.predict_instance(instance, tree[feature][value])
        else:
            return np.mean([self.predict_instance(instance, subtree) for subtree in tree[feature].values()])

    def predict(self, X):
        return np.array([self.predict_instance(row, self.tree) for _, row in X.iterrows()])


#implementarea Bayes Naiv
def train_bayes(df_train):
    features = ['Day', 'Month', 'Consum[MW]', 'Productie[MW]']
    X_train = df_train[features]
    y_train = df_train['Sold_binned']
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    return model

#functia de predictie pentru Bayes Naiv
def predict(df_test, model, scaler):
    features = ['Day', 'Month', 'Consum[MW]', 'Productie[MW]']
    X_test = df_test[features]
    
    y_pred_binned = model.predict(X_test)
    
    #inversam bucketizarea
    y_pred = scaler.inverse_transform(y_pred_binned.reshape(-1, 1))
    
    return y_pred

#calculul soldului total real si a celui prezis
def calculate_total_sold(df_test, y_pred_id3, y_pred_bayes):
    tot_sold_real = df_test['Sold[MW]'].sum()
    tot_sold_id3 = y_pred_id3.sum()
    tot_sold_bayes = y_pred_bayes.sum()
    
    return tot_sold_real, tot_sold_id3, tot_sold_bayes

#functia pentru calcularea acuratetii
def calculate_accuracy(sold_real, sold_pred):
    return (1 - abs(sold_pred - sold_real) / sold_real) * 100

#functia pentru evaluarea performantei cu rmse, mae si mse
def evaluate_model(df_test, y_pred):
    y_true = df_test['Sold[MW]'].values
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def main(file_path):
    df = get_data(file_path)
    df_train, df_test = preprocess_data(df)
    
    #se aplica bucketing pentru coloana Sold
    df_train, scaler = bucketize_sold(df_train)
    bin_midpoints = (
        scaler.bin_edges_[0][:-1] + scaler.bin_edges_[0][1:]
    ) / 2  #se calculeaza mijlocurile binurilor
    
    #antrenarea modelului ID3
    id3_model = ID3(max_depth=5, min_samples_split=10)
    id3_model.fit(
        df_train[['Day', 'Month', 'Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 
                  'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 
                  'Foto[MW]', 'Biomasa[MW]']],
        df_train['Sold_binned'],
        bin_midpoints
    )
    
    #antrenarea modelului Bayes Naiv
    naive_bayes_model = train_bayes(df_train)
    
    #predictiile pentru fiecare model
    y_pred_id3 = id3_model.predict(df_test[['Day', 'Month', 'Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 
                                                       'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]', 
                                                       'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']])
    
    y_pred_bayes = predict(df_test, naive_bayes_model, scaler)
    
    #evaluarea predictiilor
    mae_tree, mse_tree, rmse_tree = evaluate_model(df_test, y_pred_id3)
    mae_bayes, mse_bayes, rmse_bayes = evaluate_model(df_test, y_pred_bayes)
    
    #se calculeaza soldurile adevarate si prezise
    tot_sold_real, tot_sold_id3, tot_sold_bayes = calculate_total_sold(df_test, y_pred_id3, y_pred_bayes)
    
    #se calculeaza acuratetea
    accuracy_id3 = calculate_accuracy(tot_sold_real, tot_sold_id3)
    accuracy_bayes = calculate_accuracy(tot_sold_real, tot_sold_bayes)
    
    print(f"Algoritmul ID3 - MAE: {mae_tree:.2f}, MSE: {mse_tree:.2f}, RMSE: {rmse_tree:.2f}, Acuratete: {accuracy_id3:.2f}%")
    print(f"Algoritmul Bayes Naiv - MAE: {mae_bayes:.2f}, MSE: {mse_bayes:.2f}, RMSE: {rmse_bayes:.2f}, Acuratete: {accuracy_bayes:.2f}%")
    
    print(f"Sold total real in decembrie 2024: {tot_sold_real:.2f} MW")
    print(f"Sold total prezis de ID3: {tot_sold_id3:.2f} MW")
    print(f"Sold total prezis de Bayes Naiv: {tot_sold_bayes:.2f} MW")
    
    #graficul comparativ intre predictii si soldul real
    plt.figure(figsize=(10, 5))
    plt.plot(df_test['Data'], df_test['Sold[MW]'], label='Sold real', color='blue')
    plt.plot(df_test['Data'], y_pred_id3, label='Predictie ID3', color='purple', linestyle='--')
    plt.plot(df_test['Data'], y_pred_bayes, label='Predictie Naive Bayes', color='red', linestyle='--')
    plt.xlabel('Timp')
    plt.ylabel('Sold [MW]')
    plt.title('Comparatie intre predictii si soldul real in decembrie 2024')
    plt.legend()
    plt.show()

file_path = 'date_sen.xlsx'
main(file_path)
