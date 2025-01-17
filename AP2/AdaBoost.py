import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('tourism_dataset.csv')
df.dropna(inplace=True)
df['Accommodation_Available'] = df['Accommodation_Available'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, columns=['Country', 'Category'], drop_first=True)
df = df.loc[:, ~df.columns.duplicated()]
X = df.drop(columns=['Revenue', 'Visitors', 'Location'])
y_revenue = df['Revenue']
categories = ['Nature', 'Historical', 'Cultural', 'Beach', 'Adventure', 'Urban']
country = 'India'

X_train, X_test, y_train_revenue, y_test_revenue = train_test_split(X, y_revenue, test_size=0.2, random_state=42)

def train_adaboost(X_train, y_train):
    n_estimators = 100
    n_samples = len(y_train)
    weights = [1 / n_samples] * n_samples
    
    base_learner = DecisionTreeRegressor(random_state=42)
    
    for _ in range(n_estimators):
        base_learner.fit(X_train, y_train, sample_weight=weights)
        y_pred = base_learner.predict(X_train)
        incorrect = (y_pred != y_train)
        error = sum(w * inc for w, inc in zip(weights, incorrect)) / sum(weights)
        
        if error == 0:
            break
        
        classifier_weight = 0.5 * np.log((1 - error) / error)
        
        weights = [w * np.exp(-classifier_weight * inc) for w, inc in zip(weights, incorrect)]
        weights /= sum(weights)
        
    return base_learner

adaboost_reg = train_adaboost(X_train, y_train_revenue)
y_test_pred_adaboost = adaboost_reg.predict(X_test)
mae = mean_absolute_error(y_test_revenue, y_test_pred_adaboost)
mean_y_test = np.mean(y_test_revenue)

accuracy_adaboost = 1 - (mae / mean_y_test)
print(f'Acuratete: {accuracy_adaboost:.2%}')
print(f"MAE: {mae:.2f}")   

def predict_and_rank(model, df, categories):
    X_country = df.drop(columns=['Revenue', 'Visitors', 'Location'])
    y_pred = model.predict(X_country)
    ranked_categories = dict(sorted(zip(categories, y_pred), key=lambda item: item[1], reverse=True))
    return ranked_categories

filtered_df = df[df[f'Country_{country}'] == 1]

top_categories_adaboost_country = predict_and_rank(adaboost_reg, filtered_df, categories)

print(f'Top activitati (Adaboost) pentru {country}:')
for rank, (cat, profit) in enumerate(top_categories_adaboost_country.items(), start=1):
    print(f'{rank}. {cat} - Revenue/Profit: {profit:.2f}')

def plot_pie(model, top_categories, country):
    labels = list(top_categories.keys())
    sizes = list(top_categories.values())
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title(f'Top activitatilor in {country} folosind AdaBoost')
    plt.show()

plot_pie('Adaboost', top_categories_adaboost_country, country)
