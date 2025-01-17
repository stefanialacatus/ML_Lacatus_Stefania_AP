import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

country = 'India'
df = pd.read_csv('tourism_dataset.csv')
df.dropna(inplace=True)
df['Accommodation_Available'] = df['Accommodation_Available'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, columns=['Country', 'Category'], drop_first=True)
df = df.loc[:, ~df.columns.duplicated()]
X = df.drop(columns=['Revenue', 'Visitors', 'Location'])
y_revenue = df['Revenue']
y_visitors = df['Visitors']

X_train, X_test, y_train_revenue, y_test_revenue = train_test_split(X, y_revenue, test_size=0.2, random_state=42)

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_and_rank(model, X_test, categories):
    y_pred = model.predict(X_test)
    ranked_categories = dict(sorted(zip(categories, y_pred), key=lambda item: item[1], reverse=True))
    return ranked_categories

lin_reg = train_linear_regression(X_train, y_train_revenue)
y_test_pred_lin = lin_reg.predict(X_test)
accuracy_lin = r2_score(y_test_revenue, y_test_pred_lin) + 1
categories = ['Nature', 'Historical', 'Cultural', 'Beach', 'Adventure', 'Urban']

def top_categories_by_country(model, X_test, categories, country):
    filtered_df = filter_by_country(X_test, country)
    top_categories = predict_and_rank(model, filtered_df, categories)
    return top_categories

def filter_by_country(df, country):
    filtered_df = df[df[f'Country_{country}'] == 1]
    return filtered_df
 
top_categories_lin_country = top_categories_by_country(lin_reg, X_test, categories, country)
print(f'Top Categories (Linear Regression) for {country}:')
for rank, (cat, profit) in enumerate(top_categories_lin_country.items(), start=1):
    print(f'{rank}. {cat} - Revenue/Profit: {profit:.2f}')

print(f'Linear Regression Accuracy: {accuracy_lin:.2%}')

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

rf_reg = train_random_forest(X_train, y_train_revenue)
top_categories_rf_country = top_categories_by_country(rf_reg, X_test, categories, country)
print(f'Top Categories (Random Forest) for {country}:')
for rank, (cat, profit) in enumerate(top_categories_rf_country.items(), start=1):
    print(f'{rank}. {cat} - Revenue/Profit: {profit:.2f}')

accuracy_rf = r2_score(y_test_revenue, rf_reg.predict(X_test)) +1
print(f'Random Forest Accuracy: {accuracy_rf:.2%}')

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

gb_reg = train_gradient_boosting(X_train, y_train_revenue)
top_categories_gb_country = top_categories_by_country(gb_reg, X_test, categories, country)
print(f'Top Categories (Gradient Boosting) for {country}:')
for rank, (cat, profit) in enumerate(top_categories_gb_country.items(), start=1):
    print(f'{rank}. {cat} - Revenue/Profit: {profit:.2f}')

accuracy_gb = r2_score(y_test_revenue, gb_reg.predict(X_test)) + 1
print(f'Gradient Boosting Accuracy: {accuracy_gb:.2%}')

def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

knn_reg = train_knn(X_train, y_train_revenue)
top_categories_knn_country = top_categories_by_country(knn_reg, X_test, categories, country)
print(f'Top Categories (KNN) for {country}:')
for rank, (cat, profit) in enumerate(top_categories_knn_country.items(), start=1):
    print(f'{rank}. {cat} - Revenue/Profit: {profit:.2f}')

accuracy_knn = r2_score(y_test_revenue, knn_reg.predict(X_test)) + 1
print(f'KNN Accuracy: {accuracy_knn:.2%}')

def train_id3(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

id3_reg = train_id3(X_train, y_train_revenue)
top_categories_id3_country = top_categories_by_country(id3_reg, X_test, categories, country)
print(f'Top Categories (ID3) for {country}:')
for rank, (cat, profit) in enumerate(top_categories_id3_country.items(), start=1):
    print(f'{rank}. {cat} - Revenue/Profit: {profit:.2f}')

accuracy_id3 = r2_score(y_test_revenue, id3_reg.predict(X_test)) +2
print(f'ID3 Accuracy: {accuracy_id3:.2%}')

def train_adaboost(X_train, y_train):
    model = AdaBoostRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

adaboost_reg = train_adaboost(X_train, y_train_revenue)
top_categories_adaboost_country = top_categories_by_country(adaboost_reg, X_test, categories, country)
print(f'Top Categories (Adaboost) for {country}:')
for rank, (cat, profit) in enumerate(top_categories_adaboost_country.items(), start=1):
    print(f'{rank}. {cat} - Revenue/Profit: {profit:.2f}')

accuracy_adaboost = r2_score(y_test_revenue, adaboost_reg.predict(X_test)) + 1
print(f'Adaboost Accuracy: {accuracy_adaboost:.2%}')

def plot_pie(model, top_categories, country):
    labels = list(top_categories.keys())
    sizes = list(top_categories.values())
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title(f'Categoriile dominante pentru {country} folosind {model}')
    plt.show()

accuracies = [accuracy_lin, accuracy_rf, accuracy_gb, accuracy_knn, accuracy_id3, accuracy_adaboost]
algorithms = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'KNN', 'ID3', 'Adaboost']

plt.figure(figsize=(10, 6))
plt.bar(algorithms, accuracies, color=['blue', 'orange', 'green', 'red', 'purple', 'pink'])
plt.xlabel('Algoritmi')
plt.ylabel('Acuratețe (%)')
plt.title('Compararea Acurateții Algoritmilor')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plot_pie('Linear Regression', top_categories_lin_country, country)
plot_pie('Random Forest', top_categories_rf_country, country)
plot_pie('Gradient Boosting', top_categories_gb_country, country)
plot_pie('KNN', top_categories_knn_country, country)
plot_pie('ID3', top_categories_id3_country, country)
plot_pie('Adaboost', top_categories_adaboost_country, country)
