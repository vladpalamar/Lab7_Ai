import datetime
import json
import numpy as np
import yfinance as yf
from sklearn import cluster, covariance

# Завантаження прив'язок символів компаній до їх повних назв
input_file = "company_symbol_mapping.json"
with open(input_file, "r") as f:
    company_symbols_map = json.load(f)

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань за допомогою yfinance
start_date = "2003-07-03"
end_date = "2007-05-04"

quotes = {}
for symbol in symbols:
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if not data.empty:
            quotes[symbol] = data
        else:
            print(f"Дані для {symbol} недоступні.")
    except Exception as e:
        print(f"Помилка завантаження даних для {symbol}: {e}")

# Знаходження спільних дат
common_dates = set.intersection(*[set(data.index) for data in quotes.values()])
common_dates = sorted(list(common_dates))

# Вилучення котирувань за спільними датами
valid_symbols = []
opening_quotes = []
closing_quotes = []

for symbol, data in quotes.items():
    try:
        filtered_data = data.loc[common_dates]
        opening_quotes.append(filtered_data["Open"].values)
        closing_quotes.append(filtered_data["Close"].values)
        valid_symbols.append(symbol)
    except KeyError:
        print(f"Дані для {symbol} не збігаються за датами.")

opening_quotes = np.array(opening_quotes)
closing_quotes = np.array(closing_quotes)

# Обчислення різниці між двома видами котирувань
quotes_diff = closing_quotes - opening_quotes

# Фільтрація пропущених значень
quotes_diff = quotes_diff[:, ~np.isnan(quotes_diff).any(axis=0)]
X = quotes_diff.T  # Перетворення на 2-вимірний масив

# Уникнення поділу на 0
std_deviation = X.std(axis=0)
std_deviation[std_deviation == 0] = 1
X /= std_deviation

# Створення моделі графа
edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
with np.errstate(invalid="ignore"):
    edge_model.fit(X)
_, labels = cluster.affinity_propagation(edge_model.covariance_)

# Вивід кластерів
valid_names = [company_symbols_map[symbol] for symbol in valid_symbols]
for i in range(max(labels) + 1):
    cluster_names = ", ".join(np.array(valid_names)[np.array(labels) == i])
    print(f"Cluster {i + 1} ==> {cluster_names}")
