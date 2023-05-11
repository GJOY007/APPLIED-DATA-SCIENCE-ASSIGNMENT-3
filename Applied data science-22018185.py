import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv('API_19_DS2_en_csv_v2_5455435.csv')
data= df
data.info()

selected_columns = ['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969',
                    '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979',
                    '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989',
                    '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
                    '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                    '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
                    '2020', '2021', '2022']
# Create a new DataFrame with the selected columns
X = data[selected_columns]
X.fillna(0, inplace=True)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_normalized)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()

k = 4  # Choose the optimal number of clusters based on the elbow curve
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_normalized)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', label='Cluster Centers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clustering Results')
plt.legend()
plt.show()

Arab_World_data = data[(data['Country Name'] == 'Arab World') & (data['Indicator Name'] == 'CO2 emissions (kt)')]
year = [str(year) for year in range(1990, 2020)]
Arab_World_data = Arab_World_data[year].values.flatten()

# Perform exponential smoothing
model = ExponentialSmoothing(Arab_World_data, trend='add', seasonal='add', seasonal_periods=4)
fit_model = model.fit()

country = 'Arab World'

# Generate forecasts for years 2020-2041
forecast_years = [str(year) for year in range(2020, 2042)]
forecasts = fit_model.forecast(len(forecast_years))

# Plot the actual data, best fit curve, and forecasts
plt.plot(year, Arab_World_data, label='Actual')
plt.plot(year + forecast_years, fit_model.fittedvalues.tolist() + forecasts.tolist(), label='Exponential Smoothing')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.title(f'Exponential Smoothing for {country} CO2 Emissions')
plt.legend()
plt.show()

