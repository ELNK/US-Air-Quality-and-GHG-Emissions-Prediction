# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
import os

# --- Create output folders if not exist ---
os.makedirs('../data/cleaned', exist_ok=True)
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../figures', exist_ok=True)

# --- Load Raw Data ---
facilities_df = pd.read_csv('../data/raw/us_greenhouse_gas_emissions_direct_emitter_facilities.csv')
gas_type_df = pd.read_csv('../data/raw/us_greenhouse_gas_emission_direct_emitter_gas_type.csv')
aqi_2020_df = pd.read_csv('../data/raw/annual_aqi_by_county_2020.csv')

# --- Clean Column Titles ---
facilities_df.rename(columns={title: title[25:] for title in facilities_df.columns}, inplace=True)
gas_type_df.rename(columns={title: title[len('V_GHG_EMITTER_GAS.'):] for title in gas_type_df.columns}, inplace=True)

# Save cleaned data
facilities_df.to_csv('../data/cleaned/facilities.csv', index=False)
gas_type_df.to_csv('../data/cleaned/gas_types.csv', index=False)

# --- Unique Gas Names ---
gas_names = gas_type_df['GAS_NAME'].unique()
print("Unique gas names:", gas_names)

# --- Create Facility Density Map ---
latitude, longitude, zoom_level = 39.8283, -98.5795, 4
map_ = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
marker_cluster = MarkerCluster().add_to(map_)
facilities_df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

for _, row in facilities_df.iterrows():
    folium.Marker(location=[row['LATITUDE'], row['LONGITUDE']]).add_to(marker_cluster)

map_.save('../figures/facility_map.html')

# --- One-Hot Encode Gas Types ---
encoder = OneHotEncoder()
encoder.fit(gas_type_df[['GAS_NAME']])
encoded_gas_names = encoder.transform(gas_type_df[['GAS_NAME']])
gas_names_df = pd.DataFrame(encoded_gas_names.toarray(), columns=encoder.get_feature_names_out(['GAS_NAME']))
gas_type_encoded_df = pd.concat([gas_type_df, gas_names_df], axis=1)

gas_type_count = gas_type_encoded_df.groupby('COUNTY_FIPS')[gas_names_df.columns].sum()
gas_type_count.to_csv('../data/processed/gas_type_count.csv')

# --- Facility Counts by County ---
facilities_county_count = facilities_df.groupby('COUNTY_FIPS').agg({
    'COUNTY': 'first',
    'STATE': 'first',
    'FACILITY_ID': 'count'
}).reset_index()
facilities_county_count.to_csv('../data/processed/facilities_count.csv', index=False)

# --- Merge County Facility Count with Gas Type Count ---
county_facilities_count = pd.merge(facilities_county_count, gas_type_count, on='COUNTY_FIPS')
county_facilities_count.rename(columns={'FACILITY_ID': 'TOTAL_FACILITIES'}, inplace=True)

# --- AQI Percentage Processing ---
aqi_2020_grouped = aqi_2020_df.groupby(['State', 'County']).apply(lambda x: (x['Good Days']/x['Days with AQI']).mean()*100).reset_index()
aqi_2020_grouped.rename(columns={0:'PERCENTAGE_GOOD_DAYS'}, inplace=True)

county_mapping_df = pd.read_csv('../data/county_mapping.csv')
county_mapping_df.rename(columns={'StateName': 'State', 'CountyName': 'County'}, inplace=True)

aqi_2020_mapped_df = aqi_2020_grouped.merge(county_mapping_df, on=['State', 'County'], how='left')
aqi_2020_mapped_df.rename(columns={'CountyFIPS': 'COUNTY_FIPS'}, inplace=True)
aqi_2020_mapped_df = aqi_2020_mapped_df[['State', 'County', 'COUNTY_FIPS', 'PERCENTAGE_GOOD_DAYS']]

# --- Final Merged Data ---
polished_df = aqi_2020_mapped_df.merge(county_facilities_count, on='COUNTY_FIPS', how='left')
polished_df.drop(['COUNTY', 'STATE'], axis=1, inplace=True)
polished_df.dropna(subset=['TOTAL_FACILITIES'], inplace=True)
polished_df.to_csv('../data/processed/polished_merged_data.csv', index=False)

# --- Correlation Heatmap ---
corr_matrix = polished_df.corr()
plt.figure(figsize=(12,10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, square=True, linewidths=.5)
plt.title('Correlation Heatmap of Good Days Percentage vs Gas Emitter')
plt.savefig('../figures/correlation_heatmap.png')
plt.close()

# --- Modeling Prep ---
def remove_outliers(data, variable, lower=np.inf, upper=np.inf):
    return data[(data[variable]>lower) & (data[variable]<=upper)]

def remove_extreme(data, variable, indicator):
    return data[data[variable]/data['Days with AQI'] != indicator]

aqi_df_md = remove_outliers(aqi_2020_df, 'Days with AQI', lower=80)
aqi_df_md = remove_outliers(aqi_df_md, 'Days Ozone', lower=1)
aqi_df_md = remove_extreme(aqi_df_md, 'Good Days',1)

aqi_df_md['Unhealthy day total'] = aqi_df_md['Unhealthy for Sensitive Groups Days'] + aqi_df_md['Unhealthy Days'] + aqi_df_md['Very Unhealthy Days']
aqi_df_md['Good Days'] /= aqi_df_md['Days with AQI']
aqi_df_md['Moderate Days'] /= aqi_df_md['Days with AQI']
aqi_df_md['Unhealthy day total'] /= aqi_df_md['Days with AQI']
aqi_df_md.drop(columns=['Unhealthy for Sensitive Groups Days','Unhealthy Days','Very Unhealthy Days','Hazardous Days'], inplace=True)
aqi_df_md.to_csv('../data/processed/aqi_processed.csv', index=False)

# --- Modeling Functions ---
def rmse(pred, actual): return np.sqrt(np.mean((actual - pred)**2))

def train_test_split(data, train_size=0.8):
    indices = np.random.permutation(len(data))
    split = int(train_size * len(data))
    return data.iloc[indices[:split]], data.iloc[indices[split:]]

def get_X_y(data, X_cols, y_col):
    return data[X_cols], data[y_col]

def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_ann(X, y, max_epochs=100):
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=max_epochs)
    model.fit(X, y)
    return model

def plot_residuals(y_true, y_pred, weights=None):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Residuals (Proportion)')
    plt.scatter(y_true, y_pred - y_true, s=5)
    plt.axhline(0, color='k', linestyle='--')

    if weights is not None:
        plt.subplot(122)
        plt.title('Residuals (Weighted)')
        plt.scatter(y_true * weights, (y_pred - y_true) * weights, s=5)
        plt.axhline(0, color='k', linestyle='--')

    plt.tight_layout()
    plt.savefig('../figures/residual_plot.png')
    plt.close()

def AQI_model(data, X_cols, y_col, model_type='linear_regression'):
    train, test = train_test_split(data)
    X_train, y_train = get_X_y(train, X_cols, y_col)
    X_test, y_test = get_X_y(test, X_cols, y_col)

    model = train_linear_regression(X_train, y_train) if model_type == 'linear_regression' else train_ann(X_train, y_train, max_epochs=500)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"Model: {y_col} ({model_type})")
    print(f"Train RMSE: {rmse(y_train_pred, y_train):.4f}")
    print(f"Test RMSE: {rmse(y_test_pred, y_test):.4f}")
    if hasattr(model, 'intercept_'):
        print("Intercept:", model.intercept_)
    if hasattr(model, 'coef_'):
        print("Coefficients:", model.coef_)

    plot_residuals(y_test, y_test_pred, test['Days with AQI'] if 'Days with AQI' in test.columns else None)
    return model

# --- Run Models ---
X_cols_aqi = ['Max AQI', '90th Percentile AQI', 'Median AQI', 'Days CO', 'Days NO2', 'Days Ozone', 'Days SO2', 'Days PM2.5', 'Days PM10']
AQI_model(aqi_df_md, X_cols_aqi, 'Good Days')
AQI_model(aqi_df_md, X_cols_aqi, 'Moderate Days')
AQI_model(aqi_df_md, X_cols_aqi, 'Unhealthy day total')
AQI_model(aqi_df_md, X_cols_aqi, 'Good Days', model_type='ann')

X_cols_polished = ['TOTAL_FACILITIES'] + list(gas_type_count.columns)
polished_df['PERCENTAGE_GOOD_DAYS'] /= 100

AQI_model(polished_df, X_cols_polished, 'PERCENTAGE_GOOD_DAYS')
AQI_model(polished_df, X_cols_polished, 'PERCENTAGE_GOOD_DAYS', model_type='ann')
