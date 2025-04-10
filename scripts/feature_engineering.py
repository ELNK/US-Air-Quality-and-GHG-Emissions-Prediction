import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_gas(df):
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(df[['GAS_NAME']])
    encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(['GAS_NAME']))
    return pd.concat([df, encoded_df], axis=1)

def aggregate_by_county(df, encoded_columns):
    return df.groupby('COUNTY_FIPS')[encoded_columns].sum()

def compute_aqi_percent(df):
    df['Unhealthy day total'] = df[
        ['Unhealthy for Sensitive Groups Days', 'Unhealthy Days', 'Very Unhealthy Days']
    ].sum(axis=1)
    for col in ['Good Days', 'Moderate Days', 'Unhealthy day total']:
        df[col] /= df['Days with AQI']
    return df.drop(columns=['Unhealthy for Sensitive Groups Days', 'Unhealthy Days', 'Very Unhealthy Days', 'Hazardous Days'])
