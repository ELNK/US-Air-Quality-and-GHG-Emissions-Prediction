def clean_facility_columns(df):
    df.rename(columns=lambda x: x[25:], inplace=True)
    return df

def clean_gas_columns(df):
    df.rename(columns=lambda x: x[len('V_GHG_EMITTER_GAS.'):], inplace=True)
    return df
