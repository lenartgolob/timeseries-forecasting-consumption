#%% md
# Setup:
# -branje
# -preoblikovanje
# -sum na daily
#%%

#%%
import warnings
import  pandas as pd
warnings.filterwarnings('ignore')

file_path = 'LJ1_23_3-5423-15minMeritve2023-01-01-2023-12-31.xlsx'

init = pd.read_excel(file_path, sheet_name='3-5423')

init = init[['Časovna značka','Energija A+']]

init['Časovna značka'] = pd.to_datetime(init['Časovna značka'], format='mixed', dayfirst=True)

init.rename(columns={'Časovna značka': 'ds', 'Energija A+': 'y'}, inplace=True)


init

#%%
def aggregate_to_daily(init):
  #prepare for prophet
  df_1d = init.copy()
  df_1d = df_1d.rename(columns={'y': 'y15'})

  df_1d['y'] = df_1d['y15'].rolling(window=96, min_periods=1).sum()[95::96]

  df_1d = df_1d.drop(columns=['y15'])
  df_1d = df_1d.dropna(subset=['y'])
  df_1d=df_1d.reset_index(drop=True)
  #print(df_1d.head(15))

  return df_1d

df_1d = aggregate_to_daily(init)
df_1d
#%% md
# TRENIRANJE
#
#%%
from prophet import Prophet
m = Prophet(yearly_seasonality=True)

import time
t1 = time.time()
m.fit(df_1d)
t2 = time.time()
print("Time taken: ",t2-t1)

#%% md
# Prediction
#%%
future = m.make_future_dataframe(periods=255,freq='D')
t1 = time.time()
prediction = m.predict(future)
t2 = time.time()
prediction = prediction[['ds','yhat']]
print("Time taken: ",t2-t1)
prediction
#%% md
# Calculating metrics
# 
#%%
file_path = 'LJ1_23_3-5423-15minMeritve2023-01-01-2024-09-12.xlsx'

actual = pd.read_excel(file_path, sheet_name='3-5423')

actual = actual[['Časovna značka','Energija A+']]

actual['Časovna značka'] = pd.to_datetime(actual['Časovna značka'], format='mixed', dayfirst=True)

actual.rename(columns={'Časovna značka': 'ds', 'Energija A+': 'y'}, inplace=True)

actual = aggregate_to_daily(actual)
#%%
united = pd.concat([prediction['yhat'], actual['y']], axis=1)
united.columns = ['Napoved', 'Dejanska']
united
#%% md
# MAE
#%%
mae = (united['Dejanska'] - united['Napoved']).abs().mean()
print("MAE:", mae)
#%% md
# RMSE
#%%
rmse = ((united['Dejanska'] - united['Napoved']) ** 2).mean() ** 0.5
print("RMSE:", rmse)

#%% md
# 
#%% md
# MAPE
#%%
mape = ((united['Dejanska'] - united['Napoved']).abs() / united['Dejanska'].abs()).mean() * 100
print("MAPE:", mape, "%")
#%% md
# SMAPE
#%%
numerator = (united['Napoved'] - united['Dejanska']).abs()
denominator = (united['Napoved'].abs() + united['Dejanska'].abs()) / 2
smape = (numerator / denominator).mean() * 100
print("sMAPE:", smape, "%")

#%% md
# Rolling forecast origin
#%%
from prophet import Prophet
import numpy as np

def rolling_forecast(df, train_months, test_month, year=2023):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])

    results = []


    start_date = pd.Timestamp(f"{year}-{test_month:02d}-01")
    end_date = (start_date + pd.offsets.MonthEnd(0))

    current_date = start_date

    while current_date <= end_date:
        
        train_start = current_date - pd.DateOffset(months=train_months)
        train_end = current_date - pd.Timedelta(days=1)

        train_df = df[(df['ds'] >= train_start) & (df['ds'] <= train_end)]
        test_df = df[df['ds'] == current_date]

        if len(train_df) < 2 or test_df.empty:
            current_date += pd.Timedelta(days=1)
            continue

        model = Prophet(yearly_seasonality=True)
        model.fit(train_df)

        future = test_df[['ds']]
        forecast = model.predict(future)

        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

        results.append({
            'date': current_date.date(),
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'sMAPE': smape
        })

        current_date += pd.Timedelta(days=1)

    return pd.DataFrame(results)

#%%
# npr. treniraj 11 mesecev, testiraj 12. mesec
res_11m = rolling_forecast(df_1d, train_months=11, test_month=12, year=2023)
print(res_11m)
print("MAE:", res_11m['MAE'].mean())
print("RMSE:", res_11m['RMSE'].mean())
print(" MAPE:", res_11m['MAPE'].mean())
print("sMAPE:", res_11m['sMAPE'].mean())
