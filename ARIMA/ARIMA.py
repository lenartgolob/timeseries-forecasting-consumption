#%% md
# # ARIMA
#%%

!pip install pmdarima

#%%
import pandas as pd
import warnings

file_path = 'LJ1_23_3-5423-15minMeritve2023-01-01-2023-12-31.xlsx'

init = pd.read_excel(file_path, sheet_name='3-5423')

init = init[['Časovna značka','Energija A+']]

init['Časovna značka'] = pd.to_datetime(init['Časovna značka'], format='mixed', dayfirst=True)

init.rename(columns={'Časovna značka': 'ds', 'Energija A+': 'y'}, inplace=True)


init

#%%
def aggregate_to_hourly(init):
  df_1h = init.copy()
  df_1h = df_1h.rename(columns={'y': 'y15'})

  df_1h['y'] = df_1h['y15'].rolling(window=4, min_periods=1).sum()[3::4]

  df_1h = df_1h.drop(columns=['y15'])
  df_1h = df_1h.dropna(subset=['y'])
  df_1h=df_1h.reset_index(drop=True)
  #print(df_1h.head(15))

  return df_1h
df_1h= aggregate_to_hourly(init)
df_1h
#%%
from statsmodels.tsa.arima.model import ARIMA
import time
warnings.filterwarnings('ignore')
df_1h= aggregate_to_hourly(init)


df_1h = df_1h.set_index('ds')

t1=time.time()
model = ARIMA(df_1h['y'], order=(4, 0, 0))
model_fit = model.fit()
t2=time.time()
print("Čas za trening: ",t2-t1)

# Napoved za 48 ur naprej (2 dni)
t1=time.time()
forecast = model_fit.get_forecast(steps=72)
forecast_df = forecast.summary_frame()
t2=time.time()
print("Čas za napoved: ",t2-t1)

forecast_df


#%% md
# # Rolling forecast
#%%
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np

def rolling_forecast_hourly(p,d,q,df, train_days, test_month, year=2023):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds')

    results = []

    start_date = pd.Timestamp(f"{year}-{test_month:02d}-01 00:00:00")
    end_date = (start_date + pd.offsets.MonthEnd(0)).replace(hour=23)

    current_time = start_date

    while current_time <= end_date:
        train_start = current_time - pd.Timedelta(days=train_days)
        train_end = current_time - pd.Timedelta(hours=1)

        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        test_df = df[df.index == current_time]

        if len(train_df) < 24 or test_df.empty:
            current_time += pd.Timedelta(hours=1)
            continue

        try:
            model = ARIMA(train_df['y'], order=(p, d, q))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=1)

            y_true = test_df['y'].values
            y_pred = forecast.values

            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

            results.append({
                'datetime': current_time,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'sMAPE': smape
            })
        except Exception as e:
            print(f"Napaka pri {current_time}: {e}")

        current_time += pd.Timedelta(hours=1)

    return pd.DataFrame(results)

#%%


res_11m = rolling_forecast_hourly(4,0,0,init, train_days=7, test_month=1, year=2023)
print("Povprečni MAE:", res_11m['MAE'].mean())
print("Povprečni RMSE:", res_11m['RMSE'].mean())
print("Povprečni MAPE:", res_11m['MAPE'].mean())
print("Povprečni sMAPE:", res_11m['sMAPE'].mean())
#%%

def najdi_najboljsi_arima(df, train_days, test_month, leto=2023, max_p=5, max_d=5, max_q=5):
    najboljse_metrike = {
        'p': None, 'd': None, 'q': None,
        'MAE': float('inf'),
        'RMSE': None,
        'MAPE': None,
        'sMAPE': None
    }

    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                print(f"Testiram ARIMA({p},{d},{q}) ...")
                try:
                    rezultati = rolling_forecast_hourly(p, d, q, df, train_days, test_month, leto)
                    povprecni_mae = rezultati['MAE'].mean()
                    if povprecni_mae < najboljse_metrike['MAE']:
                        najboljse_metrike.update({
                            'p': p, 'd': d, 'q': q,
                            'MAE': povprecni_mae,
                            'RMSE': rezultati['RMSE'].mean(),
                            'MAPE': rezultati['MAPE'].mean(),
                            'sMAPE': rezultati['sMAPE'].mean()
                        })
                except Exception as e:
                    print(f"Napaka za ({p},{d},{q}): {e}")
                    continue

    print("\nNajboljša kombinacija:")
    print(f"ARIMA({najboljse_metrike['p']},{najboljse_metrike['d']},{najboljse_metrike['q']})")
    print(f"MAE:   {najboljse_metrike['MAE']:.2f}")
    print(f"RMSE:  {najboljse_metrike['RMSE']:.2f}")
    print(f"MAPE:  {najboljse_metrike['MAPE']:.2f}%")
    print(f"sMAPE: {najboljse_metrike['sMAPE']:.2f}%")

    return najboljse_metrike

#%%
results = najdi_najboljsi_arima(init, train_days=7, test_month=1, leto=2023, max_p=4, max_d=1, max_q=4)
#za dane parametre traja 2h 30min+ več odsvetujem (CPU: AMD 5.4Ghz 6core 12 thread)
#Best params (p,d,q): 3,0,1