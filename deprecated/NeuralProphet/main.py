from neuralprophet import NeuralProphet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")

           # adjust
DT_FORMAT     = None                    # e.g. "%d.%m.%Y %H:%M:%S" or just None
FREQ          = "15min"                 # <- NeuralProphet frequency string
FORECAST_HORIZON = 96 




class Prediction:
    def __init__(self,data_name,prediction_collumn_name,timestamp_collumn_name):
        df=pd.read_csv(f"{data_name}")
        self.prediction_collumn_name=prediction_collumn_name
        self.timestamp_collumn_name=timestamp_collumn_name
        self.graph=False
        self.rmse_mean=0
        
        df.rename(columns={f"{self.timestamp_collumn_name}":"ds",f"{self.prediction_collumn_name}":"y"},inplace=True)
        df['ds'] = pd.to_datetime(df['ds'], format="%d. %m. %Y %H:%M:%S")
        
        # Then fill missing values explicitly
        df['y'] = df['y'].interpolate()  # or .fillna(method='ffill')
        df.sort_values("ds", inplace=True)
        df = df[['ds', 'y']]
        df=self.add_flags(df)  #add weekend days weekdays and seasons
        self.df=df



    def add_flags(self, _df: pd.DataFrame) -> pd.DataFrame:
        d = _df.copy()
        # season flags
        d["spring"] = d["ds"].dt.month.isin([3, 4, 5]).astype(int)
        d["summer"] = d["ds"].dt.month.isin([6, 7, 8]).astype(int)
        d["fall"]   = d["ds"].dt.month.isin([9,10,11]).astype(int)  # autumn
        d["winter"] = d["ds"].dt.month.isin([12,1,2]).astype(int)
        # weekday / weekend flags
        d["weekday"] = (d["ds"].dt.dayofweek < 5).astype(int)
        d["weekend"] = (d["ds"].dt.dayofweek >= 5).astype(int)
        return d
  
    def make_model(self,n_lags,n_forecasts):   #creating neural prophet model
        self.m = NeuralProphet(
            n_lags       = n_lags,        
            n_forecasts  = n_forecasts,
            yearly_seasonality = False, 
            weekly_seasonality = False,
            daily_seasonality  = False,
            epochs=40
        )
        # seasonality per season (weekly because period=7 days)
        for s in ["spring", "summer", "fall", "winter"]:
            self.m.add_seasonality(name=f"{s}_weekly", period=7, fourier_order=15,
                            condition_name=s)
        # weekday vs weekend modifiers (also weekly period)
        self.m.add_seasonality(name="weekday_effect", period=7, fourier_order=15,
                        condition_name="weekday")
        self.m.add_seasonality(name="weekend_effect", period=7, fourier_order=15,
                        condition_name="weekend")
        
        
    def day_rolling_prediction(self, end_day):
      
        self.daily_rmse = []
        train_df = self.df[(self.df["ds"] < end_day)] #data before the day
        test_df = self.df[(self.df["ds"] >= end_day) & (self.df["ds"] < end_day + pd.Timedelta(days=1))]
        self.make_model(n_lags=96,n_forecasts=96)
        fit_start = time.time()
        self.m.fit(train_df, freq=FREQ) ##train model
        fit_end = time.time()
        future = self.m.make_future_dataframe(train_df, periods=FORECAST_HORIZON, n_historic_predictions=False)
        future = self.add_flags(future)
        print(f"future tail{future.tail()}")
        predict_start = time.time()
        fcst = self.m.predict(future)
        predict_end = time.time()
        fcst_day = self.prepare_forecast_data(fcst, day=end_day,n_days=1)
        merged = fcst_day.merge(test_df[["ds", "y"]], on="ds")
        print(f"merged tail{merged.tail()}")
        metrics = self.calculate_metrics(merged=merged)        
        self.plot_graph(actual_day=test_df, forecast_day=fcst_day)
        print(f"Fit time: {fit_end - fit_start:.2f} seconds, Prediction time: {predict_end - predict_start:.2f} seconds")
        return metrics


    def plot_graph(self, actual_day, forecast_day):
        plt.plot(actual_day['ds'], actual_day['y'], label='Actual', linewidth=2)
        plt.plot(forecast_day['ds'], forecast_day['prediction'], label='Forecast', linestyle='--', linewidth=2)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Forecast vs Actual on ")
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.legend()

    def prepare_forecast_data(self,forecast,day,n_days):
        yhat_cols = [f"yhat{i}" for i in range(1, (96)*n_days+1)]
        df = forecast[['ds'] + yhat_cols].copy()
        df['ds'] = pd.to_datetime(df['ds'])

        df=df[df['ds']>=day]                     
        df["prediction"]=0
        i=1
        for idx,row in df.iterrows():
            df.at[idx, 'prediction'] = row[f"yhat{i}"]
            i=i+1

        df_final = df[["ds", "prediction"]]
        return df_final
    

    def calculate_metrics(self,merged): #return metrics in order [mae,rmse,mape,smape]
        rmse   = np.sqrt(np.mean((merged["y"] - merged["prediction"]) ** 2))
        print(f"rmse:{rmse}")
        mean_actual = np.mean(merged['y'])
        print(f"mean_actual:{mean_actual}")
        rmse_percent = (rmse / mean_actual) * 100
        mae=np.mean(merged["y"] - merged["prediction"])
        mae_percent=mae/mean_actual * 100
        mape=np.mean((merged["y"] - merged["prediction"])/merged['y']) * 100
        smape=np.mean((merged["y"] - merged["prediction"])/(np.mean(merged["y"] + merged["prediction"])*2)) * 100
        print(f"mae:{mae_percent},rmse:{rmse_percent},mape:{mape},smape:{smape}")
        return [mae_percent,rmse_percent,mape,smape]
        


    def month_prediction(self, month, number_of_days):
        day = month
        metrics_list = []
        days_list = []
        while day != month + pd.Timedelta(days=number_of_days):
            metrics = self.day_rolling_prediction(end_day=day)
            metrics_list.append(metrics)
            days_list.append(day.strftime('%Y-%m-%d'))  # Store as string for clarity
            day = day + pd.Timedelta(days=1)
        self.monthly_metrics_df = pd.DataFrame(
            metrics_list,
            columns=["mae_percent", "rmse_percent", "mape_percent", "smape_percent"]
        )
        self.monthly_metrics_df.to_csv("monthly_metrics.csv", index=False)
        self.monthly_metrics_df.insert(0, "day", days_list)
        print(self.monthly_metrics_df)
        plt.show()

    def calculate_average_metrics(self):
        df = pd.read_csv("monthly_metrics.csv")
        # Replace inf and -inf with 0
        df = df.replace([np.inf, -np.inf], 0)
        # Take absolute values for all metric columns
        metric_cols = ['mae_percent', 'rmse_percent', 'mape_percent', 'smape_percent']
        df[metric_cols] = df[metric_cols].abs()
        stats = pd.DataFrame({
            'std': df[metric_cols].std(),
            'median': df[metric_cols].median(),
            'min': df[metric_cols].min(),
            'max': df[metric_cols].max(),
            'quantile_25': df[metric_cols].quantile(0.25),
            'quantile_75': df[metric_cols].quantile(0.75)
        })
        stats = stats.T  # Transpose so rows are stats, columns are metrics
        stats.to_csv("monthly_metrics_summary.csv")
        print(stats)

    def split_data_month_prediction(self, month, number_of_days):
        """
        Forecasts for num_days after start_date in one prediction (not rolling).
        Plots the forecast and returns the forecast DataFrame.
        """
        self.daily_rmse = []
        train_df = self.df[(self.df["ds"] < month)] #data before the day
        test_df = self.df[(self.df["ds"] >= month) & (self.df["ds"] < month + pd.Timedelta(days=number_of_days))]
        self.make_model(n_lags=96,n_forecasts=96*30)
        fit_start = time.time()
        self.m.fit(train_df, freq=FREQ) ##train model
        fit_end = time.time()
        future = self.m.make_future_dataframe(train_df, periods=number_of_days*96, n_historic_predictions=False)
        future = self.add_flags(future)
        print(f"future tail{future.tail()}")
        predict_start = time.time()
        fcst = self.m.predict(future)
        print(f"fcst tail{fcst.tail()}")
        predict_end = time.time()
        fcst_data = self.prepare_forecast_data(fcst, day=month,n_days=number_of_days)
        merged = fcst_data.merge(test_df[["ds", "y"]], on="ds")
        print(f"merged tail{merged.tail()}")
        metrics = self.calculate_metrics(merged=merged)        
        self.plot_graph(actual_day=test_df, forecast_day=fcst_data)
        print(f"Fit time: {fit_end - fit_start:.2f} seconds, Prediction time: {predict_end - predict_start:.2f} seconds")
        plt.show()
        return metrics
        
        
if __name__ == "__main__":
    p = Prediction(
        data_name="data_15min_measurements.csv",
        prediction_collumn_name="Energija A+",
        timestamp_collumn_name="Časovna značka"
    )
    day = pd.Timestamp("2023-04-20 00:00:00")
    p.month_prediction(month=day, number_of_days=1)


