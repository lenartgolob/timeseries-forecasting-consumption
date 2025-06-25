pip install requirements


Guide:
Create predictions:
    p = Prediction(
        data_name="data_15min_measurements.csv",
        prediction_collumn_name="P+ Prejeta delovna moč",
        timestamp_collumn_name="Časovna značka"
    )
Now you can make rolling forecast for the next day, by training model on all previous data using day_rolling_prediction(self,end_day) , and whole month predictions using month_prediction(self, month, number_of_days).

Whole metrics are exported to a csv file.