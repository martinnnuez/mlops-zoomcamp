import pickle
import pandas as pd
import numpy as np
import sys
from dateutil.relativedelta import relativedelta
from datetime import datetime

from prefect import task, flow, get_run_logger


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def read_data(filename):
    
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    return df


def prepare_dicts(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    return dicts



def save_results(df, y_pred, output_file):

    
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
    )



def get_paths(run_date, taxi_type):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month 

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'{taxi_type}_{year:04d}-{month:02d}.parquet'

    return input_file, output_file, year, month




@task
def apply_model(input_file, output_file, year, month):
    logger = get_run_logger()

    logger.info(f'reading the data from {input_file}...')
    df = read_data(input_file)
    dicts = prepare_dicts(df)

    logger.info(f'applying the model...')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(f'Mean Predicted Duration for {year:04d}-{month:02d}: {np.mean(y_pred)}')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    logger.info(f'saving the result to {output_file}...')
    save_results(df, y_pred, output_file)

    return output_file


@flow
def ride_duration_prediction(taxi_type: str, run_date: datetime):
    
    input_file, output_file, year, month = get_paths(run_date, taxi_type)
    print(f'INPUT FILE = {input_file}')

    apply_model(
        input_file=input_file,
        output_file=output_file,
        year=year,
        month=month
    )



def run():
    taxi_type = sys.argv[1] #'yellow'
    year = int(sys.argv[2]) #2022
    month = int(sys.argv[3]) #2

    ride_duration_prediction(
        taxi_type=taxi_type,
        run_date=datetime(year=year, month=month, day=1)
    )




if __name__ == '__main__':
    run()




