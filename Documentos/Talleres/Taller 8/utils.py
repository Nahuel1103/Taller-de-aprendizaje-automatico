from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from datetime import datetime
import calendar
import tensorflow as tf


# Definir las columnas categóricas y numéricas

def FilledIn(df):
    df_aux = df.copy()
    df_out = pd.DataFrame(columns=df_aux.columns)
    df_aux['datetime'] = pd.to_datetime(df_aux['datetime'])
    df_aux = df_aux.set_index('datetime')

    for year in [2011, 2012]:
      for month in range(12):
        start_date = datetime(year, month+1, 1, 0, 0, 0)
        last_day_of_month = calendar.monthrange(year, month+1)[1]
        end_date = datetime(year, month+1, last_day_of_month, 23, 0, 0)
        # Se agregan las marcas de tiempo que faltan
        df_month = df_aux[start_date:end_date]
        df_month = df_month.resample('H').asfreq()
        
        # Rellenar los datos faltantes===========

        categorical_cols = ['season', 'holiday', 'workingday', 'weather']
        numeric_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']

        df_month[numeric_cols] = df_month[numeric_cols].interpolate(method='linear')

        df_month[categorical_cols] = df_month[categorical_cols].fillna(method='ffill')

        #========================================
        df_month = df_month.reset_index()

        df_out = pd.concat([df_out,df_month])
    df_out = df_out.reset_index(drop=True)
    return df_out


class TimeFeatures(BaseEstimator, TransformerMixin):
  def __init__(self):
    self
  def fit(self, X, y=None):
    # X debe ser un DataFrame
    return self
  def transform(self, X):
    X_aux = X.copy()
    X_aux['datetime'] = pd.to_datetime(X_aux['datetime'])
    X_aux['month'] = X_aux['datetime'].dt.month
    X_aux['weekday'] = X_aux['datetime'].dt.weekday
    X_aux['hour'] = X_aux['datetime'].dt.hour
    X_aux = X_aux.drop('datetime', axis=1)
    return X_aux

def create_sequence_datasets(X_train, y_train, X_val, y_val, dates_train, dates_val, sequence_length=24, batch_size=8):
    # Crear DataFrames para train y val
    train_df = pd.DataFrame(X_train, index=pd.to_datetime(dates_train))
    val_df = pd.DataFrame(X_val, index=pd.to_datetime(dates_val))
    
    # Asegurarse de que y_train y y_val sean Series
    y_train = y_train['count'] if isinstance(y_train, pd.DataFrame) else y_train
    y_val = y_val['count'] if isinstance(y_val, pd.DataFrame) else y_val
    
    y_train = pd.Series(y_train.values.ravel(), index=pd.to_datetime(dates_train))
    y_val = pd.Series(y_val.values.ravel(), index=pd.to_datetime(dates_val))
    
    X_train_seq, y_train_seq = [], []
    X_val_seq, y_val_seq = [], []

    for year in [2011, 2012]:
        for month in range(1, 13):
            start_date = datetime(year, month, 1)
            last_day_of_month = calendar.monthrange(year, month)[1]
            end_date = datetime(year, month, last_day_of_month, 23, 59, 59)
            
            # Crear df temporal para train y val
            df_month_train = train_df[start_date:end_date]
            df_month_val = val_df[start_date:end_date]
            
            # Obtener etiquetas para train y val
            y_month_train = y_train[start_date:end_date]
            y_month_val = y_val[start_date:end_date]
            
            if not df_month_train.empty and not y_month_train.empty:
                # Crear secuencias para train
                for i in range(len(df_month_train) - sequence_length):
                    X_train_seq.append(df_month_train.iloc[i:i+sequence_length].values)
                    y_train_seq.append(y_month_train.iloc[i+1:i+1+sequence_length].values)
            
            if not df_month_val.empty and not y_month_val.empty:
                # Crear secuencias para val
                for i in range(len(df_month_val) - sequence_length):
                    X_val_seq.append(df_month_val.iloc[i:i+sequence_length].values)
                    y_val_seq.append(y_month_val.iloc[i+1:i+1+sequence_length].values)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_seq, y_val_seq)).batch(batch_size)
    
    return train_dataset, val_dataset