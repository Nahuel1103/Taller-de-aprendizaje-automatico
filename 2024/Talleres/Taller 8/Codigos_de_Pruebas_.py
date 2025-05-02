def Secuencia_Armada(data=None, y=None, df=None):
    
  df_aux = df.copy()
  df_out = pd.DataFrame(columns=df_aux.columns)
  df_aux['datetime'] = pd.to_datetime(df_aux['datetime'])
  df_aux = df_aux.set_index('datetime')

  for year in [2011, 2012]:
    for month in range(12):
      start_date = datetime(year, month+1, 1, 0, 0, 0)
      last_day_of_month = calendar.monthrange(year, month+1)[1]
      end_date = datetime(year, month+1, last_day_of_month, 23, 0, 0)
      df_month = df_aux[start_date:end_date]
      df_month = df_month.resample('H').asfreq()

      mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)

      indices = df.loc[mask].index.tolist()
      
      target_month = y[indices[0]+24:indices[-1]+2]

      secuencia_actual = tf.keras.preprocessing.timeseries_dataset_from_array(data=data[indices], targets=target_month,
                                                                              sequence_length=24, 
                                                                              sequence_stride=1, 
                                                                              sampling_rate=1,
                                                                              shuffle =True)
      df_month = df_month.reset_index()

      df_out = pd.concat([df_out,df_month])
  df_out = df_out.reset_index(drop=True)
  return df_out