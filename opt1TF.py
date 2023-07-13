import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

train_data = pd.read_csv('/kaggle/input/install-future-program-istanbul-hackathon/train.csv')
test_data = pd.read_csv('/kaggle/input/install-future-program-istanbul-hackathon/test.csv')

encoder = LabelEncoder()
train_data['SERVER'] = encoder.fit_transform(train_data['SERVER'])
test_data['SERVER'] = encoder.transform(test_data['SERVER'])

train_data['DATETIME'] = pd.to_datetime(train_data['DATETIME'])
train_data['HOUR'] = train_data['DATETIME'].dt.hour
train_data['MINUTE'] = train_data['DATETIME'].dt.minute
train_data['WEEKEND'] = train_data['DATETIME'].dt.weekday
train_data['DAY_HOUR'] = train_data['HOUR'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

test_data['DATETIME'] = pd.to_datetime(test_data['DATETIME'])
test_data['HOUR'] = test_data['DATETIME'].dt.hour
test_data['MINUTE'] = test_data['DATETIME'].dt.minute
test_data['WEEKEND'] = test_data['DATETIME'].dt.weekday
test_data['DAY_HOUR'] = test_data['HOUR'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

X = train_data[['SERVER', 'HOUR', 'MINUTE', 'WEEKEND', 'DAY_HOUR']].values
y = train_data['CPULOAD'].values

scaler = MinMaxScaler()

X = scaler.fit_transform(X)
Q1 = train_data['CPULOAD'].quantile(0.25)
Q3 = train_data['CPULOAD'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
anomalies = train_data[(train_data['CPULOAD'] < lower_bound) | (train_data['CPULOAD'] > upper_bound)]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(5,), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])
X_test = scaler.transform(test_data[['SERVER', 'HOUR', 'MINUTE', 'WEEKEND', 'DAY_HOUR']].values)
y_test_pred = model.predict(X_test)

predictions = pd.DataFrame({'DATETIME': test_data['DATETIME'], 'SERVER': test_data['SERVER'], 'CPULOAD': y_test_pred.flatten()})
predictions.to_csv('sample_submissionTF2.csv', index=False)

train_data['DATETIME'] = pd.to_datetime(train_data['DATETIME'])
test_data['DATETIME'] = pd.to_datetime(test_data['DATETIME'])
train_data['WEEKDAY'] = train_data['DATETIME'].dt.dayofweek
weekday_data = train_data[train_data['WEEKDAY'] < 5]  
weekend_data = train_data[train_data['WEEKDAY'] >= 5]
day_hour_data = train_data[(train_data['DATETIME'].dt.hour >= 6) & (train_data['DATETIME'].dt.hour <= 18)]
night_hour_data = train_data[(train_data['DATETIME'].dt.hour < 6) | (train_data['DATETIME'].dt.hour > 18)]
plt.figure(figsize=(10, 6))
plt.plot(train_data['DATETIME'], train_data['CPULOAD'], label='All Data')
plt.plot(weekday_data['DATETIME'], weekday_data['CPULOAD'], label='Weekdays')
plt.plot(weekend_data['DATETIME'], weekend_data['CPULOAD'], label='Weekends')
plt.plot(day_hour_data['DATETIME'], day_hour_data['CPULOAD'], label='Day Hours')
plt.plot(night_hour_data['DATETIME'], night_hour_data['CPULOAD'], label='Night Hours')
plt.xlabel('Datetime')
plt.ylabel('CPULOAD')
plt.title('CPULOAD - Time Analysis')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(train_data['DATETIME'], train_data['CPULOAD'], label='CPULOAD')
plt.scatter(anomalies['DATETIME'], anomalies['CPULOAD'], color='red', label='Anomalies')
plt.xlabel('Datetime')
plt.ylabel('CPULOAD')
plt.title('CPULOAD - Anomaly Detection (IQR)')
plt.legend()
plt.grid(True)
plt.show()