import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

# Function to convert to superscript
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

# GPU Initialization
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(f"Error initializing GPUs: {e}")
else:
    print("No GPU detected. Running on CPU.")

# Constants
P0 = 1013.25  # Sea level standard atmospheric pressure in hPa
HersheyAltitude = 411  # Initial reference altitude in feet

# Load Detector 2 Data
detector_2_file_path = 'Time vs Pressure Detector 2.xlsx'  # Replace with your file path
detector_2_data = pd.read_excel(detector_2_file_path)

# Filter and clean data
detector_2_data = detector_2_data[pd.to_numeric(detector_2_data['Hour'], errors='coerce').notnull()]
detector_2_data = detector_2_data[pd.to_numeric(detector_2_data['Min'], errors='coerce').notnull()]
detector_2_data = detector_2_data[pd.to_numeric(detector_2_data['Sec'], errors='coerce').notnull()]
detector_2_data = detector_2_data[pd.to_numeric(detector_2_data['Pressure'], errors='coerce').notnull()]

start_time = "09:30"
end_time = "13:30"

# Format and calculate altitude for Detector 2
detector_2_data['Time'] = pd.to_datetime(detector_2_data['Hour'].astype(int).apply(lambda x: f"{x:02d}") + ':' +
                                         detector_2_data['Min'].astype(int).apply(lambda x: f"{x:02d}") + ':' +
                                         detector_2_data['Sec'].astype(int).apply(lambda x: f"{x:02d}"),
                                         format='%H:%M:%S')
detector_2_data['Altitude_m'] = (44330 / 2) * (1 - (detector_2_data['Pressure'] / P0) ** (1 / 5.255))
detector_2_data['Altitude_ft'] = detector_2_data['Altitude_m'] * 3.28084 - min(detector_2_data['Altitude_m'] * 3.28084) + HersheyAltitude

# Detect takeoff and landing points using altitude differences
altitude_diff = detector_2_data['Altitude_ft'].diff()

# Takeoff time: First significant increase in altitude
takeoff_index = altitude_diff.gt(3).idxmax()  # Adjusted threshold for "sudden increase"
takeoff_time = detector_2_data.loc[takeoff_index, 'Time']

# Peak time: Point with maximum altitude
peak_index = detector_2_data['Altitude_ft'].idxmax()
peak_time = detector_2_data.loc[peak_index, 'Time']

# Landing time: Last significant decrease in altitude to near ground level
landing_index = altitude_diff.lt(-3).iloc[::-1].idxmax()  # Adjusted threshold for "significant decrease"
landing_time = detector_2_data.loc[landing_index, 'Time']

# Load Hess_007 Data
hess_007_file_path = 'Hess_007.xlsx'  # Replace with your file path
hess_data = pd.read_excel(hess_007_file_path)

# Format and calculate altitude for Hess_007
hess_data['TimeDelta'] = pd.to_timedelta(hess_data['Time[s]'], unit='s')
hess_data['Altitude_m'] = (44330 / 2) * (1 - (hess_data['Pressure[hPa]'] / P0) ** (1 / 5.255))
hess_data['Altitude_ft'] = hess_data['Altitude_m'] * 3.28084 - min(hess_data['Altitude_m'] * 3.28084) + HersheyAltitude

# Adjust Hess_007 Time based on peak time synchronization
hess_peak_time = hess_data.loc[hess_data['Altitude_ft'].idxmax(), 'TimeDelta']
hess_data['Time'] = peak_time + (hess_data['TimeDelta'] - hess_peak_time)

# Calculate min and max for altitude and pressure
min_altitude = min(detector_2_data['Altitude_ft'].min(), hess_data['Altitude_ft'].min())
max_altitude = max(detector_2_data['Altitude_ft'].max(), hess_data['Altitude_ft'].max())
min_pressure = min(detector_2_data['Pressure'].min(), hess_data['Pressure[hPa]'].min())
max_pressure = max(detector_2_data['Pressure'].max(), hess_data['Pressure[hPa]'].max())

# Calculate Events per Second for Hess_007 Data using the synchronized time
hess_data['Second'] = hess_data['Time'].dt.floor('S')  # Round to the nearest second
events_per_second = hess_data.resample('S', on='Time').count()['Event']

# Identify events at takeoff, peak, and landing times
takeoff_events = events_per_second.get(takeoff_time.floor('S'), 0)
peak_events = events_per_second.get(peak_time.floor('S'), 0)
landing_events = events_per_second.iloc[-1]

# Split events_per_second into before and after the peak time
events_before_peak = events_per_second[events_per_second.index < peak_time]
events_after_peak = events_per_second[events_per_second.index > peak_time]

# Group data by the specified interval (per second)
hess_data['Interval'] = hess_data['Time'].dt.floor('S')  # Round to the nearest second
events_per_interval = hess_data.groupby('Interval').size()  # Count events per interval

# Prepare data for neural network
events_data = pd.DataFrame({
    'Time_numeric': (events_per_second.index - events_per_second.index.min()).total_seconds(),
    'Events_per_sec': events_per_second.values
}).dropna()

X_events = events_data[['Time_numeric']]
y_events = events_data['Events_per_sec']

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_events_scaled = scaler_X.fit_transform(X_events)
y_events_scaled = scaler_y.fit_transform(y_events.values.reshape(-1, 1))

# Split data into training and testing
X_train_full, X_test, y_train_full, y_test = train_test_split(X_events_scaled, y_events_scaled, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# Define the best configuration
layers = [1024, 512, 256]  # Layer sizes
dropout = 0.2              # Dropout rate
learning_rate = 0.0001     # Learning rate
batch_size = 8             # Batch size
epochs = 100               # At least 100 epochs as per your request

print(f"Using Configuration: Layers={layers}, Dropout={dropout}, LR={learning_rate}, Batch Size={batch_size}")

# Build the model
model = Sequential([Input(shape=(X_train.shape[1],))])
for neurons in layers:
    model.add(Dense(neurons, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

# Define early stopping
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_valid, y_valid),
                    callbacks=[early_stopping_cb], verbose=0)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss={loss:.4f}, MAE={mae:.4f}")

# Save the model
model_save_path = 'best_model.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")