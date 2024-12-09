import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os

# Increase global font size for readability
plt.rcParams['font.size'] = 16

# ========================
# Constants & Functions
# ========================
P0 = 1013.25  # Sea level standard atmospheric pressure in hPa
HersheyAltitude = 411  # Initial reference altitude in feet
A_max = 66058  # Altitude of the Pfotzer maximum in feet (~20 km)
initial_params = [0.00005, 0.00003]

def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ৯ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def calculate_muon_flux(altitudes, A_max, k1, k2):
    flux = []
    for A in altitudes:
        if A <= A_max:
            flux.append(np.exp(k1 * A))
        else:
            flux.append(np.exp(k1 * A_max) * np.exp(-k2 * (A - A_max)))
    return np.array(flux)

def loss_function(params, altitudes, measured_events):
    k1, k2 = params
    predicted_events = calculate_muon_flux(altitudes, A_max, k1, k2)
    penalty = np.sum(np.maximum(0, predicted_events - 62)**2)
    mse = np.mean((predicted_events - measured_events) ** 2)
    return mse + penalty

# ========================
# Load and Process Detector 2 Data
# ========================
detector_2_file_path = 'Time vs Pressure Detector 2.xlsx'  # Replace with your file path
detector_2_data = pd.read_excel(detector_2_file_path)

detector_2_data = detector_2_data[
    pd.to_numeric(detector_2_data['Hour'], errors='coerce').notnull() &
    pd.to_numeric(detector_2_data['Min'], errors='coerce').notnull() &
    pd.to_numeric(detector_2_data['Sec'], errors='coerce').notnull() &
    pd.to_numeric(detector_2_data['Pressure'], errors='coerce').notnull()
]

# Format and calculate altitude
detector_2_data['Time'] = pd.to_datetime(
    detector_2_data['Hour'].astype(int).apply(lambda x: f"{x:02d}") + ':' +
    detector_2_data['Min'].astype(int).apply(lambda x: f"{x:02d}") + ':' +
    detector_2_data['Sec'].astype(int).apply(lambda x: f"{x:02d}"),
    format='%H:%M:%S'
)
detector_2_data['Altitude_m'] = (44330 / 2) * (1 - (detector_2_data['Pressure'] / P0) ** (1 / 5.255))
detector_2_data['Altitude_ft'] = detector_2_data['Altitude_m'] * 3.28084
detector_2_data['Altitude_ft'] = detector_2_data['Altitude_ft'] - detector_2_data['Altitude_ft'].min() + HersheyAltitude

# Detect times of takeoff, peak, and landing
altitude_diff = detector_2_data['Altitude_ft'].diff()
takeoff_index = altitude_diff.gt(3).idxmax()  # Takeoff time
takeoff_time = detector_2_data.loc[takeoff_index, 'Time']
peak_index = detector_2_data['Altitude_ft'].idxmax()  # Peak time
peak_time = detector_2_data.loc[peak_index, 'Time']
landing_index = altitude_diff.lt(-3).iloc[::-1].idxmax()  # Landing time
landing_time = detector_2_data.loc[landing_index, 'Time']

# ========================
# Load and Process Hess_007 Data
# ========================
hess_007_file_path = 'Hess_007.xlsx'  # Replace with your file path
hess_data = pd.read_excel(hess_007_file_path)

hess_data['TimeDelta'] = pd.to_timedelta(hess_data['Time[s]'], unit='s')
hess_data['Altitude_m'] = (44330 / 2) * (1 - (hess_data['Pressure[hPa]'] / P0) ** (1 / 5.255))
hess_data['Altitude_ft'] = hess_data['Altitude_m'] * 3.28084
hess_data['Altitude_ft'] = hess_data['Altitude_ft'] - hess_data['Altitude_ft'].min() + HersheyAltitude

# Synchronize Hess times with peak time
hess_peak_time = hess_data.loc[hess_data['Altitude_ft'].idxmax(), 'TimeDelta']
hess_data['Time'] = peak_time + (hess_data['TimeDelta'] - hess_peak_time)

# Compute events per second from Hess data
hess_data['Second'] = hess_data['Time'].dt.floor('S')
events_per_second = hess_data.resample('S', on='Time').count()['Event']

# Identify events at key times
takeoff_events = int(events_per_second.get(takeoff_time.floor('S'), 0))
peak_events = int(events_per_second.get(peak_time.floor('S'), 0))
landing_events = int(events_per_second.iloc[-1])

# Split events before and after peak
events_before_peak = events_per_second[events_per_second.index < peak_time]
events_after_peak = events_per_second[events_per_second.index > peak_time]

# Group data by interval
hess_data['Interval'] = hess_data['Time'].dt.floor('S')
events_per_interval = hess_data.groupby('Interval').size()

# ========================
# Prepare Data for NN
# ========================
events_data = pd.DataFrame({
    'Time': events_per_second.index,
    'Time_numeric': (events_per_second.index - events_per_second.index.min()).total_seconds(),
    'Events_per_sec': events_per_second.values
}).dropna()

X_events = events_data[['Time_numeric']]
y_events = events_data['Events_per_sec']

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_events_scaled = scaler_X.fit_transform(X_events)
y_events_scaled = scaler_y.fit_transform(y_events.values.reshape(-1, 1))

X_train_full, X_test, y_train_full, y_test = train_test_split(X_events_scaled, y_events_scaled, random_state=42)

# ========================
# Load and Evaluate Model
# ========================
model = load_model('best_model.h5', compile=False)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Loss={loss:.4f}, MAE={mae:.4f}")

y_pred_events_scaled = model.predict(X_events_scaled)
y_pred_events = scaler_y.inverse_transform(y_pred_events_scaled)
events_data['Predicted_Events_per_sec'] = y_pred_events.flatten()

# ========================
# Align Altitude and Optimize k1, k2
# ========================
detector_time_seconds = (detector_2_data['Time'] - detector_2_data['Time'].min()).dt.total_seconds()
aligned_altitudes_model = np.interp(
    events_data['Time_numeric'],
    detector_time_seconds,
    detector_2_data['Altitude_ft']
)

time_shift = timedelta(minutes=14, seconds=50)
muon_events_data = events_data['Time'] - time_shift

result = minimize(
    loss_function,
    initial_params,
    args=(aligned_altitudes_model, events_data['Events_per_sec'].values),
    bounds=[(0, 0.001), (0, 0.001)],
    method='L-BFGS-B'
)
optimized_k1, optimized_k2 = result.x
events_data['Muon_Events_per_sec'] = calculate_muon_flux(aligned_altitudes_model, A_max, optimized_k1, optimized_k2)

# Identify maxima in predicted data (NN model predictions)
predicted_before_peak = events_data[events_data['Time'] < peak_time]
predicted_after_peak = events_data[events_data['Time'] > peak_time]

predicted_max_before = predicted_before_peak.loc[predicted_before_peak['Predicted_Events_per_sec'].idxmax()]
predicted_max_after = predicted_after_peak.loc[predicted_after_peak['Predicted_Events_per_sec'].idxmax()]

pfotzer_max_before = events_before_peak.idxmax()
pfotzer_max_after = events_after_peak.idxmax()
pfotzer_max_before_count = int(events_per_second[pfotzer_max_before])
pfotzer_max_after_count = int(events_per_second[pfotzer_max_after])

max_altitude = max(detector_2_data['Altitude_ft'].max(), hess_data['Altitude_ft'].max())

start_time_str = "09:45"
end_time_str = "13:15"
start_time = pd.to_datetime(start_time_str, format="%H:%M")
end_time = pd.to_datetime(end_time_str, format="%H:%M")

# Font sizes
title_fontsize = 20
label_fontsize = 18
tick_fontsize = 16
annotation_fontsize = 14
legend_fontsize = 16

# Convert all relevant event counts to int in annotations
predicted_max_before_count = int(predicted_max_before['Predicted_Events_per_sec'])
predicted_max_after_count = int(predicted_max_after['Predicted_Events_per_sec'])

# ========================
# First Plot
# ========================
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.scatter(events_per_interval.index, events_per_interval.values, color='blue', label='Events/sec', s=80, zorder=3, alpha=0.4)
ax1.scatter(
    events_data['Time_numeric'].apply(lambda x: events_per_second.index[0] + pd.Timedelta(seconds=x)),
    y_pred_events.flatten(),
    color='red',
    label='Events/sec best fit',
    alpha=0.4,
    s=80,
    zorder=3
)

ax1.text(0.95, 0.75, f"Loss: {loss:.4f}\nMAE: {mae:.4f}", transform=ax1.transAxes, fontsize=annotation_fontsize,
         verticalalignment='top', horizontalalignment='right', color='red',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax1.set_xlabel('Time (H:M)', fontsize=label_fontsize)
ax1.set_ylabel(f'Measured Count Rate [sec{get_super(str(-1))}]', color='blue', fontsize=label_fontsize)
ax1.set_xlim([start_time, end_time])
ax1.set_ylim(0, max(events_per_second.max(), y_pred_events.max())+5)
ax1.tick_params(axis='y', labelcolor='blue', labelsize=tick_fontsize)
ax1.tick_params(axis='x', labelsize=tick_fontsize)

# Annotations on first figure with integer values
ax1.annotate(f"Maxima 1\n({predicted_max_before_count} events/sec)",
             xy=(predicted_max_before['Time'], predicted_max_before['Predicted_Events_per_sec']),
             xytext=(-160, -10), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='black'), color='red', fontsize=annotation_fontsize)

ax1.annotate(f"Maxima 2\n({predicted_max_after_count} events/sec)",
             xy=(predicted_max_after['Time'], predicted_max_after['Predicted_Events_per_sec']),
             xytext=(100, -10), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='black'), color='red', fontsize=annotation_fontsize)

ax1.annotate(f'Takeoff {takeoff_time.strftime("%I:%M")} am \n({takeoff_events} events/sec)',
             xy=(takeoff_time, takeoff_events), textcoords="offset points",
             xytext=(0, 40), ha='center', arrowprops=dict(arrowstyle="->", color='black'),
             color='blue', fontsize=annotation_fontsize)

ax1.annotate(f'Landing {landing_time.strftime("%I:%M")} pm \n({landing_events} events/sec)',
             xy=(landing_time, landing_events), textcoords="offset points",
             xytext=(30, 40), ha='center', arrowprops=dict(arrowstyle="->", color='black'),
             color='blue', fontsize=annotation_fontsize)

ax1.annotate(f'Maxima 1\n({pfotzer_max_before_count} events/sec)', xy=(pfotzer_max_before, pfotzer_max_before_count),
             xytext=(-100, -10), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='black'), color='blue', fontsize=annotation_fontsize)

ax1.annotate(f'Maxima 2\n({pfotzer_max_after_count} events/sec)', xy=(pfotzer_max_after, pfotzer_max_after_count),
             xytext=(100, -10), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='black'), color='blue', fontsize=annotation_fontsize)

ax2 = ax1.twinx()
ax2.plot(detector_2_data['Time'], detector_2_data['Altitude_ft'], label='Altitude', color='green', linewidth=4, zorder=2)
ax2.set_ylabel('Altitude (ft)', color='green', fontsize=label_fontsize)
ax2.set_xlim([start_time, end_time])
ax2.set_ylim(0, max_altitude + 40000)
ax2.tick_params(axis='y', labelcolor='green', labelsize=tick_fontsize)

pfotzer_max_before_nearest_time = hess_data.iloc[(hess_data['Time'] - pfotzer_max_before).abs().argsort()[:1]]['Time'].values[0]
pfotzer_max_after_nearest_time = hess_data.iloc[(hess_data['Time'] - pfotzer_max_after).abs().argsort()[:1]]['Time'].values[0]
pfotzer_max_before_altitude = hess_data.loc[hess_data['Time'] == pfotzer_max_before_nearest_time, 'Altitude_ft'].values[0]
pfotzer_max_after_altitude = hess_data.loc[hess_data['Time'] == pfotzer_max_after_nearest_time, 'Altitude_ft'].values[0]

ax2.scatter([pfotzer_max_before], [pfotzer_max_before_altitude], color='purple', s=80, zorder=3)
ax2.scatter([pfotzer_max_after], [pfotzer_max_after_altitude], color='purple', s=80, zorder=3)
ax2.scatter([peak_time], [max_altitude], color='purple', s=80, zorder=3)

ax1.scatter([predicted_max_before['Time'], predicted_max_after['Time']],
            [predicted_max_before['Predicted_Events_per_sec'], predicted_max_after['Predicted_Events_per_sec']],
            color='orange', s=80, zorder=4)

ax1.scatter([takeoff_time], [takeoff_events], color='yellow', label='Takeoff & Landing', s=80, zorder=3)
ax1.scatter([landing_time], [landing_events], color='yellow', s=80, zorder=3)
ax1.scatter([pfotzer_max_before, pfotzer_max_after],
            [pfotzer_max_before_count, pfotzer_max_after_count],
            color='orange', label='Max particle flux', s=80, zorder=3)

ax2.annotate(f'Altitude at Peak\n({int(max_altitude)} ft)',
             xy=(peak_time, max_altitude), xytext=(-40, 80), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", color='black'), ha='center', color='green', fontsize=annotation_fontsize)

ax2.annotate(f'Altitude at Maxima 1\n({int(pfotzer_max_before_altitude)} ft)',
             xy=(pfotzer_max_before, pfotzer_max_before_altitude),
             xytext=(-200, -10), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", color='black'), ha='center', color='green', fontsize=annotation_fontsize)

ax2.annotate(f'Altitude at Maxima 2\n({int(pfotzer_max_after_altitude)} ft)',
             xy=(pfotzer_max_after, pfotzer_max_after_altitude),
             xytext=(100, -10), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", color='black'), ha='center', color='green', fontsize=annotation_fontsize)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=legend_fontsize)

plt.title('Particle Flux for Cosmic Watch', fontsize=title_fontsize)
plt.show()

# ========================
# Second Plot
# ========================
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.scatter(events_per_interval.index, events_per_interval.values, color='blue', label='Events/sec', s=80, zorder=3, alpha=0.4)

ax1.scatter(
    events_data['Time_numeric'].apply(lambda x: events_per_second.index[0] + pd.Timedelta(seconds=x)),
    y_pred_events.flatten(),
    color='red',
    label='Events/sec best fit',
    alpha=0.4,
    s=80,
    zorder=3
)

ax1.text(0.95, 0.40, f"Loss: {loss:.4f}\nMAE: {mae:.4f}",
         transform=ax1.transAxes, fontsize=annotation_fontsize, verticalalignment='top',
         horizontalalignment='right', color='red',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax1.set_xlabel('Time (H:M)', fontsize=label_fontsize)
ax1.set_ylabel(f'Measured Count Rate [sec{get_super(str(-1))}]', color='blue', fontsize=label_fontsize)
ax1.set_xlim([start_time, end_time])
ax1.set_ylim(0, max(events_per_second.max(), y_pred_events.max())+5)
ax1.tick_params(axis='y', labelcolor='blue', labelsize=tick_fontsize)
ax1.tick_params(axis='x', labelsize=tick_fontsize)

ax1.annotate(f"Maxima 1\n({predicted_max_before_count} events/sec)",
             xy=(predicted_max_before['Time'], predicted_max_before['Predicted_Events_per_sec']),
             xytext=(-160, -10), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='black'), color='red', fontsize=annotation_fontsize)

ax1.annotate(f"Maxima 2\n({predicted_max_after_count} events/sec)",
             xy=(predicted_max_after['Time'], predicted_max_after['Predicted_Events_per_sec']),
             xytext=(100, -10), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='black'), color='red', fontsize=annotation_fontsize)

ax1.annotate(f'Takeoff {takeoff_time.strftime("%I:%M")} am \n({takeoff_events} events/sec)',
             xy=(takeoff_time, takeoff_events), textcoords="offset points",
             xytext=(0, 40), ha='center', arrowprops=dict(arrowstyle="->", color='black'),
             color='blue', fontsize=annotation_fontsize)

ax1.annotate(f'Landing {landing_time.strftime("%I:%M")} pm \n({landing_events} events/sec)',
             xy=(landing_time, landing_events), textcoords="offset points",
             xytext=(30, 40), ha='center', arrowprops=dict(arrowstyle="->", color='black'),
             color='blue', fontsize=annotation_fontsize)

ax1.annotate(f'Maxima 1\n({pfotzer_max_before_count} events/sec)',
             xy=(pfotzer_max_before, pfotzer_max_before_count),
             xytext=(-100, -10), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='black'), color='blue', fontsize=annotation_fontsize)

ax1.annotate(f'Maxima 2\n({pfotzer_max_after_count} events/sec)',
             xy=(pfotzer_max_after, pfotzer_max_after_count),
             xytext=(100, -10), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='black'), color='blue', fontsize=annotation_fontsize)

ax2 = ax1.twinx()
ax2.plot(muon_events_data, events_data['Muon_Events_per_sec'], label='Muon Events/sec', color='green', linewidth=4)
ax2.set_ylabel(f'Muon Count Rate [sec{get_super(str(-1))}]', color='green', fontsize=label_fontsize)
ax2.set_xlim([start_time, end_time])
ax2.set_ylim(0, max(events_per_second.max(), y_pred_events.max())+5)
ax2.tick_params(axis='y', labelcolor='green', labelsize=tick_fontsize)

pfotzer_max_before_nearest_time = hess_data.iloc[(hess_data['Time'] - pfotzer_max_before).abs().argsort()[:1]]['Time'].values[0]
pfotzer_max_after_nearest_time = hess_data.iloc[(hess_data['Time'] - pfotzer_max_after).abs().argsort()[:1]]['Time'].values[0]
pfotzer_max_before_altitude = int(hess_data.loc[hess_data['Time'] == pfotzer_max_before_nearest_time, 'Altitude_ft'].values[0])
pfotzer_max_after_altitude = int(hess_data.loc[hess_data['Time'] == pfotzer_max_after_nearest_time, 'Altitude_ft'].values[0])

desired_time = pfotzer_max_before - time_shift
closest_idx = (hess_data['Time'] - desired_time).abs().idxmin()
height_at_desired_time = int(hess_data.loc[closest_idx, 'Altitude_ft'])
print(f"Height at time_shift + pfotzer_max_before: {height_at_desired_time} ft")

ax1.scatter([predicted_max_before['Time'], predicted_max_after['Time']],
            [predicted_max_before['Predicted_Events_per_sec'], predicted_max_after['Predicted_Events_per_sec']],
            color='orange', s=80, zorder=4)

ax1.scatter([takeoff_time], [takeoff_events], color='yellow', label='Takeoff & Landing', s=80, zorder=3)
ax1.scatter([landing_time], [landing_events], color='yellow', s=80, zorder=3)
ax1.scatter([pfotzer_max_before, pfotzer_max_after],
            [pfotzer_max_before_count, pfotzer_max_after_count],
            color='orange', label='Max particle flux', s=80, zorder=3)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=legend_fontsize)

plt.title('Particle Flux for Cosmic Watch', fontsize=title_fontsize)

# ========================
# Extend the Muon Flux Line
# ========================
last_predicted_time = events_data['Time'].iloc[-1]
extension_duration = pd.Timedelta(minutes=15)
extension_end_time = last_predicted_time + extension_duration

extended_time_index = pd.date_range(
    start=last_predicted_time + pd.Timedelta(seconds=1),
    end=extension_end_time, freq='S'
)

extended_time_numeric = (extended_time_index - events_per_second.index.min()).total_seconds()
extended_data = pd.DataFrame({'Time': extended_time_index, 'Time_numeric': extended_time_numeric})

original_altitudes = detector_2_data['Altitude_ft'].values
landing_altitude = int(detector_2_data.loc[detector_2_data['Time'] == landing_time, 'Altitude_ft'].values[0])
landing_time_secs = (landing_time - detector_2_data['Time'].min()).total_seconds()
window_before = 2 * 60
time_before_landing = landing_time_secs - window_before
alt_before_landing = np.interp(time_before_landing, detector_time_seconds, original_altitudes)

# Interpolate altitude for extended times
extended_aligned_altitudes_model = np.interp(
    extended_data['Time_numeric'],
    detector_time_seconds,
    original_altitudes
)

# Apply sigmoid function after landing
scale = 60.0
after_landing_mask = extended_data['Time_numeric'] > landing_time_secs
t_rel = extended_data.loc[after_landing_mask, 'Time_numeric']
extended_aligned_altitudes_model[after_landing_mask] = alt_before_landing + \
    (landing_altitude - alt_before_landing) / (1 + np.exp(-(t_rel - landing_time_secs)/scale))

extended_data['Muon_Events_per_sec'] = calculate_muon_flux(
    extended_aligned_altitudes_model, A_max, optimized_k1, optimized_k2
)

combined_events_data = pd.concat([events_data[['Time', 'Time_numeric', 'Muon_Events_per_sec']], extended_data], ignore_index=True)
combined_events_data = combined_events_data.sort_values('Time').reset_index(drop=True)
combined_muon_events_data = combined_events_data['Time'] - time_shift

# Remove previously plotted Muon lines if needed
for line in ax2.get_lines():
    if 'Muon Events/sec' in line.get_label():
        line.remove()

ax2.plot(
    combined_muon_events_data,
    combined_events_data['Muon_Events_per_sec'],
    label='Muon Events/sec',
    color='green',
    linewidth=4,
    linestyle='-'
)

# Find maxima for the muon fitted line after extending
muon_predicted_before_peak = combined_events_data[combined_events_data['Time'] < peak_time]
muon_predicted_after_peak = combined_events_data[combined_events_data['Time'] > peak_time]

muon_predicted_max_before = muon_predicted_before_peak.loc[muon_predicted_before_peak['Muon_Events_per_sec'].idxmax()]
muon_predicted_max_after = muon_predicted_after_peak.loc[muon_predicted_after_peak['Muon_Events_per_sec'].idxmax()]

muon_predicted_max_before_count = int(muon_predicted_max_before['Muon_Events_per_sec'])
muon_predicted_max_after_count = int(muon_predicted_max_after['Muon_Events_per_sec'])

# Annotate muon maxima in green as integers
ax2.annotate(
    f"Muon Maxima 1\n({muon_predicted_max_before_count} events/sec)",
    xy=(muon_predicted_max_before['Time'] - time_shift, muon_predicted_max_before['Muon_Events_per_sec']),
    xytext=(-160, -10),
    textcoords='offset points',
    ha='center',
    arrowprops=dict(arrowstyle="->", color='black'),
    color='green',
    fontsize=annotation_fontsize
)

ax2.annotate(
    f"Muon Maxima 2\n({muon_predicted_max_after_count} events/sec)",
    xy=(muon_predicted_max_after['Time'] - time_shift, muon_predicted_max_after['Muon_Events_per_sec']),
    xytext=(100, -10),
    textcoords='offset points',
    ha='center',
    arrowprops=dict(arrowstyle="->", color='black'),
    color='green',
    fontsize=annotation_fontsize
)

ax1.set_xlim([start_time, end_time])
ax2.set_xlim([start_time, end_time])

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=legend_fontsize)

plt.draw()
plt.show()
