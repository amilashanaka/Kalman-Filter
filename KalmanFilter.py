import numpy as np
import pandas as pd

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, transition_matrix, observation_matrix, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def predict(self):
        # Predict the next state
        predicted_state = self.transition_matrix @ self.state
        predicted_covariance = self.transition_matrix @ self.covariance @ self.transition_matrix.T + self.process_noise
        return predicted_state, predicted_covariance
    
    def update(self, measurement):
        # Update the state based on measurement
        innovation = measurement - self.observation_matrix @ self.state
        innovation_covariance = self.observation_matrix @ self.covariance @ self.observation_matrix.T + self.measurement_noise
        kalman_gain = self.covariance @ self.observation_matrix.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        self.covariance = (np.identity(self.state.shape[0]) - kalman_gain @ self.observation_matrix) @ self.covariance
        return self.state, self.covariance

# Example usage
# Create a DataFrame with some measurements
data = {'Time': [1, 2, 3, 4, 5],
        'Measurement': [1.2, 1.8, 3.6, 4.1, 5.2]}
df = pd.DataFrame(data)

# Initialize Kalman filter parameters
initial_state = np.array([0])
initial_covariance = np.array([1])
transition_matrix = np.array([1])
observation_matrix = np.array([1])
process_noise = np.array([0.01])
measurement_noise = np.array([1])

# Create a Kalman filter object
kalman_filter = KalmanFilter(initial_state, initial_covariance, transition_matrix, observation_matrix, process_noise, measurement_noise)

# Iterate over the measurements and update the filter
filtered_values = []
for index, row in df.iterrows():
    measurement = np.array([row['Measurement']])
    updated_state, updated_covariance = kalman_filter.update(measurement)
    filtered_values.append(updated_state[0])
    kalman_filter.state = updated_state
    kalman_filter.covariance = updated_covariance

# Add the filtered values to the DataFrame
df['Filtered'] = filtered_values

print(df)
