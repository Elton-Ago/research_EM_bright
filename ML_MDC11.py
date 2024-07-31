import os
import h5py
import numpy as np
import pandas as pd
from ligo.em_bright import em_bright
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import configparser

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-P", "--plot", action="store_true", default=False, help="Toggle plot")
parser.add_argument("-V", "--verbose", action="store_true", default=False, help="Print stdout")
args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')
csv_file_path_to_PE_events = config['paths']['csv_PE_events']
csv_file_path_to_crossmatch = config['paths']['csv_crossmatch']

# Load CSV data
PE_events_df = pd.read_csv(csv_file_path_to_PE_events)
crossmatch_df = pd.read_csv(csv_file_path_to_crossmatch)

class RandomForestPredictor:
    def __init__(self, df, train_size_ratio=0.7, test_size=None):
        self.df = df
        self.train_size_ratio = train_size_ratio
        self.test_size = test_size
        self.model = None
        self.predictions = None
        self.event_names_test = None

    def prepare_data(self):
        df = self.df[['event', 'mass1_source_inj', 'mass2_source_inj', 'spin1z_inj', 'spin2z_inj']]
        
        # Check for NaN values
        if args.verbose:
            print("NaN values in dataset:\n", df.isna().sum())
        
        X = df.drop(columns=['event']).values
        y = df.drop(columns=['event']).values
        event_names = df['event'].values
        return X, y, event_names

    def predictor(self, random_state=42, test_result=True):
        X, y, event_names = self.prepare_data()
        
        total_size = X.shape[0]
        
        if self.test_size is None:
            self.test_size = int(0.2 * total_size)  # Default to 20% if not specified
        
        # Calculate train size based on the ratio
        train_size = int(self.train_size_ratio * total_size)
        
        # Ensure sizes are within bounds
        train_size = min(train_size, total_size - self.test_size)
        test_size = self.test_size
        
        if args.verbose:
            print(f"Total data size: {total_size}")
            print(f"Training set size: {train_size}")
            print(f"Test set size: {test_size}")

        # Split data
        X_train, X_test, y_train, y_test, _, event_names_test = train_test_split(
            X, y, event_names, train_size=train_size, test_size=test_size, random_state=random_state
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)

        if not test_result:
            return self.model

        self.predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, self.predictions)
        r2 = r2_score(y_test, self.predictions)
        
        if args.verbose:
            print(f"Mean Squared Error: {mse}")
            print(f"R-squared: {r2 * 100:.2f}%")

        self.event_names_test = event_names_test
        return X_test, self.predictions, self.model

    def save_predictions(self, filename='predictions.csv'):
        if self.predictions is not None and self.event_names_test is not None:
            results_df = pd.DataFrame(self.predictions, columns=['mass1_source_pred', 'mass2_source_pred', 'spin1z_source_pred', 'spin2z_source_pred'])
            results_df['event'] = self.event_names_test
            
            if args.verbose:
                print(f"Number of predictions: {len(self.predictions)}")
                print(f"Number of events: {len(self.event_names_test)}")
            
            results_df.to_csv(filename, index=False)
        else:
            if args.verbose:
                print("No predictions to save.")

# Initialize and use RandomForestPredictor
train_size_ratio = 0.7  # Ratio of training data, helps with overfitting
test_size = 398         # Number of rows you want to predict

rf_predictor = RandomForestPredictor(crossmatch_df, train_size_ratio=train_size_ratio, test_size=test_size)

# Train the model and make predictions
X_test, predictions, model = rf_predictor.predictor()

# Save the predictions to a CSV file
rf_predictor.save_predictions()

# Calculate the number of rows being predicted
n_total_rows = len(crossmatch_df)
n_predicted_rows = test_size

# Load HDF5 data from all files
def load_all_hdf5_data(base_directory):
    mass_1_source_all = []
    mass_2_source_all = []
    spin_1z_source_all = []
    spin_2z_source_all = []

    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.hdf5') and file.startswith('Bilby.fast_test'):
                file_path = os.path.join(root, file)
                with h5py.File(file_path, "r") as f:
                    posterior_samples = f['posterior_samples']
                    mass_1_source_all.extend(posterior_samples['mass_1_source'])
                    mass_2_source_all.extend(posterior_samples['mass_2_source'])
                    spin_1z_source_all.extend(posterior_samples['spin_1z'])
                    spin_2z_source_all.extend(posterior_samples['spin_2z'])

    return np.array(mass_1_source_all), np.array(mass_2_source_all), np.array(spin_1z_source_all), np.array(spin_2z_source_all)

base_directory = "MDC11_PE/1360608000_1364064000"
posterior_samples_all = load_all_hdf5_data(base_directory)
mass_1_source_all, mass_2_source_all, spin_1z_source_all, spin_2z_source_all = posterior_samples_all

# True injected values
mass1_inj_source = np.array(crossmatch_df['mass1_source_inj'])
mass2_inj_source = np.array(crossmatch_df['mass2_source_inj'])
spin1z_inj_source = np.array(crossmatch_df['spin1z_inj'])
spin2z_inj_source = np.array(crossmatch_df['spin2z_inj'])

# Calculate means
mean_mass1_inj = mass1_inj_source.mean()
mean_mass2_inj = mass2_inj_source.mean()
mean_spin1z_inj = spin1z_inj_source.mean()
mean_spin2z_inj = spin2z_inj_source.mean()

mean_mass1_pred = predictions[:, 0].mean()
mean_mass2_pred = predictions[:, 1].mean()
mean_spin1z_pred = predictions[:, 2].mean()
mean_spin2z_pred = predictions[:, 3].mean()

result_inj = em_bright.source_classification(mean_mass1_inj, mean_mass2_inj, mean_spin1z_inj, mean_spin2z_inj, 11)
result_pred = em_bright.source_classification(mean_mass1_pred, mean_mass2_pred, mean_spin1z_pred, mean_spin2z_pred, 11)

# Print means
if args.verbose:
    print(f"Mean of true injected Mass 1: {mean_mass1_inj}")
    print(f"Mean of true injected Mass 2: {mean_mass2_inj}")
    print(f"Mean of true injected Spin 1z: {mean_spin1z_inj}")
    print(f"Mean of true injected Spin 2z: {mean_spin2z_inj}")

    print(f"Mean of predicted Mass 1: {mean_mass1_pred}")
    print(f"Mean of predicted Mass 2: {mean_mass2_pred}")
    print(f"Mean of predicted Spin 1z: {mean_spin1z_pred}")
    print(f"Mean of predicted Spin 2z: {mean_spin2z_pred}")
    
    print(f"Mean of true injected EM_bright results: {result_inj}")
    print(f"Mean of true predicted EM_bright results: {result_pred}")
    
    print(f"Number of rows being predicted: {n_predicted_rows}")

# Plot histograms
def plot_histogram(posterior_samples, mean_inj, mean_pred, parameter_name):
    plt.figure(figsize=(14, 7))
    
    num_bins = 60

    plt.hist(posterior_samples, bins=num_bins, color='blue', edgecolor='black', alpha=0.5, density=True, label='Injected Samples of Events')
    plt.axvline(mean_inj, color='red', linestyle='dashed', linewidth=2, label='Injected Mean')
    plt.axvline(mean_pred, color='purple', linestyle='dashed', linewidth=2, label='Predicted Mean')
    
    plt.xlabel(parameter_name)
    plt.ylabel('Probability')
    plt.title(f'Histogram of {parameter_name}')
    plt.legend()
    plt.xlim([min(posterior_samples), max(posterior_samples)])
    plt.grid(True)
    plt.show()

if args.plot:
    plot_histogram(mass_1_source_all, mean_mass1_inj, mean_mass1_pred, 'Mass 1')
    plot_histogram(mass_2_source_all, mean_mass2_inj, mean_mass2_pred, 'Mass 2')
    plot_histogram(spin_1z_source_all, mean_spin1z_inj, mean_spin1z_pred, 'Spin 1z')
    plot_histogram(spin_2z_source_all, mean_spin2z_inj, mean_spin2z_pred, 'Spin 2z')
