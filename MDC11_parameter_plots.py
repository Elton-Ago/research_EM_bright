import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ligo.em_bright import em_bright
import argparse
import sys
import configparser

parser = argparse.ArgumentParser()
parser.add_argument("-S", "--eventname", action="store", help="Name of event we want to cross match")
parser.add_argument("-P", "--plot", action="store_true", default=False, help="Toggle plot")
parser.add_argument("-V", "--verbose", action="store_true", default=False, help="Print stdout")
#searched event
args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')
csv_file_path_to_PE_events = config['paths']['csv_PE_events']
csv_file_path_to_crossmatch = config['paths']['csv_crossmatch']

# File paths
#function default variable, 
#config parser files 

# Input for the event to search
#searched_event = args.eventname
# Load CSV data
#PE_events_df = pd.read_csv(file_PE_events)
#crossmatch_df = pd.read_csv(file_crossmatch)

PE_events_df = pd.read_csv(csv_file_path_to_PE_events)
crossmatch_df = pd.read_csv(csv_file_path_to_crossmatch)

# Find matching rows in CSV dataframes
# Define the base directory to search for HDF5 files
base_directory="MDC11_PE/1360608000_1364064000"
searched_event=args.eventname


def find_event_file(base_directory, searched_event):
    """
    Function to find the HDF5 file corresponding to the searched event in the base directory.

    Parameters:
    - base_directory (str): Directory to search for HDF5 files.
    - searched_event (str): Event name to search for in HDF5 file names.

    Returns:
    - str or None: File path of the HDF5 file if found, None if not found.
    """
    event_directory = os.path.join(base_directory, searched_event)
    #os.path.join just concatinates directories.
    if os.path.exists(event_directory):
        #checking if this path, in this case, the event we look up in the 1360608000_1364064000 directory, because we need to find the event directory itself... 
        #... for the bilby posterior files
        for root, _, files in os.walk(event_directory):
        #root, dirs, and files are to include the root directories so we can use shorthand, because I dont use dirs I can just _ for everything included, and...
        #...files to find the exsiting files in the event directories we designated specifically
        #os.walk is a OS package that makes it possible to traverse through the local host (my computer) directories files within my OS
            for file in files:
            #for the amount of files in file, iterate through them
                if file.endswith('.hdf5') and file.startswith('Bilby.fast_test'):
                #then evaluate that if this is an .hdf5 files, we want that and especially if its the Bilby.fast_test file we want.
                #ends with and starts with are needed to highlight the parse the file with these key words to find it more directly, available from OS package imported
                    return os.path.join(root, file)
                    #return the joined root and file (bilby one we got) to make available for the histogram plot we will make of the mass distribution later

matching_rows_true_event = crossmatch_df[crossmatch_df['event'].str.contains(args.eventname, case=False, na=False)]
matching_rows_distribution_event = PE_events_df[PE_events_df['event'].str.contains(args.eventname, case=False, na=False)]    
# Find the HDF5 file for the searched event
file_path = find_event_file(base_directory, searched_event)
#most importantly this lets us call the function we created above to get the base_directory which is base_directory = "MDC11_PE/1360608000_1364064000" that we set 
#prior to the function and also use the searched event inputted by the user to find that specific event

if file_path:
    if args.verbose: print(f"Found event file: {file_path}")  # Debug print
    # Load mass_1_source data from hdf5 file
    with h5py.File(file_path, "r") as file:
        posterior_samples = file['posterior_samples']
        mass_1_source = np.array(posterior_samples['mass_1_source'])
        mass_2_source = np.array(posterior_samples['mass_2_source'])
        spin_1z_source = np.array(posterior_samples['spin_1z'])
        spin_2z_source = np.array(posterior_samples['spin_2z'])
        #check mass1 vs mass_1_source label for correct mass red shifted
else:
    if args.verbose: print(f"No hdf5 file found for event '{searched_event}' in directory '{base_directory}'")
    sys.exit(0)
    #Raise value error in case

# Print matching rows from CSV data
if not matching_rows_true_event.empty:
    if args.verbose: print("\nMatching event:")
    if args.verbose: print(matching_rows_true_event)
else:
    if args.verbose: print("\nNo matches found for true event.")

if not matching_rows_distribution_event.empty:
    if args.verbose: print("\nMatching event for distribution:")
    if args.verbose: print(matching_rows_distribution_event)
else:
    if args.verbose: print("\nNo matches found for distribution event.")

'''
True injected values
'''
mass1_inj_source = np.array(matching_rows_true_event['mass1_source_inj'])
mass2_inj_source = np.array(matching_rows_true_event['mass2_source_inj'])
spin1z_inj_source = np.array(matching_rows_true_event['spin1z_inj'])
spin2z_inj_source = np.array(matching_rows_true_event['spin2z_inj'])

mass1_inj_source = mass1_inj_source.item()
mass2_inj_source = mass2_inj_source.item()
spin1z_inj_source = spin1z_inj_source.item()
spin2z_inj_source = spin2z_inj_source.item()

'''
distribution of posteriors
'''
mass_1_source = np.array(mass_1_source)
mass_2_source = np.array(mass_2_source)
spin_1z_source = np.array(spin_1z_source)
spin_2z_source = np.array(spin_2z_source)

#mass_2_source = mass_2_source.item()
#spin_1z_source = spin_1z_source.item()
#spin_2z_source = spin_2z_source.item()

if args.verbose: print("true: ", mass1_inj_source)
if args.verbose: print("posterior: ", mass_1_source)

if args.verbose: print("true: ", mass2_inj_source)
if args.verbose: print("posterior: ", mass_2_source)

if args.verbose: print("true: ", spin1z_inj_source)
if args.verbose: print("posterior: ", spin_1z_source)

if args.verbose: print("true: ", spin2z_inj_source)
if args.verbose: print("posterior: ", spin_2z_source)

result = (em_bright.source_classification(mass1_inj_source, mass2_inj_source, spin1z_inj_source, spin2z_inj_source, 11))
if args.verbose: print("result: ", result)

# Plot histogram of mass_1_source data with vertical line for mass1_inj_source

def plot_histogram_of_parameter(parameter_source, injected_source, color, injected_color, injected_label, label, title):
    plt.hist(parameter_source, bins=30, color=color, alpha=0.5, density=True, label=label)
    plt.axvline(injected_source, color=injected_color, linestyle='dashed', linewidth=2, label=injected_label)
    plt.xlabel('Parameter')
    plt.ylabel('Probabillity')
    plt.title(title)
    plt.legend()
    plt.show()

if args.plot:
    plots = [
        {
            "parameter_source": mass_1_source,
            "injected_source": mass1_inj_source,
            "color": 'red',
            "injected_color": 'purple',
            "injected_label": 'Injected (M1) Parameter Source',
            "label": 'Parameter (M1) Source',
            "title": 'Mass 1'
        },
        {
            "parameter_source": mass_2_source,
            "injected_source": mass2_inj_source,
            "color": 'orange',
            "injected_color": 'black',
            "injected_label": 'Injected (M2) Parameter Source',
            "label": 'Parameter (M2) Source',
            "title": 'Mass 2'
        },
        {
            "parameter_source": spin_1z_source,
            "injected_source": spin1z_inj_source,
            "color": 'blue',
            "injected_color": 'black',
            "injected_label": 'Injected (S1) Parameter Source',
            "label": 'Parameter (S1) Source',
            "title": 'Spin 1'
        },
        {
            "parameter_source": spin_2z_source,
            "injected_source": spin2z_inj_source,
            "color": 'green',
            "injected_color": 'purple',
            "injected_label": 'Injected (S2) Parameter Source',
            "label": 'Parameter (S2) Source',
            "title":  'Spin 2'
        }
    ]

    title = f'Posterior Samples for {searched_event}'

    for plot in plots:
        plot_histogram_of_parameter(
            parameter_source=plot["parameter_source"],
            injected_source=plot["injected_source"],
            color=plot["color"],
            injected_color=plot["injected_color"],
            injected_label=plot["injected_label"],
            label=plot["label"],
            title=title
        )
