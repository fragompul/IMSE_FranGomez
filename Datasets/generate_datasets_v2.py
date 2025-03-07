import numpy as np
import pandas as pd
from scipy.stats import uniform
import cbadc as cb
from itertools import product
from numpy.linalg import LinAlgError
from scipy.stats import qmc
import time



def generate_dataset(form, order, implementation, t_min, num_samples, min_snr, parameter_ranges, discrete_params):
    """
    Generate a dataset based on the selected form, order, implementation
    
    Args:
    - form (str): The (continuous) form of the filter ('FB', 'FF')
    - order (int): The order of the filter (2, 3, 4)
    - implementation (str): The implementation type ('Active-RC', 'Gm-C')
    - t_min (float): Time in seconds to execute the function
    - num_samples (int): Initial number of samples to run LHS
    - min_snr (float): Minimum SNR accepted.
    - parameter_ranges (list): List with the range of each parameter
    - discrete_parameters (list): List with the discrete parameters and each values
    
    Returns:
    - dataset (pd.DataFrame): A DataFrame containing the generated dataset,
                              named: dataset_form_order_implementation.csv
    """
    
    start_time = time.time() # record start time
    
    # initialize lists to store the parameters and corresponding SNR values after simulation:
    snr_values = []
    all_params = []
    
    while (time.time() - start_time) < t_min:
        # Latin Hypercube Sampling (LHS) to sample the parameter space:
        lhs_samples = lhs_sampling(parameter_ranges, num_samples, discrete_params)

        # iterate through each set of parameters from the LHS samples:
        for sample in lhs_samples:
            # map the sampled values to parameter names:
            if implementation == 'Active-RC':
                params = {'Bw': sample[0], 'osr': int(sample[1]), 'nlev': int(sample[2]), 'Hinf': sample[3], 
                          'Cint': sample[4], 'gm': sample[5], 'Ro': sample[6], 'Co': sample[7]}
            elif implementation == 'Gm-C':
                params = {'Bw': sample[0], 'osr': int(sample[1]), 'nlev': int(sample[2]), 'Hinf': sample[3], 
                          'Cint': sample[4], 'Ro': sample[5], 'Cp': sample[6], 'v_n': sample[7], 
                          'slew_rate': sample[8], 'output_swing': sample[9]}

            # apply parameter relationships filters to ensure validity (not implemented yet):
            if not filter_parameter_relationships(params, implementation):
                continue # skip this sample if it doesn't pass the filter

            # simulate the circuit and get the SNR value:
            snr = simulate_circuit(form, order, implementation, params)

            # store the the parameters and corresponding SNR value:
            if snr is not None and np.isfinite(snr) and snr > min_snr:
                snr_values.append(snr)
                all_params.append(params)
    
    # create a DataFrame and save it as CSV:
    dataset = pd.DataFrame(all_params)
    dataset['SNR'] = snr_values
    filename = f"datasets/dataset_{form}_{order}_{implementation}.csv"
    dataset.to_csv(filename, index=False)
    print(f"\n\tDataset saved as {filename}\n")
    
    return dataset



def lhs_sampling(parameter_ranges, num_samples, discrete_params):
    """
    Latin Hypercube Sampling (LHS) for a given set of parameter ranges,
    handling both continuous and discrete parameters

    Args:
    - parameter_ranges (list): List of tuples [(min, max), ...] defining the range for each parameter
    - num_samples (int): Number of samples to generate
    - discrete_params (list): List of tuples [(index, values)] where:
        - index: The position of the discrete parameter in parameter_ranges
        - values: A list of possible discrete values

    Returns:
    - samples (list): A NumPy array of shape (num_samples, len(parameter_ranges)) containing the generated samples
    """

    # separate continuous and discrete parameters:
    continuous_params = [p for i, p in enumerate(parameter_ranges) if i not in [d[0] for d in discrete_params]]
    discrete_indices = {d[0]: d[1] for d in discrete_params}

    # create an LHS design only for continuous parameters:
    sampler = qmc.LatinHypercube(d=len(continuous_params)) # d = number of continuous parameters
    sample_cont = sampler.random(num_samples) # generate LHS samples in [0,1] range
    
    # rescale LHS samples to match the specified ranges for continuous parameters:
    cont_scaled = qmc.scale(sample_cont, 
                            [p[0] for p in continuous_params], # min values
                            [p[1] for p in continuous_params]) # max values
    
    # initialize the result matrix:
    samples = np.zeros((num_samples, len(parameter_ranges)))

    # insert continuous values into the result matrix:
    cont_idx = 0 # index to track continuous parameters in cont_scaled
    for i, p in enumerate(parameter_ranges):
        if i in discrete_indices: # skip discrete parameters
            continue
        samples[:, i] = cont_scaled[:, cont_idx] # assign LHS-sampled values
        cont_idx += 1

    # insert discrete parameters using random selection with replacement:
    for idx, values in discrete_indices.items():
        samples[:, idx] = np.random.choice(values, size=num_samples, replace=True)

    return samples



def simulate_circuit(form, order, implementation, params):
    """
    Simulate the circuit and return the SNR value based on the selected
    form, order and implementation
    
    Args:
    - form (str): The (continuous) form of the filter ('FB', 'FF')
    - order (int): The order of the filter (2, 3, 4)
    - implementation (str): The implementation type ('Active-RC', 'Gm-C')
    - params (dict): A dictionary containing the parameters for the simulation
    
    Returns:
    - max_snr (float): The maximum SNR value obtained from the simulation
    """
    
    # define parameters:
    Bw = params.get('Bw')
    osr = params.get('osr')
    nlev = params.get('nlev')
    Hinf = params.get('Hinf')
    f0 = 0.
    tdac = [0, 1]
    
    # compute the sampling frequency:
    fs = Bw * osr * 2
    
    # synthesize the optimal NTF (Noise Transfer Function)
    ntf = cb.delsig.synthesizeNTF(order, osr, 2, Hinf, f0)
    
    # create analog frontend:
    dt = 1.0 / fs
    ABCDc, tdac2 = cb.delsig.realizeNTF_ct(ntf, form, tdac)
    analog_frontend = cb.AnalogFrontend.ctsdm(ABCDc, tdac2, dt, nlev)
    analog_frontend.dt = 1.0 / fs
    
    # run the SNR simulation (depending on the implementation):
    if implementation == 'Active-RC':
        max_snr = simulate_snr_active_rc(params, order, osr, analog_frontend)
    elif implementation == 'Gm-C':
        max_snr = simulate_snr_gm_c(params, order, osr, analog_frontend)    
    else:
        raise ValueError(f"Unknown implementation type: {implementation}")
    
    return max_snr



def simulate_snr_active_rc(params, order, osr, analog_frontend):
    """
    Simulate the SNR for Active-RC implementation
    
    Args:
    - params (dict): The parameter set
    - order (int): The order of the filter (2, 3, 4)
    - osr (int): The oversampling ratio
    
    Returns:
    - snr (float): The simulated SNR value.
    """
    
    # define parameters:
    Cint = np.ones(order) * params.get('Cint')
    gm = np.ones(order) * params.get('gm')
    Ro = np.ones(order) * params.get('Ro')
    Co = np.ones(order) * params.get('Co')
    
    # simulate SNR:
    try:
        ActiveRC_analog_frontend = cb.ActiveRC(
            analog_frontend, Cint, gm, Ro, Co
        )
                
        snr_ActiveRC, amp_ActiveRC, _ = ActiveRC_analog_frontend.simulateSNR(osr)
        max_snr_ActiveRC = max(snr_ActiveRC)
        
        if not np.isnan(max_snr_ActiveRC):
            print(f"Achieved SNR: {max_snr_ActiveRC} db")
            return max_snr_ActiveRC
        else:
            return None
        
    except (LinAlgError, ValueError) as e:
        print(f"Error with Active-RC parameters {params}: {e}")
        return None



# Function to simulate SNR for Gm-C
def simulate_snr_gm_c(params, order, osr, analog_frontend):
    """
    Simulate the SNR for Gm-C implementation
    
    Args:
    - params (dict): The parameter set
    - order (int): The order of the filter (2, 3, 4)
    - osr (int): The oversampling ratio
    
    Returns:
    - snr (float): The simulated SNR value
    """ 
    
    # define parameters:
    Cint = np.ones(order) * params.get('Cint')
    Ro = np.ones(order) * params.get('Ro')
    Cp = np.ones(order) * params.get('Cp')
    v_n = np.ones(order) * params.get('v_n')
    slew_rate = np.ones(order) * params.get('slew_rate')
    v_out_max = np.ones(order) * params.get('output_swing')
    v_out_min = -v_out_max
    
    # simulate SNR:
    try:
        GmC_analog_frontend = cb.GmC(
            analog_frontend, Cint, Ro, Cp, v_n, v_out_min, v_out_max, slew_rate
        )
    
        snr_GmC, amp_GmC, _ = GmC_analog_frontend.simulateSNR(osr)
        max_snr_GmC = max(snr_GmC)

        if not np.isnan(max_snr_GmC):
            print(f"Achieved SNR: {max_snr_GmC} db")
            return max_snr_GmC
        else:
            return None
    
    except (LinAlgError, ValueError) as e:
        print(f"Error with Gm-C parameters {params}: {e}")
        return None



def filter_parameter_relationships(params, implementation):
    """
    Apply relationships-based filters to ensure that the parameters 
    comply with physical and stability constraints

    Args:
    - params (dict): The parameters set
    - implementation (str): The implementation type ('Active-RC', 'Gm-C')

    Returns:
    - cond (bool): True if the parameters are valid, False otherwise
    """
    
    # add relationships between parameters as needed:
    
    """
    Example:
    
    if implementation == 'Active-RC':
        gm = params.get('gm')
        Ro = params.get('Ro')

        if gm * Ro > 1e3:
            return False # invalid relationship

    elif implementation == 'Gm-C':
        gm = params.get('gm')
        Ro = params.get('Ro')

        if gm * Ro < 1e-2:
            return False # invalid relationship
    """
    
    return True # if no issues, return True to indicate the parameters are valid



if __name__ == "__main__":
    # dataset generator test:

    form = 'FB'
    order = 2
    implementation = 'Active-RC'
    t_min = 600
    num_samples = 50
    min_snr = 50

    # define parameter ranges (set as function args better?):
    if implementation == 'Active-RC':
        parameter_ranges = [
            (10e3, 10e9), # Bw (Bandwidth)
            [8, 16, 32, 64, 128, 256], # osr (Oversampling Ratio) - Discrete
            [2, 3, 4, 5], # nlev (Number of Levels) - Discrete
            (1.0, 2.0), # Hinf (Infinity Gain)
            (100e-15, 5e-12), # Cint (Integration Capacitance)
            (1e-6, 1e-3), # gm (Transconductance)
            (100e3, 10e6), # Ro (Output Resistance)
            (50e-15, 1e-12), # Co (Output Capacitance)
        ]
    elif implementation == 'Gm-C':
        parameter_ranges = [
            (10e3, 10e9), # Bw (Bandwidth)
            [8, 16, 32, 64, 128, 256], # osr (Oversampling Ratio) - Discrete
            [2, 3, 4, 5], # nlev (Number of Levels) - Discrete
            (1.0, 2.0), # Hinf (Infinity Gain)
            (100e-15, 2e-12), # Cint (Integration Capacitance)
            (10e3, 1e6), # Ro (Output Resistance)
            (1e-15, 100e-15), # Cp (Capacitor)
            (10e-9, 1e-6), # v_n (Noise Voltage)
            (1e-6, 100e-6), # slew_rate (Slew Rate)
            (0.5, 3.0), # output_swing (Output Swing)
        ]

    # define discrete parameters:
    discrete_params = [
        (1, [8, 16, 32, 64, 128, 256]), # osr (discrete)
        (2, [2, 3, 4, 5]) # nlev (discrete)
    ]

    dataset = generate_dataset(form, order, implementation, t_min, num_samples, min_snr, parameter_ranges, discrete_params)
    print(dataset.head())
