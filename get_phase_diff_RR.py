import numpy as np
import argparse
import functions as fct

# Parameter dictionary
param = {
    
    # Integration
    "total_t"   : 405.,
    "dt"        : 0.1,
    
    # Initial conditions
    "init_phi_A" : 0.,
    "init_phi_B" : 0.,
    
    # Frequencies
    "omega_A" : 0.0457,
    "omega_B" : 0.0457,

    # Coupling
    "c"       : 0.015,
    "phi*"    : 0.,
    "s_sigma" : 0.1,
    "beta"    : 0.
    
}

# The beta parameter and the coupling strength must be specified when running this code
parser = argparse.ArgumentParser(description="Generates phase difference data for visualizing the synchronization outcome of the rectified pulsed-coupling model")
parser.add_argument('-p', '--param', nargs=2, type=float, help='Beta parameter & coupling strength')
args = parser.parse_args()
param["beta"] = args.param[0]
param["c"] = args.param[1]

# Initial conditions
n_init = 100
init_phiA = np.linspace(-np.pi, np.pi, n_init)
init_phiB = np.linspace(-np.pi, np.pi, n_init)
phase_diff = np.zeros((3, n_init*n_init))
for j in range(n_init):
    for k in range(n_init):

        # A
        param["init_phi_A"] = init_phiA[j]
        param["init_phi_B"] = init_phiA[j]
        phi_A1, phi_A2 = fct.integrate_last(fct.H_RR, param)

        # B
        param["init_phi_A"] = init_phiB[k]
        param["init_phi_B"] = init_phiB[k]
        phi_B1, phi_B2 = fct.integrate_last(fct.H_RR, param)

        # AB (mix)
        param["init_phi_A"] = init_phiA[j]
        param["init_phi_B"] = init_phiB[k]
        phi_AB1, phi_AB2 = fct.integrate_last(fct.H_RR, param)

        # Phases
        phi_A = fct.compute_phase_mix(phi_A1, phi_A2)
        phi_B = fct.compute_phase_mix(phi_B1, phi_B2)
        phi_AB = fct.compute_phase_mix(phi_AB1, phi_AB2)
        
        # Phase differences
        phase_diff[0,j*n_init+k] = (phi_A-phi_B) %(2.*np.pi)
        phase_diff[1,j*n_init+k] = (phi_A-phi_AB) %(2.*np.pi)
        phase_diff[2,j*n_init+k] = (phi_B-phi_AB) %(2.*np.pi)
        
# Cast the phase differences between -π and π
phase_diff[phase_diff > np.pi] -= 2.*np.pi

# Label which oscillator is ahead / behind
phase_diff_label = phase_diff.copy()
phase_diff_label[1,phase_diff[0]< 0.] = phase_diff[2,phase_diff[0]< 0.]
phase_diff_label[2,phase_diff[0]< 0.] = phase_diff[1,phase_diff[0]< 0.]

# Output file
np.savetxt('Phase_diff_sync_outcome_RR_beta='+str(args.param[0])+'_c='+str(args.param[1])+'.txt', phase_diff_label, fmt='%.5f')