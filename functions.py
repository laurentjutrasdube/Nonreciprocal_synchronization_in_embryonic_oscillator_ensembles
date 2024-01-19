import numpy as np
import matplotlib.pyplot as plt
import glob
from pyboat import WAnalyzer



def gaussian(x, mu, sigma):
    
    return np.exp(-0.5*((x-mu)/sigma)**2)





# 1. Retrieve info from file names

def get_file_names():
    
    data_dir = './RAFL_data/'
    return np.array(glob.glob(data_dir+'*.txt'))



def get_exp_info():

    # File names
    data_dir = './RAFL_data/'
    file_names = np.array(glob.glob(data_dir+'*.txt'))
    n_exp = len(file_names)

    # Experiment info
    exp_type = np.zeros(n_exp, dtype='<U6')
    exp_date = np.zeros(n_exp, dtype='<U8')
    exp_dish = np.zeros(n_exp, dtype='<U6')
    exp_dt = np.zeros(n_exp)
    exp_t0 = np.zeros(n_exp)

    for k in range(n_exp):

        temp = file_names[k].split('/')[2].split('.')[0].split('_')

        exp_type[k] = temp[0]
        exp_date[k] = temp[1]
        exp_dish[k] = temp[2]
        exp_dt[k] = temp[3]
        exp_t0[k] = temp[4]
        
    return exp_type, exp_date, exp_dish, exp_dt, exp_t0





# 2. Retrieve time series data and compute oscillation phase

def get_phase(file_name, roi=False):
    
    # Raw signal
    data = np.genfromtxt(file_name,delimiter=' ').T
    if (len(data) == len(data.T)):    data = np.expand_dims(data,axis=0)

    # Time points
    dt = float(file_name.split('/')[2].split('.')[0].split('_')[3])
    t0 = float(file_name.split('/')[2].split('.')[0].split('_')[4])
    times = np.arange(t0, t0+dt*len(data[0]), dt)
     
    # Detrended signal
    p_max = 200.
    periods = np.linspace(100., 180., 150)
    wAn = WAnalyzer(periods=periods,dt=dt,p_max=p_max)

    for j in range(len(data)):
        trend = wAn.sinc_smooth(data[j],T_c=p_max)
        data[j] = (data[j]-trend)
    
    # Regions of interests, detrended phase
    if roi:
        for j in range(len(data)):
            wAn.compute_spectrum(data[j], do_plot=False)
            wAn.get_maxRidge()
            data[j] = np.array(wAn.ridge_data['phase'])
        
        return times, data
    
    # Phase of the mean detrended signal
    t_min_i = np.where(times >= max(200., min(times)))[0][0]
    t_max_i = np.where(times >= min(1000., max(times)))[0][0]

    data = np.mean(data, axis=0)
    data = data/np.max(np.abs(data[t_min_i:t_max_i]))

    wAn.compute_spectrum(data, do_plot=False)
    wAn.get_maxRidge()
    data = np.array(wAn.ridge_data['phase'])
 
    return times, data



def get_data(exp_date, exp_type='AB', output_type=None, phase=False):
    
    data_dir = './RAFL_data/'

    # AB experiments
    if exp_type == 'AB':
        file_names = np.array([glob.glob(data_dir+'AB_'+exp_date+'_1*.txt')[0],
                               glob.glob(data_dir+'AB_'+exp_date+'_2*.txt')[0],
                               glob.glob(data_dir+'AB_'+exp_date+'_M*.txt')[0]])
        
    # Twin experiments
    elif exp_type == 'twin':
        file_names = np.array([glob.glob(data_dir+'AA_'+exp_date+'_1*.txt')[0],
                               glob.glob(data_dir+'AA_'+exp_date+'_2*.txt')[0]])
        
    # ABC experiments
    elif exp_type == 'ABC':
        file_names = np.array([glob.glob(data_dir+'ABC_'+exp_date+'_1*.txt')[0],
                               glob.glob(data_dir+'ABC_'+exp_date+'_2*.txt')[0],
                               glob.glob(data_dir+'ABC_'+exp_date+'_3*.txt')[0],
                               glob.glob(data_dir+'ABC_'+exp_date+'_M*.txt')[0]])
        
    # Any other experiment type returns empty arrays
    else:
        print('exp_type can be either AB, twin or O. Returning empty arrays.')
        return [], []

    # Raw signal
    data = []
    for file_name in file_names:
        temp = np.genfromtxt(file_name, delimiter=' ').T
        if (len(temp) == len(temp.T)):    temp = np.expand_dims(temp,axis=0)
        data.append(temp)

    # Time points
    dt = float(file_names[0].split('/')[2].split('.')[0].split('_')[3])
    t0 = float(file_names[0].split('/')[2].split('.')[0].split('_')[4])
    times = np.arange(t0, t0+dt*len(data[0][0]), dt)
    
    # Regions of interest, raw
    if (output_type == 'raw'):    
        
        if phase:
            print('The raw signal needs to be detrended before\n' 
                 +'calculating the phase. Returning the raw signal.')
        
        return times, data
     
    # Detrended signal
    p_max = 200.
    periods = np.linspace(100., 180., 150)
    wAn = WAnalyzer(periods=periods,dt=dt,p_max=p_max)

    for k in range(len(data)):
        for j in range(len(data[k])):
            trend = wAn.sinc_smooth(data[k][j],T_c=p_max)
            data[k][j] = (data[k][j]-trend)
    
    # Regions of interests, detrended
    if (output_type == 'detrended'):
        
        # Phase of the regions of interests
        if phase:
            for k in range(len(data)):
                for j in range(len(data[k])):
                    wAn.compute_spectrum(data[k][j], do_plot=False)
                    wAn.get_maxRidge()
                    data[k][j] = np.array(wAn.ridge_data['phase'])
                
        return times, data
    
    # Mean detrended signal
    t_min_i = np.where(times >= max(200., min(times)))[0][0]
    t_max_i = np.where(times >= min(1000., max(times)))[0][0]

    for k in range(len(data)):
        data[k] = np.mean(data[k], axis=0)
        data[k] = data[k]/np.max(np.abs(data[k][t_min_i:t_max_i]))
    data = np.array(data)

    # Phase of the mean detrended signal
    if phase:
        for k in range(len(data)):
            wAn.compute_spectrum(data[k], do_plot=False)
            wAn.get_maxRidge()
            data[k] = np.array(wAn.ridge_data['phase'])
 
    return times, data





# 3. Generate plots

def ts_plot(times, data, file_name, t=400., sim=False, legend=True):

    # Initialize the figure
    plt.rcParams["figure.figsize"] = 10., 2.6
    fig, ax = plt.subplots()

    # Plot the time series: controls
    colors = ['orangered', 'deepskyblue', 'gold']
    labels = ['A', 'B', 'C']
    for k in range(len(data)-1):
        ax.plot(times, data[k], c=colors[k], label=labels[k], lw=6, zorder=k+3)

    # Plot the time series: mix
    ax.plot(times, data[-1], c='purple', label='Mix', lw=6, ls='--', zorder=5)
    
    # Line at the specified time point (polar plot)
    ax.plot([t, t], [-1.1, 1.1], c='k', lw=5, zorder=2)

    # Axes, ticks, grid, legend
    ylabel = ' \nSignal (a.u.)'
    if sim:    ylabel = ' \ncos($\phi$)'
    ax.set_xlabel('Time (min.)', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim([200., 800.])
    ax.set_ylim([-1.1, 1.1])
    ax.tick_params(labelsize=18)
    ax.grid(which='major')
    if legend:    ax.legend(loc=6, fontsize=18)

    # Save the figure
    fig.tight_layout()
    fig.savefig('Figures/'+file_name+'.pdf', dpi=300)



def polar_plot(times, data, file_name, t=400.):
    
    # Initialize the figure
    plt.rcParams["figure.figsize"] = 7., 7.
    fig, ax = plt.subplots()

    # Time point for the polar plot
    t_index = np.where(times == t)[0][0]

    # Axes
    ax.plot([-1.4,1.4], [0.,0.], c='k', lw=8, zorder=0)
    ax.plot([0.,0.], [-1.4,1.4], c='k', lw=8, zorder=0)

    # Axes' labels
    ax.text(-1.55, 1.35, '$\sin(\phi)$', color='k', fontsize=54)
    ax.text(1.35, -0.45, '$\cos(\phi)$', color='k', fontsize=54)

    # Circle
    x = np.linspace(-1.,1.,1000)
    ax.plot(x, np.sqrt(1-x**2), c='k', lw=8, zorder=0)
    ax.plot(x, -np.sqrt(1-x**2), c='k', lw=8, zorder=0)

    # Polar plot: controls
    colors = ['orangered', 'deepskyblue', 'gold']
    for k in range(len(data)-1):
        ax.plot(np.cos(data[k,t_index]), np.sin(data[k,t_index]), marker='o', markeredgecolor=colors[k], 
                markerfacecolor=colors[k], markersize=50., markeredgewidth=12., zorder=2*k+1)
        ax.plot([0., np.cos(data[k,t_index])], [0., np.sin(data[k,t_index])], c=colors[k], lw=20, zorder=2*k+2)

    # Polar plot: mix
    ax.plot(np.cos(data[-1,t_index]), np.sin(data[-1,t_index]), marker='o', markeredgecolor='purple', 
                markerfacecolor='none', markersize=50., markeredgewidth=12., zorder=20)
    ax.plot([0., 0.7*np.cos(data[-1,t_index])], [0., 0.7*np.sin(data[-1,t_index])], c='purple', lw=20, zorder=21)

    # Aspect
    ax.set_aspect('equal')
    ax.axis('off')

    # Save the figure
    fig.tight_layout()
    fig.savefig('Figures/'+file_name+'.pdf', dpi=300)





# 4. Simulate models: coupling functions and integrator

def H_plus(phi_1, phi_2, param):

    dphi = phi_2-phi_1
    if (dphi > 0.):    H = np.sin(dphi)
    else:    H = 0.

    return H



def H_minus(phi_1, phi_2, param):

    dphi = phi_2-phi_1
    if (dphi < 0.):    H = np.sin(dphi)
    else:    H = 0.

    return H



def H_KS(phi_1, phi_2, param):
    
    return np.sin(phi_2-phi_1+param["alpha"])-np.sin(param["alpha"])



def H_RK(phi_1, phi_2, param):
    
    sin = np.sin(phi_2-phi_1)
    
    return sin*(1.-np.tanh(param["beta"]*sin))



def H_RK_linear(phi_1, phi_2, param):
    
    dphi = phi_2-phi_1
    sign = np.sign(dphi)
    
    H = (1. -param["gamma"]*sign)*dphi

    if (np.abs(dphi) > np.pi*0.5):
        H = (1. -param["gamma"]*sign)*(np.pi*sign -dphi)
    
    return 0.65*H



def H_RR(phi_r, phi_s, param):
    
    # Pulse phase
    phi_r -= param["phi*"]
    phi_s -= param["phi*"]
    
    # Modulo 2π
    phi_r = phi_r %(2.*np.pi)
    phi_s = phi_s %(2.*np.pi)
    
    # Between -π and π
    if (phi_r > np.pi):    phi_r -= 2.*np.pi
    if (phi_s > np.pi):    phi_s -= 2.*np.pi
        
    # Stimulus
    s = gaussian(phi_s, 0., param["s_sigma"])
    
    # Response
    r = np.sin(-phi_r)*(1.-np.tanh(param["beta"]*np.sin(-phi_r)))
    
    return r*s



def H_RR_linear(phi_r, phi_s, param):
    
    # Pulse phase
    phi_r -= param["phi*"]
    phi_s -= param["phi*"]
    
    # Modulo 2π
    phi_r = phi_r %(2.*np.pi)
    phi_s = phi_s %(2.*np.pi)
    
    # Between -π and π
    if (phi_r > np.pi):    phi_r -= 2.*np.pi
    if (phi_s > np.pi):    phi_s -= 2.*np.pi
        
    # Stimulus
    s = gaussian(phi_s, 0., param["s_sigma"])
    
    # Response
    sign_r = np.sign(-phi_r)
    r = (1. -param["gamma"]*sign_r)*-phi_r
    if (np.abs(phi_r) > np.pi*0.5):    
        r = (1. -param["gamma"]*sign_r)*(np.pi*sign_r +phi_r)
    
    return 0.65*r*s



def integrate(coupling_fct, param):
    
    # Times
    times = np.arange(0., param["total_t"], param["dt"])
    times_plot = [0.]
    
    # Initial conditions
    phi_A = param["init_phi_A"]
    phi_B = param["init_phi_B"]
                
    # Results' lists
    results_phi_A = [phi_A]
    results_phi_B = [phi_B]
   
    # Integration
    counter = 0
    n_skr = param["n_skr"]
    for t in times[1:]:
        
        # Coupling term
        dphi_A = param["omega_A"] +param["c"]*coupling_fct(phi_A, phi_B, param)
        dphi_B = param["omega_B"] +param["c"]*coupling_fct(phi_B, phi_A, param)
        
        # Updated phases
        phi_A += dphi_A*param["dt"]
        phi_B += dphi_B*param["dt"]
        
        # Gaussian noise
        phi_A += np.random.normal(loc=0., scale=param["noise_std"])*np.sqrt(param["dt"])
        phi_B += np.random.normal(loc=0., scale=param["noise_std"])*np.sqrt(param["dt"])
    
        # Save results
        counter += 1
        if (counter %n_skr == 0):
            results_phi_A.append(phi_A)
            results_phi_B.append(phi_B)
            times_plot.append(t)
      
    # Output
    return np.array(times_plot), np.array([results_phi_A, results_phi_B])



def integrate_last(coupling_fct, param):
    
    # Initialize the array of times
    times = np.arange(0., param["total_t"], param["dt"])
    
    # Set the initial conditions for the two oscillators
    phiA = param["init_phi_A"]
    phiB = param["init_phi_B"]
   
    # Perform the integration
    for t in times[1:]:
        
        # Coupling term
        dphiA = param["omega_A"] +param["c"]*coupling_fct(phiA, phiB, param)
        dphiB = param["omega_B"] +param["c"]*coupling_fct(phiB, phiA, param)
        
        # Update the state of the phase oscillators
        phiA += dphiA*param["dt"]
        phiB += dphiB*param["dt"]
    
    return phiA, phiB





# 5. Compute phase differences (Note: these work only for phase diff. models such as KS & RK)

def compute_phase_mix(phi1, phi2):

    x = 0.5*(np.cos(phi1)+np.cos(phi2))
    y = 0.5*(np.sin(phi1)+np.sin(phi2))

    return np.arctan2(y,x)



def get_phase_diff(coupling_fct, param, label=True):

    # Initial conditions
    n_init = 100
    init_phiB = np.linspace(0., np.pi, n_init)
    phase_diff = np.zeros((3, n_init))
    for k in range(n_init):

        # A
        param["init_phi_A"] = 0.
        param["init_phi_B"] = 0.
        phi_A1, phi_A2 = integrate_last(coupling_fct, param)

        # B
        param["init_phi_A"] = init_phiB[k]
        param["init_phi_B"] = init_phiB[k]
        phi_B1, phi_B2 = integrate_last(coupling_fct, param)

        # AB (mix)
        param["init_phi_A"] = 0.
        param["init_phi_B"] = init_phiB[k]
        phi_AB1, phi_AB2 = integrate_last(coupling_fct, param)

        # Phases
        phi_A = compute_phase_mix(phi_A1, phi_A2)
        phi_B = compute_phase_mix(phi_B1, phi_B2)
        phi_AB = compute_phase_mix(phi_AB1, phi_AB2)
        
        # Phase differences
        phase_diff[0,k] = (phi_A-phi_B) %(2.*np.pi)
        phase_diff[1,k] = (phi_A-phi_AB) %(2.*np.pi)
        phase_diff[2,k] = (phi_B-phi_AB) %(2.*np.pi)
        
    # Cast the phase differences between -π and π
    phase_diff[phase_diff > np.pi] -= 2.*np.pi
    if not label:    return phase_diff

    # Label which oscillator is ahead / behind
    phase_diff_label = phase_diff.copy()
    phase_diff_label[1,phase_diff[0]<0.] = phase_diff[2,phase_diff[0]<0.]
    phase_diff_label[2,phase_diff[0]<0.] = phase_diff[1,phase_diff[0]<0.]

    return phase_diff_label













