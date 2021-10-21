import numpy as np
import matplotlib.pyplot as plt

def step_function(t, ts, te):
    array_len = len(t)
    tstep = t[1] - t[0]
    array = np.zeros(array_len)
    ts_num = int(ts/tstep)
    te_num = int(te/tstep)
    try:
        array[ts_num: te_num] = 1
    except ts_num - te_num:
        raise ValueError("ending time should be bigger than start time!!")
    return array

def MW_Rabi_coeff(t, args):
## this MW is meant to solve time-dependent master equation by function method
    H1 = args['H1']
    omega_rf = args['omega_rf']
    phi = args['phi']
    t_mw = args['t_mw']
    H1_coeff = H1* np.sin(2*np.pi*omega_rf* t + phi)* step_function(t, 0, t_mw)
    return H1_coeff


def MW_sequence_plot(tlist):
    H1 = 80
    omega_rf = 2000
    phi = 0
    t_mw = 1
    argdict = {'H1': H1, 'omega_rf': omega_rf, 'phi': phi, 't_mw': t_mw}
    seq = MW_Rabi_coeff(tlist, argdict)
    plt.plot(tlist, seq)
    plt.show()
    return

tlist = np.linspace(0,2,20000)


MW_sequence_plot(tlist)