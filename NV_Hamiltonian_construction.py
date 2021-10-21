import qutip as Q
import numpy as np
import matplotlib.pyplot as plt

def H1_dist(inputpath):
    H1_pdf = np.loadtxt(inputpath + "histogramDataStraight150.txt", delimiter=",")
    H1_pdf=np.transpose(H1_pdf)
    ## converting H1 into Omega
    gamma = 28025 ## MHz/T
    H1_pdf[0] = H1_pdf[0]* gamma/10000
    ## normalization
    H1_pdf[1] = H1_pdf[1]/sum(H1_pdf[1])
    return np.transpose(H1_pdf)


def step_function(t, ts, te):
    if isinstance(t, np.ndarray):
        array_len = len(t)
        tstep = t[1] - t[0]
        array = np.zeros(array_len)
        ts_num = int(ts/tstep)
        te_num = int(te/tstep)

        array[ts_num: te_num] = 1
         
        return array
    else:
        if ts < t < te:
            return 1
        else:
            return 0
    

def NV_H_single(spin, H0, D = 2870):
## at first, only the zero field splitting, external zeeman is included
    gamma = 28025 ## MHz/T
    HH = D* Q.jmat(1, 'z')**2 + gamma* H0* Q.jmat(1, 'z')
    if spin == 1:
        return HH
    else:
        pMat = Q.Qobj([[0, 0],[1, 0],[0, 1]])
        HH = pMat.dag()* HH* pMat
    return HH


def MW_Rabi_coeff(t, args):
## this MW is meant to solve time-dependent master equation by function method
    H1 = args['H1']
    omega_rf = args['omega_rf']
    phi = args['phi']
    t_mw = args['t_mw']

    H1_coeff = H1* np.sin(omega_rf* t + phi)* step_function(t, 0, t_mw)

    return H1_coeff


def MW_Ramsey_coeff(t, args):
    H1 = args['H1']
    omega_rf = args['omega_rf']
    phase_l = args['phase_l']
    t_90 = args['t_90']
    t_wait = args['t_wait']

    H1_coeff = H1* (np.sin(omega_rf* t + phase_l[0])* step_function(t, 0, t_90) + 
    np.sin(omega_rf* t + phase_l[1])* step_function(t, t_wait, t_wait + t_90))

    return H1_coeff


def MW_echo_coeff(t, args):
    H1 = args['H1']
    omega_rf = args['omega_rf']
    phase_l = args['phase_l']
    t_90 = args['t_90']
    tau_l = args['tau_l']

    H1_coeff = H1* (np.sin(omega_rf* t + phase_l[0])* step_function(t, 0, t_90) + 
        np.sin(omega_rf* t + phase_l[1])* step_function(t, tau_l[0] - t_90/2, tau_l[0] + 3* t_90/2) + 
        np.sin(omega_rf* t + phase_l[2])* step_function(t, tau_l[0] + tau_l[1], tau_l[0] + tau_l[1] + t_90))
    return H1_coeff

def MW_XY8N_coeff(t, args):
    XY8_phase = [0, np.pi/2, 0, np.pi/2, np.pi/2, 0, np.pi/2, 0]

    H1 = args['H1']
    omega_rf = args['omega_rf']
    XY8num = args['XY8num']
    phase_l = args['phase_l']
    t_90 = args['t_90']
    tau = args['tau']

    t_pointer = 0
    ## first 90deg pulse
    H1_coeff = H1* np.sin(omega_rf* t + phase_l[0])* step_function(t, 0, t_90)
    t_pointer += tau/2 - t_90/2
    
    for num in range(XY8num):
        for xyphase in XY8_phase:
            H1_coeff += H1* np.sin(omega_rf* t + xyphase)* step_function(t, t_pointer, t_pointer + t_90* 2)
            t_pointer += tau

    ## last 90 deg pulse
    t_pointer += tau/2 + t_90/2
    H1_coeff += H1* np.sin(omega_rf* t + phase_l[1])* step_function(t, t_pointer, t_pointer + t_90)

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


def lindblad_solver(spin, MW_coeff, dictargs, tlist):


    HHint = Q.jmat(spin, 'x')/spin
    HH0 = NV_H_single(spin, H0, D)
    ##print(HH0)
    HHtot = [HH0, [HHint, MW_coeff]]
    psi0 = Q.basis(int(spin*2 + 1), 0)
    rho0 = psi0* psi0.dag()
    ##print(rho0)

    
    # if spin == 1/2:
    #     T2z = 148.15 ##us
    #     T1 = 100 ##us
    #     ## The real T2 is: 1/T2 = 1/T2z + 1/(2*T1)
    #     T2_rate = np.sqrt(2/T2z)
    #     T1_rate = np.sqrt(1/T1)
    #     splus = Q.Qobj([[0, 1],[0, 0]])
    #     c_ops = [T2_rate* Q.jmat(spin, 'z'), T1_rate* splus]
    # else:
    #     c_ops = []

    c_ops = []

    
    ## mesolve means Lindblad master equation solver, which includes collapse operators to interact with enviroment
    projectionz = Q.mesolve(HHtot, rho0, tlist, c_ops, [Q.jmat(spin, 'x')/spin], args = dictargs)
    return projectionz




savepath = "f:/NMR/NMR/py_projects/Qutip/data_txt/"
inputpath = "f:/NMR/NMR/py_projects/Qutip/"


## parameters
D = 2870 ## zero field splitting
H0 = 870/28025 ## external field; which makes reasonance frequency@2GHz
H1 = 2*np.pi* 10 ## MW Rabi frequency, which has same unit as gamma*H; MHz
omega_rf = 2000 ## MW pulse frequency; MHz
spin = 0.5

## Rabi test
# tlist = np.linspace(0,2,2000)
# t_mw = 1 ## us
# phi = 0
# MW_dict = {'H1': H1, 'omega_rf': omega_rf, 'phi': phi, 't_mw': t_mw}
# result = lindblad_solver(spin, MW_Rabi_coeff, MW_dict, tlist)
# plt.plot(tlist, result.expect[0])
# plt.show()

# ## Ramsey test
# tlist = np.linspace(0,2,2000)
# phase = [0, -np.pi/2]
# t_90 = 2* np.pi/(4* H1) ## us
# wait_list = [1]##np.linspace(t_90, 1.8, 256)
# sz_measure = []
# for t_wait in wait_list:
#     MW_dict = {'H1': H1, 'omega_rf': omega_rf, 'phase_l': phase, 't_90': t_90, 't_wait': t_wait}
#     result = lindblad_solver(spin, MW_Ramsey_coeff, MW_dict, tlist)
#     sz_measure.append(result.expect[0][-1])
# sz_measure = np.array(sz_measure)
# plt.plot(tlist, result.expect[0], 'o-b')
# plt.show()
# ##data = np.transpose(np.array([wait_list, sz_measure]))

## echo test
t_90 = 2* np.pi/(4* H1) ## us
phase = [0, 0, 0]
tau_list = np.linspace(1,10,10)
t_delay = 1
dt = 5e-3
sz_measure = []
for tau in tau_list:
    tlist = np.linspace(0, 25, 20000)
    tau_l = [tau, tau]
    MW_dict = {'H1': H1, 'omega_rf': omega_rf, 'phase_l': phase, 't_90': t_90, 'tau_l': tau_l}
    result = lindblad_solver(spin, MW_echo_coeff, MW_dict, tlist)
    sz_measure.append(np.mean(result.expect[0][-1]))

sz_measure = np.array(sz_measure)

plt.plot(tau_list, sz_measure, 'o-b')
#plt.plot(tlist, result.expect[0], 'o-b')
plt.show()

# data = np.transpose(np.array([tau_list, sz_measure]))

##np.savetxt(savepath + "echo_sz_vs_tau_2.txt", data)