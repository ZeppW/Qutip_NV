import qutip as Q
import numpy as np
import matplotlib.pyplot as plt

spin = 0.5
def T2_decay(tlist):
    rf = 2*np.pi*0 ##MHz
    H0 = rf* Q.jmat(spin, 'z')
    rho0 = Q.Qobj([[0.5, 0.5],[0.5, 0.5]]) ## pure state of sx=+1
    ##print(rho0)
    T2 = 1 ##us
    c_rate = np.sqrt(2/T2)
    c_ops = [c_rate* Q.jmat(spin, 'z')]
    e_ops = [Q.jmat(spin, 'x'), Q.jmat(spin, 'y'), Q.jmat(spin, 'z')]
    result = Q.mesolve(H0, rho0, tlist, c_ops, e_ops)
    return result

def T1andT2_decay(tlist):
    rf = 2*np.pi*10 ##MHz
    H0 = rf* Q.jmat(spin, 'z')
    rho0 = Q.Qobj([[0.5, 0.5],[0.5, 0.5]]) ## pure state of sz=+1
    ##print(rho0)
    T2 = 1 ##us
    T1 = 1 ##us
    ## The real T2 is: 1/T2 = 1/T2 + 1/(2*T1)
    T2_rate = np.sqrt(2/T2)
    T1_rate = np.sqrt(1/T1)
    splus = Q.Qobj([[0, 1],[0, 0]])
    print(splus)
    c_ops = [T2_rate* Q.jmat(spin, 'z'), T1_rate* splus]
    e_ops = [Q.jmat(spin, 'x'), Q.jmat(spin, 'y'), Q.jmat(spin, 'z')]
    result = Q.mesolve(H0, rho0, tlist, c_ops, e_ops)
    return result

tlist = np.linspace(0,2,2000)
result = T1andT2_decay(tlist)
plt.plot(tlist, result.expect[0])
plt.plot(tlist, result.expect[1])
plt.plot(tlist, result.expect[2])
plt.show()
data = np.transpose(np.array([tlist, result.expect[0], result.expect[1], result.expect[2]]))
path = "f:/NMR/NMR/py_projects/Qutip/data_txt/"
np.savetxt(path + "T1andT2_decay_10MHz_1usT2_1usT1.txt", data)