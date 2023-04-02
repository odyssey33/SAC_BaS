import numpy as np

"""
TODO: add continuous bas functions and make choosing discrete or continuous an option.
Based on the option, bas_dyn and bas_grad should be sent to embed_dynamics instead of discrete_bas and grad as now. 
"""


class BaSDynamics:
    def __init__(self, safety_function, init_state, des_state, gamma=0, barrier_type='inverse_barrier', n_bas=1):
        #self.dynamics = dynamics                                        # systems dynamics
        self.safety_function = safety_function                          # safety functions h (array)
        self.gamma = gamma
        self.n_bas = n_bas                                              # number of barrier states
        if barrier_type == 'inverse_barrier':
            self.barrier = self.inverse_barrier
            self.barrier_derivative = inverse_barrier_derivative
        elif barrier_type == 'log_barrier':
            self.barrier = self.log_barrier
            self.barrier_derivative = log_barrier_derivative
        self.bas_initial = self.barrier(init_state)[0]            # bas initial condition
        self.bas_terminal = self.barrier(des_state)[0]              # bas terminal condition

    def inverse_barrier(self, x):  # returns inverse barrier value given x
        """ This creates inverse barrier function column vector Beta.
        n_bas is either 1 or equal to number of constraints/h functions """
        
        #beta = np.zeros((self.n_bas, 1))
        h = self.safety_function(x)[0] # return h1 h2 h3 for each obstacles
        const_violation_flag = False
        if x.ndim == 1:
            numIter = 1
            beta = np.zeros((self.n_bas, 1))
            h_dim = h.shape[0]
        
            for jj in range(self.n_bas):
                for ii in range(int(h_dim/self.n_bas)):
                    beta[jj] += 1 / h[ii+jj]
                    if 1 / h[ii + jj] < - 1e-10:
                        const_violation_flag = True
        else:
            numIter = x.shape[0]
            beta = np.zeros((numIter, self.n_bas))
            h_dim = h.shape[1]
            for n in range(numIter):
                for jj in range(self.n_bas):
                    for ii in range(int(h_dim/self.n_bas)):
                        beta[n, jj] += 1 / h[n, ii+jj]
                        if 1 / h[n, ii + jj] < - 1e-10:
                            const_violation_flag = True
                            #np.log(np.maximum(1.001, 1 + beta))
        return beta, const_violation_flag

    def log_barrier(self, x):  # returns log barrier value given x
        """ This creates log barrier function column vector Beta.
        n_bas is either 1 or equal to number of constraints/h functions """
        beta = np.zeros((self.n_bas, 1))
        h = self.safety_function(x)[0]
        if x.ndim == 1:
            numIter = 1
            beta = np.zeros((self.n_bas, 1))
            h_dim = h.shape[0]
            for jj in range(self.n_bas):
                for ii in range(int(h.shape[0]/self.n_bas)):
                    beta[jj] += np.nan_to_num(- np.log(h[ii+jj]/(1+h[ii+jj])))
            return beta if beta != 0 else 1e20
        else:
            numIter = x.shape[0]
            beta = np.zeros((numIter, self.n_bas))
            h_dim = h.shape[1]
            for n in range(numIter):
                for jj in range(self.n_bas):
                    for ii in range(int(h_dim/self.n_bas)):
                        beta[n, jj] += np.nan_to_num(- np.log(h[n, ii+jj] / (1 + h[n, ii+jj])))
                        if beta[n, jj] == 0:
                            beta[n, jj] == 1e20 
            return beta


    # def discrete_bas_dynamics(self, x, u):
    #     """ This returns the discrete barrier state given current state and control, which uses the system's model
    #     to compute the value of the next barrier state fw = w_next = Beta(h(f(x,u))) """
    #     x_next = self.dynamics.system_propagate(x[0:self.dynamics.n, :], u)
    #     return self.barrier(x_next)
    def discrete_bas_dyn(self, x, x_next, bas_indx):
        """ This returns the (next) discrete barrier state given current state and control, which uses the system's
        model to compute the value of the next barrier state fw = w_next = Beta(h(f(x,u))) - gamma (wk - B(h(xk)))"""
        #x_next = self.dynamics.system_propagate(x[:self.dynamics.n, :], u)
        #if self.barrier_type == 'inverse_barrier':
        # if x.ndim == 1:
        #     w = x[bas_indx-1]#x[bas_indx-self.n_bas, :]
        #     w_next = self.barrier(x_next)[0] - self.gamma * (w - self.barrier(x)[0])
        # else:
        #     w = x[:, bas_indx-1]
        #     w_next = self.barrier(x_next)[0].reshape(-1) - self.gamma * (w - self.barrier(x)[0].reshape(-1))
        if x.ndim == 1:
            w = x[bas_indx-1]#x[bas_indx-self.n_bas, :]
            w_next = self.barrier(x_next) - self.gamma * (w - self.barrier(x))
        else:
            w = x[:, bas_indx-1]
            w_next = self.barrier(x_next).reshape(-1) - self.gamma * (w - self.barrier(x).reshape(-1))
        return w_next, self.barrier(x_next)
    
    # def discrete_bas_dyn(self, x, u, bas_indx):
    #     """ This returns the (next) discrete barrier state given current state and control, which uses the system's
    #     model to compute the value of the next barrier state fw = w_next = Beta(h(f(x,u))) - gamma (wk - B(h(xk)))"""
    #     x_next = self.dynamics.system_propagate(x[:self.dynamics.n, :], u)
    #     w = x[bas_indx-self.n_bas, :]
    #     w_next = self.barrier(x_next)[0] - self.gamma * (w - self.barrier(x)[0])
    #     return w_next, self.barrier(x_next)[1]

    def discrete_bas_grad(self, x, u):
        """ This returns the discrete barrier state's **analytical** gradients with respect to the state and control
        (fw_x and fw_u) given current state and control, which uses the system's gradients fx and fu """
        x_next = self.dynamics.system_propagate(x[0:self.dynamics.n, :], u)
        fw_x = np.zeros((self.n_bas, x.shape[0]))
        fw_u = np.zeros((self.n_bas, self.dynamics.m))
        h, hx = self.safety_function(x_next)
        h_next, hx_next = self.safety_function(x_next)
        for jj in range(self.n_bas):
            for ii in range(int(h_next.shape[0]/self.n_bas)):
                B_h = self.barrier_derivative(h[ii+jj])
                B_h_f = self.barrier_derivative(h_next[ii+jj])
                h_x = hx[ii+jj]
                h_f = hx_next[ii+jj]
                f_x = self.dynamics.system_grad(x, u)[0]
                f_u = self.dynamics.system_grad(x, u)[1]
                fw_w = - self.gamma * np.ones((1, len(x) - self.dynamics.n))
                fw_x[jj, :] += np.concatenate((B_h_f * h_f @ f_x + self.gamma * B_h * h_x, fw_w), axis=None)
                fw_u[jj, :] += B_h_f * h_f @ f_u
        return fw_x, fw_u


def inverse_barrier_derivative(h):
    """ This returns the inverse barrier function's derivative with respect to its argument, the safety function """
    return -1/h**2


def log_barrier_derivative(h):
    """ This returns the log barrier function's derivative with respect to its argument, the safety function """
    return -1/(h**2+h)
