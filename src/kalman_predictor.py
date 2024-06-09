import numpy as np
from scipy.stats import multivariate_normal

class Params:
    def __init__(self, pi, Sigma, a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi):
        # pi and Sigma go into $x_t ~ \mathcal{N}(\pi_t, \Sigma_t)$
        self.pi = pi
        self.Sigma = Sigma
        # a_eta and a_mu define the state transition matrix A = [a_eta 0; 0 a_mu]
        self.a_eta = a_eta
        self.a_mu = a_mu
        # sigma_eta and sigma_mu define the covariance matrix Q = [sigma_eta^2 0; 0 sigma_mu^2]
        # for the Gaussian noise in the state transition w_t ~ \mathcal{N}(0, Q_t)
        self.sigma_eta_sq = sigma_eta_sq
        self.sigma_mu_sq = sigma_mu_sq
        # r goes into v_t ~ \mathcal{N}(0,r) where v_t is the noise in observation t
        self.r = r
        # phi is the seasonality parameter.
        # It's a vector in $\mathbb{R}^T$ where T is the number of intraday observations in a day
        self.phi = phi
        
    def A(self, tau):
        a1 = 1.0
        a2 = 1.0
        if tau % 13 == 0: # tau = kT for some integer K, T is the # of observations in a day
            a1 = self.a_eta
            a2 = self.a_mu
        return np.array([[a1, 0.0], [0.0, a2]])
    
    def Q(self, tau):
        a1 = 0.0
        a2 = 0.0
        if tau % 13 == 0: # tau = kT for some integer K, T is the # of observations in a day
            a1 = self.sigma_eta_sq
            a2 = self.sigma_mu_sq
        return np.array([[a1, 0.0], [0.0, a2]])

def kalman_filtering(tau, x_hat_tau, y_tau_plus, Sigma_tau_tau, params):
    A = params.A(tau)
    C = np.ones((1,2))
    x_hat_tau_plus = A @ x_hat_tau # predict mean
    Sigma_tau_plus = A @ Sigma_tau_tau @ A.T + params.Q(tau) # predict covariance
    
    # compute Kalman gain
    K_tau_plus = Sigma_tau_plus @ C.T @ np.linalg.inv(C @ Sigma_tau_plus @ C.T + params.r)
    
    # correct conditional mean
    x_hat_next = x_hat_tau_plus + K_tau_plus @ (y_tau_plus - params.phi[tau%13] - C@x_hat_tau_plus)
    Sigma_next = Sigma_tau_plus - K_tau_plus @ C @ Sigma_tau_plus
    #print("x_hat_next", x_hat_next.shape, "Sigma_next", Sigma_next.shape)
    return x_hat_next, Sigma_next


def kalman_smoothing(x_t, ys, Sigma_t, params):
    # this uses the outputs from the filtering algorithm
    # NOTE THAT x_t is a shorthand in the next few lines for x_{t|t} and Sigma_t := Sigma_{t|t}
    N = ys.shape[0]
    C = np.ones((1,2))
    x_ts = []
    Sigma_ts = []
    
    # this is an unsightly way to code it but I think it makes more sense
    for t in range(0, N):
        x_t, Sigma_t = kalman_filtering(t, x_t, ys[t-1], Sigma_t, params)
        x_ts.append(x_t)
        Sigma_ts.append(Sigma_t)
        
    x_N, Sigma_N = x_ts[-1], Sigma_ts[-1]
    # Now we have x_{N|N}, Sigma_{N|N}
    
    x_tau_n = x_N # this is the initialization of x_{t+1|N} and Sigma_{t+1|N}
    Sigma_tau_n = Sigma_N
    Lt = np.zeros_like(Sigma_t)
    
    for t in range(N-1, 0, -1):
        A = params.A(t)
        # in here, Sigma_ts[t-1] is Sigma_{t|t} because of 0-indexing
        Sigma_tau_plus = A @ Sigma_ts[t-1] @ A.T + params.Q(t)
        x_hat_tau_plus = A @ x_ts[t-1]
        
        Lt = Sigma_ts[t-1] @ A.T @ np.linalg.inv(Sigma_tau_plus)
        x_tau_n = x_ts[t-1] + Lt @ (x_tau_n - x_hat_tau_plus)
        Sigma_tau_n = Sigma_ts[t-1] + Lt @ (Sigma_tau_n - Sigma_tau_plus) @ Lt.T
    return np.reshape(x_tau_n, (2,1)), Sigma_tau_n, Lt, x_ts, Sigma_ts # this is x_{t|N} and Sigma_{t|N}
    

def compare(params1, params2):
    err = np.mean(np.abs(params1.pi - params2.pi)) + np.mean(np.abs(params1.Sigma - params2.Sigma)) + \
          np.abs(params1.a_eta - params2.a_eta) + np.abs(params1.a_mu - params2.a_mu) + \
          np.abs(params1.sigma_eta_sq - params2.sigma_eta_sq) + \
          np.abs(params1.sigma_mu_sq - params2.sigma_mu_sq) + \
          np.abs(params1.r - params2.r) + np.mean(np.abs(params1.phi - params2.phi))
    return err/8.0 # since there are 8 terms
        


def sum_matrices(Ps):
    result = np.zeros_like(Ps[0])
    for Pi in Ps:
        result += Pi
    return result

def em(x_1, ys, params, maxsteps=10, tol=1e-1, I=13):
    step = 0; err = np.Inf
    C = np.ones((1,2))
    N = ys.shape[0]
    N_days = int(N/I)
    print("N = {}, N_days={}".format(N, N_days))
    
    while step < maxsteps and err > tol:
        
        Ps = [] # this is an array of P_t in REVERSE order
        P_minuses = [] # this is an array of P_{t|t-1} in REVERSE order, e.g. P_{N|N-1} is first and P_{1|0} is last
        xs = []
        x_tau = x_1
        Sigma_tau = params.Sigma
        Sigma_tau_minus_N = Sigma_tau # IDK how to initialize this
        
        for tau in range(N, 1, -1): # this is [N, N-1,...,2] because otherwise the indexing doesn't make sense for kalman smoothing
            # this line is x_{tau|N}, Sigma_{tau|N}, L_tau and x_ts = list of x_{tau|tau}, Sigma_ts = list of Sigma_{tau|tau}
            x_tau_n, Sigma_tau_n, L_tau, x_ts, Sigma_ts = kalman_smoothing(x_1, ys[0:tau], params.Sigma, params)
            x_tau = x_tau_n # line 5
            P_tau = Sigma_tau_n + x_tau_n @ x_tau_n.T # line 6
            
            # this line gets x_{tau-1|N}, Sigma-{tau-1|N} and L_{tau-1}
            x_tau_minus_n, Sigma_tau_minus_N, L_tau_minus, _, _ = kalman_smoothing(x_1, ys[0:tau-1], params.Sigma, params)
            # This is line 7 of Algorithm 3 which computes Sigma_{tau, tau-1|N}
            # Note that Sigma_{tau, tau-1|N} has a dependency on Sigma_{tau+1, tau|N} which makes sense except
            # we do not know what to initialize it to
            Sigma_tau_minus_N = Sigma_ts[tau-1] @ L_tau_minus.T + \
                                L_tau @ (Sigma_tau_minus_N - params.A(tau) @ Sigma_ts[tau-1]) @ L_tau_minus.T
            P_tau_minus = Sigma_tau_minus_N + x_tau_n @ x_tau_minus_n.T # line 8
            
            xs.append(x_tau)
            Ps.append(P_tau)
            P_minuses.append(P_tau_minus)
        Ps.reverse()
        P_minuses.reverse()
        print("{} first step done {}".format(step, len(Ps)))
        
        N = N - 1
        # now we have a list of P_t and P_{t|t-1}
        pi = x_tau # equation 17
        Sigma = Ps[-1] - x_tau @ x_tau.T # equation 18
        print("\tpi, sigma = {}, {}".format(pi, Sigma))
        P_sum = sum_matrices([Ps[I*i + 1] for i in range(N_days)]) # for equation 19
        P_minus_sum = sum_matrices([P_minuses[I*i + 1] for i in range(N_days)]) # for equation 19
        a_eta = P_minus_sum[0,0] / P_sum[0,0]
        
        P_sum2 = sum_matrices([Ps[i] for i in range(2, N)]) # for equation 20
        P_minus_sum2 = sum_matrices([P_minuses[i] for i in range(1, N-1)]) # for equation 20
        a_mu = P_minus_sum2[1,1] / P_sum2[1,1]
        
        sigma_eta_sq = 0.0 # equation 21
        print("\ta_eta = {}, a_mu = {}".format(a_eta, a_mu))
        for i in range(N_days):
            t = i*I+1
            sigma_eta_sq += (Ps[t] + a_eta**2.0 * Ps[t-1] - 2.0 * a_eta * P_minuses[t-1])[0,0]
        sigma_eta_sq /= (N_days - 1) # 
        
        sigma_mu_sq = 0.0 # equation 22
        for t in range(2, N):
            sigma_mu_sq = (Ps[t] + a_mu**2.0 * Ps[t-1] - 2.0 * a_mu * P_minuses[t-1])[1,1]
        sigma_mu_sq *= 1.0/(N - 1.0)
        print("\tsigma_eta_sq = {}, sigma_mu_sq = {}".format(sigma_eta_sq, sigma_mu_sq))
        
        phi = np.zeros(13)
        for i in range(N):
            phi[i%I] += ys[i] - (C@xs[i])[0,0]
        phi /= N_days
        
        r = 0.0
        for t in range(N): # equation 23, probably
            r += ys[t]**2 + np.sum(Ps[t]) - 2.0*ys[t] * np.sum(xs[t]) + \
                (params.phi[t%I])**2.0 - 2.0 * ys[t] * params.phi[t%I] + \
                2.0*params.phi[t%I] * np.sum(xs[t])
        r /= N
        
        print("\tr = {}, phi = {}".format(r, phi))
        print("{} second step done".format(step))
        
        
        params1 = Params(pi, Sigma, a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi)
        err = compare(params, params1)
        params = params1
        
        print("{}, Error={}".format(step, err))
        step += 1
        N += 1
        
    return params


class KalmanPredictor:
    def __init__(self, params:Params, I=13):
        self.params = params
        self.I = I
        self.C = np.ones((1,2))

    def x1(self):
        return multivariate_normal(self.params.pi.flatten(), self.params.Sigma).rvs(1)
    
    def train(self, log_returns, maxsteps=25, tol=0.01):
        x_1 = self.x1()
        self.params = em(x_1, log_returns, self.params, maxsteps=maxsteps, tol=tol)
    
    def predict_alike(self, y_observed, x_t=None, start_time=0):
        C = self.C
        if x_t is None:
            x_t = self.x1()

        # This will return a vector of y_hat that is the same size as y_observed
        # The assumption is we start at a hidden state x1 described by the distribution in self.params
        N = y_observed.size
        predictions = np.zeros_like(y_observed)
        Sigma_t = self.params.Sigma
        for i in range(N):
            y_t = y_observed[i]
            x_t, Sigma_t = kalman_filtering(i+start_time, x_t, y_t, Sigma_t, self.params)
            predictions[i] = (C@x_t)[0] + self.params.phi[(i+start_time)%self.I]
        return predictions
    
    # This function takes as input N independent days containing T steps where T < I.
    # It fills in the remaining steps.
    def predict(self, y_observed):
        C = self.C
        N,T = y_observed.shape
        predictions = np.zeros((N, self.I-T))
        for n in range(N):
            # initialize
            x_t = self.x1()
            Sigma_t = self.params.Sigma
            
            # move through the T intraday steps
            for t in range(T):
                y_t = y_observed[n,t]
                x_t, Sigma_t = kalman_filtering(i, x_t, y_t, Sigma_t, self.params)
            
            # make a multi-step prediction for the rest of the day
            for t in range(self.I-T):
                x_t, Sigma_t = kalman_filtering(i, x_t, (C@x_t)[0], Sigma_t, self.params)
                predictions[n,t] = (C@x_t)[0] + self.params.phi[i%self.I]
        return predictions