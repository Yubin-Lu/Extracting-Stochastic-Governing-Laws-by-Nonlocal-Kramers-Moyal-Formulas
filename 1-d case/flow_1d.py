import numpy as np
import itertools
import logging
import matplotlib.pyplot as plt

import time
import scipy.io as scio
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import scipy.io as scio
import seaborn as sns
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal
from scipy.special import gamma

from nf.flows import *
from nf.models import NormalizingFlowModel
sns.set()

try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # enable for GPU
except:
    pass



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

def gen_data(n=512):
    return np.r_[np.random.randn(n // 2, 1) + np.array([2]),
                 np.random.randn(n // 2, 1) + np.array([-2])]


def StableVariable(m, alpha):
     V = np.pi/2 * (2*np.random.rand(m)-1)
     W = np.random.exponential(scale=1, size=m)
     y = np.sin(alpha * V) / (np.cos(V)**(1/alpha) ) * (np.cos( V*(1-alpha)) / W )**((1-alpha)/alpha)
     return y

def GeneratingData(T, dt, n_samples, X0):
    t = np.arange(0, T, dt)
    Nt = len(t)
    alpha = 1.5
    # X0 = 1.5 * np.ones([10000])
    # X0 = np.random.randn(500)
    x0 = X0 * np.ones([n_samples])
    # N = len(x0)
    N = x0.size
    x = np.zeros((Nt, N))
    x[0, :] = x0
    for i in range(Nt-1):
        Ut = dt**(1/alpha) * StableVariable(N, alpha)
        UUt = dt**(1/2) * np.random.randn(N)
        # print(Ut.shape)
        
        #double-well systems with Brownian motion
        x[i+1, :] = x[i, :] + 1*(3*x[i, :] - 1*x[i, :]**3)*dt + 1*Ut + 0*(1*x[i, :]+0)*UUt
    return t, x[-1,:]

def plot_data(x, bandwidth = 0.2, **kwargs):
    # kde = sp.stats.gaussian_kde(x[:,0])
    kde = sp.stats.gaussian_kde(x)
    x_axis = np.linspace(-10, 10, 200)
    plt.plot(x_axis, kde(x_axis), **kwargs)


if __name__ == "__main__":

    setup_seed(123)
    tis1 = time.perf_counter()
    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    # argparser.add_argument("--flows", default=32, type=int)
    argparser.add_argument("--flows", default=32, type=int)
    # argparser.add_argument("--flow", default="OneByOneConv", type=str)
    argparser.add_argument("--flow", default="NSF_AR", type=str)
    argparser.add_argument("--iterations", default=200, type=int)
    args = argparser.parse_args()


    T = 0.02
    dt = 0.01
    x_init = np.linspace(-2, 2, 21)
    drift = np.zeros(len(x_init))
    sigma = np.zeros(len(x_init))
    resampling_data = np.zeros([0,1])
    count = 0
    for x0 in x_init:
        
        time_set, position_x = GeneratingData(T, dt, 5000, x0)
        
        
        mu_x = np.mean(position_x)
        sigma_x = 1/5*np.std(position_x)
        position_x = (position_x - mu_x) / sigma_x
        
        P_x = np.reshape(position_x, position_x.size, order='C').reshape(-1, 1)
        x = torch.Tensor(P_x)
        # x.requires_grad_()
        flow = eval(args.flow)
        flows = [flow(dim=1) for _ in range(args.flows)]
        prior = MultivariateNormal(torch.zeros(1), torch.eye(1))
        model = NormalizingFlowModel(prior, flows)
    
        optimizer = optim.Adam(model.parameters(), lr=0.005)

    
        # plot_data(x, color = "black")
        # plt.show()
        
        
    
        loader = data.DataLoader(
        dataset = x,
        batch_size=50000,             
        shuffle=True,             
        )
        
        Loss = np.array([])
        for epoch in range(args.iterations):    
            # for i in range(args.iterations):
            for step, batch_x in enumerate(loader): 
                optimizer.zero_grad()
                z, prior_logprob, log_det = model(x)
                logprob = prior_logprob + log_det
                loss = -torch.mean(prior_logprob + log_det)
                # loss = -torch.mean(torch.log(px))
                loss.backward()
                optimizer.step()
                tmp = loss.cpu().detach().numpy()
                Loss = np.append(Loss, tmp)
        print("count:", count)

        ##System Identification
        ##  pure jump LM 0.5, BM+LM 1.5
        a, b = ((-1.+x0) - mu_x) / sigma_x, ((1.+x0) - mu_x) / sigma_x
        u0 = np.linspace(a, b, 201)
        
        # ## 
        # u0 = np.linspace(x0-1, x0+1, 201)
        
        du = 2/200
        u1 = np.reshape(u0, u0.size, order='C').reshape(-1, 1)
        uu = torch.Tensor(u1)
        z, prior_logprob, log_det = model(uu)
        samples_px = torch.exp(prior_logprob + log_det)
        # px_estimated = samples_px.cpu().detach().reshape(u0.shape).numpy()
        px_estimated = samples_px.cpu().detach().numpy()
        

        q=np.arange(0,len(Loss))
        plt.plot(q,Loss,'r')
        plt.show()
        
        # plt.figure(figsize=(8, 3))
        # plt.subplot(1, 2, 1)
        # # plot_data(x.cpu(), color="black", alpha=0.5)
        # plot_data(position_x, color="black", alpha=0.5)
        # plt.title("Training data")
        # plt.subplot(1, 2, 2)
        # # samples = model.sample(500).cpu().data
        # # plot_data(samples, color="black", alpha=0.5)
        # plt.plot(u0, px_estimated, 'b')
        # plt.title("Generated samples")
        # plt.show()
        
        p_learned = px_estimated 
        # z = 1.5
        # f = 0.2992/np.abs((x_sample[120:]-z)**2.5)
        # plt.plot(x_sample[120:], f, 'r')
        # plt.plot(x_sample[120:], p_learned[120:], 'b')
        # plt.show()
        
        ## For drift
        integral = np.sum(p_learned*(u0*sigma_x+mu_x-x0))*du / (dt * sigma_x)
        drift[count] = integral
        
        ## For BM
        ##
        eps, alpha_tmp, sigma_tmp =1., 1.5, 1. ## coupling MultiBM+LM
        C_alpha_tmp = alpha_tmp*gamma(1/2+alpha_tmp/2) / (2**(1-alpha_tmp)*(np.pi**0.5)*gamma(1-alpha_tmp/2))
        sigma[count] = np.sum(p_learned*((u0*sigma_x+mu_x-x0)**2))*du / (dt * sigma_x) - 2 * (sigma_tmp**alpha_tmp) * (eps**(2-alpha_tmp)) * C_alpha_tmp / (2-alpha_tmp)

        ## for kernel
        M = 10000
        samples = model.sample(M).data.cpu().detach().numpy()
        samples = samples * sigma_x + mu_x - x0
        resampling_data = np.append(resampling_data, samples, axis=0)
        
        count += 1
        
    ## Plotting drift
    drift_coeff = np.polyfit(x_init, drift, 3)
    p1 = np.poly1d(drift_coeff)
    drift_fitting = p1(x_init)  
    plt.style.use('classic')
    plt.figure(figsize=(12,8), facecolor='white', edgecolor='black')
    
    l1, = plt.plot(x_init, drift_fitting, 'r', linewidth=3)
    # ax1.set_facecolor('#eafff5')
    drift_T = 3*x_init - x_init**3
    l2, = plt.plot(x_init, drift_T, 'b', linewidth=3)
    plt.xlabel("x1",fontsize=20)
    plt.ylabel("x2",fontsize=20)
    # plt.title("drift",fontsize=20)
    plt.legend(handles=[l1,l2],labels=['learned drift','true drift'],loc='upper right',fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(-2,2)
    plt.ylim(-3.,3.)
    # plt.plot(x_init, drift, 'g')
    plt.grid(False)
    plt.show()
    
    
    
    
    
    ## Plotting BM
    sigma_coeff = np.polyfit(x_init, sigma, 2)
    p1 = np.poly1d(sigma_coeff)
    sigma_fitting = p1(x_init)  
    plt.plot(x_init, sigma_fitting, 'r')
    sigma_T = (x_init+0)**2
    plt.plot(x_init, sigma_T, 'b')
    plt.plot(x_init, sigma, 'g')
    plt.show()
    
    # m, eps = 2.2, 0.2 ## for pureLM
    # m, eps = 2.5, 0.2 ## for multi BM+LM
    m, eps = 2., 0.047 ## for pureLM + NSF_AR
    # m, eps = 2., 0.1 ## for multi BM+LM + NSF_AR
    N = 1
    M = 21 * 10000
    alpha_estimated = np.zeros(N)
    sigma_estimated = np.zeros(N)
    for k in range(N):
        k = k + 1
        
        
        result1 = pow(resampling_data, 2) < (eps)**2
        result2 = pow(resampling_data, 2) < (m * eps)**2
        n_0 = np.sum(result2) - np.sum(result1)
        result3 = pow(resampling_data, 2) < (m**k * eps)**2
        result4 = pow(resampling_data, 2) < (m**(k+1) * eps)**2
        n_k = np.sum(result4) - np.sum(result3)
        
        ## Estimating alpha
        alpha_estimated[k-1] = 1/(k*np.log(m)) * np.log(n_0/n_k)
    
        ## Estimating  sigma
        tmp1 = alpha_estimated[k-1] * eps**alpha_estimated[k-1] * m**(k*alpha_estimated[k-1]) *n_k
        C_alpha = alpha_estimated[k-1]*gamma(1/2+alpha_estimated[k-1]/2) / (2**(1-alpha_estimated[k-1])*(np.pi**0.5)*gamma(1-alpha_estimated[k-1]/2))
        tmp2 = 2 * C_alpha * dt * M * (1 - 1/m**alpha_estimated[k-1])
        sigma_estimated[k-1] = (tmp1/tmp2)**(1/alpha_estimated[k-1])


    print('alpha:',np.average(alpha_estimated))
    print('sigma:',np.average(sigma_estimated))
    
    
    tis2 = time.perf_counter()
    print("Time used:", tis2-tis1)
    

