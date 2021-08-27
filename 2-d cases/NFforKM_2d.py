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
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal
from sympy import *
from scipy.special import gamma

from nf.flows import *
from nf.models import NormalizingFlowModel

from scipy import integrate






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

def StableVariable(m, alpha):
     V = np.pi/2 * (2*np.random.rand(m)-1)
     W = np.random.exponential(scale=1, size=m)
     y = np.sin(alpha * V) / (np.cos(V)**(1/alpha) ) * (np.cos( V*(1-alpha)) / W )**((1-alpha)/alpha)
     return y

def GeneratingData(T, dt, n_samples, x_init, y_init):
    t = np.arange(0, T, dt)
    
    Nt = len(t)
    # #multimodal initial distribution
    # mu = np.array([[2, 3]])
    # sigma = np.eye(2)
    # X0 = 1*np.random.multivariate_normal(mu[0],sigma,250)  + 0.5*np.random.multivariate_normal(-mu[0],sigma,250)
    # XX0 = 1*np.random.multivariate_normal(mu[0],sigma,250)  + 0.5*np.random.multivariate_normal(-mu[0],sigma,250)
    
    
    # #single-model initial distribution
    # X0 = np.random.randn(n_samples//2,2) * np.sqrt(0.01) + x_init
    # XX0 = np.random.randn(n_samples//2,2) * np.sqrt(0.01) + y_init
    
    #fixed initial value
    X0 = np.ones([n_samples//2,2]) * np.array([[x_init, y_init]])
    XX0 = np.ones([n_samples//2,2]) * np.array([[x_init, y_init]])
    
    
    x0 = X0[:,0:1]
    xx0 = XX0[:,0:1]
    y0 = X0[:,1:]
    yy0 = XX0[:,1:]
    N = len(x0) + len(xx0)
    alpha = 1.5
    x = np.zeros((Nt, N))
    y = np.zeros((Nt, N))
    x[0, 0:n_samples//2] = x0.squeeze()
    x[0, n_samples//2:n_samples] = xx0.squeeze()
    y[0, 0:n_samples//2] = y0.squeeze()
    y[0, n_samples//2:n_samples] = yy0.squeeze()
    for i in range(Nt-1):
        Ut = dt**(1/alpha) * StableVariable(N, alpha)
        Vt = dt**(1/alpha) * StableVariable(N, alpha)
        UUt = dt**(1/2) * np.random.randn(N)
        VVt = dt**(1/2) * np.random.randn(N)
        x[i+1, :] = x[i, :] + (1*x[i, :] - 1*x[i, :]**3)*dt + 0*x[i, :]*UUt+ 1*Ut
        y[i+1, :] = y[i, :] + (1*y[i, :] - 1*y[i, :]**3)*dt + 0*(1*y[i, :]+1)*VVt + 1*Vt
        # x[i+1, :] = x[i, :] + (-1*x[i, :])*dt + 1*x[i, :]*UUt+ 0*UUt
        # y[i+1, :] = y[i, :] + (-1*y[i, :])*dt + (1*y[i, :])*VVt + 0*VVt
        # x[i+1, :] = x[i, :] + 1*(4*x[i, :] - 1*x[i, :]**3)*dt + x[i, :]*Ut
        # y[i+1, :] = y[i, :] - x[i, :]*y[i, :]*dt + y[i, :]*Vt
        # x[i+1, :] = x[i, :] - x[i, :]*dt + x[i, :]*UUt+ x[i, :]*Ut
        # y[i+1, :] = y[i, :] + (x[i, :]**2 + y[i, :])*dt + y[i, :]*VVt + y[i, :]*Vt
        b=np.empty(0).astype(int)
        for j in range(n_samples):
            if (np.abs(x[:,j])>1e4).any() or (np.abs(y[:,j])>1e4).any():
                b = np.append(b,j)
        x1 = np.delete(x,b,axis=1)
        y1 = np.delete(y,b,axis=1)
        # t2 = t[:-1]
        # x2 = x1[:-1,:]
        # y2 = y1[:-1,:]
        # tt = t[-1]
        # xx = x1[-1,:]
        # yy = y1[-1,:]

    return t, x1[-1,:], y1[-1,:]# tt, xx, yy




# def plot_data(x, **kwargs):
#     plt.scatter(x[:,0], x[:,1], marker="x", **kwargs)
#     plt.xlim((-3, 3))
#     plt.ylim((-3, 3))
def plot_data(x, **kwargs):
    # plt.scatter(x[:,0], x[:,1], **kwargs)
    plt.scatter(x[:,0], x[:,1], s=1, marker="o", **kwargs)
    # plt.xlim((-5, 5))
    # plt.ylim((-20, 20))
    # EX3
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.xlabel("x",fontsize=20)
    plt.ylabel("y",fontsize=20)
    
    
def sample2density(x, u, v, du, dv):
    m, n = u.shape
    l, s =x.shape
    count = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            for k in range(l):
                if x[k,0]>=u[i,j]- du/2 and x[k,0]<u[i,j] + du/2 and x[k,1]>=v[i,j]- dv/2 and x[k,1]<v[i,j]+ dv/2:
                    count[i,j] += 1
    return count/(l*du*dv)
    

if __name__ == "__main__":
    # setup_seed(12) ## 例2
    setup_seed(123) ## 例3
    tis1 = time.perf_counter()
    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--flows", default=1, type=int)
    argparser.add_argument("--flow", default="RealNVP", type=str)
    argparser.add_argument("--iterations", default=500, type=int)
    argparser.add_argument("--use-mixture", action="store_true")
    argparser.add_argument("--convolve", action="store_true")
    argparser.add_argument("--actnorm", action="store_true")
    args = argparser.parse_args()




    # ## for drift
    # x_init = np.linspace(-2, 2, 6)
    # y_init = np.linspace(-2, 2, 6)

    
    ## for kernel
    x_init = np.linspace(-2, 2, 6)
    y_init = np.linspace(-2, 2, 6)
    
    

    x_init_grid, y_init_grid = np.meshgrid(x_init, y_init)
    data_init = np.concatenate((x_init_grid.reshape(-1,1), y_init_grid.reshape(-1,1)), axis=1)
    # kernel = np.zeros([len(data_init), 201, 201])
    drift1 = np.zeros(len(data_init))
    drift2 = np.zeros(len(data_init))
    sigma1 = np.zeros(len(data_init))
    sigma2 = np.zeros(len(data_init))
    resampling_data = np.zeros([0,2])
    count = 0
    for (x0, y0) in data_init:
        
        flow = eval(args.flow)
        flows = [flow(dim=2) for _ in range(args.flows)]
        prior = MultivariateNormal(torch.zeros(2), 1*torch.eye(2))
        # prior = MultivariateNormal(torch.zeros(2), 0.35*torch.eye(2))
        model = NormalizingFlowModel(prior, flows)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        # optimizer = optim.Adagrad(model.parameters(), lr=0.005, lr_decay=0., weight_decay=0)
        

        # dataFile = 'DW_sde.mat' #come from SDE
        count_tmp = count + 1
        dataFile = "Data_"+ "%d"  %count_tmp #come from SDE
        SDE_data = scio.loadmat(dataFile)
        position_x = SDE_data['x_end']
        position_y = SDE_data['y_end']
        # time_set, position_x, position_y = GeneratingData(T, dt, 2000, x0, y0)
        
        ##标准化
        mu_x, mu_y = np.mean(position_x), np.mean(position_y)
        sigma_x, sigma_y = np.std(position_x), np.std(position_y)
        position_x = (position_x - mu_x) / sigma_x
        position_y = (position_y - mu_y) / sigma_y
        
        P_x = np.reshape(position_x, position_x.size, order='C').reshape(-1, 1)
        P_y = np.reshape(position_y, position_y.size, order='C').reshape(-1, 1)
        
        x = torch.Tensor(np.concatenate((P_x,P_y),axis=1))
        
        
        
        
        loader = data.DataLoader(
        dataset = x,
        batch_size=60000,             # 每批提取的数量
        shuffle=True,             # 要不要打乱数据（打乱比较好）
        # num_workers=2             # 多少线程来读取数据
        )
        # Loss = np.zeros([args.iterations, 1])
        Loss = np.array([])
        rho = 1
        for epoch in range(args.iterations):    # 对整套数据训练3次
            # for i in range(args.iterations):
            for step, batch_x in enumerate(loader): 
                optimizer.zero_grad()
                z, prior_logprob, log_det = model(x)
                logprob = prior_logprob + log_det
                loss = -torch.mean(prior_logprob + log_det) + 0*np.log(rho) - 0*torch.log(torch.std(logprob)) + 0*torch.std(logprob)
                
                # loss = -torch.mean(torch.log(px))
                loss.backward()
                optimizer.step()
                tmp = loss.cpu().detach().numpy()
                # tmp = loss.detach().numpy()
                Loss = np.append(Loss, tmp)
            # print("mean:", torch.mean(prior_logprob + log_det))
            # print("std:", torch.log(torch.std(logprob)))
            # print("rho:", np.log(rho))
        q=np.arange(0,len(Loss))
        plt.plot(q,Loss,'r')
        # plt.plot(q[100:],Loss[100:],'r')
        plt.show()
        print("count:", count)


        ##System Identification
        ##标准化  例2：1, 例3： 1.5
        a, b = ((-1.5+x0) - mu_x) / sigma_x, ((1.5+x0) - mu_x) / sigma_x
        c, d = ((-1.5+y0) - mu_y) / sigma_y, ((1.5+y0) - mu_y) / sigma_y
        u0, v0 = np.linspace(a, b, 301), np.linspace(c, d, 301)

        # ## 无标准化
        # u0, v0 = np.linspace(-0.5+x0, 0.5+x0, 301), np.linspace(-0.5+y0, 0.5+y0, 301)
        # # index = [96,97,98,99,100,101,102,103,104]
        # # index = np.arange(90,111)
        # # u0 = np.delete(u0, index)
        # # v0 = np.delete(v0, index)

        du = 3/300
        dv = 3/300
        u, v = np.meshgrid(u0, v0)
        u1 = np.reshape(u, u.size, order='C').reshape(-1, 1)
        v1 = np.reshape(v, v.size, order='C').reshape(-1, 1)
        uu = torch.Tensor(np.concatenate((u1,v1),axis=1))
        ## 例2：0.05， 例3：0.01
        t_star = 0.01
        z, prior_logprob, log_det = model(uu)
        samples_px = torch.exp(prior_logprob + log_det)
        px_estimated = samples_px.cpu().detach().reshape(u.shape).numpy()
        # px_estimated = samples_px.detach().reshape(u.shape).numpy()
        
        
        # ## Plotting density
        # dataFile = "DataT_"+ "%d"  %count_tmp #come from SDE
        # SDE_data = scio.loadmat(dataFile)
        # positionT_x = SDE_data['x_end']
        # positionT_y = SDE_data['y_end']
        
        # PT_x = np.reshape(positionT_x, positionT_x.size, order='C').reshape(-1, 1)
        # PT_y = np.reshape(positionT_y, positionT_y.size, order='C').reshape(-1, 1)
        
        # xT = torch.Tensor(np.concatenate((PT_x,PT_y),axis=1))
        # px_true = sample2density(xT.numpy(), u, v, du, dv)
        
        # plt.figure(figsize=(22,12))
        # plt.subplot(1, 2, 1)
        # c1 = plt.pcolormesh(u, v, px_estimated, cmap='jet', shading='gouraud')
        # plt.colorbar(c1)
        # plt.subplot(1, 2, 2)
        # c2 = plt.pcolormesh(u, v, px_estimated, cmap='jet',shading='gouraud')
        # plt.colorbar(c2)
        # plt.show()
        
        
        
        # ## for kernel
        # kernel = px_estimated / t_star
        # f = 0.1712/(((u-x0)**2 + (v-y0)**2)**(3.5/2))
        # plt.figure(figsize=(22,22))
        # plt.subplot(2, 2, 1)
        # k=0
        # c1 = plt.pcolormesh(u[k:,k:], v[k:,k:], f[k:,k:], cmap='jet', shading='gouraud')
        # plt.xlabel("x",fontsize=20)
        # plt.ylabel("y",fontsize=20)
        # plt.title("kernel_T",fontsize=20)
        # plt.colorbar(c1)
        # plt.subplot(2, 2, 2)
        # c2 = plt.pcolormesh(u[k:,k:], v[k:,k:], kernel[k:,k:], cmap='jet', shading='gouraud')
        # plt.xlabel("x",fontsize=20)
        # plt.ylabel("y",fontsize=20)
        # plt.title("kernel_L",fontsize=20)
        # plt.colorbar(c2)

        
        
        # plt.subplot(2, 2, 3)
        # k=170
        # c3 = plt.pcolormesh(u[k:,k:], v[k:,k:], f[k:,k:], cmap='jet', shading='gouraud')
        # # plt.pcolormesh(u[k:,k:], v[k:,k:], f[k:,k:], cmap='jet', shading='gouraud')
        # plt.xlabel("x",fontsize=20)
        # plt.ylabel("y",fontsize=20)
        # plt.title("kernel_T",fontsize=20)
        # plt.colorbar(c3)
        # plt.subplot(2, 2, 4)
        # c4 = plt.pcolormesh(u[k:,k:], v[k:,k:], kernel[k:,k:], cmap='jet', shading='gouraud')
        # plt.xlabel("x",fontsize=20)
        # plt.ylabel("y",fontsize=20)
        # plt.title("kernel_L",fontsize=20)
        # plt.colorbar(c4)
        # plt.show()
        
        
        
        ## for kernel
        M = 10000
        samples = model.sample(M).data.cpu().detach().numpy()
        samples[:,0] = samples[:,0] * sigma_x + mu_x - x0
        samples[:,1] = samples[:,1] * sigma_y + mu_y - y0
        resampling_data = np.append(resampling_data, samples, axis=0)


        
        
        
        
        
        
        ## for drift
        # ##标准化
        # px_estimated = px_true
        # drift1[count] = du*dv*sum(sum((u0*sigma_x+mu_x-x0)*px_estimated)) / (t_star * sigma_x * sigma_y)
        # drift2[count] = du*dv*sum(sum((v0*sigma_y+mu_y-y0)*(px_estimated.T))) / (t_star * sigma_x * sigma_y)
        
        # ## 无标准化
        # drift1[count] = du*dv*sum(sum((u0-x0)*px_estimated)) / t_star
        # drift2[count] = du*dv*sum(sum((v0-y0)*(px_estimated.T))) / t_star
        
        ## 复化梯形公式
        ## 标准化
        tmp1 = (u0*sigma_x+mu_x-x0)*px_estimated
        tmp2 = (v0*sigma_y+mu_y-y0)*(px_estimated.T)
        
        # ## 无标准化
        # tmp1 = (u0-x0)*px_estimated
        # tmp2 = (v0-y0)*(px_estimated.T)
        
        tmp1_j = np.r_[tmp1[1:, :], np.zeros([1, tmp1[0,:].size])]
        tmp1_i = np.c_[tmp1[:, 1:], np.zeros([tmp1[:, 0].size, 1])] 
        tmp1_ij_tmp = np.r_[tmp1[1:, 1:], np.zeros([1, tmp1[0,:].size-1])]
        tmp1_ij = np.c_[tmp1_ij_tmp, np.zeros([tmp1[:, 0].size, 1])]
        
        tmp2_j = np.r_[tmp2[1:, :], np.zeros([1, tmp2[0,:].size])]
        tmp2_i = np.c_[tmp2[:, 1:], np.zeros([tmp2[:, 0].size, 1])] 
        tmp2_ij_tmp = np.r_[tmp2[1:, 1:], np.zeros([1, tmp2[0,:].size-1])]
        tmp2_ij = np.c_[tmp2_ij_tmp, np.zeros([tmp2[:, 0].size, 1])]
        drift1[count] = du*dv*np.sum((tmp1 + tmp1_j + tmp1_i + tmp1_ij)) / (4*t_star * sigma_x * sigma_y)
        drift2[count] = du*dv*np.sum((tmp2 + tmp2_j + tmp2_i + tmp2_ij)) / (4*t_star * sigma_x * sigma_y)
        
        ## for BM
        # ##矩形公式
        # eps, alpha_tmp, sigma_tmp =1.5, 1.47, 1.11 ## coupling MultiBM+LM
        # C_alpha_tmp = alpha_tmp*gamma(1+alpha_tmp/2) / (2**(1-alpha_tmp)*np.pi*gamma(1-alpha_tmp/2))
        # sigma1[count] = du*dv*sum(sum(((u*sigma_x+mu_x-x0)**2)*px_estimated)) / t_star - np.pi * sigma_tmp**alpha_tmp * eps**(2-alpha_tmp) * C_alpha_tmp / (2-alpha_tmp)
        # sigma2[count] = du*dv*sum(sum(((v*sigma_y+mu_y-y0)**2)*(px_estimated))) / t_star - np.pi * sigma_tmp**alpha_tmp * eps**(2-alpha_tmp) * C_alpha_tmp / (2-alpha_tmp)
        
        ## 复化梯形公式
        ## 标准化
        tmp3 = ((u*sigma_x+mu_x-x0)**2) * px_estimated
        tmp4 = ((v*sigma_y+mu_y-y0)**2) * px_estimated
        
        # ## 无标准化
        # tmp1 = (u0-x0)*px_estimated
        # tmp2 = (v0-y0)*(px_estimated.T)
        
        tmp3_j = np.r_[tmp3[1:, :], np.zeros([1, tmp3[0,:].size])]
        tmp3_i = np.c_[tmp3[:, 1:], np.zeros([tmp3[:, 0].size, 1])] 
        tmp3_ij_tmp = np.r_[tmp3[1:, 1:], np.zeros([1, tmp3[0,:].size-1])]
        tmp3_ij = np.c_[tmp3_ij_tmp, np.zeros([tmp3[:, 0].size, 1])]
        
        tmp4_j = np.r_[tmp4[1:, :], np.zeros([1, tmp4[0,:].size])]
        tmp4_i = np.c_[tmp4[:, 1:], np.zeros([tmp4[:, 0].size, 1])] 
        tmp4_ij_tmp = np.r_[tmp4[1:, 1:], np.zeros([1, tmp4[0,:].size-1])]
        tmp4_ij = np.c_[tmp4_ij_tmp, np.zeros([tmp4[:, 0].size, 1])]
        
        # eps, alpha_tmp, sigma_tmp =0.5, 1.58, 1.13 ## AddiBM+LM
        # eps, alpha_tmp, sigma_tmp =1.5, 1.43, 0.86 ## MultiBM+LM
        # eps, alpha_tmp, sigma_tmp =1.5, 1.46, 1.08 ## coupling MultiBM+LM
        eps, alpha_tmp, sigma_tmp =1.5, 1.58, 1.23 ## coupling MultiBM+LM  例3
        C_alpha_tmp = alpha_tmp*gamma(1+alpha_tmp/2) / (2**(1-alpha_tmp)*np.pi*gamma(1-alpha_tmp/2))
        
        
        sigma1[count] = du*dv*np.sum((tmp3 + tmp3_j + tmp3_i + tmp3_ij)) / (4*t_star * sigma_x * sigma_y) - np.pi * (sigma_tmp**alpha_tmp) * (eps**(2-alpha_tmp)) * C_alpha_tmp / (2-alpha_tmp)
        sigma2[count] = du*dv*np.sum((tmp4 + tmp4_j + tmp4_i + tmp4_ij)) / (4*t_star * sigma_x * sigma_y) - np.pi * (sigma_tmp**alpha_tmp) * (eps**(2-alpha_tmp)) * C_alpha_tmp / (2-alpha_tmp)
        count += 1


    tis2 = time.perf_counter()
    print("Time used:", tis2-tis1)
    
    ##Plotting coefficients
    drift1_learned = np.reshape(drift1, x_init_grid.shape)
    drift2_learned = np.reshape(drift2, x_init_grid.shape)
    # drift1_true = 3*x_init_grid - x_init_grid**3
    # drift2_true = 3*y_init_grid - y_init_grid**3
    # drift1_true = (x_init_grid - x_init_grid**3 - 5*x_init_grid*y_init_grid**2)
    # drift2_true = -(1 + x_init_grid**2) * y_init_grid
    drift1_true = 0.001*x_init_grid - x_init_grid*y_init_grid
    drift2_true = -6*y_init_grid + 0.25*x_init_grid**2


    plt.figure(figsize=(27,17), facecolor='white', edgecolor='black')
    plt.subplot(2, 2, 1)
    # c1 = plt.pcolormesh(x_init_grid, y_init_grid, drift1_true, cmap='jet')
    c1 = plt.pcolormesh(x_init_grid, y_init_grid, drift1_true, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("drift1",fontsize=25)
    plt.colorbar(c1)
    
    plt.subplot(2, 2, 2)
    # c2 = plt.pcolormesh(x_init_grid, y_init_grid, drift2_true, cmap='jet')
    c2 = plt.pcolormesh(x_init_grid, y_init_grid, drift2_true, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("drift2",fontsize=25)
    plt.colorbar(c2)
    
    plt.subplot(2, 2, 3)
    # c5 = plt.pcolormesh(x_init_grid, y_init_grid, drift1_learned, cmap='jet')
    c5 = plt.pcolormesh(x_init_grid, y_init_grid, drift1_learned, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.colorbar(c1)
    
    plt.subplot(2, 2, 4)
    # c6 = plt.pcolormesh(x_init_grid, y_init_grid, drift2_learned, cmap='jet')
    c6 = plt.pcolormesh(x_init_grid, y_init_grid, drift2_learned, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.colorbar(c2)
    

    
    
    
    
    ## plotting diffusion term
    sigma1_learned = np.reshape(sigma1, x_init_grid.shape)
    sigma2_learned = np.reshape(sigma2, x_init_grid.shape)
    sigma1_true = (x_init_grid+0)**2
    sigma2_true = (y_init_grid+0)**2
    # sigma1_true = (y_init_grid+1)**2
    # sigma2_true = (x_init_grid+0)**2

    plt.figure(figsize=(27,17), facecolor='white', edgecolor='black')
    plt.subplot(2, 2, 1)
    c1 = plt.pcolormesh(x_init_grid, y_init_grid, sigma1_true, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("diffusion1",fontsize=25)
    plt.colorbar(c1)
    
    plt.subplot(2, 2, 2)
    c2 = plt.pcolormesh(x_init_grid, y_init_grid, sigma2_true, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("diffusion2",fontsize=25)
    plt.colorbar(c2)
    
    plt.subplot(2, 2, 3)
    c5 = plt.pcolormesh(x_init_grid, y_init_grid, sigma1_learned, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.colorbar(c1)
    
    plt.subplot(2, 2, 4)
    c6 = plt.pcolormesh(x_init_grid, y_init_grid, sigma2_learned, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.colorbar(c2)
    
    
    
    
    
    
    
    

    # m, eps = 2, 1. ## for LM  例2
    # m, eps = 2, 1.5 ## for multi BM+LM
    m, eps = 2, 1. ## for coupling multi BM+LM 例3
    N = 1
    M = 36 * 10000
    alpha_estimated = np.zeros(N)
    sigma_estimated = np.zeros(N)
    for k in range(N):
        k = k + 1
        
        ##标准化
        result1 = pow(resampling_data[:,0], 2) + pow(resampling_data[:,1], 2) < (eps)**2
        result2 = pow(resampling_data[:,0], 2) + pow(resampling_data[:,1], 2) < (m * eps)**2
        n_0 = np.sum(result2) - np.sum(result1)
        result3 = pow(resampling_data[:,0], 2) + pow(resampling_data[:,1], 2) < (m**k * eps)**2
        result4 = pow(resampling_data[:,0], 2) + pow(resampling_data[:,1], 2) < (m**(k+1) * eps)**2
        n_k = np.sum(result4) - np.sum(result3)
        
        # ##无标准化
        # result1 = pow(samples[:,0]-x0, 2) + pow(samples[:,1]-y0, 2) < (eps)**2
        # result2 = pow(samples[:,0]-x0, 2) + pow(samples[:,1]-y0, 2) <= (m * eps)**2
        # n_0 = np.sum(result2) - np.sum(result1)
        # result3 = pow(samples[:,0]-x0, 2) + pow(samples[:,1]-y0, 2) < (m**k * eps)**2
        # result4 = pow(samples[:,0]-x0, 2) + pow(samples[:,1]-y0, 2) <= (m**(k+1) * eps)**2
        # n_k = np.sum(result4) - np.sum(result3)
        
        alpha_estimated[k-1] = 1/(k*np.log(m)) * np.log(n_0/n_k)
    
    
        ## Estimating  sigma
        tmp1 = alpha_estimated[k-1] * eps**alpha_estimated[k-1] * m**(k*alpha_estimated[k-1]) *n_k
        C_alpha = alpha_estimated[k-1]*gamma(1+alpha_estimated[k-1]/2) / (2**(1-alpha_estimated[k-1])*np.pi*gamma(1-alpha_estimated[k-1]/2))
        tmp2 = 2*np.pi * C_alpha * t_star * M * (1 - 1/m**alpha_estimated[k-1])
        sigma_estimated[k-1] = (tmp1/tmp2)**(1/alpha_estimated[k-1])


    print('alpha:',np.average(alpha_estimated))
    print('sigma:',np.average(sigma_estimated))



        

    # # Resampling
    # plt.figure(figsize=(22,12))
    # plt.subplot(1, 2, 1)
    # xxx = np.concatenate((position_x.T, position_y.T), axis=1)
    # plot_data(xxx, color="black", alpha=0.5)
    # plt.title("True",fontsize=30)

    # plt.subplot(1, 2, 2)
    # xxx = np.concatenate((positionS_x.T, positionS_y.T), axis=1)
    # plot_data(xxx, color="black", alpha=0.5)
    # plt.title("Learned",fontsize=30)
    # plt.show()








    # for i in range(args.iterations):
    #     optimizer.zero_grad()
    #     z, prior_logprob, log_det = model(x)
    #     logprob = prior_logprob + log_det
    #     loss = -torch.mean(prior_logprob + log_det)
    #     loss.backward()
    #     optimizer.step()
    #     if i % 100 == 0:
    #         logger.info(f"Iter: {i}\t" +
    #                     f"Logprob: {logprob.mean().data:.2f}\t" +
    #                     f"Prior: {prior_logprob.mean().data:.2f}\t" +
    #                     f"LogDet: {log_det.mean().data:.2f}")

    # plt.figure(figsize=(8, 3))
    # plt.subplot(1, 3, 1)
    # plot_data(x, color="black", alpha=0.5)
    # plt.title("Training data")
    # plt.subplot(1, 3, 2)
    # plot_data(z.data, color="darkblue", alpha=0.5)
    # plt.title("Latent space")
    # plt.subplot(1, 3, 3)
    # samples = model.sample(500).data
    # plot_data(samples, color="black", alpha=0.5)
    # plt.title("Generated samples")
    # plt.savefig("./examples/ex_2d.png")
    # plt.show()

    # for f in flows:
    #     x = f(x)[0].data
    #     plot_data(x, color="black", alpha=0.5)
    #     plt.show()




    # alpha = Symbol('alpha')
    # sigma = Symbol('sigma')
    # ind1, ind2 = 180, 185
    # # C_alpha = alpha*gamma(1+alpha/2) / (2**(1-alpha)*np.pi*gamma(1-alpha/2))
    # C_alpha = 0.1712
    # # solved_value = solve([(sigma**(alpha-1))/((2*(u[ind1,ind1]-x0)**2)**(alpha/2+1))-f[ind1,ind1]/C_alpha, (sigma**(alpha-1))/((2*(u[ind2,ind2]-y0)**2)**(alpha/2+1))-f[ind2,ind2]/C_alpha], [alpha, sigma])
    # solved_value = solve([(sigma**(alpha-1))/((2*(u[ind1,ind1]-x0)**2)**(alpha/2+1))-kernel[ind1,ind1]/C_alpha, (sigma**(alpha-1))/((2*(u[ind2,ind2]-y0)**2)**(alpha/2+1))-kernel[ind2,ind2]/C_alpha], [alpha, sigma])
    # print(solved_value)

    # ## Estimating alpha
    # # M, m, eps = 50000, 2, 0.2 #for pure jump Levy
    # # M, m, eps = 50000, 2, 1 #for constant BM + Levy
    # M, m, eps = 10000, 2, 1
    # N = 1
    # samples = model.sample(M).data.cpu().detach().numpy()
    # alpha_estimated = np.zeros(N)
    # sigma_estimated = np.zeros(N)
    # for k in range(N):
    #     k = k + 1
        
    #     ##标准化
    #     result1 = pow(samples[:,0]*sigma_x+mu_x-x0, 2) + pow(samples[:,1]*sigma_y+mu_y-y0, 2) < (eps)**2
    #     result2 = pow(samples[:,0]*sigma_x+mu_x-x0, 2) + pow(samples[:,1]*sigma_y+mu_y-y0, 2) <= (m * eps)**2
    #     n_0 = np.sum(result2) - np.sum(result1)
    #     result3 = pow(samples[:,0]*sigma_x+mu_x-x0, 2) + pow(samples[:,1]*sigma_y+mu_y-y0, 2) < (m**k * eps)**2
    #     result4 = pow(samples[:,0]*sigma_x+mu_x-x0, 2) + pow(samples[:,1]*sigma_y+mu_y-y0, 2) <= (m**(k+1) * eps)**2
    #     n_k = np.sum(result4) - np.sum(result3)
        
    #     # ##无标准化
    #     # result1 = pow(samples[:,0]-x0, 2) + pow(samples[:,1]-y0, 2) < (eps)**2
    #     # result2 = pow(samples[:,0]-x0, 2) + pow(samples[:,1]-y0, 2) <= (m * eps)**2
    #     # n_0 = np.sum(result2) - np.sum(result1)
    #     # result3 = pow(samples[:,0]-x0, 2) + pow(samples[:,1]-y0, 2) < (m**k * eps)**2
    #     # result4 = pow(samples[:,0]-x0, 2) + pow(samples[:,1]-y0, 2) <= (m**(k+1) * eps)**2
    #     # n_k = np.sum(result4) - np.sum(result3)
        
    #     alpha_estimated[k-1] = 1/(k*np.log(m)) * np.log(n_0/n_k)
    #     # print(alpha_estimated)
    
    
    #     ## Estimating 
    #     tmp1 = alpha_estimated[k-1] * eps**alpha_estimated[k-1] * m**(k*alpha_estimated[k-1]) *n_k
    #     C_alpha = alpha_estimated[k-1]*gamma(1+alpha_estimated[k-1]/2) / (2**(1-alpha_estimated[k-1])*np.pi*gamma(1-alpha_estimated[k-1]/2))
    #     tmp2 = 2*np.pi * C_alpha * t_star * M * (1 - 1/m**alpha_estimated[k-1])
    #     sigma_estimated[k-1] = (tmp1/tmp2)**(1/alpha_estimated[k-1])
    #     # print(sigma_estimated)


    # print('alpha:',np.average(alpha_estimated))
    # print('sigma:',np.average(sigma_estimated))

