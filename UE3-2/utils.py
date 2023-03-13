import torch
import torch.nn.functional as F
import math
from math import pi, gamma, sqrt
import numpy as np
from torch.utils.data import Dataset

def get_grid3d(S, T, time_scale=1.0, device='cpu'):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt

# Gaussian Random Field generator
class GRF_Mattern(object):
    def __init__(self, dim, size, length=1.0, nu=None, l=0.1, sigma=1.0, boundary="periodic", constant_eig=None, device=None):

        self.dim = dim
        self.device = device
        self.bc = boundary

        a = sqrt(2/length)
        if self.bc == "dirichlet":
            constant_eig = None
        
        
        if nu is not None:
            kappa = sqrt(2*nu)/l
            alpha = nu + 0.5*dim
            self.eta2 = size**dim * sigma*(4.0*pi)**(0.5*dim)*gamma(alpha)/(kappa**dim * gamma(nu))
        else:
            self.eta2 = size**dim * sigma*(sqrt(2.0*pi)*l)**dim
        # if sigma is None:
        #     sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2
        if self.bc == "periodic":
            const = (4.0*(pi**2))/(length**2)
        else:
            const = (pi**2)/(length**2)

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            k2 = k**2
            if nu is not None:
                # self.sqrt_eig = (size**dim)*sqrt(2.0)*sigma*((const*(k**2) + tau**2)**(-alpha/2.0))
                eigs = 1.0 + (const/(kappa*length)**2*k2)
                self.sqrt_eig = self.eta2/(length**dim) * eigs**(-alpha/2.0)
            else:
                self.sqrt_eig = self.eta2/(length**dim)*torch.exp(-(l)**2*const*k2/4.0)

            if constant_eig is not None:
                self.sqrt_eig[0] = constant_eig # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            k2 = k_x**2 + k_y**2 
            if nu is not None:
                # self.sqrt_eig = (size**dim)*sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
                eigs = 1.0 + (const/(kappa*length)**2*k2)
                self.sqrt_eig = self.eta2/(length**dim) * eigs**(-alpha/2.0)
            else:
                self.sqrt_eig = self.eta2/(length**dim)*torch.exp(-(l)**2*const*k2/4.0)

            if constant_eig is not None:
                self.sqrt_eig[0,0] = constant_eig # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            k2 = k_x**2 + k_y**2 + k_z**2
            if nu is not None:
                # self.sqrt_eig = (size**dim)*sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
                eigs = 1.0 + (const/(kappa*length)**2*k2)
                self.sqrt_eig = self.eta2/(length**dim) * eigs**(-alpha/2.0)
            else:
                self.sqrt_eig = self.eta2/(length**dim)*torch.exp(-(l)**2*const*k2/4.0)

            if constant_eig is not None:
                self.sqrt_eig[0,0,0] = constant_eig # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        if self.bc == 'dirichlet':
            coeff.real[:] = 0
        if self.bc == 'neumann':
            coeff.imag[:] = 0
        coeff = self.sqrt_eig*coeff

        u = torch.fft.irfftn(coeff, self.size, norm="backward")
        return u
    
    
# 2D Wave equation solver
class WaveEq2D():
    def __init__(self,
                 xmin=0,
                 xmax=1,
                 ymin=0,
                 ymax=1,
                 Nx=100,
                 Ny=100,
                 c=1.0,
                 dt=1e-3,
                 tend=1.0,
               device=None,
                 dtype=torch.float64,
                 ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.Nx = Nx
        self.Ny = Ny
        x = torch.linspace(xmin, xmax, Nx+1, device=device, dtype=dtype)
        y = torch.linspace(ymin, ymax, Ny+1, device=device, dtype=dtype)
        self.x = x
        self.y = y
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.X, self.Y = torch.meshgrid(x,y,indexing='ij')
        self.c = c
        self.phi = torch.zeros_like(self.X[:Nx,:Ny], device=device)
        self.psi = torch.zeros_like(self.phi, device=device)
        self.phi0 = torch.zeros_like(self.phi, device=device)
        self.dt = dt
        self.tend = tend
        self.t = 0
        self.it = 0
        self.Phi = []
        self.T = []
        self.device = device
        
    

    # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
    def CD_i(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
        return data_diff_i

    def CD_ij(self, data, axis_i, axis_j, dx, dy):
        data_diff_i = self.CD_i(data,axis_i,dx)
        data_diff_ij = self.CD_i(data_diff_i,axis_j,dy)
        return data_diff_ij

    def CD_ii(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
        return data_diff_ii

    def Dx(self, data):
        data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
        return data_dx

    def Dy(self, data):
        data_dy = self.CD_i(data=data, axis=1, dx=self.dy)
        return data_dy


    def Dxy(self, data):
        data_dxy = self.CD_ij(data, axis_i=0, axis_j=1, dx=self.dx, dy=self.dy)
        return data_dxy
        

    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
        return data_dxx

    def Dyy(self, data):
        data_dyy = self.CD_ii(data,axis=1, dx=self.dy)
        return data_dyy

    
    def wave_calc_RHS(self, phi, psi):
        phi_xx = self.Dxx(phi)
        phi_yy = self.Dyy(phi)
        
        psi_RHS = self.c**2 * (phi_xx + phi_yy) # it is usually c^2, but c is consistent with simflowny code
        phi_RHS = psi
        return phi_RHS, psi_RHS
        
    def update_field(self, field, RHS, step_frac):
        field_new = field + self.dt*step_frac*RHS
        return field_new
        


    def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
        field_new = field + self.dt/6.0*(RHS1 + 2*RHS2 + 2.0*RHS3 + RHS4)
        return field_new

    def wave_rk4(self, phi, psi, t=0):
        phi_RHS1, psi_RHS1 = self.wave_calc_RHS(phi, psi)
        t1 = t + 0.5*self.dt
#         display(phi.shape)
#         display(phi_RHS1.shape)
        phi1 = self.update_field(phi, phi_RHS1, step_frac=0.5)
        psi1 = self.update_field(psi, psi_RHS1, step_frac=0.5)
        
        phi_RHS2, psi_RHS2 = self.wave_calc_RHS(phi1, psi1)
        t2 = t + 0.5*self.dt
        phi2 = self.update_field(phi, phi_RHS2, step_frac=0.5)
        psi2 = self.update_field(psi, psi_RHS2, step_frac=0.5)
        
        phi_RHS3, psi_RHS3 = self.wave_calc_RHS(phi2, psi2)
        t3 = t + self.dt
        phi3 = self.update_field(phi, phi_RHS3, step_frac=1.0)
        psi3 = self.update_field(psi, psi_RHS3, step_frac=1.0)
        
        phi_RHS4, psi_RHS4 = self.wave_calc_RHS(phi3, psi3)
        
        t_new = t + self.dt
        psi_new = self.rk4_merge_RHS(psi, psi_RHS1, psi_RHS2, psi_RHS3, psi_RHS4)
        phi_new = self.rk4_merge_RHS(phi, phi_RHS1, phi_RHS2, phi_RHS3, phi_RHS4)
        
        return phi_new, psi_new, t_new
    
    def plot_data(self, cmap='jet', vmin=None, vmax=None, fig_num=0, title='', xlabel='', ylabel=''):
        plt.ion()
        fig = plt.figure(fig_num)
        plt.cla()
        plt.clf()
        
        c = plt.pcolormesh(self.X, self.Y, self.phi, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        fig.colorbar(c)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis('equal')
        plt.axis('square')
        plt.draw() 
        plt.pause(1e-17)
        plt.show()

        
    def wave_driver(self, phi0, save_interval=10, plot_interval=0):
        # plot results
        # t,it = get_time(time)
#         display(phi0[:self.Nx,:self.Ny].shape)
        self.phi0 = phi0[:self.Nx,:self.Ny]
        self.phi = self.phi0
        
        if plot_interval != 0 and self.it % plot_interval == 0:
            self.plot_data(vmin=-1,vmax=1,title=r'\{phi}')
        if save_interval != 0 and self.it % save_interval == 0:
            self.Phi.append(self.phi)
            # self.Psi.append(self.psi)
            self.T.append(self.t)
        # Compute equations
        while self.t < self.tend:
#             print(f"t:\t{self.t}")
            self.phi, self.psi, self.t = self.wave_rk4(self.phi, self.psi, self.t)
            
            self.it += 1
            if plot_interval != 0 and self.it % plot_interval == 0:
                self.plot_data(vmin=-1,vmax=1,title=r'\{phi}')
            if save_interval != 0 and self.it % save_interval == 0:
                self.Phi.append(self.phi)
                # self.Psi.append(self.psi)
                self.T.append(self.t)

        return torch.stack(self.Phi)
    

# Dataset loader for 2D samples
class DataLoader2D(object):
    def __init__(self, data, nx=128, nt=100, sub=1, sub_t=1):
        self.sub = sub
        self.sub_t = sub_t            
        s = nx
        # if nx is odd
        if (s % 2) == 1:
            s = s - 1
        self.S = s // sub
        self.T = nt // sub_t
        self.T += 1
        data = data[:, 0:self.T:sub_t, 0:self.S:sub, 0:self.S:sub]
        self.data = data.permute(0, 2, 3, 1)
        
    def make_loader(self, n_sample, batch_size, start=0, train=True, only_data=True, repeat=False):
        a_data = self.data[start:start + n_sample, :, :, 0].reshape(n_sample, self.S, self.S)
        u_data = self.data[start:start + n_sample].reshape(n_sample, self.S, self.S, self.T)
        gridx, gridy, gridt = get_grid3d(self.S, self.T)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 1).repeat([1, 1, 1, self.T, 1])
        if repeat:
            a_data = torch.cat((gridx.repeat([n_sample, 1, 1, 1, 1]),
                                gridy.repeat([n_sample, 1, 1, 1, 1]),
                                gridt.repeat([n_sample, 1, 1, 1, 1]),
                                a_data), dim=-1)
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        if only_data:
#             a_data = torch.permute(a_data, (0, 4, 3, 1, 2))
#             u_data = torch.permute(u_data, (0, 3, 1, 2))
            a_data = torch.permute(a_data, (0, 4, 1, 2, 3))
            return {"IC": a_data.numpy()}, {"sol": u_data.numpy()}
        if train:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader
