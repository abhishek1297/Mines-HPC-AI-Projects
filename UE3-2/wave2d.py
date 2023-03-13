"""
Abhishek Purandare
"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.key import Key
from modulus.models.layers.spectral_layers import fourier_derivatives
from modulus.node import Node

from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import SupervisedGridConstraint
from modulus.domain.validator import GridValidator
from modulus.dataset import DictGridDataset
from modulus.utils.io.plotter import GridValidatorPlotter

from functorch import vmap, grad
from utils import *
# ***************************************************
                    # DEFAULTS
# ***************************************************
import os
import sys
ROOT_DIR = os.path.abspath(os.getcwd())
sys.path.append(ROOT_DIR)
C = 1.0
DT = 1e-4

class Wave2D_PINO(torch.nn.Module):
    
    """
    PINO solver for 2D wave equation
    """
    def __init__(self, gradient_method="exact", time_dependent=False, c=C, dt=DT):
        super().__init__()
        self.gradient_method = gradient_method
        self.c = c
        self.dt = dt
        if time_dependent and gradient_method == "exact":
            raise Exception("Time dependent exact method implementation not available yet.")
            
        self.time_dependent = time_dependent
    
    def forward(self, input_var: Dict[str, torch.Tensor]):
        
        u = input_var["sol"]
        # a = input_var["IC"]
        
        # The second dimension is NOT for time. But since that's the only available
        # I use it to store all u(x,y) till the solution
        _, dim_u_t, dim_u_x, dim_u_y = u.shape

        dxf = 1.0 / dim_u_x
        dyf = 1.0 / dim_u_y
        # **************************************************************************************
                                            # EXACT DERIVATIVES
        # **************************************************************************************
        if self.gradient_method == "exact":
            ddudtt = 0.
            dduddx = input_var["sol__x__x"]
            dduddy = input_var["sol__y__y"]
        
        # **************************************************************************************
                                            # FOURIER DERIVATIVES
        # **************************************************************************************
        elif self.gradient_method == "fourier":
            padding = [0, dim_u_y - 1, 0, dim_u_x - 1]
            dims = [2.0, 2.0]
            # Note that we do not need fourier derivatives for temporal dimension.
            # These are calculated below (dudt, ddudtt)
            # However, modulus fourier_derivatives function expects dimension values to be
            # len(u.shape) - 2 == len(dims)
            # I am passing the following dt just in case, but it doesn't matter what you pass here.
            if self.time_dependent:
                padding += [0, 0]
                dims += [self.dt]
            u = F.pad(u, padding, mode="reflect")
            f_du, f_ddu = fourier_derivatives(u, dims)
            if self.time_dependent:
                # dudt = (u[:, :, 1:, :dim_u_x, :dim_u_y] - u[:, :, :-1, :dim_u_x, :dim_u_y]) / self.dt
                ddudtt = (u[:, :, 2:, :dim_u_x, :dim_u_y]
                          - 2.0*u[:, :, 1:-1, :dim_u_x, :dim_u_y]
                          + u[:, :, :-2, :dim_u_x, :dim_u_y]) / (self.dt**2)
                dudx_fourier = f_du[:, 0:1, 1:dim_u_t-1, :dim_u_x, :dim_u_y]
                dudy_fourier = f_du[:, 1:2, 1:dim_u_t-1, :dim_u_x, :dim_u_y]
                dduddx_fourier = f_ddu[:, 0:1, 1:dim_u_t-1, :dim_u_x, :dim_u_y]
                dduddy_fourier = f_ddu[:, 1:2, 1:dim_u_t-1, :dim_u_x, :dim_u_y]
            else:
                # No temporal dimension
                ddudtt = 0. 
                # dudx_fourier = f_du[:, 0:1, :dim_u_x, :dim_u_y]
                # dudy_fourier = f_du[:, 1:2, :dim_u_x, :dim_u_y]
                dduddx_fourier = f_ddu[:, 0:1, :dim_u_x, :dim_u_y]
                dduddy_fourier = f_ddu[:, 1:2, :dim_u_x, :dim_u_y]
            dduddx = dduddx_fourier
            dduddy = dduddy_fourier
                
        else:
            raise Exception(f"Invalid value for gradient method: {self.gradient_method}")
        # **************************************************************************************
                                            # COMPUTE WAVE
        # **************************************************************************************
        wave2d = (
                ddudtt
                - self.c ** 2 * (dduddx + dduddy)
        )
        # Zero outer boundary
        tmp = wave2d[:, :, 2:-2, 2:-2]
        padding_size = [2, 2, 2, 2]
        if self.time_dependent:
            padding_size += [0, 0]
        
        wave2d = F.pad(tmp, padding_size, "constant", 0)
        output_var = {
            "wave2d": dxf * wave2d,
        }  # weight boundary loss higher

        return output_var
        
        
@modulus.main(config_path=".", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # **************************************************************************************
                                        # CONFIGURATIONS
    # **************************************************************************************
    datapath = cfg.custom.datapath
    dim = cfg.arch.fno.dimension
    n = cfg.custom.n
    nx = cfg.custom.nx
    ny = cfg.custom.ny
    nt = cfg.custom.nt
    c = C
    sub = cfg.custom.sub
    sub_t = cfg.custom.sub_t
    l = cfg.custom.l
    L = cfg.custom.L
    sigma = cfg.custom.sigma
    nu = None
    n_train = cfg.custom.n_train
    n_test = cfg.custom.n_test
    n_samples = n_train + n_test
    dt = DT
    time_dependent = False # Don't know how to pass all the time dependent values and evolve the in FNO
    save_int = int(1e-2/dt)
    device = torch.device(f'cuda:{cfg.custom.device}' if torch.cuda.is_available() else 'cpu')
    print("[INFO] running on ", torch.cuda.get_device_name(device=device))
    
    if time_dependent and cfg.custom.gradient != "fourier":
        raise Exception(f"[ERROR] Only Fourier derivatives are implemented with time dependent derivatives")
    # **************************************************************************************
                                    # DATASET LOADING/GENERATION
    # **************************************************************************************
    p = os.path.join(ROOT_DIR, datapath)
    if datapath and os.path.exists(p):
        u = torch.load(p)
        assert u.shape[-2:] == (nx, ny)
        print("[INFO] Loaded", p, u.shape)
    else:
        print("[INFO] Generating {n_samples} dataset")
        grf = GRF_Mattern(dim, nx, length=L, nu=nu, l=l, sigma=sigma, boundary="periodic", device=device)
        U0 = grf.sample(n_samples)
        print("[INFO] GRF done")

        wave_eq = WaveEq2D(Nx=nx, Ny=ny, dt=dt, device=device)
        U = vmap(wave_eq.wave_driver, in_dims=(0, None))(U0, save_int)
        u = U.cpu()
        del U0, U
    u = u.numpy()
    st = 0 if time_dependent else -1
    end = u.shape[1]
    n_test_end = n_train + n_test
    invar_train = {"IC": u[:n_train, :1, :, :]}
    outvar_train = {"sol": u[:n_train, st:end, :, :]}
    invar_test = {"IC": u[n_train:n_test_end, :1, :, :]}
    outvar_test = {"sol": u[n_train:n_test_end, st:end, :, :]}
    print("[INFO] IC shape", invar_train["IC"].shape, invar_test["IC"].shape)
    print("[INFO] Sol shape", outvar_train["sol"].shape, outvar_test["sol"].shape)
    print("[INFO] Data loaded")
    
    del u
    
    input_keys = [
        Key("IC", scale=(np.mean(invar_train["IC"]), np.std(invar_train["IC"])))
    ]
    output_keys = [
        Key("sol", scale=(np.mean(outvar_train["sol"]), np.std(outvar_train["sol"])))
    ]

    # add additional constraining values for wave2d variable
    outvar_train["wave2d"] = np.zeros_like(outvar_train["sol"])
    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset = DictGridDataset(invar_test, outvar_test)
    del invar_train, outvar_train, invar_test, outvar_test
    # **************************************************************************************
                                            # MODELS
    # **************************************************************************************
    # Define FNO model
    decoder_net = instantiate_arch(
        cfg=cfg.arch.decoder,
        output_keys=output_keys,
    )
    fno = instantiate_arch(
        cfg=cfg.arch.fno,
        input_keys=[input_keys[0]],
        decoder_net=decoder_net,
    )
    # Make custom Wave2d residual node for PINO
    inputs = [
        "sol",
        "IC",
    ]
    if cfg.custom.gradient == "exact":
        derivatives = [
            Key("sol", derivatives=[Key("x"), Key("x")]),
            Key("sol", derivatives=[Key("y"), Key("y")]),
        ]
    
        fno.add_pino_gradients(derivatives=derivatives,
                                domain_length=[1.0, 1.0])
    
        inputs += ["sol__x__x",
                "sol__y__y"]

        
    wave2d_node = Node(
        inputs=inputs,
        outputs=["wave2d"],
        evaluate=Wave2D_PINO(gradient_method=cfg.custom.gradient),
        name="Wave2d Node",
    )
    nodes = [fno.make_node('fno'), wave2d_node]

    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised, "supervised")

    # add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        requires_grad=True,
    )
    domain.add_validator(val, "test")
    # make solver
    slv = Solver(cfg, domain)

    print("[INFO] All set")
    # start solver
    if cfg.custom.debug:
        with torch.autograd.detect_anomaly():
            slv.solve()
    else:
        slv.solve()
    
if __name__ == "__main__":
    run()
