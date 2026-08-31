from typing import Dict, Union

import math
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.autograd import grad

# Conditional Flow Matching model for inference, adapted from https://github.com/cosmostatistics/Inflation21cmSBI/tree/main

class ConditionalCFM(nn.Module):
    def __init__(
        self,
        n_dim: int,
        summary_dim: int,
        n_layers: int,
        hidden_dim: int,
        alpha: float = 0.0,
        p_drop: float = 0.0,
    ):
        super().__init__()
        self.param_dim = n_dim
        self.alpha = alpha
        input_dim = self.param_dim + summary_dim + 1
        glu_cond_dim = self.param_dim + 1
        layers = [CondGLUMLP(input_dim, glu_cond_dim, hidden_dim)]
        for _ in range(n_layers):
            layers.append(CondGLUMLP(hidden_dim, glu_cond_dim, hidden_dim, p_drop=p_drop))
        layers.append(CondGLUMLP(hidden_dim, glu_cond_dim, self.param_dim, p_drop=0))
        self.layers = nn.ModuleList(layers)
        
    def velocity(self, x, cond):
        # First block: (θ, cond) -> hidden
        inp = torch.cat([x, cond], dim=1)
        h = self.layers[0](inp, cond)  # no skip here

        # Middle blocks: hidden -> hidden + skip
        for block in self.layers[1:-1]:
            delta = block(h, cond)      # new features
            h = h + delta               # residual add

        # Final block: hidden -> θ
        out = self.layers[-1](h, cond)
        return out
        
    def batch_loss(
        self,
        theta: torch.Tensor,   # samples of θ, shape (batch, data_dim)
        data: torch.Tensor,   # corresponding observations, (batch, cond_dim)
    ) -> torch.Tensor:
        # sample a random time [0,1] from a power law distribution with alpha

        u = torch.rand(theta.shape[0], 1, device=theta.device)  # uniform [0,1]
        t = u.pow(1.0 / (1.0 + self.alpha))   # sample such that p(t) ∝ t^α
 
        # perturb θ → θₜ
        noise = torch.randn_like(theta)
        theta_t = (1 - t) * theta + t * noise

        # concatenate time and data to condition the network
        glu_cond = torch.cat([t, theta_t], dim=1)  # (batch, cond_dim)
 
        # predict instantaneous velocity
        pred_v = self.velocity(data, glu_cond)

        # the true velocity is noise − x
        true_v = noise - theta

        return ((pred_v - true_v)**2).mean()

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        data: torch.Tensor,       # (cond_dim,) or (n_samples, cond_dim)
    ) -> torch.Tensor:
        # 1) draw θ₁ ~ N(0,I)
        device = data.device
        dtype  = data.dtype
        theta_1 = torch.randn(n_samples, self.param_dim, device=device, dtype=dtype)

        # make data_obs shape (n_samples, cond_dim)
        if data.ndim == 1:
            data_rep = data.unsqueeze(0).repeat(n_samples, 1)
        else:
            data_rep = data
            
        # wrap the net so it sees (t, theta, data)
        def net_wrapper(t, theta_t):
            # t: scalar tensor
            tt = t * torch.ones_like(theta_t[:, :1])
            glu_cond = torch.cat([tt, theta_t], dim=1)  # (batch, cond_dim)
            return self.velocity(data_rep, glu_cond)  # (batch, param_dim)

        # integrate from t=1 → 0
        t_space = torch.tensor([1., 0.], device=device, dtype=dtype)
        theta_path = odeint(net_wrapper, theta_1, t_space, rtol=1e-5, atol=1e-5)
        # θ_path shape: (2, n_samples, data_dim)
        theta_0 = theta_path[-1]
        return theta_0
    
    def latent_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of latent variable z under a standard
        Gaussian prior.

        Parameters
        ----------
        z : torch.Tensor
            Latent variable tensor, shape (batch, data_dim).

        Returns
        -------
        torch.Tensor
            Log probability of z, shape (batch,).
        """
        return - (z**2 /2 +0.5 * math.log(2 * math.pi)).sum(dim=1)
    
    def log_prob(self, theta: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of the model given parameters theta and
        observations data.

        Parameters
        ----------
        theta : torch.Tensor
            Model parameters, shape (batch, data_dim).
        data : torch.Tensor
            Observations, shape (batch, cond_dim).

        Returns
        -------
        torch.Tensor
            Log probability of the model, shape (batch,).
        """
        batch_size = theta.size(0)
        dtype = theta.dtype
        device = theta.device
        
        if data.ndim == 1:
            data = data.unsqueeze(0).repeat(batch_size, 1)
        else:
            data = data
        # Wrap the network such that the ODE solver can call it
        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                # Prepare the network inputs
                x_t = state[0].detach().requires_grad_(True)
                tt = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device).requires_grad_(False)
                glu_cond = torch.cat([tt, x_t], dim=1)  # (batch, cond_dim)
                # Predict v
                v = self.velocity(data, glu_cond)
                # Calculate the jacobian trace
                dlogp_dt = -autograd_trace(v, x_t).view(-1, 1) 
            return v.detach(), dlogp_dt.detach()
        # Set initial conditions for the ODE
        logp_diff_1 = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        states = (theta, logp_diff_1)
        x_t, logp_diff_t = odeint(
                net_wrapper,
                states,
                torch.tensor([0, 1], dtype=dtype, device=device),
                atol=1e-5,
                rtol=1e-5,
                )
        # Extract the latent space points and the jacobians
        x_1 = x_t[-1].detach()
        jac = logp_diff_t[-1].detach()
        return self.latent_log_prob(x_1).squeeze() - jac.squeeze()
        
    def transfer_overlapping_inputs(self, src_ckpt_path: str) -> None:
        """
        Partially load a *smaller-input* checkpoint into the current
        (larger-input) ConditionalCFM. We copy:

        1. All parameters whose shapes match exactly.
        2. The overlapping columns of the first CondGLUMLP layer
           (``layers[0].fc``).

        Parameters
        ----------
        src_ckpt_path : str
            Path to a checkpoint saved from a model with a *smaller*
            ``summary_dim`` / ``input_dim``.
        """
        ckpt = torch.load(src_ckpt_path)

        if "inference_state_dict" not in ckpt:
            raise KeyError(
                f"'inference_state_dict' not found in checkpoint keys: {ckpt.keys()}"
            )

        src_sd: Dict[str, torch.Tensor] = ckpt["inference_state_dict"]
        dst_sd: Dict[str, torch.Tensor] = self.state_dict()

        # ---------------------------------------------------------------------
        # 1) Copy overlapping columns of the very first FC in layers[0]
        # ---------------------------------------------------------------------
        first_fc_prefix = "layers.0.fc."

        src_w = src_sd[first_fc_prefix + "weight"]   # [2*H, old_in+cond]
        src_b = src_sd[first_fc_prefix + "bias"]     # [2*H]

        dst_w = dst_sd[first_fc_prefix + "weight"]   # [2*H, new_in+cond]
        dst_b = dst_sd[first_fc_prefix + "bias"]     # [2*H]

        n_overlap_in = min(src_w.size(1), dst_w.size(1))

        with torch.no_grad():  # purely initialisation
            dst_w[:, :n_overlap_in].copy_(src_w[:, :n_overlap_in])
            dst_b.copy_(src_b)

        # ---------------------------------------------------------------------
        # 2) Copy all other parameters that match exactly
        # ---------------------------------------------------------------------
        for name, src_param in src_sd.items():
            if name.startswith(first_fc_prefix):
                continue  # first layer already handled
            if name in dst_sd and dst_sd[name].shape == src_param.shape:
                dst_sd[name].copy_(src_param)

        # ---------------------------------------------------------------------
        # 3) Commit the updated state dict
        # ---------------------------------------------------------------------
        self.load_state_dict(dst_sd)
    
class CondGLUMLP(nn.Module):
    def __init__(self, in_dim, cond_dim, hidden_dim, p_drop=0.0):
        super().__init__()   
        # we’re going to split into two halves, so output 2*hidden_dim
        self.fc = nn.Linear(in_dim + cond_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()


    def forward(self, x, y):
        # x: [batch, in_dim], y: [batch, cond_dim]
        h = torch.cat([x, y], dim=-1)       # [batch, in + cond]
        gates = self.fc(h)                  # [batch, 2*hidden_dim]
        A, B = gates.chunk(2, dim=-1)       # each [batch, hidden_dim]
        return A * torch.sigmoid(B)         # gated output

def autograd_trace(x_out, x_in, drop_last=False):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    if drop_last:
        for i in range(x_out.shape[1]-1):
            trJ += grad(x_out[:, i].sum(), x_in,
                        retain_graph=True)[0].contiguous()[:, i].contiguous().detach()
    else:
        for i in range(x_out.shape[1]):
            trJ += grad(x_out[:, i].sum(), x_in,
                        retain_graph=True)[0].contiguous()[:, i].contiguous().detach()
    return trJ.contiguous()