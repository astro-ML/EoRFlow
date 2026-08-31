import os
import sys
sys.path.append('/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/src/models')
sys.path.append('/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/src/data_tools')
import math
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from torch.utils.data import DataLoader
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset


# Noise-trained model checkpoint
#MODEL_PATH = '/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/output/noise_augmented/Pk_window_10_512_std5/best_model.pth'
MODEL_PATH = '/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/output/pure/Pk_10_512_Gaussian0.05/best_model.pth'

# Challenge noise folders 
DATASETS = {
    "PS1": './PS1/pure',
    "PS2": './PS2/pure'
}

# Posterior sampling settings
SAMPLE_SIZE = 10000   
BINS        = 100     # number of bins per axis
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sigmoid flag & epsilon (if you used logit during training)
USE_SIGMOID = True
EPS         = 1e-5

# Temperature scaling (1.0 = none)
TEMPERATURE = 1.0

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def load_flow(path):
    params = {
        'flow': {
            'n_dim': 3,
            'n_blocks': 10,
            'n_nodes': 512,
            'cond_dims': 303,
            'load': False,
            'model_location': '',
            'dropout': 0.0,
            'sigmoid': False
        }
    }
    block = ConditionalInvertibleBlock(params)
    flow = block.flow.to(DEVICE)
    flow.load_state_dict(torch.load(path, map_location=DEVICE))
    flow.eval()
    return flow

def sample_all(flow, data_dir):
    """
    Uses PowerSpectrumDataset on `data_dir` (folder of .npz files),
    draws SAMPLE_SIZE posterior samples per file, and returns
    an array shape (N_files * SAMPLE_SIZE, 3).
    """
    ds = PowerSpectrumDataset(
        [data_dir],
        add_noise=False,
        augment_noise=False,
        add_gaussian=True,
        gaussian_std=0.05,
        std_strength=1.0,
        k_scale=False,
        convert_dimensionless=False,
        logit=False
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    all_samps = []
    for cond, _ in loader:
        cond = cond.to(DEVICE)
        cond_rep = cond.repeat(SAMPLE_SIZE, 1)
        z = torch.randn(SAMPLE_SIZE, 3, device=DEVICE) * math.sqrt(TEMPERATURE)
        x, _ = flow(z, c=[cond_rep], rev=True)
        if USE_SIGMOID:
            x = torch.sigmoid(x)
            x = (x - EPS) / (1 - 2*EPS)
        all_samps.append(x.cpu().detach().numpy())
    return np.concatenate(all_samps, axis=0)

def store_posterior_fits(samples, bins, dataset_name):
    """
    Build a 3D histogram over [0,1]^3, normalize, and write FITS.
    """
    # 3D histogram
    hist, edges = np.histogramdd(
        samples,
        bins=bins,
        range=[[0,1],[0,1],[0,1]]
    )
    hist_norm = hist / np.sum(hist)
    fits_data = hist_norm 
    fits_data = np.transpose(hist_norm, (2,1,0))  # NAXIS3,2,1

    # Build header
    hdr = fits.Header()
    hdr['SIMPLE']  = True
    hdr['BITPIX']  = -64
    hdr['NAXIS']   = 3
    hdr['NAXIS1']  = fits_data.shape[2]
    hdr['NAXIS2']  = fits_data.shape[1]
    hdr['NAXIS3']  = fits_data.shape[0]
    hdr['HIERARCH AXIS1-FREQ'] = 'xHI 151-166 MHz'
    hdr['HIERARCH AXIS2-FREQ'] = 'xHI 166-181 MHz'
    hdr['HIERARCH AXIS3-FREQ'] = 'xHI 181-196 MHz'
    hdr['CRPIX1']  = 1
    hdr['CRPIX2']  = 1
    hdr['CRPIX3']  = 1
    hdr['CRVAL1']  = 0.005
    hdr['CRVAL2']  = 0.005
    hdr['CRVAL3']  = 0.005
    hdr['CDELT1']  = 0.01
    hdr['CDELT2']  = 0.01
    hdr['CDELT3']  = 0.01
    hdr['CUNIT1']  = ''
    hdr['CUNIT2']  = ''
    hdr['CUNIT3']  = ''
    hdr['BUNIT']   = ''
    hdr['BTYPE']   = 'Posterior Probability'
    hdr['ORIGIN']  = 'SKA Observatory'
    hdr['TELESCOP']= 'Example Telescope'
    hdr['OBSERVER']= 'SKAO Observer'
    hdr['DATE-OBS']= datetime.datetime.now().strftime('%Y-%m-%d')
    hdr.add_comment("3D posterior probability distribution normalized to 1.")
    
    hdu = fits.PrimaryHDU(data=fits_data, header=hdr)
    out_fname = f'{dataset_name}_posterior_pure_trained_ska.fits'
    hdu.writeto(out_fname, overwrite=True)
    print(f"→ Saved FITS: {out_fname}")
    return out_fname

# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__=='__main__':
    flow_noise = load_flow(MODEL_PATH)

    for name, dirpath in DATASETS.items():
        print(f"\nProcessing {name}...")
        samples = sample_all(flow_noise, dirpath)
        fits_file = store_posterior_fits(samples, BINS, name)
  