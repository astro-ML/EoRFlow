#!/usr/bin/env python
import os
import glob
import sys
import numpy as np
import h5py
from py21cmfast_tools import calculate_ps

# computes 1d and 2d power spectra for lightcones in a directory

# MPI rank/size (from OpenMPI environment vars, optional)
rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))

# --- User settings ---
input_directory  = '/pfs/10/work/hd_pt254-eorflow/database/imnn/deriv0'
TOTAL            = None     # if None, all files
ps_redshifts     = np.linspace(5.0, 12.0, 15)
# ----------------------

def process_file(h5_filename):
    infile = os.path.join(input_directory, h5_filename)
    print(f"[{rank}] Processing {h5_filename}")
    try:
        with h5py.File(infile, 'a') as f:
            # 1) Load the truncated lightcone and its redshifts
            bt = f['lightcones/brightness_temp'][()]       # (140,140,2350)
            lc_z = f['lightcone_redshifts'][()] # (2350,)

        # 2) Compute the power spectra
        ps = calculate_ps(
            lc=bt,
            lc_redshifts=lc_z,
            zs=ps_redshifts,
            box_length=200,
            box_side_shape=140,
            calc_2d=True,
            calc_1d=True,
            calc_global=False,
            log_bins=True,
            nbins=10,
            kpar_bins=10
        )

        # 3) Extract & sanitize
        k       = ps['k']
        k       = np.nan_to_num(k, nan=0.0)
        k_perp  = ps['final_kperp']
        k_par   = ps['final_kpar']
        PS_1D   = ps['ps_1D']
        PS_2D   = ps['final_ps_2D']
        PS_1D   = np.nan_to_num(PS_1D, nan=0.0, posinf=1e10, neginf=-1e10)
        PS_2D   = np.nan_to_num(PS_2D, nan=0.0, posinf=1e10, neginf=-1e10)
        zout    = ps['redshifts']

        # 4) Write back into the HDF5 under /ps group
        with h5py.File(infile, 'a') as f:
            if 'ps' in f:
                del f['ps']
            grp = f.create_group('ps')

            grp.create_dataset('k',            data=k,        dtype='f4')
            grp.create_dataset('k_perp',       data=k_perp,   dtype='f4')
            grp.create_dataset('k_par',        data=k_par,    dtype='f4')
            grp.create_dataset('ps_redshifts', data=zout,     dtype='f4')
            grp.create_dataset('ps1d',         data=PS_1D,    dtype='f4', compression='gzip')
            grp.create_dataset('ps2d',         data=PS_2D,    dtype='f4', compression='gzip')

        print(f"[{rank}] Embedded PS into {h5_filename}")

    except Exception as e:
        print(f"[{rank}] ERROR on {h5_filename} — {e}", file=sys.stderr)


if __name__ == "__main__":
    # gather all files
    all_files = sorted(f for f in os.listdir(input_directory) if f.endswith('.h5'))
    if TOTAL is None or TOTAL > len(all_files):
        TOTAL = len(all_files)
    all_files = all_files[:TOTAL]

    # split by rank
    base, rem = divmod(TOTAL, size)
    n_local   = base + (1 if rank < rem else 0)
    start     = rank * base + min(rank, rem)
    my_files  = all_files[start : start + n_local]

    print(f"[{rank}/{size}] processing {len(my_files)} files (idx {start}→{start+n_local-1})")
    for fname in my_files:
        process_file(fname)