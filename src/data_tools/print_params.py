import numpy as np

# Replace this with your actual NPZ file path.
fid_mid = '0_simrun_3783.npz'
fid_late = 'lightcone_5z25CDMOMm0.316E0222.325LX40.993Tvir5.532Zeta83.402.npz'
fid_early = 'run3249.npz'
fid_mid_2 = 'run2322.npz'
npz_file = "/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/test/" + fid_late
data = np.load(npz_file)
# Assumes that the NPZ file contains a key 'params'
params = data['params']
xH = data['label']

# Define labels in the correct order.
labels = [
    r"m$_{\mathrm{WDM}}$",
    r"$\Omega_{\mathrm{M}}$",
    r"E$_0$",
    r"L$_{\mathrm{X}}$",
    r"T$_{\mathrm{vir}}$",
    r"$\zeta$"
]
#print(xH)
print(f"model = {npz_file}")
for label, value in zip(labels, params):
    print(f"{label} = {value}")