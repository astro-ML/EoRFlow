# EoRFlow

**Fast and robust inference of the Epoch of Reionization from 21cm 2D power spectra using deep learning and normalizing flows.**

![EoRFlow banner](docs/eorflow_banner.png)

## Overview

EoRFlow is a machine learning pipeline designed to reconstruct the global reionization history from simulated 21cm 2D power spectra. 

- Simulation-based inference (SBI) for likelihood-free problems
- Conditional invertible neural networks (cINNs) for posterior estimation
- easily adjustable to other physical or learned summaries of the 21cm signal 
- optionally with Convolutional neural networks (CNNs) for feature extraction
