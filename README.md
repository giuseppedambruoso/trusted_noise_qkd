# Trusted Noise QKD

A Python-based simulation framework for analyzing Quantum Key Distribution (QKD) protocols involving trusted noise. This repository contains the tools needed to configure, generate parameters, run large-scale simulations on a compute cluster, and visualize the results.

## 📁 Repository Structure

* `main.py`: The core simulation script that models the trusted noise QKD protocol.
* `config.py`: Global configuration and simulation settings.
* `generate_params.py`: Script used to define and generate the parameter space for sweeps.
* `params.txt`: An auto-generated text file containing the list of parameters for each simulation job.
* `job_submission.sub`: The submit file for running batch jobs on a high-throughput computing cluster (e.g., HTCondor).
* `wrapper.sh`: The shell script wrapper executed by the cluster nodes to set up the environment and run `main.py`.
* `make_plots.py`: Utility script for parsing simulation outputs and generating visualizations/graphs.

## 🚀 Getting Started

### Prerequisites

* **Python 3.x**
* Standard scientific computing libraries (e.g., `numpy`, `scipy`, `matplotlib`, `pandas`)
* Access to an HTCondor-compatible computing cluster (optional, but required for distributed batch processing)

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/giuseppedambruoso/trusted_noise_qkd.git](https://github.com/giuseppedambruoso/trusted_noise_qkd.git)
   cd trusted_noise_qkd