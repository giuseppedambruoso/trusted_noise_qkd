# Trusted Noise QKD

A Python-based simulation framework for analyzing Quantum Key Distribution (QKD) protocols involving trusted noise.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
* **Python 3.12+**
* **Poetry** (for dependency management)
* **HTCondor** (only required if running batch jobs on a cluster)

## 🛠️ Installation

This project uses Poetry for dependency management. To set up the project locally:

1. Clone the repository and navigate to the project root.
2. Install the dependencies:
   ```bash
   poetry install
3. (Optional) If you plan to contribute to the code, install the pre-commit hooks:
   ```bash
   poetry run pre-commit install
   
## 🚀 Usage
The main entry point for the simulation is main.py. The script requires three positional arguments:

N (integer): The finite-size parameter (or 1 if running asymptotic phase1).

p (float): The noise parameter.

alpha (float): The alpha parameter.

Outputs are saved as CSV files automatically in the results/ directory.

Running a Single Job locally
To run a single simulation on your local machine, use Poetry to execute the script within the virtual environment:
   ```bash
   poetry run python main.py <N> <p> <alpha>

Example:

Bash
poetry run python main.py 1000 0.1 0.95
Running Batch Jobs (HTCondor)
To run multiple parameter combinations on an HTCondor cluster, you can use the provided job_submission.sub and wrapper.sh scripts.

1. Prepare your parameters:
Ensure your parameters are listed in src/trusted_noise_qkd/config/params.txt. The submission file expects them in the format N, p, alpha per line.

2. Setup directories:
HTCondor requires the log directories to exist before submission. Create them if they don't:

```Bash
mkdir -p logs results
3. Check cluster paths:
⚠️ Important: Open wrapper.sh and ensure the path to the virtual environment matches your cluster's specific setup. Currently, it points to a specific Lustre filesystem path (/lustrehome/giosca/SCALAG/trusted_noise2/qkd_venv/bin/activate). Update this to your local Poetry environment or cluster environment if needed.

4. Make the wrapper executable:

```Bash
chmod +x wrapper.sh
5. Submit the jobs:

Bash
condor_submit job_submission.sub
You can monitor your jobs using condor_q and check the logs/ directory for standard output and error files.

📁 Project Structure
src/trusted_noise_qkd/ - Main package source code.

config/ - Configuration logic and parameters.

cvx_optimization/ - Convex optimization solvers.

frank_wolfe/ - Frank-Wolfe algorithm implementations.

key_rate/ - Key rate calculation functions (asymptotic and finite-size).

objective_and_gradients/ - Objective functions for optimizations.

utils/ - Mathematical and helper utilities.

main.py - Core execution script.

job_submission.sub / wrapper.sh - HTCondor batch submission scripts.

pyproject.toml / poetry.lock - Poetry configuration and lockfiles.

📄 License
This project is licensed under the MIT License.
