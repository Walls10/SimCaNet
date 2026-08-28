# SimCaNet: High-Dimensional Data Simulator based on Directed Acyclic Graphs and Structural Causal Models

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SimCaNet** provides a high-dimensional data generation framework that explicitly models structural causal dependencies to benchmark large-scale process monitoring, fault detection and diagnosis 

> ### Cite
> ```bibtex
> @article{paredes2026,
>   title={ High-Dimensional Decentralized Causal Process Monitoring: Impact of Network Topology on Fault Detection Sensitivity },
>   author={ Rodrigo Paredes , Wei-Ting Yang , Marco S. Reis},
>   journal={Submitted for },
>   year={2026}
> }
> ```

---

## Simulation Workflow

SimCaNet consists of three main stages.

### 1. Network Generation
A modified Barabási–Albert model is used to generate scale-free directed acyclic graphs. Unlike the classical Barabási–Albert model, where each node has a fixed number of outgoing connections, SimCaNet assigns an individual out-degree to each node sampled from a discrete Poisson distribution

The resulting networks preserve the heterogeneous connectivity patterns commonly observed in large-scale systems.

### 2. Normal Operating Conditions (NOC) Data Generation
After generating a valid DAG, variables are simulated according to a topological ordering. Root nodes are sampled independently:
  $$X_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$$

Non-root nodes are generated using linear structural causal models:
  $$X_i = \alpha_i + \sum_{X_j \in PA_i} \beta_{ji} X_j + e_i$$

where:

- $\alpha_i$ is the intercept;
- $\beta_{ji}$ represents the causal effect of parent variable $X_j$ on $X_i$;
- coefficients are sampled from a Student's t-distribution to introduce heterogeneous causal strengths;
- $e_i \sim \mathcal{N}(0,\sigma_{e_i}^2)$ is the model noise.

Measurement noise can be added to all variables according to a user-defined signal-to-noise ratio (SNR).

### 3. Faulty Data Generation
SimCaNet supports the injection of structurally consistent anomalies directly into the SCM mathematical equations at specified magnitudes ($k$):
1. **Process Perturbations:** A step change is applied to a root variable ($\mu_i = \mu_i + k \cdot \sigma_i$). The disturbance propagates naturally through the causal network and affects downstream variables
2. **Sensor Biases:** The standard deviation of the measurement noise of a variable is scaled by the fault magnitude: ($\sigma_{e_i} = (k+1) \cdot \sigma_{e_i}$). The underlying process remains unchanged
3. **Correlation Changes:** A structural change is introduced by modifying a causal coefficient according to the fault magnitude: $$\beta_{ji}^{fault} = \beta_{ji}\left(1-\frac{k}{100}\right)$$, where $k$ denotes the fault magnitude expressed as a percentage. This simulates changes in the strength of the causal relationship between two connected variables while preserving the underlying network topology.

---

## Repository Structure
* `SimCaNet.py`: Core implementation of network generation, SCM initialization, data simulation, and fault injection
* `SimCaNet_example_usage.ipynb`: Example notebook illustrating dataset generation and visualization
* `requirements.txt`: Required python dependencies

---

## Prerequisites 
Python 3.9 or newer. Install the dependencies using `pip install -r requirements.txt`
