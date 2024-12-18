# **Resultant Force Measurement on Barchan Dunes Using Convolutional Neural Networks**

This repository contains the code for a convolutional neural network (CNN) designed to estimate the resultant forces acting on a barchan dune. The model has been trained using numerical data to predict force distributions based on the morphological features of dunes.


### **Overview**

Dunes are widespread across various terrains on Earth, Mars, and beyond, yet understanding their dynamics at the grain scale remains a challenging task. The approach presented in our paper addresses this challenge by employing a convolutional neural network (CNN) trained on numerical data to predict the force distribution on experimental dune grains based on their morphological features. This breakthrough not only advances our understanding of granular dynamics but also has the potential to provide a powerful methodology for analyzing and monitoring landscape dynamics across diverse environments, including agricultural, geological, and planetary contexts.

The method we present holds great potential for revolutionizing how we measure and understand the forces acting on granular materials, with wide-reaching applications across various fields, from planetary exploration to resource management and environmental conservation. Moreover, the proposed procedure opens new possibilities for measuring the resultant force on relatively small elements that are imaged over time, such as rocks, boulders, rovers, and human-built constructions photographed by satellites on terrestrial and Martian landscapes.


### **Key Features:**

- **Force Prediction**: Estimate resultant forces acting on barchan dunes using CNN-based models.
- **Training Dataset**: Built on image data from subaqueous experiments and high-resolution simulations.
- **Generalization**: The model demonstrates the ability to generalize to unseen dune morphologies, making it applicable to a wide range of landscape configurations.
- **Potential Applications**: While the method has not yet been tested in practical applications, its potential extends to various fields, such as agriculture, housing, forest conservation, and planetary exploration, through remote sensing and landscape monitoring.


### **Installation**

> :warning: **macOS and Windows platforms**: CUDA-enabled builds only work on Linux.
> For macOS and Windows, the project must be configured to use specific accelerators (See UV-Pytorch integration in the [UV guides](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index))

**Using UV:**
We recommend using UV as Python package and project manager. It replaces many tools making the workflow easier.

See [here](https://docs.astral.sh/uv/getting-started/installation/) how to install UV in your machine.

To install and run this code locally, follow these instructions:
1. Clone the repository:
```bash
git clone https://github.com/las-unicamp/measuring_forces_on_sand_dunes.git
```
1. Install required dependencies and run script:
```bash
uv run my_script.py
```

<!-- 2. Install required dependencies:
```bash
uv sync
```
1. Activate virtual environment
```bash
source .venv/bin/activate
``` -->

### **Usage**

Once installed, the model can be trained on your own data or applied to the existing datasets.
Instructions for training, testing, and running the model is provided in this section (TO BE ADDED)

### **License**

This project is licensed under the MIT License.


### **Contributors**

Renato F. Miotto
Carlos A. Alvarez
Danilo S. Borges
William R. Wolf
Erick M. Franklin
