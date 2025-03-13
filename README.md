# MRI Reconstruction Methods
Welcome to the MRI Reconstruction Methods repository! This project is the culmination of my research efforts in MRI image reconstruction, developed through my Ph.D. at UCLA and postdoctoral work at the Cleveland Clinic Foundation. My research has focused on improving MRI acquisition and reconstruction techniques to accelerate imaging and enhance clinical diagnostics.

During my work, I encountered key limitations in MRI acquisition—particularly signal corruption and aliasing artifacts in accelerated scans. These challenges motivated me to develop novel reconstruction techniques that improve image fidelity while reducing scan times. This repository brings together some of the methods I have worked on, with the goal of making them accessible to researchers and practitioners in the field.

Table of Contents
Introduction
Features
Installation
Usage
Methods Included
Data
License
Contact
# Introduction
MRI is a powerful but inherently slow imaging modality. Reducing scan time without compromising image quality is a long-standing challenge, especially in applications like cardiac imaging where motion artifacts and signal loss are common. My research has explored ways to accelerate imaging by reconstructing high-fidelity images from undersampled data using both physics-driven and machine learning-based approaches.

This repository contains implementations of various MRI reconstruction techniques, including parallel imaging, region-of-interest-based methods, and deep learning approaches for artifact removal. My goal is to provide an open resource for the community to build upon, experiment with, and apply in their own research.

# Features
Diverse Reconstruction Techniques: Implementations of multiple MRI reconstruction methods.
Modular Codebase: Organized structure for easy integration and extension.
Sample Data: Example datasets included for testing and validation.

# Installation
To use the methods in this repository:

Clone the repository:

bash:
git clone https://github.com/faa5115/reconstruction.git  
Navigate to the project directory:

bash:
cd reconstruction  
Install necessary dependencies:

Ensure you have MATLAB installed. Additional toolboxes may be required for specific methods.



# Usage
Each reconstruction method has its own directory with scripts and instructions. Generally, to apply a reconstruction method:

Navigate to the method's directory:

bash: 
cd method_name  
Run the main script:

matlab: 
main_script.m  
Replace method_name and main_script.m with the actual method’s directory and script name.


Each directory listed below contains :

Code: MATLAB scripts implementing the reconstruction algorithm.
Documentation: Detailed descriptions and usage instructions.
Examples: Sample data and scripts demonstrating the method’s application. (For most methods)

# Methods Included
These are all my implementation of these methods

## Coil Combination
Topics I go over: 
* Whitening
* Square Root Sum of Squares
* Adaptive Coil Combine (Walsh's Method)
  * I will later go over Mark Bydder's generalization of Walsh's method for time series data.

## Parallel Imaging
Topics in parallel imaging that I will go over: 
* Image domain unaliasing.
  * Sensitivity Encoding (SENSE)  
* k-Space based unaliasing.
  * SMASH:  I will show how phased-array data can be used to estimate spatial harmonics used in Fourier encoding
  * GRAPPA:  I will then show how SMASH concepts can be generalized to estimate missing k-space entries for each coil.
  * SPIRiT:  show how the linear dependence of local k-space neighbors across all channels can be enforced with data consistency to accurately estimate a complete k-space.
  * "Simplified E-SPIRiT":  I will show how by using the concepts of spatial harmonics, one can estimate coil sensitivity maps.
  * Low-Rank (or "subspace") Based Parallel Imaging methods:  show how the linear dependence of multi-channel k-space neighbors can be generalized to treat them as a low-rank system.
 ## Spatial Beamforming
 For now I will discuss ROVir, which was a novel adaptation of Walsh's adaptive coil combine method to combine the acquired channel data in different linear combination to generate a new set of "virtual" channels that optimize the signal power within the specific region of interest (ROI) over the signal power outside of that region of interest.  
 I will later upload my new approach once it is published or filed for patent; whichever comes first. 
 ## SMS Imaging.
 Topics in SMS Imaging: 
 * Controlled aliasing in parallel imaging (CAIPI).
 * SENSE based reconstruction
 * GRAPPA based CAIPI for 3D imaging.
 * Split-slice GRAPPA.

## Non-Cartesian Reconstruction 
Topics I will go over: 
* The "problem statement" in trying to get an image from non-Cartesian sampled data.
* Gridding (often called the adjoint)
* Least-squares gridding. 

# DATA
Sample datasets for testing and validation are included in the data directory:

Brain Imaging Data: brain_8ch.mat, brain_alias_8ch.mat
Hip Imaging Data: hipSlice.mat, noiseForHipSlice.mat

# Concepts and Demonstrations
Here I go over the concepts listed in the "Methods Included" section. 

## Coil Combination
This section covers three key methods: **whitening**, **square root sum of squares (Sq. SOS)**, and **adaptive coil combination (Walsh's method)**. The code uses the dataset **"hipSliceAndNoise.mat"**, which contains:
- **"raw"**: k-space data of a single hip slice (`Nx x Ny x Nz x Nc x Ns`).
- **"noise"**: noise data (`Nt x Nc`) acquired at the same readout bandwidth as the raw scan.

### Understanding the Data Structure
- **Nx**: Number of readout points.
- **Ny**: Number of phase encoding lines.
- **Nz**: Number of partitions (1 for 2D scans).
- **Nc**: Number of coil channels.
- **Ns**: Number of excited slices (1 here).
- **Nt**: Number of time points in the noise scan.

---

## Data Whitening
**Code Demonstration**: [`coilCombine/main_demonstrateWhitening.m`](coilCombine/main_demonstrateWhitening.m)

Mutual inductance causes noise correlation across channels, leading to variations in noise power across the combined image. Whitening transforms the data to:
- **Decorrelate** the channels.
- **Normalize** noise variance across channels.

The cross-correlation matrix for the noise is:
$$ R_n = N^H N $$
where \( N \) is the noise matrix (`Nt x Nc`), and \( ^H \) indicates the Hermitian conjugate.

#### Noise Correlation Before Whitening
![](/figures/ChannelCrossCorrelation.jpg)

The whitening transform \( W \) is chosen such that the whitened noise satisfies:
$$ N_w^H N_w = I $$
where:
$$ N_w = N W $$
Solving for \( W \), we get:
$$ W = \Lambda_N^{-1/2}V_N $$
where \( V_N \) and \( \Lambda_N \) are the eigenvector and eigenvalue matrices from eigen-decomposition of \( R_n \).

The function `func_whitenMatrix(noise)` computes \( W \), returning:
- `W`: Whitening matrix.
- `V`, `D`: Eigenvectors and eigenvalues.

#### Noise Correlation After Whitening
![](/figures/WhitenedChannelsCrossCorrelation.jpg)

Whitening is applied to k-space data as:
$$ d_w(\mathbf{k}) = d(\mathbf{k})^T W $$

---

## Square Root Sum of Squares (Sq. SOS)
**Code Demonstration**: [`coilCombine/main_demonstrateCoilCombine.m`](coilCombine/main_demonstrateCoilCombine.m)

Sq. SOS combines multi-channel images by computing the vector magnitude at each voxel:
$$
Im_{sos}(\mathbf{r})  = \sqrt{ Im^T(\mathbf{r}) Im(\mathbf{r}) }
$$
where \( Im(\mathbf{r}) \) is the Nc-length vector of channel intensities at voxel \( \mathbf{r} \).

**Function**: `func_sqSOS(multi_channel_images, noise)`
- **Inputs**:
  - `multi_channel_images`: Size `Nx x Ny x Nz x Nc`
  - `noise`: Size `Nt x Nc` (leave as `[]` if already whitened)
- **Output**: Sq. SOS image.

#### Results
**Individual channel images**:
![](/figures/HipChannelImages.jpg)

**Channel phases**:
![](/figures/HipChannelPhases.jpg)

**Final Sq. SOS reconstruction**:
![](/figures/HipSqSOSRecon.jpg)

---

## Adaptive Coil Combination (Walsh's Method)
**Code Demonstration**: [`coilCombine/main_demonstrateCoilCombine.m`](coilCombine/main_demonstrateCoilCombine.m)

Walsh’s method estimates **spatially varying coil sensitivity profiles** to perform optimal coil combination. The sensitivity map \( c(\mathbf{r}) \) is estimated adaptively from a local voxel patch.

### Theory
Given:
- **Noise covariance matrix**: \( R_n \)
- **Signal covariance matrix**: \( R_s(\mathbf{r}) \)

The optimal coil combination weights \( m \) are found from:
$$ m = \text{eigenvector corresponding to the largest eigenvalue of } R_n^{-1} R_s(\mathbf{r}) $$

This **double-diagonalizes** noise and signal covariance matrices, ensuring:
- Maximum **SNR** improvement.
- Phase preservation by normalizing weights using \( e^{-i\theta_{max}} \).

### Function: `func_WalshMethod(imRaw, noise, patchSize)`
- **Inputs**:
  - `imRaw`: Multi-channel image (`Nx x Ny x Nz x Nc x Nm`).
  - `noise`: Noise scan (`Nt x Nc`).
  - `patchSize`: Patch size (`Npatchx x Npatchy x Npatchz x Nm`). Leave as `[]` for auto-selection (~250 voxels).
- **Output**: Coil-combined image (`Nx x Ny x Nz x 1 x Nm`).

#### Results
**Magnitude and phase of Walsh-combined reconstruction**:
![](/figures/WalshCombine_signal_and_phase.jpg)

This method is especially useful for **multi-echo MRI** (e.g., R2* mapping), as shown in work by **Mark Bydder**.

---

## Summary
- **Whitening** removes channel noise correlation.
- **Sq. SOS** provides a simple but suboptimal combination method.
- **Walsh's method** is the most advanced, yielding SNR-optimal reconstructions.

Further demonstrations (e.g., multi-echo data) will be added in future updates.


## Parallel Imaging  
The following subsections describe two common parallel imaging techniques: SENSE and GRAPPA.  

### SENSE (Sensitivity Encoding)  
SENSE is an image-domain method for parallel imaging that leverages the sensitivity profiles of individual coil elements to reconstruct missing k-space data. The undersampled k-space data from each coil is transformed to the image domain, where the aliasing artifacts due to undersampling appear as structured overlaps. These artifacts are resolved by solving a system of linear equations that incorporate the coil sensitivity maps.  

Mathematically, let $$\mathbf{m}$$ be the fully sampled image, and let $$\mathbf{S}_c$$ represent the sensitivity profile of the $$c$$th coil. The measured signal for coil $$c$$, $$\mathbf{y}_c$$, is modeled as:  

$$
\mathbf{y}_c = \mathbf{S}_c \mathbf{m}
$$  

With multiple coils, we construct a system of equations:  

$$
\mathbf{y} = \mathbf{S} \mathbf{m}
$$  

where $$\mathbf{y}$$ is the vector of undersampled coil images, and $$\mathbf{S}$$ is the concatenated sensitivity maps. The desired image $$\mathbf{m}$$ is then obtained by solving this system using least-squares or regularized inversion techniques.  

I provide an implementation of SENSE in `SENSE_fa1D.m`. This function takes as input the undersampled multi-channel image data and the coil sensitivity maps, then solves for the fully sampled reconstruction.  

### Spatial Harmonics in k-Space  
Before discussing GRAPPA, it is useful to examine the concept of **spatial harmonics** in k-space. Because phased-array coils have distinct spatial sensitivity patterns, the k-space data from each coil contains modulated versions of the underlying object. This modulation creates additional harmonics in k-space that allow for reconstruction of missing k-space lines.  

The image below illustrates these spatial harmonics, showing how coil sensitivity variations introduce structured modulations in k-space:  

![Spatial Harmonics in k-Space](/figures/SpatialHarmonics.jpg)  

Understanding these harmonics is crucial for GRAPPA, as they enable the estimation of missing k-space lines from acquired data.  

### GRAPPA (GeneRalized Autocalibrating Partial Parallel Acquisition)  
GRAPPA is a k-space-based parallel imaging method that estimates missing k-space lines by using the fully sampled region of k-space as an auto-calibration signal (ACS). Unlike SENSE, GRAPPA does not require explicit sensitivity maps.  

In GRAPPA, each missing k-space point is reconstructed as a weighted sum of nearby acquired points across all coils. The reconstruction weights are determined from the ACS region by solving a least-squares system. If $$\mathbf{k}_\text{miss}$$ represents a missing k-space point, it is estimated as:  

$$
\mathbf{k}_\text{miss} = \sum_{c=1}^{N_c} \sum_{j} w_{c,j} \mathbf{k}_{c,j}
$$  

where $$w_{c,j}$$ are the learned weights, $$\mathbf{k}_{c,j}$$ are sampled k-space points from coil $$c$$, and $$N_c$$ is the number of coils. The weights are computed by minimizing the error between the acquired and estimated ACS data.  

My implementation of GRAPPA is available in `func_GRAPPA.m`, which reconstructs the missing k-space data using the ACS region and outputs the combined image.  

---

### Future Topics  
The implementations for SPIRiT, Simplified E-SPIRiT, low-rank parallel imaging, beamforming, simultaneous multi-slice (SMS) imaging, and non-Cartesian reconstruction have also been uploaded. I will provide detailed explanations for these topics later, but in the meantime, anyone interested is welcome to explore and use the code.


# License
I appreciate a shoutout if my code is used for generating data or figures for conferences and publications.  Thanks!

# Contact
Contact
If you have questions, suggestions, or would like to collaborate, feel free to reach out:

Name: Fadil Ali
Email: faa5115@g.ucla.edu 
Institution: The Cleveland Clinic Foundation



