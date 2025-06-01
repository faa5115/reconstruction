<!--
Welcome to the Image Reconstruction Methods repository. This project is currently a work in progress. It will eventually hold my entire body of work in MRI image reconstrion, developed during my Ph.D. at UCLA and my postdoctoral fellowship at the Cleveland Clinic Foundation. While this work focuses on magnetic resonance imaging, the techniques and principles applied here: digital signal processing, sensor array combination, aliasing mitigation, and inverse problem formulation—are foundational to Synthetic Aperture Radar (SAR) as well.

My goal in sharing this repository is to demonstrate the depth of my experience in solving complex imaging problems that are closely analogous to those in SAR. Though my work has been in the medical imaging domain, the core challenges, such as sparse sampling, spatial aliasing, beamforming, non-Cartesian data acquisition, and image reconstruction from frequency-domain measurements are fundamentally similar.
-->

# Signal Reconstruction Implementations and their Demos
This repository presents a growing collection of signal modeling and image reconstruction methods I’ve implemented from scratch, originally motivated by problems in medical imaging (MRI). Many of the underlying principles, such as inverse problem formulation, array signal modeling, and low-rank matrix recovery, are also foundational in synthetic aperture radar (SAR). While the repository is still under development, it is intended as a practical reference for applying these shared concepts across domains.

## Currently demonstrated methods include:

Sensitivity encoding (SENSE) for spatial matched filter reconstruction of coil-combined images and can be extended to unalias collapsed signals with different channel sensitivities

Stochastic spatial matched filter for reconstruction with probabilistic weighting

GRAPPA, a k-space-based convolutional method that complements SENSE for unaliasing

Non-uniform FFT (NUFFT) and gridding-based reconstructions for non-Cartesian sampling trajectories

Projection reconstruction using radial sampling

Structured low-rank matrix recovery from undersampled data





## Upcoming additions will include:

Spatial beamforming demonstrations, including ROVir (an MRI adaptation of SAR transmit beamforming)

A novel CLEAN-style deconvolution technique (results to be shared pending publication)

E-SPIRiT-based parallel reconstruction (used to generate localized sensitivity maps for each receive channel of a phased array)

Spiral readout reconstruction via NUFFT

Calibrationless structured low-rank matrix recovery (SAKE)

Calibrated structured low-rank recovery and other hybrid methods

Random matrix theory based denoising

My aim is to highlight the shared language and mathematical structures that unite MRI and SAR signal modeling, with an emphasis on physical interpretability and implementation transparency.


---
## Relevance to Radar Signal Processing

<!--
Key skills demonstrated in this repository that translate directly to SAR include:

Multi-sensor data fusion: Combining spatially varying signals from array elements (MRI coils) using methods such as SENSE and adaptive beamforming (David Walsh).
Aliasing mitigation from sub-Nyquist sampling: Solving inverse problems to reconstruct unaliased images, akin to resolving spatial aliasing in radar aperture synthesis.
Image reconstruction from frequency-space data: Processing raw k-space (analogous to SAR phase history data) with gridding, NUFFT, and backprojection algorithms.
Low-rank and structured signal recovery: Leveraging low-dimensional structure in undersampled measurements for robust image recovery, as in compressed sensing radar.
This repository brings together multiple reconstruction strategies, each accompanied by code and visual examples, to highlight my hands-on DSP proficiency and adaptability across imaging domains.

-->

Although this repository was developed in the context of MRI, many of the reconstruction techniques here align closely with the signal modeling challenges faced in SAR:

Phased-array processing and unaliasing techniques correspond to SAR beamforming and resolution enhancement.

Low-rank matrix models are applicable to SAR denoising, missing data interpolation, and compressive sensing.

Matched filtering and gridding methods resemble SAR pulse compression and image formation pipelines.

Inverse problem approaches reflect similar issues in SAR: ill-posedness, limited data coverage, and trade-offs between resolution and artifacts.



---

# Table of Contents
**Introduction**\
**Features**\
**Installation**\
**Usage**\
**Methods Included**\
**Data**\
**License**\
**Contact**\

---
# Introduction

<!--
MRI is a highly flexible imaging modality, but its performance is limited by slow acquisition times and sensitivity to artifacts caused by undersampling and motion. These challenges mirror many of the signal reconstruction issues found in radar imaging.

My work addresses these problems by developing robust reconstruction algorithms based on signal processing theory and array systems engineering. The techniques presented here reflect both a deep theoretical understanding and hands-on implementation experience in advanced reconstruction pipelines.

This repository is not only a showcase of MRI reconstruction techniques—it is a demonstration of transferable digital signal processing skills applicable to radar and remote sensing applications.

-->


This repository is a work in progress aimed at unifying concepts from signal processing, inverse problems, and physics-based modeling as they apply to image reconstruction tasks. While many demonstrations originate from MRI, the mathematical frameworks and practical implementations are deeply relevant to radar imaging—particularly SAR.

Each example emphasizes:

Conceptual clarity and documentation

Transparent, from-scratch implementations

Connections to physical modeling and array processing

Relevance to real-world reconstruction workflows

I welcome feedback and plan to continue expanding the repository with more SAR-relevant signal models and reconstruction tools.

---

# Features

**DSP-Rich Algorithms:** Implementations of SENSE, GRAPPA, beamforming (ROVir), low-rank matrix recovery, and NUFFT—all of which apply core DSP techniques.
**Array Data Processing:** Multi-channel coil combination strategies (e.g., adaptive combining and sensitivity encoding).
**Modular Codebase:** Easily extensible Python/Matlab code with demos and results.
**Visual Artifacts + Solutions:** Side-by-side illustrations of aliasing, beam pattern formation, and artifact suppression.
**Sample Data:** Toy datasets provided for easy experimentation and reproducibility.

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
These are all my implementation of these methods:

## Coil Combination
Topics I go over: 
* Whitening
* Square Root Sum of Squares
* Adaptive Coil Combine (Walsh's Method)
  * I will later go over Mark Bydder's generalization of Walsh's method for time series data.
* Sensitivity Encoding (SENSE)

## Solving the Aliasing Problem using phased-arrays
<!--
Topics in parallel imaging that I will go over: 
* Image domain unaliasing.
  * Sensitivity Encoding (SENSE)  
* k-Space based unaliasing.
  * SMASH:  I will show how phased-array data can be used to estimate spatial harmonics used in Fourier encoding
  * GRAPPA:  I will then show how SMASH concepts can be generalized to estimate missing k-space entries for each coil.
  * SPIRiT:  show how the linear dependence of local k-space neighbors across all channels can be enforced with data consistency to accurately estimate a complete k-space.
  * "Simplified E-SPIRiT":  I will show how by using the concepts of spatial harmonics, one can estimate coil sensitivity maps.
  * Low-Rank (or "subspace") Based Parallel Imaging methods:  show how the linear dependence of multi-channel k-space neighbors can be generalized to treat them as a low-rank system.
  -->
  
 ### Parallel Imaging:  Use localized channel sensitivies to unfold aliased signal
 <!--
 For now I will discuss ROVir, which was a novel adaptation of Walsh's adaptive coil combine method to combine the acquired channel data in different linear combination to generate a new set of "virtual" channels that optimize the signal power within the specific region of interest (ROI) over the signal power outside of that region of interest.  
 I will later upload my new approach once it is published or filed for patent; whichever comes first. 
  -->

* Subsampling and Aliasing
* SENSE Unaliasing
* GRAPPA Reconstruction
* SENSE vs GRAPPA: Side Lobe Artifacts
* Structured Low-Rank Matrix Completion

### Spatial Beamforming:  Use phased array data filter signal from unwanted regions.
* Region-of-Interest Beamforming (ROVir)
* I will briefly go over my new approach Coil Localized Elimination of Aliased NMR (CLEAN). It is not yet published and currently filed a provisional patent application), so I will not go over any details on the implementation.
<!--
 ## SMS Imaging.
 Topics in SMS Imaging: 
 * Controlled aliasing in parallel imaging (CAIPI).
 * SENSE based reconstruction
 * GRAPPA based CAIPI for 3D imaging.
 * Split-slice GRAPPA.
   -->

## Non-Cartesian Reconstruction 
<!--
Topics I will go over: 
* The "problem statement" in trying to get an image from non-Cartesian sampled data.
* Gridding (often called the adjoint)
* Least-squares gridding. 
-->
* Radial Sampling & Filtered Backprojection
* Gridding Reconstruction
* NUFFT-Based Reconstruction

## Image slice interference
* Simultaneous acquisition of mutliple slices.
   * Controlled Aliasing in Parallel Imaging (CAIPI).
   * Using standard parallel imaging based approach.
   * A demonstration of our CLEAN approach to spatially beamform for the specific slice.
* Signal leakage from the desired slice
  * Demonstrate the on-slice projection of the out of slice signal.
  * Show how 3D Fourier ("k-space") sampling can resolve this.
  * Use channel sensitivity profiles to unfold the image.
  * Use CLEAN to spatially beamform for the leaked signal. 
# DATA
Sample datasets for testing and validation are included in the data directory:

Brain Imaging Data: brain_8ch.mat, brain_alias_8ch.mat
Hip Imaging Data: hipSlice.mat
  The hip image data has a separte noise scan:  noiseForHipSlice.mat

The data has the following size dimensions: Nx x Ny x Nz x Nc x Ns x Nm
### Understanding the Data Structure
#### raw data
- **Nx**: Number of readout points.
- **Ny**: Number of phase encoding lines.
- **Nz**: Number of partitions (1 for 2D scans).
- **Nc**: Number of coil channels.
- **Ns**: Number of excited slices (1 here).

#### noise data
- **Ntp**: Number of time-points sampled
- **Nc**:  Number of channels, same as **raw data**. 

  


This discussion is in the context of MRI.  In MRI the the raw data is the Fourier transform of the image, which is referred to as "k-space." 
That means if you have a fully sampled (satisfying Nyquist-Shannon sampling criteria which we will discuss below in the "Aliasing and Parallel Imaging" subsection) raw data, you must do a 3D spatial inverse fast Fourier Transform (IFFT) for each channel, slice. 

# Concepts and Demonstrations
Here I go over the concepts listed in the "Methods Included" section. 

## Coil Combination
This section covers three key methods: **whitening**, **square root sum of squares (Sq. SOS)**, and **adaptive coil combination (Walsh's method)**. The code uses the dataset **"hipSliceAndNoise.mat"**, which contains:
- **"raw"**: k-space data of a single hip slice (`Nx x Ny x Nz x Nc x Ns`).
- **"noise"**: noise data (`Nt x Nc`) acquired at the same readout bandwidth as the raw scan.

<!---
### Understanding the Data Structure
- **Nx**: Number of readout points.
- **Ny**: Number of phase encoding lines.
- **Nz**: Number of partitions (1 for 2D scans).
- **Nc**: Number of coil channels.
- **Ns**: Number of excited slices (1 here).
- **Nt**: Number of time points in the noise scan.
---> 

---

### Data Whitening
**Code Demonstration**: [`coilCombine/main_demonstrateWhitening.m`](coilCombine/main_demonstrateWhitening.m)
To follow, load the raw k-space and noise data: 
```
load('hipSlice.mat')
load('noiseForHipSlice.mat')
```

This loads the raw data (``` raw ```) and noise (``` noise ```). 

Mutual inductance causes noise correlation across channels, leading to variations in noise power across the combined image. Whitening transforms the data to:
- **Decorrelate** the channels.
- **Normalize** noise variance across channels.

The cross-correlation matrix for the noise is:
$$R_n = N^H N$$
where \( N \) is the noise matrix (`Nt x Nc`), and \( ^H \) indicates the Hermitian conjugate.

#### Noise Correlation Before Whitening
The following block of code and figure show how to generate the noise correlation matrix and displays it. 
```
Rn = noise' * noise;
figure,
imagesc(abs(Rn))
```
![](/figures/ChannelCrossCorrelation.jpg)

Ideal, *whitened* , noise ($$N_w$$) would have a its correlation matrix be the identity matrix:  $$N_w^H N_w = I$$. 

To decorellate the multi-channel data, a ***whitening transform*** ($$W$$) must be determined.   

If $$W$$ is known, $$N_w$$ can be determined by $$N_w = N W$$

To determine $$W$$:\
$$N_w^H N_w = I$$\
$$[NW]^H[NW] = I$$\
$$W^HN^HNW = I$$\
$$W^HR_nW = I$$\
$$WV_N \Lambda_N V_N^HW = I$$ where $$V_N$$ and $$\Lambda_N$$ are the eigenvector and eigenvalue matrices from eigen-decomposition of   $$R_n$$.\
$$WV_N \Lambda_N^{1/2} \Lambda_N^{1/2} V_N^HW = I$$\
$$\Lambda_N^{1/2} V_N^HW = I$$\
$$W = V_N \Lambda_N^{-1/2}$$\
\
\
\
Having the whitening transform be $$W = V_N \Lambda_N^{-1/2}$$ will leave the channel data in a different vector space, so to bring it back to the space, I post multiply by $$V_N^H$$, which is just a personal preference of mine:  $$W = V_N \Lambda_N^{-1/2}$V_N^H$$.  

The following block of code illustrates how to determine the $$W$$ from the noise data ``` noise ```: 
```
[V, D] = eig(Rn); % V (eigenvector matrix) W (diagonal matrix of eigenvalues). 
W = V * diag(diag(D).^(-0.5)) * V';
```






The function ```func_whitenMatrix(noise)``` computes \( W \), returning:
- ```W```: Whitening matrix.
- ```V```, ```D```: Eigenvectors and eigenvalues.
```
function [W, V, D] = func_whitenMatrix(noise)
% Input :  noise is number of timepoints x Nc
% Output:  W is the whitening matrix.  V and D are the eigenvector and diagonal eigenvalue matrix respectively.   
Rn = noise' * noise;
[V, D] = eig(Rn);% eigen value decomposition of noise correlation matrix
W = V * diag(diag(D).^(-0.5)) * V';
```

#### Noise Correlation After Whitening

To demonstrate the that noise is whitened, the correlation matrix should be the identity matrix.  That means all off-diagonal entries should be zeros.\
This guarantees independence between the data of each channel. 

The following block of code and figure illustrate the whitened noise correlation matrix:  
```
noisew = noise * W;
Rnw = noisew'*noisew;

figure,
imagesc(abs(Rnw))
```

![](/figures/WhitenedChannelsCrossCorrelation.jpg)

One can then use $$W$$ to whiten the raw data at each acquried k-space index $$k_n$$ (where $$k_n \in [1, N_x \cdot N_y \cdot N_z]$$ ) by $$\textbf{d}_w[k_n]=[\textbf{d}[k_n]]^TW$$ where $$\textbf{d}[k_n]$$ is a $$N_cx1$$ length vector of the acquired k-space of each channel.

The same can be done in the image domain   $$\textbf{Im}_w[r_n]=[\textbf{Im}[r_n]]^TW$$ where $$r_n$$ is an index in the image domain and $$\textbf{Im}[r_n]$$ is a $$N_cx1$$ length vector of the channel data in the image domain.  

The following snippet of code demonstrates how we take the correlate raw k-space data ```raw``` and whiten the data in the image domain: 

```
imRaw = zeros(size(raw));
imRaw_vect = zeros(Nx*Ny*Nz, Nc  );
imRaw_w = zeros(size(raw));
for chiter = 1 : Nc
    imRawiter = ifftnc(squeeze(raw(:, :, :, chiter)));
    imRaw(:, :, :, chiter ) = imRawiter;
    imRaw_vect(:, chiter) = imRawiter(:);
end

imRaw_w_vect(:, :) = imRaw_vect(:, :) * W;
for chiter = 1 : Nc
    imRaw_w(:, :, :, chiter) = ...
        reshape(imRaw_w_vect(:, chiter), [Nx, Ny, Nz]);

end

% imRaw_w (size Nx x Ny x Nz x Nc) is the whitened raw data.  

```
Now that this multi-channel data is whitened, we have to combine them. This will be discussed in the following three subsections.  

---

### Square Root Sum of Squares (Sq. SOS)
**Code Demonstration**: [`coilCombine/main_demonstrateCoilCombine.m`](coilCombine/main_demonstrateCoilCombine.m)

Even if one whitens his data, he still has multiple images, each weighted by the sensity of the channel it was received from.  Each individual channels image has a magnitude and phase, as shown below: 

**Individual channel images**:
![](/figures/HipChannelImages.jpg)

**Channel phases**:
![](/figures/HipChannelPhases.jpg)

One would like to combine the multi-channel data to form a single image that represents the signal.\
One simple way to do that is by square-root sum of squares (Sq.sos).\
Conceptually, the Sq.sos image at each voxel index $$r_n$$, $$Im_{sos}[r_n]$$, is basically a magntidue of the all the channel values:\ 
$$Im_{sos}(\mathbf{r})  = \sqrt{ Im^H(\mathbf{r}) Im(\mathbf{r}) }$$


The following snippet of code shows my square-root sum of squares function, ```func_sqSOS```.   \
Its inputs are the raw image data and noise data, which is optional.  If you do not have or wish to use noise data, simply place ```[]``` instead. \
If noise is included, the data will be whitened.  
```

function imRaw_sos = func_sqSOS(imRaw, noise)

[Nx, Ny, Nz, Nc] = size(imRaw);

imRaw_w      = zeros(Nx, Ny, Nz, Nc);
imRaw_vect   = zeros(Nx*Ny*Nz, Nc  );
imRaw_w_vect = zeros(Nx*Ny*Nz, Nc  );

for chIter = 1 : Nc
    imRawIter   = imRaw  (:, :, :, chIter);
    imRaw_vect  ( :, chIter) = imRawIter(:);
end

if ~isempty(noise)
    disp('with noise vector')
    [W, V, D] = func_whitenMatrix(noise);
    noise_w = noise * W;
    Rn_w = noise_w' * noise_w;
    
    
    imRaw_w_vect(:, :) = imRaw_vect(:, :) * W;
    for chIter = 1 : Nc
        imRaw_w(:, :, :, chIter) = reshape(imRaw_w_vect(:, chIter), [Nx, Ny, Nz]);
    end
    
else
    disp('no noise vector')
    Rn_w = eye(Nc);
    imRaw_w = imRaw;
end



imRaw_sos = sqrt(sum(abs(imRaw_w).^2, 4));

```

The following snippet is an example using ```func_sqSOS``` on the multi-channel hip data above and the final coil-combined image: 

```
imRawsos = func_sqSOS(imRaw, noise);

figure, 
imshow(flip(abs(imRawsos), 1), [])
title('Sq. SOS. Recon.')

```


**Final Sq. SOS reconstruction**:\
![](/figures/HipSqSOSRecon.jpg)


Unfortunately, this approach is not the most signal-to-noise (SNR) optimal approach.  It also loses phases information of the data, which can be crucial for many applications, such as observing moving signal, assessing for spectral components (different chemical shifts), etc.  

---
### Using Channel Sensitivity Encoding ("SENSE") to approximate the Spatial Matched Filter Reconstruction.
**Code Demonstration**: [`coilCombine/main_demonstrateCoilCombine.m`](coilCombine/main_demonstrateCoilCombine.m)

In many cases, such as with NMR or MRI, the signal is complex-valued.  This information is lost in square-root sum-of-squares.  Furthermore. it does not yield the SNR optimal signal reconstruction.  A spatial matched filter would yield the SNR optimal reconstruction, as Roemer et al discussed when writing the NMR Phased Array.  We will first discuss the spatial matched filter as an inverse problem and then proceed modify the inverse problem into the format of the Matched Filter.  

The multi-channel image data at voxel $$\mathbf{r}_n$$ can be described with the following channel-sensitivty encoding scheme: \

<!--
$$\mathbf{Im}(\mathbf{r}_n)=[\mathbf{S}(\mathbf{r}_n)] Imcc(\mathbf{r}_n)$$
-->
$$[\mathbf{Im}(\mathbf{r}_n)]=[\mathbf{S}(\mathbf{r}_n)] M(\mathbf{r}_n)$$

  $$M(\mathbf{r}_n)$$ is the original complex-valued signal at location  $$\mathbf{r}_n$$.\

  $$[\mathbf{Im}(\mathbf{r}_n)]$$ is a vector whose entries are each complex-valued  image from each of the $$N_c$$ channels at position $$\mathbf{r}_n$$.  \
  $$[\mathbf{Im}(\mathbf{r}_n)] = [Im_1(\mathbf{r}_n), Im_2(\mathbf{r}_n), ... , Im _{Nc}(\mathbf{r}_n) ]^T$$, where $$Im_j(\mathbf{r}_n)$$ is the channel image of channel $$j$$.  \
  
  
  $$[\mathbf{S}(\mathbf{r}_n)]$$ is a matrix $$N_c \times 1$$ that holds how sensitive each channel is from signal at position $$\mathbf{r}_n$$.  \
   $$[\mathbf{S}(\mathbf{r}_n)] = [S_1(\mathbf{r}_n), S_2(\mathbf{r}_n), ... , S _{Nc}(\mathbf{r}_n) ]^T$$ where $$S_j(\mathbf{r}_n)$$ is the channel sensitivty of channel $$j$$. 
  
 
This is the forward problem describing how one gets the individual channel images from a spatially varying signal $$M(\mathbf{r}_n)$$.

If one had the sensitivity values, one could compute an estimate of $$M$$, $$M_{est}$$  solve **inverse problem**  of the above formula: 
$$M_{est} = Im_{cc}(\mathbf{r}_n) = [\mathbf{S}(\mathbf{r}_n)]^{-1}[\mathbf{Im}(\mathbf{r}_n)]$$

Here $$Im_{cc}(\mathbf{r}_n)$$ is introduced because the estimated signal, $$M _{est}$$, is the coil combined image. 

To "match" the matched filter formulation, this can be expanded to the spatial matched filter, if noise data is provided:  \



$$Im_{cc}(\mathbf{r}_n) = [\mathbf{S}(\mathbf{r}_n)]^{-1}\mathbf{Im}(\mathbf{r}_n)$$  \


$$[\mathbf{U}_{sense}]=[S^HRn^{-1}S]^{-1}[S^H]Rn^{-1}$$


The difficulty remains in determining the sensitivity values.  One simple way to approximate for sensitivities is by taking a low-resolution version of each channel image, and then dividing each low-resolution image by the square-root sum of squares of the low resolution images.  Getting a low-resolution channel images can be done by applying a low pass filter to the k-space (Fourier transform) of each channel image, thereby keeping low-frequency components of k-space and penalizing the high-resolution edges.   By the convolution theorem of the Fourier transform, this is equivalent to convolving the channel images by kernel. 

A common low-pass filter for k-space used is a rect function that only selects the center k-space lines.  But this could problematic becuase it will cause Gibbs ringing in the sensitivity maps.  

Here is a snippet on how one can use my sensitivity encoding function to generate images: 

```
senseMapsOption = 1; % I have two options in generating sensitivity maps.  
                     % Option 1 simply makes low resolution images from  
                     % the center k-space lines and divides each low
                     % resolution image by their square root sum of squares.
                     % Option 2 uses a method called E-SPIRiT which I will
                     % discuss in a later section. 
R = [1, 1];  % This is "acceleration factor" along ky and kz encoding which I will discuss in the parallel imaging section.  
             % We did not accelerate the acceleration in this example, so
             % the values for both is 1. 


% func_SENSE can be found in ../parallelImaging.
[imRawSense, senseMaps] = func_SENSE(imRaw, imCalib, R, noise, senseMapsOption);

```
Here I show sensitivity maps generated from simply selecting the center region of k-space, where I took only the center 28 lines, and the resulting image reconstructions from sensitivity encoding: 

![](/figures/senseMapsMagnitude.jpg)

![](/figures/senseRecon.jpg)

I will later discuss, in parallel imaging an alternative way to generate the sensitivity maps.  


Given the difficulty in proper assessment of channel sensitivity, I will go into Adaptive coil combination.  This approach exploits local correlation statistics across all channels to generate a stochastic mathced filter.   




---

### Adaptive Coil Combination:  The Stochastic Matched Filter Coil Combination ("Walsh's Method")
**Code Demonstration**: [`coilCombine/main_demonstrateCoilCombine.m`](coilCombine/main_demonstrateCoilCombine.m)

<!-- Walsh’s method estimates **spatially varying coil sensitivity profiles** to perform optimal coil combination. The sensitivity map \( c(\mathbf{r}) \) is estimated adaptively from a local voxel patch. -->
The adaptive coil combine method (which I sometimes called "Walsh's Method") combines the channels using a ***stochastic matched filter***.  

This signal reconstruction method was based off of a polametric technique used to optimize signal power to clutter power ratio, or in other contexts Signal to noise power (SNR), in SAR imagery (see SM Verbout "Polarimetric techniques for enhancing SAR imagery" *Synthetic Aperture Radar SPIE*, Vol. 1630 1992).  

In the context of SAR imaging, the data is the measurement of a time-domain signal, which is assumed to be *stochastic*.  This measurement consists of the desired signal s(t) and the undesired noise process n(t). 

In this section I will summarize the proof used to generate the method and then go over my implementation 

#### Proof

With an $$N_c$$ element phased-array system, the signal and noise received for each channel can be described as:  \

$$\mathbf{s}(t) = [s_1(t), s_t(t), ..., s_{N_c}(t)]^T$$
$$\mathbf{n}(t) = [n_1(t), n_t(t), ..., n_{N_c}(t)]^T$$

$$N_c \times N_c$$ correlation matrices can be constructed:  \

$$R_s = E[\mathbf{s}(t) \textbf{s}^H(t)]$$
$$R_n = E[\mathbf{n}(t) \textbf{n}^H(t)]$$

$$E[]$$ is the expectation value of the term within the brackets.  Expectation value is evaluated over time.  

The stochastic matched filter is a vector $$\mathbf{m}$$ that maximizes signal power to noise power: \

$$\frac{E[signal power]}{E[noise power]} = \frac{E[|\mathbf{m}^H \mathbf{s}|^2]}{E[|\mathbf{m}^H \mathbf{n}|^2]}$$

The goal is to find $$\mathbf{m}$$ that maximizes this SNR objective function, $$\mathbf{m}_{max}$$.

This can be expanded as 
$$\frac{E[signal power]}{E[noise power]} = \frac{E[|\mathbf{m}^H \mathbf{s} \mathbf{s}^H \mathbf{m}|]}{E[|\mathbf{m}^H \mathbf{n} \mathbf{n}^H \mathbf{m}|]}$$
$$= \frac{\mathbf{m}^HR_s\mathbf{m}}{\mathbf{m}^H R_n \mathbf{m}}$$

This objective function can turn into an eigenvalue problem by exploiting the simultaneous diagonalization of $$R_s$$ and $$R_n$$:


$$R_n^{-1}R_s P= PD$$ 

where the columns of $$P$$ are the eigenvectors of $$R_n^{-1}R_s$$ and $$D$$ is a diagonal matrix consisting of the eigenvalues to the corresponding eigenvector column of $$P$$.  
For convenience, the eigenvalues are arranged in decending order. 

Because $$R_n^{-1}R_s$$ is not necessarily Hermitian, the columns of $$P$$ do not make an orthonormal basis.  Rather, $$P$$ is orthogonal with $$R_NP$$: \


$$P^HR_NP = I$$ where $$I$$ is the identity matrix.  

Because of this, one can plug this equality in $$R_n^{-1}R_s P= PD$$ :

$$R_n^{-1}R_s P= PD$$ \
$$R_s P = R_n PD$$ \
$$P^H R_s P =P ^H R_n PD = I D$$ 
$$P^H R_s P = D$$

This gives use two equalities we can use in the SNR objective function: \


$$P^H R_s P = D$$  $$R_s = (P^H)^{-1}DP^{-1}$$ and \
$$P^HR_NP = I$$  $$R_N = (P^H)^{-1}P^{-1}$$

With this in mind, we can return to our SNR objective function: 

$$\frac{E[signal power]}{E[noise power]} = \frac{\mathbf{m}^HR_s\mathbf{m}}{\mathbf{m}^H R_n \mathbf{m}}$$
$$= \frac{\mathbf{m}^H(P^H)^{-1}DP^{-1} \mathbf{m}}{\mathbf{m}^H (P^H)^{-1}P^{-1} \mathbf{m}}$$

For convenience, let's write $$q = P^{-1}\mathbf{m}$$: \
$$\frac{E[signal power]}{E[noise power]} = \frac{\mathbf{q}^HD \mathbf{q}}{\mathbf{q}^H \mathbf{q}}$$


If  $$\mathbf{q}$$ optimizes SNR, then so can any scalar multiple of $$\mathbf{q}$$ because the scalars will cancel out.  If we normalize the objective function, ($$\mathbf{q}^H\mathbf{q}=1$$), then we are finding the vector $$\mathbf{q}$$ that maximizes:


$$\frac{E[signal power]}{E[noise power]} = \mathbf{q}^HD \mathbf{q}$$

Because the eigenvalues in $$D$$ are arranged in descending order, then $$\mathbf{q}_{max} = [1, 0, 0, 0, ..., 0]^T$$.\

The vector that combines the channels with the optimal SNR output is therefore: 

$$\mathbf{m} = P \mathbf{q}_{max}$$\

which is the column of $$P$$ that corresponds to the highest eigenvalue in $$D$$.  

Therefore, the optimal matched filter is the vector $$\mathbf{m}$$ that is the eigenvector corresponding to the largest eigen value of $$R_n^{-1}R_s$$
A reconstruction with approximately uniform noise variance can be achieved by scaling $$\mathbf{m}=\mathbf{m}_{max}$$ with $$\alpha = \frac{1}{\sqrt{\mathbf{m}^HR_n\mathbf{m}}}$$.

#### Adaptation to MRI
This described determining the matched filter from a stochastic time-varying process.  In MRI, we deal with images, which can be treated as spatially varying random variables.  
Therefore, the correlation statistics for the signal must be applied to the channel data in a spatially adaptive fashion for each voxel coordinate $$\mathbf{r}_c$$.  The signal correlation of channels $$j$$ and $$l$$ can be approximated for each spatial voxel $$\mathbf{r}_c$$:  \

$$R_s(j,l) (r_c) = \sum_{\mathbf{r}_n \in patch[\mathbf{r}_c]} [\mathbf{Im}(\mathbf{r}_n) \mathbf{Im}^H(\mathbf{r}_n)]$$

where $$patch[\mathbf{r}_c]$$ defines a local patch of voxels centered around voxel $$\mathbf{r}_c$$. 

$$R_s(j,l) (r_c)$$ is evaluated locally for each $$\mathbf{r}_n$$ and used in place of $$R_s$$.

Therefore this recosntruction determines a spatailly adaptive stochastic matched filter.  The matched filter is determined for each voxel.  The following steps are carried out: 

1. Use $$Nt \times Nc$$ noise data to compute $$R_n$$:  $$R_n = N^HN$$
2. Iterate through all voxels $$\mathbf{r}_c$$:
  2a. compute:  
   
   $$R_s(j,l) (r_c) = \sum_{\mathbf{r}_n \in patch[\mathbf{r}_c]} [[\mathbf{Im}(\mathbf{r}_n)] [\mathbf{Im}(\mathbf{r}_n)]^H]$$

   2.b compute:
   $$R_n^{-1}R_s(j,l) (r_c)$$

   2.c $$\mathbf{m}_{max}(\mathbf{r}_c)$$ is the highest eigenvector of $$R_n^{-1}R_s(j,l) (r_c)$$

   2.d Scale:

    $$\mathbf{m}_{max}(r_c) = \alpha \mathbf{m} _{max}(r_c)$$

   2.c

   $$Im_{cc}(\mathbf{r}_c) = \mathbf{m}(r_c) \cdot [\mathbf{Im}(r_c)]$$

   
In my implementation of the stochastic matched filter ( titled ```func_WalshMethod```), my scaling factor, $$\alpha$$ involves a spatially varying phase term: 

$$\alpha(\mathbf{r}_c) = \frac{e^{-i \theta(r_c)} }{\sqrt{\mathbf{m}^HR_n\mathbf{m}}}$$


where $$\theta(r_c)$$ is the phase of the index in $$\mathbf{m}$$ that corresponds to the channel with the highest signal power.  


#### Function
The snippet of code below shows how one can use my implementation: 
```
patch = [5, 5, 1, size(imRaw, 4)]; 
[imRawcc] = func_WalshMethod(imRaw, noise, patch) ;

```

- **Inputs**:
  - `imRaw`: Multi-channel image (`Nx x Ny x Nz x Nc x Nm`). `Nm` is the number of echoes timepoints acquired for each spatial coordinate.  
  - `noise`: Noise scan (`Nt x Nc`). 
  - `patchSize`: Patch size (`Npatchx x Npatchy x Npatchz x Nm`). Leave as `[]` for auto-selection (~250 voxels).
- **Output**: Coil-combined image (`Nx x Ny x Nz x 1 x Nm`).

  #### Results
**Magnitude and phase of adaptive stochastic matched filterreconstruction**:
I called this the "walsh method" out of respect for David Walsh who adapted this SAR image reconstruction process to MRI.  
![](/figures/WalshCombine_signal_and_phase.jpg)

You may notice in my ```func_WalshMethod``` code that I have a time-domain measurement in the image data.  This is because Mark Bydder has a paper titled "Optimal phased-array combination for spectroscopy" where he uses the stochastic matched filter to combine NMR spectroscopy data. So I modified my implementation of the stochastic matched filter to accomodate for time-varying signal as well.  
<!--
#### Theory
Given:
- **Noise covariance matrix**: \( R_n \)
- **Signal covariance matrix**: \( R_s(\mathbf{r}) \)

The optimal coil combination weights \( m \) are found from:
$$ m = \text{eigenvector corresponding to the largest eigenvalue of } R_n^{-1} R_s(\mathbf{r}) $$

This **double-diagonalizes** noise and signal covariance matrices, ensuring:
- Maximum **SNR** improvement.
- Phase preservation by normalizing weights using \( e^{-i\theta_{max}} \).

#### Function: `func_WalshMethod(imRaw, noise, patchSize)`
- **Inputs**:
  - `imRaw`: Multi-channel image (`Nx x Ny x Nz x Nc x Nm`).
  - `noise`: Noise scan (`Nt x Nc`).
  - `patchSize`: Patch size (`Npatchx x Npatchy x Npatchz x Nm`). Leave as `[]` for auto-selection (~250 voxels).
- **Output**: Coil-combined image (`Nx x Ny x Nz x 1 x Nm`).

#### Results
**Magnitude and phase of Walsh-combined reconstruction**:
![](/figures/WalshCombine_signal_and_phase.jpg)

This method is especially useful for **multi-echo MRI** (e.g., R2* mapping), as shown in work by **Mark Bydder**.

-->

---

### Summary
- **Whitening** removes channel noise correlation.
- **Sq. SOS** provides a simple but suboptimal combination method.
- **Matched Filter** yields SNR optimal coil combination by using **channel sensitivity** encoding.
- **Difficul to estimate channel sensitivity neded**
- **Walsh's method adaptive coil combine** is the most advanced, yielding SNR-optimal reconstructions, and determines a **stochastic** matched filter **locally**.  

Further demonstrations (e.g., multi-echo data) will be added in future updates.


## Aliasing and Parallel Imaging  
As discussed in the ```faa5115/blochSimulations``` repository, the signal readout in NMR and MRI is done in the Fourier domain.  This sampling domain is called "k-space".  The MRI acquisition process tries to  reconstruct an image of the spatial signal distribution.  Let's call this signal $$\mathbf{M}(\mathbf{r})$$.  The NMR/MR system employs time-varying gradients in the magnetic field that causes time-varying spatial harmonics in the signal $$\mathbf{M}(\mathbf{r})$$, which can be described as $$e^{-i2\pi\int \mathbf{G}(t)dt\cdot \mathbf{r}} = e^{-i2\pi \mathbf{k}(t) \cdot \mathbf{r}}$$.  Here $$\mathbf{k}(t)$$ is the sampled k-space coordinate at time $$t$$.  The receiver $$j$$ hear a time-varying signal that is a vector sum of the entire signal profile undergoing this spatially and time-varying precession:\

***k-space signal model***


$$d_j(k)=\int S_j(\mathbf{r}) \mathbf{M}(\mathbf{r}) e^{-i 2 \pi \mathbf{k}(t) \cdot \mathbf{r}} dr$$

Here $$S_j(\mathbf{r})$$ is the spatial sensitivity profile of channel $$j$$ at position $$\mathbf{r}$$.  For the sake of simplicity this discussion is only focusing on k-space sampling because I want to demonstrate some reconstruction code later.  I want to make it clear that this formula is not complete when you consider other effects, such as signal decay (T2) of $$\mathbf{M}(\mathbf{r})$$ during your k-space sampling, which can cause blurring in your image domain (by Fourier convolution theorem k-space samples times a decay function equals a convolution of the image of $$\mathbf{M}(\mathbf{r})$$ with the Fourier transform of the decay function), does not consider spatially varing inhomogeneities in the magnetic field (which impacts the real spatial harmonics in the system and not having that in account causes spatial shifts and distortions in the image domain).  

<!-- FADIL ALI COME BACK HERE LATER AND DISCUSS FREQUENCY AND PHASE ENCODING
With that in mind, we can continue.  k-Space sampling is often done in spatially uniform, Cartesian, rectilinear coordinates.  
-->


It can be seen that sampling k-space sufficiently, as in sampling a wide enough range to achieve a great resolution and sampling at a high enough density to resolve all the signal, takes time.  
If you do not sample a wide enough range of high density samples in k-space, you are left with a low-resolution image, as shown below.  This is because not enough because you need to sample the k-space coordinates that impose high order harmonics on the signal.  The high order harmonics will cause a significant enough phase differences between signals located close together.  The image resolution along a specific direction, say $$\mathbf{r}$$ to be arbitrary, is the reciprocal of the total range of k-space coordinates along that axis: $$\Delta \mathbf{r} = \frac{1}{mathbf{K}_{max} - \mathbf{k} _{min}}$$.

Meanwhile, if you skip lines in k-space, the image will result in aliased signal.  This is because the distance between adjacent k-space coordinates is reciprocal to the encoded FOV $$FOV_{enc}$$, $$\Delta \mathbf{k}=\frac{1}{FOV_{enc}}$$.  If the spread of the signal $$\mathbf{M}(\mathbf{r})$$ is wider than the range of the encoded FOV, then the residual signal will alias over the signal within the encoded FOV.  This is illustrated in the figure below.  

![](/figures/brainFullLowResAliased.jpg)  


<!--
![](/figures/PhaseEncoding.jpg)  
-->

The skipping $$R$$ k-space lines reduces the encoded FOV by a factor $$R$$, which also reduces the total scan time by $$R$$ as well.  If the total signal had a *real* FOV of $$FOV_{sig}$$, a reduction factor of $$R$$ will result in at most signal from at most $$R$$ different locations being aliased at the same voxel.  Signal at location $$\mathbf{r}$$ will receive aliasing from locations $$\mathbf{r} + b \frac{FOV_{sig}}{R}$$ where $$b \in [0, R-1]$$.
Because imaging is done with a phased-array, which each element of the phased-array having its own localized sensitivity profile, once could notice that aliased signals have different channel profiles. This is illustrated below: 


![](/figures/DemoAliasedSensitivities.jpg)  

If the channel sensitivity profiles between aliased voxels is enough, one could exploit the difference in channel sensitivities to unalias the signal.  Using phased-array data to unalias images is called "parallel imaging" because we are using the the data acquired by multiple phased-array channels in parallel.  Channel sensitivity profiles can be determined from the acquisition of a low-resolution data set.  

We will first discuss how to unalias this in the image domain.  This is called SENSE (Sensitivity Encoding), and achieves the unaliasing by adapting the spatial matched filter.  We will then discuss how to unalias in the k-space domain.  In the k-space domain we will discuss a method called GRAPPA which uses the phased-array data to mimic estimate the spatial harmonics achieved in the image acquisition to complete the k-space grid.  We will then generalize k-space completion as a structured low rank matrix recovery problem.  

---
The following subsections describe two common parallel imaging techniques: SENSE and GRAPPA.  

### Image Domain Unaliasing using SENSE (Sensitivity Encoding) 
Code demonstrating this concept is found in ```./parallelImaging/main_demonstrateSENSE.m```

Let's revisit the channel sensitivity encoding forward problem for a voxel at locatoin $$\mathbf{r}_n$$: 

$$[\mathbf{Im}(\mathbf{r}_n)]=[\mathbf{S}(\mathbf{r}_n)] M(\mathbf{r}_n)$$
Where just as before:  

$$[\mathbf{Im}(\mathbf{r}_n)]$$ is a vector whose entries are each complex-valued  image from each of the $$N_c$$ channels at position $$\mathbf{r}_n$$.  \
$$[\mathbf{Im}(\mathbf{r}_n)] = [Im_1(\mathbf{r}_n), Im_2(\mathbf{r}_n), ... , Im _{Nc}(\mathbf{r}_n) ]^T$$, where $$Im_j(\mathbf{r}_n)$$ is the channel image of channel $$j$$.
  


 If $$R$$ k-space lines were skipped along the $$k_r$$ axis, then the encoded FOV along the along the $$r$$ axis in the image domain was reduced to $$\frac{FOV_r}{R}$$.  This modifies the forward channel sensitivity encoding model for a specific channel $$j$$ to: 



 **Aliased SENSE Equation**

 
 $$Im_{j,R}(\mathbf{r}_n) = [S_j(\mathbf{r}_n),  S_j(\mathbf{r}_n + \frac{N_F}{R}), ...,  S_j(\mathbf{r}_n + (R-1)\frac{N_F}{R})][\mathbf{M}(\mathbf{r})]$$\
 
 The subscript $$_R$$ is included to indicate that the resulting image is aliased. $$N_F$$ is the number of voxels across the original (not reduced) FOV and  $$[\mathbf{M}(\mathbf{r})]$$ is a $$R\times1$$ length vector with the following entries: 
 
 $$[\mathbf{M}(\mathbf{r})] = [M(\mathbf{r}), M(\mathbf{r} + \frac{FOV}{R}), ... M(\mathbf{r} + (R-1)\frac{FOV}{R})]^T$$\

The Aliased SENSE equation can be written more compactly as: 


$$[\mathbf{Im}(\mathbf{r}_n)]_R = [ [\mathbf{S}(\mathbf{r}_n + 0 \frac{N_F}{R})] [\mathbf{S}(\mathbf{r}_n + 1 \frac{N_F}{R})] ... [\mathbf{S}(\mathbf{r}_n + (R-1) \frac{N_F}{R})]][\mathbf{M}(\mathbf{r})]$$


$$[\mathbf{Im}(\mathbf{r}_n)]_R = [S]_R [\mathbf{M}(\mathbf{r})]$$

where $$[S]_R$$ is an $$N_c \times R$$ matrix where each column contains the channel sensitivities of the voxel positioned at $$\mathbf{r}_n+b\cdot \frac{N_F}{R}$$, $$b\in[0,(R-1)]$$.
Just as with using channel sensitivity maps to combine the channels, the sensitivity map can be estimated from a fully sampled low resolution acquisition.  

Then an unfolding matrix $$[U]_R$$ can be determiend as $$[U]_R = ([S]^H_R R_n^{-1}[S]_R)^-1[S]^H_R R_n^{-1}$$ to obtain a coil-combined unaliased image:  

$$[M_{est}] = [Im_{cc}(\mathbf{r}_n)] = [U]_R[\mathbf{Im}(\mathbf{r}_n)]_R$$

where the evaluation of $$[U]_R[\mathbf{Im}(\mathbf{r}_n)]_R$$ is a $$R\times 1$$ vector of the unfolded signals located at positions $$\mathbf{r} + b \frac{N_F}{R}$$ for $$b\in[0, (R-1)]$$.  
 Below show the sensitivty maps, orignal image of a fully sampled acquisition, and 2x and 4x SENSE reconstructions of a brain dataset. 

 
![](/figures/BrainSenseRecons2xand4x.jpg)  

The snippet of code below shows how to use my implementation of SENSE when trying to unalias images.  The output is a coil-combined reconstruction of an unalised image.  
```

imCalibInput = imCalib; % ImCalib is the image from the low resolution region of k-space.  In this example, 


senseMapsOption = 1; % this option takes the imCalib images and divide them by their square root sum of squares.  if you make it =2, I use E-SPIRiT to generate the sense maps.  I will discuss E-SPIRiT later. 
noise = []; % I do not have noise data for this example. 

Rinput = [R , 1];
% Rinput is a vector of length 2.  The first index is the acceleration factor along ky and the second dimension is the acceleration along kz.
% this is a 2D image example, so kz is fully sampled.  In this snippet of code R is either 2 or 4.  

tic
[imRawSense, senseMaps] = func_SENSE(imRawUs, imCalibInput, Rinput, noise, senseMapsOption);
toc
```

<!--
*************************
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
-->

---
### k-Space Domain Unaliasing

Unaliasing can be achieved in the k-space domain.  Just as with unaliasing in the image domain, a low resolution fully sampled calibration region of k-space must be acquired to somehow exploit localized channel sensitivities.  For aliasing to be achieved in the k-space domain, the calibration region must somehow be used to estimate the missing k-space lines, resulting in a complete k-space grid.  This was first shown possible by Dan Sodickson's work called SMASH (Simultaneous Acquisition of Spatial Harmonics).  If all of the channel sensitivity profiles have a Gaussian-like or half-harmonic sensitivity profile, and there are enough varying sensitivity profiles along the undersampled directions, one could use the channel profiles to estimate the missing k-space harmonics.  In SMASH, the input was undersampled multi-channel k-space data and the low resolution channel images, and the output was an unaliased coil-combined image.  However this coil-combination was not necessarily optimal.  

If i have time, I will go over Sodickson's proof with some plots on why SMASH works.  but for now, I just want to get the point across and put together all of my implemented reconstruction code online (which is the reason why I am putting together this repository).  so for now I will take a different route in explaining k-space domain parallel imaging.   


We can multiply the SENSE reconstructed images by the senstivitiy maps to generate channel images that look strikingly similar to the channel images of a fully sampled acquisition:



![](/figures/SenseReconTimesSenseMaps.jpg)

Let's write out the math of multiplying a SENSE reconstructed image of Rx subsampled data for a given channel $$j$$:

***Channel by channel SENSE images***

$$[ \mathbf{Im} (\mathbf{r} _n)] _{est,N_c} = [S] _{ R,N_c}[U]_R [\mathbf{Im}(\mathbf{r}_n)]_R$$


where $$[\mathbf{Im}(\mathbf{r})]_{est,N_c}$$ is a $$R N_c \times 1$$ length vector that has the estimate of the $$R$$ unaliased voxel values at locations $$\mathbf{r}_n + b \frac{N_F}{R}$$ for $$b \in [0, R-1]$$ for each channel and $$[S] _{R,N_c}$$ is a $$R N_c \times R$$ matrix that has the sensitivity values at each of those locations.  I have the subscript "est,Nc" to mean "estimated channel images."

Essentially, the unaliased channel images $$[ \mathbf{Im} (\mathbf{r} _n)] _{est,N_c}$$ is the product of a low spatial resolution operator $$[S] _{ R,N_c}[U]_R$$ and the aliased channel images $$[\mathbf{Im}(\mathbf{r}_n)]_R$$.  


Let's take a look at the spatial Fourier transform for each term of the channel by channel SENSE images:  
- Let  $$[\mathbf{K} (\mathbf{k} _n)] _{est,N_c} = F_s [[ \mathbf{Im} (\mathbf{r} _n)] _{est,N_c}]$$ where $$F_s$$ is the Fourier transform in the spatial dimensions only.  This consists of sampled k-space lines and the estimated skipped k-space lines. 
- $$K_{ker}$$ is the spatial Fourier transform of $$[S] _{R,N_c}[U]_R$$, $$K _{ker} = F_S [[S] _{R,N_c}[U]_R]$$.  The subscript "ker" stands for "kernel" which I will discuss shortly.
- $$[\mathbf{K}(\mathbf{k}_n)]_R =F_s [[\mathbf{Im}(\mathbf{r}_n)]_R]$$.  This is only consists of the sampled k-space lines, with 0s for the skipped k-space lines.  

Becuase the channel sensitivities are low resolution, $$K_{ker}$$ is a narrow range kernel in k-space.  Because of the convolution theorem of the Fourier transform, the channel by channel SENSE image equation translates to estimating a full k-space grid by convolving a narrow bandwidth convolution kernel with the acquired (undersampled) k-space. This is shown in the following equation: 

***Estimate k-space by convolution***

 $$[\mathbf{K}(\mathbf{k} _n)] _{est,N _c} = conv [ K _{ker}, [\mathbf{K}] _R ]$$
 
Because the convolution kernel is a narrow bandwidth  function in k-space, this essentially means that the k-space index of a missing $$\mathbf{k}_n$$ of channel $$j$$ is essentially a linear combination of all neighboring k-space indices across all channels.  

Transitioning to the k-space domains has several advantages, as we will see later.  One very important advantage relates to the fact that, as I mentioned before, estimating channel sensitivities is difficult.  This is especially the case if your image acquisition consisted of a low SNR system.  

On the other hand, the entries of a low-resolution kernel can be determined as the linear coefficients that fit the k-space index of one channel of your calibration data as a linear combination of the surrounding k-space indices across all channels.  

A few years after SMASH was published, Mark Griswold published GRAPPA (Generalized autocalibrating partially parallel acquisitions), which estimated the kernel using only the surrounding k-space entries that were acquired. The kernel used in GRAPPA only considered acquired k-space neighbors:  An unacquired k-space index $$k_n$$ of channel $$j$$ is estimated as a linear combination of acquired k-space neighbors across all channels. 

***GRAPPA linear dependence formula***
$$d_j(\mathbf{k}_n)=\sum^{N_c} _{l=1}  \sum^{N_b} _{b=1} w _{j,l,b} d_l(\mathbf{k} _{n,b})$$

where $$\mathbf{k} _{n,b}$$ is any acquired k-space neighbor of target coordinate $$\mathbf{k}_n$$ $$w _{j,l,b}$$ is the kernel-weight fitting neighbor $$b$$ of channel $$l$$ to target channel $$j$$.  In this case $$N_b$$ is the number of neighbors in the kernel contributing to estimating the target.  

This can be expressed in matrix form as the following: 

<!--
$$[\mathbf{d}]_j = [W]_j [\mathbf{d}]$$


Here $$[\mathbf{d}]$$ is a tall vector consisting of all 

$$[\mathbf{d}]_j$$ is long list of all unacquired k-space indices of channel $$j$$.  If full sampling consists of $$N_x \cdot N_y \cdot N_z$$ voxels, and the total reduction factor was $$R$$, then $$[\mathbf{d}]_j$$ has size $$\frac{(R-1)N_x N_y N_z}{R}$$. 

-->
$$[\mathbf{d}]_{GRAPPA} = [W] [\mathbf{d}]$$


Where $$[\mathbf{d}]$$ is a tall list of all vectorized k-space entries across all channels, having size $$N_c \cdot N_x \cdot N_y \cdot N_z \times 1$$.  

Next, $$[\mathbf{d}] _{GRAPPA}$$ is the GRAPPA reconstructed k-space of size $$N_c \cdot N_x \cdot N_y \cdot N_z \times 1$$.  

Finally, $$W$$ is a sparse  $$N_c \cdot N_x \cdot N_y \cdot N_z \times N_c \cdot N_x \cdot N_y \cdot N_z$$ matrix that consists of the weights needed to estimate the unacquired entries of $$[\mathbf{d}]$$.  Each row of $$W$$ has $$N_bN_c$$ nonzero members.  Because the weights in $$W$$ are shift invariant, which means that the same weights are used to estimate a k-space index at any location, it can be seen that $$W$$ is a circulant matrix.  Because $$W$$ has repeated entries, it can be determined from the calibration dataset with high precision.  

If the weights in $$W$$ were accurately chosen, and applied on a **fully sampled** k-space $$[\mathbf{d}]_{Full}$$, then

***GRAPPA Marix Formula***
$$[\mathbf{d}]_{GRAPPA} = [W] [\mathbf{d}]$$

That is an important thing to consider, and this is core to determining channel sensitivity maps using E-SPIRiT. This is also important because this is the premise of how we determine the weights from the calibration data.  Given the calibration data, $$[\mathbf{d}]_{cal}$$,  of size $$N _{xc} \cdot N _{yc} \cdot N _{zc} \cdot N_c \times 1$$the terms of $$W$$ must best approximate:


$$[\mathbf{d}]_{cal} = [W] [\mathbf{d}] _{cal}$$

Because the $$N_b$$ weights appear in each row of $$W$$, one could restructure the the equation above to solve for the $$N_b$$ weights: 

***Structured Calibration Formula***
$$[\mathbf{d} _{cal}] = [D _{sources, cal}] [\mathbf{w}]$$

where $$[D_{sources, cal}]$$ is a $$N_{xc} N_{yc}N_{zc}N_c \times N_b N_c$$ matrix where each row contains the neighbor $$N_b$$ k-space indices across all $$N_c$$ channels of each target in $$[\mathbf{d}_{cal}]$$.  The vector $$[\mathbf{w}]$$ is an $$N_b \cdot N_c$$ member long list of the kernel weights.  

The terms in $$[\mathbf{w}]$$ can be solved by


$$pinv([D_{sources}]) [\mathbf{d}_{cal}]$$

where $$pinv()$$ is the Moore-Pensrose pseudo-inverse of the matrix within the brackets.  This can be approximated from  the SVD of $$[D_{sources}]$$.  



The complete k-space can then be estimated by properly arrange the weights in $$[mathbf{w}]$$ to their respective location in the ***GRAPPA Marix Formula***, or one can restructure this formula to the same format as the ***Structured Calibration Formula***:  


$$[\mathbf{d}] = [D _{sources}] [\mathbf{w}]$$

subject that the acquired coordinates do not change.  





The figure below illustrates the procedure.



![](/figures/GRAPPA_Diagram.jpg)


<!-- 
TALK ABOUT HARMONIC FITTING LATER!!!!!!!!
One could plug in the  ***k-space signal model*** into the ***GRAPPA linear dependence formula*** to get the following:


$$$$

$$d(k)=\int \mathbf{M}(\mathbf{r}) e^{-i 2 \pi \mathbf{k}(t) \cdot \mathbf{r}} dr$$

***GRAPPA linear dependence formula***
$$d_j(\mathbf{k}_n)=\sum^{N_c} _{l=1}  \sum^{N_b} _{b=1} w _{j,l,b} d_l(\mathbf{k} _{n,b})$$

-->

##### GRAPPA code

Demonstration of doing parallel imaging along 1 undersampled direction (such as just ky or just kz) is found in  ```../parallelImaging/main_demonstrateGRAPPA2D3x.m```
A demonstration of 2 undersampled directions (such as ky and kz) is found in  ```../parallelImaging/main_demonstrateGRAPPA3D.m```  The dataset used for the latter is too large to be uplaoded.  You could replace the line where i load my data in ```main_demonstrateGRAPPA3D.m``` with your own raw data to see for yourself.  Just make sure the size of the data is arranged in the following order Nx x Ny x Nz x Nc.  


I have two functions:  ```func_grappa_recon.m``` and ```func_complete_grappa_recon.m```.  You can use ```func_grappa_recon``` for to solve any undersampling done under one dimension (say just $$ky$$ or just $$kz$$).  ```func_complete_grappa_recon.m``` should be used if you did undersampling in two dimensions, such as both $$ky$$ and $$kz$$.  

Let's take a look at func_grappa_recon

```
[ rawRecon, weightsKernels] = func_grappa_recon(rawUs, calib, kernelShape, kSolveIndices)
%{
input:
rawUs:  size Nx x Ny x Nz x Nc.  Undersampled raw k-space data.  If the acceleration factor along a dimension is R, then along one k-space axis, every R line is skipped.

calib:  size Nxc x Nyc x Nzc x Nc.  Fully sampled low resolution calibraiton data.  Used to determine the kernels.

kernelShape:  Defines the size of the kernel and has 1s for "source" k-space indices in the kernel and 0 otherwise.  size is the dimension of the kernel width in kspace:  Nkx x Nky x Nkz x numKernelShapes.  numKernelShapes is R-1. 
The minimum size along the undersampled directoin should be R+1. For example, if undersampling is done along ky (the second dimension), then Nky should be R+1.
I usually make Nkx =3 because  harmonic fitting is most accurate when the distance is at most +/1 one harmonic away. Nkz should be 1 if the dataset is 2D.  if the dataset is 3D and you have Nz >3, then i would make Nkz = 3.  
An example for R = 3 along the y direction for a 2D dataset:

---> ky
|         1 0 0 1
|         1 0 0 1
v kx      1 0 0 1
kernelShape has the same values for each of the numKernelShapes entries.  I know this is redundant, but I have it this way so I can be consistent with kSovleIndices.  because it does not cause an issue, I have not yet changed it. 

kSolveIndices:  a 3 x numKernelShapes list that tells you the kx, ky, and kz coordinate in the kernel of the target.  For the example above:
kSolveIndices(3, 1) = [2, 2, 1].'
kSolveIndices(3, 2) = [2, 3, 1].'

Output:
rawRecon: Nx x Ny x Nz x Nc reconstructed k-space dataset.
weightsKernels:  the weights for each kernel.  People do not need this.  I output this so that I could plot the spatial harmonics caused by the weights.  
 
%}
```

Here is some code where I demonstrate preparing the kernel for R=3 along ky for a 2D k-space dataset: 

```
%% Prepare the kernel. 
% * - refers to acquired points providing the "source"
% 0 - refers to unacquired
% x - refers to unacquired point that is the target of the kernel
% kernel shape 1: 
% * * * 
% 0 x 0
% 0 0 0
% * * *

% kernel shape 1: 
% * * * 
% 0 0 0
% 0 x 0
% * * *

Nkx = 3; % length of kernel along the kx direction.
Nky = R+1; % length of kernel along the ky direction.
Nkz = 1; % length of kernel along the kz direction.
numKernelShapes = R - 1;
% kernelShape is just a binary indicating the source.
% in this example, the kernel is 
kernelShape = zeros(Nkx, Nky, Nkz, numKernelShapes); 

% indicating that the edge along ky are the sources.
kernelShape(:, 1, 1, :) = 1;
kernelShape(:, R+1, 1, :) = 1;

% kSolveIndices tells you the indices along each direction where the
% target.  In this example, kernel shape 1 has its target at [2, 2, 1] and
% kernel shape 2 has its target at [2, 3, 1]. 
kSolveIndices = zeros(3, R-1);  % 3 : kx, ky, kz.  7 refers to 8-1 different targets.
kSolveIndices(1, 1) = 2; kSolveIndices(2, 1) = 2; kSolveIndices(3, 1) = 1;
kSolveIndices(1, 2) = 2; kSolveIndices(2, 2) = 3; kSolveIndices(3, 2) = 1;


 [recon, ~] =  ...
        func_grappa_recon(rawUs(:, :, :, :), ...
                          calib(:, :, :, :), ...
                          kernelShape, kSolveIndices);
```

I had to make adjustments for 3D parallel imaging with undersampling in 2 dimensions.  
This is why i have a separate function called ```func_complete_grappa_recon```.  Consider the following ky (vertical) and kz (horizontal) sampling grid, where kx is fully sampled and goes into the page:
```
% x 0 x 0 x 0 x 0 x 0 x 0
% 0 0 0 0 0 0 0 0 0 0 0 0 
% x 0 x 0 x 0 x 0 x 0 x 0
% 0 0 0 0 0 0 0 0 0 0 0 0
% x 0 x 0 x 0 x 0 x 0 x 0
% 0 0 0 0 0 0 0 0 0 0 0 0
```
If your kernel is 3x3 for the ky and kz directions, you will need three different ky,kz kernel structures are necessary when convolving through this k-space: 
```
% the first kernel: 
% x 0 x
% 0 T 0
% x 0 x
% the second kernel: 
% 0 x 0
% 0 T 0
% 0 x 0
% the second kernel: 
% 0 0 0
% x T x
% 0 0 0
```

The first kenrel structure hasa four nonzero entries and second two just have two nonzero entries.  
my previous GRAPPA ```func_getWeights``` and ```func_grappa_recon``` functions works where the kernel shapes all have the same number of nonzero entries. We do that so that for a given kernel the nonzero entries can be vectorized, and then we can carry out the solution for the missing k-space entries by multiplying the kernel (in vector form) by a matrix (that consists only of the source elements).  We cannot do that with the kernel shapes having different sizes.  so we should have a separate variable, called "kernel structure."  

In ```func_complete_grappa_recon```, ```func_grappa_recon``` is called for each of these individual structures.  In each iteration, one of the structures is input as the kernel shape. 


---
<!--
---
### Spatial Harmonics in k-Space  

#### Phase Encoding and Frequency Encoding  
Before diving into parallel imaging techniques like SMASH and GRAPPA, it’s essential to understand the two primary ways in which spatial information is encoded in MRI data: **phase encoding** and **frequency encoding**. These techniques are fundamental for reconstructing images from k-space data.

- **Frequency Encoding**: A gradient is applied along an axis  during the time of the readout.  This creates a gradient of frequencies along that axis while the receivers are listening to the time-varying signal. The FFT of this signal shows the relative "amount" each frequency existed during the readout.  Because the frequencies depend on the position along the readout axis, the FFT therefore gives a projection of the image along the frequency encoding axis.  Here is an illustration below. 
![](/figures/FrequencyEncoding.jpg)

- **Phase Encoding**:  Frequency encoding only maps the signal onto one axis:  it does nothing to split up the signal sourced from different locations that map onto the same frequency encoding direction. Multiple readouts must be made, with each having a different phase-offset along a direction perpendicular to frequency encoding in order to distinguish signal sourced from different locations that project onto the same frequency encoding axis.  This is "phase-encoding." This technique involves applying a gradient along a perpendicular axis to the frequency encoding to encode spatial information. The resulting phase shifts correspond to different positions along the phase encoding axis in the image.  The more different phase-offsets you make, the higher the resolution you can achieve along that direction, as shown below.  


![](/figures/PhaseEncoding.jpg)  


Before discussing GRAPPA, it is useful to examine the concept of **spatial harmonics** in k-space. Because phased-array coils have distinct spatial sensitivity patterns, the k-space data from each coil contains modulated versions of the underlying object. This modulation creates additional harmonics in k-space that allow for reconstruction of missing k-space lines.  

The image below illustrates these spatial harmonics, showing how coil sensitivity variations introduce structured modulations in k-space:  

![Spatial Harmonics in k-Space](/figures/harmonicFitting.jpg)  


Understanding these harmonics is crucial for GRAPPA, as they enable the estimation of missing k-space lines from acquired data.  

### GRAPPA (GeneRalized Autocalibrating Partial Parallel Acquisition)  
GRAPPA is a k-space-based parallel imaging method that estimates missing k-space lines by using the fully sampled region of k-space as an auto-calibration signal (ACS). Unlike SENSE, GRAPPA does not require explicit sensitivity maps.  

In GRAPPA, each missing k-space point is reconstructed as a weighted sum of nearby acquired points across all coils. The reconstruction weights are determined from the ACS region by solving a least-squares system. If $$\mathbf{k}_\text{miss}$$ represents a missing k-space point, it is estimated as:  

$$
\mathbf{k}_\text{miss} = \sum_{c=1}^{N_c} \sum_{j} w_{c,j} \mathbf{k}_{c,j}
$$  

where $$w_{c,j}$$ are the learned weights, $$\mathbf{k}_{c,j}$$ are sampled k-space points from coil $$c$$, and $$N_c$$ is the number of coils. The weights are computed by minimizing the error between the acquired and estimated ACS data through least-squares. 

My implementation of GRAPPA is available in `func_grappa_recon.m`.  I also have 'func_complete_grappa_recon.m' which loops over 'func_grappa_recon.m' with different kernel structures, which is useful in 3D imaging.  I will demonstrate this later. 

-->

### Coming soon! Low-rank matrix recovery in Parallel imaging (see my "func_UNCLE_SAM_NMR" function ...
...or read my paper "Unfolding coil localized errors from an imperfect slice profile
using a structured autocalibration matrix: An application to
reduce outflow effects in cine bSSFP imaging"


### Coming soon! E-SPIRiT.  See my functions:  func_ESPIRiT_fast and func_simplified_ESPIRiT.  


---
## Spatial Beamforming:  Coming soon!  See my implementation of Region Optimized Virtual Coils (func_ROVir_nc).  I will later show results of my new spatial beamforming method later on.  
I have an approach that I previously presented at ISMRM called Low rank reduced FOV (LR-rFOV).  For my paper and pending patent submission I changed the name to Coil Localized Elimination of Aliased NMR (CLEAN).  In my paper and patent submission I compare against ROVir, which is so far the only spatial beamforming approach used in MRI.  ROVir adapted the stochastic matched filter formulation to optimize for signal power within a specific region of interest (ROI) to the signal power outside.  My approach CLEAN is different.  In short, I exploited the linearity of the Fourier transform and the complex valued nature of the raw data to vectorially cancel signal outside of the ROI.  My implementation for rovir (func_ROVir_nc) is online but my implementation of CLEAN will not be posted until I published my paper.   

---
## Non-Cartesian Reconstruction

Non-Cartesian signal reconstruciton refers to the reconstruction of signal where the sampling was not done on a rectilinear grid.  I will start my discussion with projection imaging because that is most common.  Then I will generalize to other non-uniform sampling schemes in imaging. 

When discussing this, I will demonstrate my implementation of gridding and the non-uniform Fourier transform.  

### Projection Imaging

A common example is projection imaging, where different projections of an object are taken.  The goal is to reconstruct an image of the object from those projections.  

Say you have an image described by signal profile $$M(x, y, z)$$. 


The projection of this object $$p _{\theta, \phi}(t_1, t_2) is the sum is the sum along all values of $$M(x,y,z)$$ along the plane defined by the polar and azimutha angles $$\theta$$ and $$\phi$$.  This can be described mathematically as: 


**3D Projection Function**
$$p_{\theta, \phi}(t_1, t_2) = M(x,y,z) \delta(t_1 - \mathbf{u}_1 \cdot \mathbf{r}) \delta(t_2 - \mathbf{u}_2 \cdot \mathbf{r}) dxdydz$$

where $$\mathbf{r} = [x, y, z]$$, $$t_1$$ and $$t_2$$ are  sampling coordinates on the 2D projection plane, $$\mathbf{u}_1$$ and $$\mathbf{u}_2$$ are the direction along those axes, and $$\delta$$ is the Dirac delta function. 

A common way to resolve the image is by an algorithm called Filtered Back Projection (FBP).  In the context for simple 2D backprojection the algorithm is fairly simple.  First let's write down the 2D projection function, which is simply the sum of all values of $$M(x,y)$$  along the line defined by $$x cos(\theta) + y sin(\theta)$$:  

$$p_{\theta}(t) = \int \int M(x,y) \delta(t - x cos(\theta) - y sin(\theta))dx dy$$

The FBP algorithm to generate a 2D image $$Im(x,y)$$ is fairly simple: 

1. Each projection is passed through a high-pass filter to pre-compensate for blurring that would occur if you did not filter.  this is important because a signal located at a specific $$(x,y)$$ coordinate will appear in multiple projections.  Not high-pass filtering this will result blurring across the image domain.  This can be done by multiplying the Fourier transform of the projection by some filtering function, $$f(k)$$:

  **filter projection**

  
  
  $$p^{filter} _{\theta}(t) = F (p _{\theta} (t))(k) \cdot f(k)$$

2. Back proejction.  You smear the filtered projection across the image domain at the corresponding angle $$\theta$$.
   For each $$(x,y)$$:
   a. Compute the corresponding position along the projection:

   $$t = x cos(\theta) + y sin(\theta)$$

   b.  Accumulate the filtered projection value at $$t$$ into into $$Im(x,y)$$:

   $$Im(x,y) = \int^{\pi}_{0} p^{filter} _{\theta}(t) (x cos(\theta) + y sin(\theta)) d\theta$$




I personally prefer reconstructing the data in the Fourier Domain.  Two common methods that do this are the NUFFT and "gridding" (gridding non-uniform k-space samples on a Cartesian/rectilinear Fourier grid) which are more flexible and robust to irregularly sampled data.  Specifically, FBP is limited to projections, while the NUFFT and gridding k-space domain data can be used for any sampling trajectory.  

It is convenient to solve the projection data using these k-space approach thanks to the Fourier projection theorem: the Fourier transform of the projection of the object is equal to a slice of the object's Fourier transform taken through the origin and oriented along the direction of the projection.  In other words:  projections map to slices in the Fourier space (k-space).  Collecting projections at different angles let one complete the k-space, which let's one complete the image.  The projection theorem is is illustrated with the Shepp-Logan phantom below.  


![](/figures/ProjectionTheorem.jpg)  


### NUFFT and Gridding

Before reading if you want to look at my implementation, check out ```func_Cart2nonCart_fa.m```.  This function has on option for you to choose either NUFFT or gridding reconstruction.  Default is NUFFT.  If you want to see a demonstration on how to use this function, see ```/nonCartesianReconstruction/mainDemonstrate_NonCartRecon.m```.

Both methods solve the same the same inverse problem, but differe slight:  the NUFFT solves a least-squares solution and gridding gives in approximation. 
In this discussion, the *acquired* raw k-space data is $$\mu$$, and all acquired samples are concatenated as a tall vector.  For example say each readout projection had $$N_{ro}$$ samples, and there were $$N_{spokes}$$ total projections, then $$\mu$$ is a $$N_{ro} N_{spokes} \times 1$$ tall vector.  The cartesian image $$Im(x, y, z)$$ is on a rectilinear/Cartesian grid of size $$N_x N_y N_z \times 1$$. 

Both NUFFT and gridding try to find a solution $$Im(x,y,z)$$ given the non-Cartesian samples $$mu$$.  They model the relationship of the non-Cartesian samples as in inverse problem.  Let's call the vector $$K$$ the "cartesian sampled" Fourier transform of the image $$Im$$: $$K = F[Im]$$.  In this discussion $$F$$ is the discrete uniform Fourier transform and $$F^{-1}$$ is the discrete inverse uniform Fourier transform.  The inverse problem that the NUFFT solves for and that gridding approximates for is that the acquired non-uniform k-space samples $$\mu$$ can modeled as the convolution of the Cartesian sampled k-space $$K$$ with some filter.  A common filter used is the Kaiser-Bessel window: 

***Forward NUFFT/Gridding Problem***

$$\mu = conv(K, filter)$$

This essentially treats any sampled coordinate in $$\mu$$ as a linear combination of all Cartesian indices in $$K$$.  In my implementation, I used the Kaisser-Bessell filter because Jeff Fessler's "Nonuniform Fast Fourier Transforms Using Min-Max Interpolation" showed that was the best possible solution.  The convolution operator can be described as $$T$$, giving us the following: 

***Forward Matrix NUFFT/Gridding Problem***


$$\mu = T F[Im] $$

or equivalently

$$\mu = T K $$

 $$T$$ has a large size:  $$N_{ro} N_{spokes} \times N_x N_y N_z$$, which makes inverting it difficult to computer.  Therefore, this must be approximated by bounding this filter function, as a sparse matrix $$H$$.  This then models each sampled index in $$\mu$$ as a linear combination of only neighboring Cartesian grid coordinates in $$K$$.  However convolving the Cartesian k-space that you are trying to solve for with a narrow bandwidth kernel apodizes the signal.  If this is not accounted for, the resulting image will be bright at its center, which is an inaccurate representation.  This can be accounted for by multiplying by an image domain de-apodization filter: U.  This leaves us with following sparse inverse problem:  


 ***Sparse NUFFT Inverse Problem***
 

$$\mu =  HFU Im = HFUF^{-1}K$$

Where now $$T$$ is approximated as $$HFUF^{-1}$$.  

***Approximation of T**
$$T = HFUF^{-1}$$

In my implementation of the NUFFT, I broke this down into several steps.  


***My NUFFT steps***
First I solved the following least-squares problem: 

$$\mu = H \kappa$$

where $$\kappa$$ is not my final solution.  

This is illustrated in this snippet  of code from my ```func_Cart2nonCart_fa```:

```
nufft_result = lsqr(H_matrix, mu(:), 1e-6, 20);%pcg(T_combinedMatrix, mu_combined, 1e-6, 100);
        nufft_result = reshape(nufft_result, [Nx, Ny, Nz]);


```
  "nufft_result" is what I call $$\kappa$$ above.  
  I then computed the inverse Fast discrete Fourier transform of $$\kappa$$ before multiplying it by the apodization filter:  $$UF^{1}\kappa$$: 

 ```
IF_nufft_result = ifftnc(nufft_result);
 UIF_nufft_result =  IF_nufft_result  ./ U_matrix; %ones(size(test_U));% test_U;
```
where ```UIF_nufft_result``` is my final deapodized image.  

The NUFFT employs the lsqr algorithm in order to approximate a least-squares solution.  This usually takes a few (around 15 iterations).  The number of iterations needed can be reduced by preconditioning the least-squares problem with a density compensation term, $$D_0$$.  This density compensation term penalizes high density sampled regions.  However I can live with the iterations in the NUFFT.  I only use density compensation if I decide to perform gridding instead, which I will get to next: 

#### Gridding
Gridding seeks to directly approximate an estimate of $$K$$, $$K_{est}$$ by using the **adjoint** of T: 


***Gridding Equation***


$$K_{est} = T^H D_0 \mu$$

This approximation uses the density compensation term to estimate a solution in one iteration, rather than relying on an iterative least-squares solver.   
Remembering the approximation of T, the gridding equation becomes: 

$$K_{est} = (HFUF^{-1})^H D_0 \mu$$

$$K_{est} = F U^H F^{-1}H^H D_0 \mu$$

This is shown in my ```func_Cart2nonCart_fa``` function:  

```
grid_result = H_matrix' * D_0 * mu(:);
grid_result = reshape(grid_result, [Nx, Ny, Nz]);
IF_grid_result = ifftnc(grid_result);
UIF_grid_result =  IF_grid_result .* U_matrix;

```
where ```UIF_grid_result``` is the image of the gridded k-space. 

Gridding just implements the forward operation above: multiply the acquried non-Cartesian data $$\mu$$ by $$F U^H F^{-1}H^H D_0$$.  Without the density compensation term, it there will be a huge bias to the regions of k-space highly sampled --> they will have much more power than what is accurate.  For projection imaging, all k-space samples go through the center, giving the center high-sampling density.  If density compensation is not performed in gridding, then the high sampled density center will have much higher power than it really does, resulting in a low resolution image.  

My ```func_Cart2nonCart_fa``` function allows you to input your own density compensation term.  If empty and you decide to perform gridding, I have a default approach, where I grid and then inverse grid a sample of ones, and then take the reciprocal of the output.  

I will write more about this later, but gridding can be seen as a single iteration of the NUFFT solver.  


One last thing I want to mention before I show results, when I contstruct the interpolation matrix $$H$$, I often interpolate to a grid that is oversampled by a scaling factor in each Cartesian direction.  In otherwords, I oversample data in my interpolation.  This makes a huge difference in the final result.  It makes sense to oversample because in non-Cartesian sampling, such as in projection/radial imaging (projection images sample radial k-space coordinates), the gaps between sampled k-space coordinates near the center is tiny, encoding for a FOV that is much larger than the nominal FOV.  

In this script ```mainDemonstrate_NonCartRecon``` I load a Shepp-Logan phantom.  I then take the radon transform to generate $$N_{spokes}=610$$ projections of the phantom, linearly spaced from $$0$$ to $$\pi$$ radians.  Each individual projection is Fourier transforms to generate a spoke in k-space.  The coordinates of these spokes is shown below: 


![](/figures/sampledSpokes.jpg)  



I then carried out NUFFT and Gridding reconstructions of these sampled spokes, oversampling by a factor of 2 in the $$k_x$$ and $$k_y$$ directions. 
The results can be seen in the figure below.  The figure is divided into three rows.  The top shows the original phantom next to MATLAB's inverse radon transform of the projections.  The top row is just for display to visually compare. 
The second row shows my NUFFT results.  Both results are the same, it is just that one shows the whole oversampled reconstruction and the other shows the center region of the oversampled reconstruction cropped.  
The third row shows the gridding recon.  



![](/figures/griddingAndNUFFTResults.jpg)  


The snippet of code below shows how to use my ```func_nonCart2Cart_fa``` function. It can handle 2D and 3D datasets.  
```
[raw_grid_cropped, raw_grid_ls_osf] = ... %"_ls refers to "least squares"
    func_nonCart2Cart_fa(mu, coord_Matrix, H_and_U_andGrid_Struct, b_choice, D_0, osf, b_square);
% mu is the sampeld raw data:  size Nro * Nspokes x 1. ... or all sampled points x 1.
% coord_Matrix is the sampled kx,ky,kz coordinates of the nonuniformly sampled data.
% H_and_U_andGrid_Struct is a struct that has the sparse H (convolution) matrix and the U (deapodization) matrix.  This is optional to include.  If it is empty, the function will quickly create it. The default % convolution function in H is a truncated Kaiser Bessel function.
% b_choice:  1 - nufft.  2- gridding.  default is 1.
% D_0:  optional.  density compensation diagonal matrix.  If it is not included and you chose option 2, it will create one by griding and ungridding 1.
% osf:  oversampling factor.
% b_square:  default is 1.  just a boolean that sees if you want your entire Cartesian gride to be centered at the (kx,ky) origin.  
```

---
## Coming Soon
The implementations for SPIRiT, Simplified E-SPIRiT, low-rank parallel imaging, beamforming, simultaneous multi-slice (SMS) imaging, and non-Cartesian reconstruction have also been uploaded. I will provide detailed explanations for these topics later, but in the meantime, anyone interested is welcome to explore and use the code.


# License
I appreciate a shoutout if my code is used for generating data or figures for conferences and publications.  Thanks!

# Contact
Contact
If you have questions, suggestions, or would like to collaborate, feel free to reach out:

Name: Fadil Ali


Email: faa5115@g.ucla.edu 


Institution: The Cleveland Clinic Foundation



