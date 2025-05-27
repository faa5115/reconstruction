# MRI Reconstruction Methods
Welcome to the MRI Reconstruction Methods repository. This project represents my body of work in MRI image reconstruction, developed during my Ph.D. at UCLA and my postdoctoral fellowship at the Cleveland Clinic Foundation. While this work focuses on magnetic resonance imaging, the techniques and principles applied here—digital signal processing, sensor array combination, aliasing mitigation, and inverse problem formulation—are foundational to Synthetic Aperture Radar (SAR) as well.

My goal in sharing this repository is to demonstrate the depth of my experience in solving complex imaging problems that are closely analogous to those in SAR. Though my work has been in the medical imaging domain, the core challenges—sparse sampling, spatial aliasing, beamforming, non-Cartesian data acquisition, and image reconstruction from frequency-domain measurements—are fundamentally similar.



## Relevance to Radar Signal Processing

Key skills demonstrated in this repository that translate directly to SAR include:

Multi-sensor data fusion: Combining spatially varying signals from array elements (MRI coils) using methods such as SENSE and adaptive beamforming (David Walsh).
Aliasing mitigation from sub-Nyquist sampling: Solving inverse problems to reconstruct unaliased images, akin to resolving spatial aliasing in radar aperture synthesis.
Image reconstruction from frequency-space data: Processing raw k-space (analogous to SAR phase history data) with gridding, NUFFT, and backprojection algorithms.
Low-rank and structured signal recovery: Leveraging low-dimensional structure in undersampled measurements for robust image recovery, as in compressed sensing radar.
This repository brings together multiple reconstruction strategies, each accompanied by code and visual examples, to highlight my hands-on DSP proficiency and adaptability across imaging domains.



# Table of Contents
Introduction
Features
Installation
Usage
Methods Included
Data
License
Contact
# Introduction
MRI is a highly flexible imaging modality, but its performance is limited by slow acquisition times and sensitivity to artifacts caused by undersampling and motion. These challenges mirror many of the signal reconstruction issues found in radar imaging.

My work addresses these problems by developing robust reconstruction algorithms based on signal processing theory and array systems engineering. The techniques presented here reflect both a deep theoretical understanding and hands-on implementation experience in advanced reconstruction pipelines.

This repository is not only a showcase of MRI reconstruction techniques—it is a demonstration of transferable digital signal processing skills applicable to radar and remote sensing applications.



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

## Aliasing and Parallel Imaging
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
  
 ## Spatial Beamforming
 <!--
 For now I will discuss ROVir, which was a novel adaptation of Walsh's adaptive coil combine method to combine the acquired channel data in different linear combination to generate a new set of "virtual" channels that optimize the signal power within the specific region of interest (ROI) over the signal power outside of that region of interest.  
 I will later upload my new approach once it is published or filed for patent; whichever comes first. 
  -->

* Subsampling and Aliasing
* SENSE Unaliasing
* GRAPPA Reconstruction
* SENSE vs GRAPPA: Side Lobe Artifacts
* Structured Low-Rank Matrix Completion

## Spatial Beamforming 
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
### Using Channel Sensitivity Encoding to approximate the Spatial Matched Filter Reconstruction.
**Code Demonstration**: [`coilCombine/main_demonstrateCoilCombine.m`](coilCombine/main_demonstrateCoilCombine.m)

In many cases, such as with NMR or MRI, the signal is complex-valued.  This information is lost in square-root sum-of-squares.  Furthermore. it does not yield the SNR optimal signal reconstruction.  A spatial matched filter would yield the SNR optimal reconstruction, as Roemer et al discussed when writing the NMR Phased Array.  We will first discuss the spatial matched filter as an inverse problem and then proceed modify the inverse problem into the format of the Matched Filter.  

The multi-channel image data at voxel $$\mathbf{r}_n$$ can be described with the following channel-sensitivty encoding scheme: \

<!--
$$\mathbf{Im}(\mathbf{r}_n)=[\mathbf{S}(\mathbf{r}_n)] Imcc(\mathbf{r}_n)$$
-->
$$\mathbf{Im}(\mathbf{r}_n)=[\mathbf{S}(\mathbf{r}_n)] M(\mathbf{r}_n)$$

  $$M(\mathbf{r}_n)$$ is the original complex-valued signal at location  $$\mathbf{r}_n$$.\

  $$\mathbf{Im}(\mathbf{r}_n)$$ is a vector whose entries are each complex-valued  image from each of the $$N_c$$ channels at position $$\mathbf{r}_n$$.  \
  $$\mathbf{Im}(\mathbf{r}_n) = [Im_1(\mathbf{r}_n), Im_2(\mathbf{r}_n), ... , Im _{Nc}(\mathbf{r}_n) ]^T$$, where $$Im_j(\mathbf{r}_n)$$ is the channel image of channel $$j$$.  \
  
  
  $$[\mathbf{S}(\mathbf{r}_n)]$$ is a matrix $$N_c \times 1$$ that holds how sensitive each channel is from signal at position $$\mathbf{r}_n$$.  \
   $$[\mathbf{S}(\mathbf{r}_n)] = [S_1(\mathbf{r}_n), S_2(\mathbf{r}_n), ... , S _{Nc}(\mathbf{r}_n) ]^T$$ where $$S_j(\mathbf{r}_n)$$ is the channel sensitivty of channel $$j$$. 
  
 
This is the forward problem describing how one gets the individual channel images from a spatially varying signal $$M(\mathbf{r}_n)$$.

If one had the sensitivity values, one could compute an estimate of $$M$$, $$M_{est}$$  solve **inverse problem**  of the above formula: 
$$M_{est} = Im_{cc}(\mathbf{r}_n) = [\mathbf{S}(\mathbf{r}_n)]^{-1}\mathbf{Im}(\mathbf{r}_n)$$

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
$$= \frac{\mathbf{q}^HD \mathbf{a}}{\mathbf{q}^H \mathbf{q}}$$


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

---

### Summary
- **Whitening** removes channel noise correlation.
- **Sq. SOS** provides a simple but suboptimal combination method.
- **Walsh's method** is the most advanced, yielding SNR-optimal reconstructions.

Further demonstrations (e.g., multi-echo data) will be added in future updates.


## Aliasing and Parallel Imaging  
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



