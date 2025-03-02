# reconstruction
This is a repository of some MRI reconstruction methdos that I reconstructed, along with a brief explanation of them and individaul main files demonstrating how to use my code. I spent time on the details of these methods in the background chapter of my thesis "The Exploitation and Mitigation of Flow Effects in MRI." In this repository, I go over and demosntrate my implementations of the following:   whitening multichannel data, adaptive coil combine, sensitivity encoding, SMASH, GRAPPA,  SPIRiT, my own "simplified E-SPIRiT", SAKE/other low rank parallel imaging methods, non-Cartesian reconstruction, and spatial beamforming (right now i will just go over ROVir because I am trying to publish my own spatial beamforming work).  

## A brief story of how I got into reconstruction.  (For an MRI audience).
Starting off my PhD at UCLA, I thought I would just be a pulse sequence guy, but I eventually really got into reconstruction.  During my PhD, I was originally working on applying SEMAC to cine GRE imaging to see if it could improve imaging of subjects with pacemakers or ICDs.  I then came across Mike Markl's "Flow effects in balanced steady state free precession imaging", and I realized that outflow artifacts in balanced steady state free precession (bSSFP) is similar to metal artifacts.  When trying to acquire 2D images of an imaging slice close to metal or having through-plane flow, the signal profiles both actually span a 3D space.  For metal that is because the metallic device/implant distortes the magnetic field, so when applying RF transmission to excite a slice, the distorted magnetic field actually excites a 3D profile.  For 2D bSSFP imaging, if you have a vessel that goes through the imaged slice, the inflowing spins get excited onto the transverse plane.  As they flow out, they 

I have divided this into different sections:  coil combine, .... 

## Coil Combine
I explain whitening, square root sum of squares and adaptive coil combine (Walsh's method).
The code here works with the data found in  "hipSliceAndNoise.mat" will give you "raw" (raw k-space data of a single slice hip of size Nx x Ny x Nz x Nc x Ns) and "noise" (size Nt x Nc).  
This noise scan was acquired at the same readoutbandwidth as the scan of raw.  
The size of raw:  Nx = number of readout points.  Ny = number of phase encoding lines. Nz = Number of partitions which is one because this was a 2D scan. Ns is the number of excited slices, which is one. 
The size of noise: Nt is the number of time points received during the noise scan. As mentioned above, Nc is the number of channels. 

### Data Whitening 
I demonstrate this is the file coilCombine/main_demonstrateWhitening.m


Mutual inductance is inevitable for channels in the phased array. This has subtle effects in
the noise power amplitude of the combined image from the $$N_c$$ Channels, leaving each channel its own noise variance. Whitening the data will decorrelate the data
from the channels and result in the same noise variance for each channel.  I will explain and demonstrate this:

The cross-correlation of the channels (of size Nc x Nc) is given by $$Rn = N^HN$$, where $$N$$ is the Nt x Nc noise data and and $$^H$$ indicates the Hermitian conjugate. \
![](/figures/ChannelCrossCorrelation.jpg)\

This has noticeable off-diagonal entries, indicating significant cross-channel correlation.  A whitening transform, $$W$$$, should be applied to this noise or raw data to remove the cross-channel correlation.  Whitened noise data, $$N_w$$, would satisfy $$N^H_wN_w = I$$, where $$I$$ is the identity matrix. If $$N_w = NW$$, then the whitening transform can be determined from the following: 



$$ 
\begin{align} 
N^H_wN_w = I \\ 
(NW)^H(NW) = I \\ 
W^HN^HNW = I \\
W^HV_N\Lambda_N V^H_NW = I \\
W^HV_N\Lambda_N^{1/2}\Lambda_N^{1/2} V^H_NW=I \\
(\Lambda_N^{1/2} V^H_NW)^2 = I \\
(\Lambda_N^{1/2} V^H_NW) = I \\
W = \Lambda_N^{-1/2}V_N \\
\end{align}
$$

The function func_whitenMatrix(noise) inputs the noise data and has the following outputs:  W (the whiten matrix) and V and D (the eigenvector and diagonal eigenvalue matrix in case you want them). After applying $$NW$$ the correlation matrix of the newly whitened noise data is: \
![](/figures/WhitenedChannelsCrossCorrelation.jpg)\

\
This whitening operator can then be applied to each k-space entry across all channels, $$\textbf{d}(\textbf{k}) $$,  to result in whitened k-space data by $$\textbf{d}_w(\textbf{k}) = \textbf{d}(\textbf{k})^T W $$ where $$\textbf{k}$$ indicates k-space coordinate, $$\textbf{d}(\textbf{k})$$ is a Nc long vector giving the k-space entry of that coordinate for each channel, and $$^T$$ indicates transpose.

This can also be applied to the image domain as well because the Fourier transform is linear.  An an Nc length vector $$\textbf{Im}(\textbf{r})   $$, which holds the image voxels at location $$\textbf{r}$$ for each channel, can be whitened by post multiplying its transpose by $$W$$:  $$\textbf{Im}_w(\textbf{r}) = \textbf{Im}(\textbf{r})^T W$$.




The content of the following two sections are demonstrated in coilCombine/main_demonstrateCoilCombine.m
### Square Root Sum of Squares (Sq. SOS). 
 I demonstrate my implementation in the file coilCombine/main_demonstrateCoilCombine.m

If you treat each voxel across all channels as an Nc-length vector, $$\textbf{Im}(\textbf{r}) $$, the square-root sum of squares of that voxel is simply the magnitude of that vector:  

$$
\begin{align}
\textbf{Im}_{sos}(\textbf{r})  = \sqrt{ \textbf{Im}^T(\textbf{r}) \textbf{Im}(\textbf{r}) }
\end{align}
$$

My func_sqSOS function has two inputs:  multi-channel images (size Nx x Ny x Nz x Nc) and noise (size Nt x Nc).  If the images are already whitened, or you do not want to whiten the data, just place [] in place of noise.  The output is sq. sos. image.
Below are individual (correlated) channel images followed by, channel phases, and a Sq. SOS. recon of whitened channel images.

![](/figures/HipChannelImages.jpg)\

![](/figures/HipChannelPhases.jpg)\

![](/figures/HipSqSOSRecon.jpg)\


### Adaptive Coil Combine (Walsh's Method). 
I demonstrate my implementation in the file coilCombine/main_demonstrateCoilCombine.m

The spatial matched filter provides SNR optimal combination while removing the greatest
extent of local channel shading by using the distribution of the magnetic field generated
by each channel (the sensitivity profile of each channel). Using the sensitivity profiles to
estimate for a combined channel image preserves the relative phase the voxels. The weights
for the spatial matched filter ($$\textbf{m}(\textbf{r}) \in N_c x 1 $$) can be described as:

$$
\begin{align}
\textbf{m}(\textbf{r}) = Rn^{-1}\textbf{c}({\textbf{r}})
\end{align}
$$

where $$\textbf{c}({\textbf{r}}) \in Nc x 1$$ is an Nc long vector of the channel sensitivities of the signal at voxel/position $$\textbf{r}$$. 
The coil-combined image at position $$\textbf{r}$$, $$Im_{cc}(\textbf{r})$$ can be given as  

$$
\begin{align}
Im_{cc}(\textbf{r}) = \textbf{m}^H(\textbf{r}) \textbf{Im}(\textbf{r})
\end{align}
$$

However properly estimating $$\textbf{c}(\textbf{r})$$ is difficult.  A common way of doing this is by dividing the individual channel
images by the square root sum of squares image or by dividing by an image from the body coil.   An adaptive method to estimate for the optimal matched described by Walsh, borrowing from the stochastic matched filter formulation of temporal signal processes commonly seen in radar. Rather than time, Walsh applied the stochastic matched filter process of combining the channel images at location $$\textbf{r}$$, $$\textbf{c}(\textbf{r})$$ over a patch of voxels centered at $$\textbf{r}$$.  Given a noise scan of size $$Nt x Nc$$ and a patch of voxels centered at $$\textbf{r}$$ for each channel of size $$Np x Nc$$ ($$Np$$ is patch-size), a noise covariance matrix, $$Rn$$, and a signal covariance matrix $$Rs(\textbf{r})$$ (including "$$(\textbf{r})$$ to make it clear that $$Rs$$ is going to be different for each voxel), can be formed.  In the manuscript, Walsh demonstrates that the weights, $$\textbf{m}$$, that optimizes signal to noise power is the eigenvector corresponding the highest eigenvalue of the eigenbasis that double-diagonalizes $$Rn$$ and $$Rs$$.  By that i mean the $$Ncx1$$ weights vector $$\textbf{m}$$ is the eigenvector of the largest eigenvalue of $$Rn^{-1}Rs(\textbf{r})$$.  Because this double-diagonalizes the noise and signal covariance matrices, the input noise and signal can be correlated, but the weights vector $$\textbf{m}$$ will whiten in the summation.   To preserve relative phase before the summation, $$\textbf{m}$$ is multiplied by $$e^{-i\theta_{max}}$$ where $$\theta_{max}$$ is the phase of the  element of $$\textbf{m}$$ corresponding to the channel with the highest signal power.  Below is the magnitude and phase of a Walsh method reconstruction of the multi-channel hip data illustrated above: \

![](/figures/WalshCombine_signal_and_phase.jpg)\

My implementation of this reconstruction is func_WalshMethod. 
Inputs:  Its inputs are imRaw, noise, and patchSize.  Here, imRaw is of size Nx x Ny x Nz x Nc x Nm, where Nm is the number of echoes.  If you have multiple slices or slabs (Ns > 1) do the recon separately for each slice/slab.  patchSize is of size Npatchx x Npatchy x Npatchz x Nm.  You can have patchSize be empty ([]) and the code will figure it out for you (based on Walsh's writing that you need ~250 voxels in your patch to have the most statistically robust coil combination).  The reason why i included the echo dimension in this implementation is because Mark Bydder has a paper showing how Walsh's adaptive recon method can be used for time-series MRI.  This makes sense because the original math behind Walsh's method was for time-varying data in radar, as I mentioned earlier.  Mark Bydder's paper was regarding spectroscopy but he later used it for multi-echo data for fat and R2* mapping.  

Output:  the coil-combined data of size Nx x Ny x Nz x 1 x Nm.  

I will put a demonstration with multi-echo data later. 

## Parallel Imaging
Here I will go over parallel imaging methods that I implemented.  
