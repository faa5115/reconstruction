# reconstruction

 I HAVE NOT YET UPLOADED EVERYTHING YET. I am creating figures and "main files" that show how to use the reconstruction code I wrote.  When this is complete, this repository will have my written codes of SENSE, GRAPPA, SPIRiT, SAKE (plus other low rank methods such as p-LORAKS), "simplified E-SPIRiT", spatial beamforming (ROVir, and then my recon i developed as a post doctoral fellow (under the guidance of Dr. Xioajuan Li) once the work gets accepted), and non-Cartesian reconstruction.  With regards to "pluse other low rank methods" in the parenthesis in the previous sentence, in faa5115/uncle_sam_recon, I have my code of low-rank based k-space completion that I developed under the guidance of Dr Mark Bydder and Dr. J. Paul Finn.  We proposed using low-rank k-space based parallel imaging to unfold outflow effects (outflowing spins have different channel sensitivities than the imaged slice).  So if you want some low-rank k-space based reconstruction, you can use that code.  

 


This is a repository of some MRI reconstruction methdos that I reconstructed, along with a brief explanation of them and individaul main files demonstrating how to use my code. I spent time on the details of these methods in the background chapter of my thesis "The Exploitation and Mitigation of Flow Effects in MRI." In this repository, I go over and demosntrate my implementations of the following:   whitening multichannel data, adaptive coil combine, sensitivity encoding, SMASH, GRAPPA,  SPIRiT, my own "simplified E-SPIRiT", SAKE/other low rank parallel imaging methods, non-Cartesian reconstruction, and spatial beamforming (right now i will just go over ROVir because I am trying to publish my own spatial beamforming work).  

One thing that is important for a non-MRI audience is that the signal reception in MRI is actually the Fourier transform of the image.  We call this "k-space."  So when we are "spatially encoding", we are reading out the Fourier transform of the image (or simply "reading out k-space"). 
## A brief story of how I got into reconstruction.  (For an MRI audience).
Starting off my PhD at UCLA, I thought I would just be a pulse sequence guy, but I eventually really got into reconstruction.  During my PhD, I was originally working on applying SEMAC to cine GRE imaging to see if it could improve imaging of subjects with pacemakers or ICDs.  I then came across Mike Markl's "Flow effects in balanced steady state free precession imaging", and I realized that outflow artifacts in balanced steady state free precession (bSSFP) is similar to metal artifacts.  When trying to acquire 2D images of an imaging slice close to metal or having through-plane flow, the signal profiles both actually span a 3D space.  For metal that is because the metallic device/implant distortes the magnetic field, so when applying RF transmission to excite a slice, the distorted magnetic field actually excites a 3D profile.  This distorted excitation profile collapses on a 2D plane if you only spend time spatially encode for a 2D plane.  To resolve this issue, you must make the effort to spatially encode for 3D.  For 2D bSSFP imaging, if you have a vessel that goes through the imaged slice, the inflowing spins get excited onto the transverse plane.  As they flow out, they carry coherent signal that still contribute to the MRI signal reception. This is primarily a problem with bSSFP because there is nothing done to spoil this outflowing signal.  Because we have signal flowing out of the slice, we actually have a signal profile that spans a 3D volume instead of a 2D plane, requiring a 3D spatial encoding of the k-space.  So I thought to apply what I had implemented for metal to resolve the outflow effect issue, where I implemented slice-encoding steps to have 3D k-space encoding, allowing for spatial encoding for a volume beyond the excited slice.  Afte a 3D FFT of the raw data, the target slice will be found in the center, unfolded from the outflow effects.  I have NMR Bloch simulations illustrating this in the file "github.com/faa5115/flow_bSSFP_Bloch_Simulations". This original idea of mine was core to my PhD thesis and we published it in Magnetic Resonance in Medicine ("Slice encoding for the reduction of outflow signal artifacts in cine balanced SSFP imaging").  I am glad to have mentors (such as Drs. Peng Hu, J. Paul Finn, and Mark Bydder) who encouraged me publish this observation.  With the intellectual support of Dr. Finn and Dr. Mark Bydder, I explored means of using spatially varying channel sensitivities of the phased-array to unfold the outflow effects from the target slice.  The rational behind this is that as the spins flow out of the slice, they will have a channel sensitivity prifle that is different from the target slice (see "uncle_sam_recon").  This reconstruction method can be found in github.com/faa5115/uncle_sam_recon and I will go over it in the parallel imaging section.  Our initial approach of doing this was based off of SMASH (which we presented as an oral at the ISMRM in 2023 in Toronto, Ontario, Canada), but further developed the method to instead by based on low-rank matrix completion (and called it "UNCLE SAM").  We published this work in NMR in Biomedicine (Unfolding coil localized errors from an imperfect slice profile using a structured autocalibration matrix: An application to reduce outflow effects in cine bSSFP imaging). 

## What MRI reconstruction topics I go over.
I have divided this into different sections:  coil combine, parallel imaging, non-Cartesian reconstruction, and spatial beamforming, .... 

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
Inputs:  Its inputs are imRaw, noise, and patchSize.  Here, imRaw is of size Nx x Ny x Nz x Nc x Nm, where Nm is the number of echoes.  If you have multiple slices or slabs (Ns > 1) do the recon separately for each slice/slab.  patchSize is of size Npatchx x Npatchy x Npatchz x Nm.  You can have patchSize be empty (place "[]" instead) and the code will figure it out for you (based on Walsh's writing that you need ~250 voxels in your patch to have the most statistically robust coil combination).  The reason why i included the echo dimension in this implementation is because Mark Bydder has a paper showing how Walsh's adaptive recon method can be used for time-series MRI.  This makes sense because the original math behind Walsh's method was for time-varying data in radar, as I mentioned earlier.  Mark Bydder's paper was regarding spectroscopy but he later used it for multi-echo data for fat and R2* mapping.  

Output:  the coil-combined data of size Nx x Ny x Nz x 1 x Nm.  

I will put a demonstration with multi-echo data later. 

## Parallel Imaging
I will go over some of the parallel imaging methods I implemetned myself and some intuition behind them.  I had fun going down this rabbit hole. 

First I should go over hte importance of phase-encoding.  This will give the intuition of why scan time scales with parallel imaging, which will then be useful for understanding parallel imaging. First off, a single readout line  only provides a single projection of the image along the direction frequency encoding was applied. This readout signal is a sum of multiple spatial harmonic functions where all spins at the having the same coordinate along the frequency encoding direction same frequency,  $$\gamma \textbf{G}_{ro} \cdot \textbf{r}$$.   This does nothing to localize spins along either perpendicular phase-encoding direction.  A single frequency encoded readout is shown below.  The image on the left is the object being imaged (ACR Phantom - American College of Radiology) and the white arrow indicates the direction of frequency encoding.  On the right, the top image shows the time-varying k-space readout and beneath it is the FFT of this readout.  This FFT is just a projection of the spins along the frequency encoding direction.  I go more into detail in the background section of my thesis, but Paul Lauterbur's ground breaking paper in Nature explains this really well.  
![](/figures/FrequencyEncoding.jpg)\

In order to distinguish $$$N$ harmonic signals, $$N$$ observations must be made with each observation having a different phase offset between between the signals.  In the case for MR imaging, if $$N_{pey$}$ and $$N_{se}$$ different localizations are needed to be made along   $$\textbf{u}_{pey}$$ and  $$\textbf{u}_{se}$$ respectively, frequency encoding must be repeated $$N_{pey} \times N_{se}$$ times with each having its own spatially-varying phase offsets along these two directions. This is illustrated in the figure below for an increasing number of phase-encoding steps, notice how the resolution increases as the number of different readouts-each with a different phase-offset increases.  

![](/figures/PhaseEncoding.jpg)\



 The phase offsets needed is achieved by applying a \textit{phase encoding} gradient along $$\textbf{u}_{pey}$$ or ($$\textbf{u}_{pey}$$ and $$\textbf{u}_s$$ if 3D imaging is being done), with the phase-encoding gradient achieving a different phase dispersion across the  FOV for each readout.  Applying a brief phase-encoding gradient before the readout induces a spatial harmonic of a factor of the imaging FOV along the phase-encoding direction.  Each phase-encoding line induces a spatial harmonic of a its own factor of the FOV. A figure below explains this:   Top row:  magnitude and phase of the ACR phantom with the phase-encoding direction indicated by the white arrow.  Beneath the magnitude image is a map depicting the log of k-space with the $$ \textbf{k}_{pey} $$ pointing vertically.  The spatial harmonic phase-dispersion, induced by $$G_{pey}$$, across the phase-encoding direction is shown, with the color of the box corresponding to its indicated phase encoding step in the k-space map.

![](/figures/PhaseEncoding_Part2.jpg)\

Given this very brief discussion on phase-encoding, it is clear that the number of phase-encoding steps scales with scan time.  To reduce scan time, you must reduce the number of phase-encoding steps.  If you only spend time to acquire the k-space lines close to the center, you are only left with information that cannot distinguish many nearby spins from one another and are left with a low-resolution image.  What happens if you skip k-space lines?  

To answer this question, it must be noted that the discrete sampling process unavoidably generates a periodic representation of the non-periodic signal. This in effect produces periodic copies of the signal, separated by a distance determined by the sampling frequency. In the case of phase-encoding, the distance between adjacent phase-encoding lines is reciprocal of the the \textit{encoded} FOV along the PE direction, $$\Delta k_{pey} = \frac{1}{FOV_{pey}}$$ where $$FOV_{pey}$$ is the FOV along the phase-encoding direction.  That is the signal in the image will repeat over a length of $$FOV_{pey}$$.  If the span of signal along the phase-encoding direction (as in the "real FOV") is larger than  $$FOV_{pey}$$, that is $$FOV_{pey} < FOV_{pey,true} $$ then you will have signal overlapping, a phenomenon called "aliasing." This is illustrated below for case where the encoded FOV is one-half of the full FOV. 

![](/figures/BrainAliased.jpg)\


Parallel imaging works to unfold aliased signal to generate a full FOV image and they rely on the localized coil/channel sensitivity of the phased-array signal.  An illustration the localized spatially varying chanenel sensitivity profiles of a brain MRI is shown below. 
There are two broad approaches to parallel imaging:

- Explicity used the localized channel sensitivty profiles to unaliased the signal in the image domain.  The most famous approach is "Sensitivity Encoding" ("SENSE"). 
- Implicitly using the channel sensitivity profiles to figure out how to mimic the spatial harmonics that would have been achieved by the unacquired phase-encodings steps.  SMASH (Simultaneous Acquisition of Spatial Harmonics) was the first method to try this approach.  It has variants (such as Auto-SMASH and VD-Auto SMASH).  These methods laid the foundation for more mature approaches such as GRAPPA and SPIRiT.  I will explain how these methods work and upload my code for them soon.  After explaining them, I will then discuss how low rank mathematics can be used for parallel imaging.  To be brief,  I will say that methods like GRAPPA and SPIRiT show that any k-space entry of a given channel can be expressed as a linear combination of surrounding k-space entries across all channels.  This means a local neighbor patch of k-space across all  channels exist in some low-rank space.  By figuring out the null space, one can recover the missing k-space entries.  This gets us into methods such as PRUNO, SAKE, and p-LORAKS.



### Sensitivity Encoding (SENSE)
I demonstration this in "parallelImaging/main_demonstrateSENSE.m"

A common method to unalias the image is by explicitly exploiting the localized channel
sensitivity, called sensitivity encoding (SENSE). SENSE is a generalization of the spatial
matched filter discussed above (see "Adaptive Coil Combine" Above).  For generous simplicity when narrating SENSE, let’s
assume the channels are whitened, meaning that the noise covariance matrix is the identity matrix. The formulat for the matched filter then becomes 

$$
\begin{align}
Im_{comb}(\textbf{r}) = \textbf{c}^H(\textbf{r}) \textbf{Im}(\textbf{r})
\end{align}
$$

where $$\textbf{c}^H(\textbf{r})$$ is an $$N_cx1$$ vector containing hte channel sensitivy at position $$\textbf{r}$$ and I write $$Im_{comb}(\textbf{r})$$ here to indicate the coil combined image. 
This can be expanded into matrix format (a screenshot from my Chatper 2 of my thesis): 

![](/figures/MatchedFilterEquationMatrix.jpg)\

where Nvox  is the total number of voxels in an image and $$\textbf{r}_j$$ , with $$ j \in [1, Nvox] $$
refers to the location of each voxel. The second line of the image above is written in a block matrix
format to conserve space. Each block matrix $$ \textbf{C}(\textbf{r})$$ is a diagonal matrix with each being
the complex channel sensitivities.  This equation  can be generalized as the following to describe aliasing when skipping each $$R$$ k-space lines (another screenshot from my thesis):

![](/figures/SENSEFilterEquationMatrix.jpg)\
Where $$Im_{comb}(\textbf{r})$$ can be solved for my least-squares. A common way to estimate for the entries
of the sensitivity matrix is to simply divide the image of each channel by their sum of
squares, followed by some filtering to smooth the data and threshold only
for voxels with anatomical support.  In the example shown in "parallelImaging/main_demonstrateSENSE.m", I use my own implementation of E-SPIRiT (which I call "simplified E-SPIRiT").  The magnitude and (relative). phase of the sensitivity maps are shown below.  I say "relative" phase because I rotated the phase of all channels negative to the phase of channel two in order to show a smooth phase variation in channel senseitivity profile. 

![](/figures/senseMaps.jpg)\

As mentioned earlier, the channel sensitivity profiles are explicitly used to achieve the necessary spatial encoded to unalias the image. Although this narration featured a simplified
explanation of the spatial matched filter, a thorough investigation on the ideal channel sensitivities used for the matched filter has the potential to achieve an SNR optimal parallel imaging scheme. SENSE is the generalization of the spatial matched filter for subsampled imaging.  This is different from the Walsh method, which does not consider aliasing.  Because they are both coil combine methods based on the principles of the matched filter I included four images below to compare the magnitude and phase of a SENSE R=2 reconstruction with a fully sampled Walsh coil combination. 

![](/figures/SENSE_and_Walsh.jpg)\

## k-Space based parallel imaging approaches
This section covers k-space based approaches.  I want to go into detail but for now I will be brief for the sake of time - I just want to get my code uploaded first before typing the detail on the theory behind these approaches.  To be brief, these k-space methods use the channel sensitivity profiles to mimic the effects of phase-encoding lines.  Phase-encoding gradients add a phase-dispersion across the FOV along the direction of the phase-encoding gradient, $$\textbf{r}_{pe}$$.  The signal for a given MR channel of sensitivity $$C$$ can be described as (this is very simple and ignoring relaxation and recovery, $$\Delta B_0$$ and $$\Delta B_1$$ effects : 


$$
\begin{align}
S(\textbf{k}) = \int M(\textbf{r}) e^{i 2 \pi  \textbf{k} \textbf{r}} d \textbf{r}   
= \int M(\textbf{r}) e^{-i 2 \pi  (\textbf{k} + m \Delta \textbf{k}) \textbf{r}} e^{-i 2 \pi  (- m \Delta \textbf{k}) \textbf{r}} d \textbf{r} 
\end{align}
$$

This implies if $$S(\textbf{k})$$ is not explicitly acquired, it can be estimated from an
acquired k-space $$S(\textbf{k} + m \Delta \textbf{k})$$ if a $$-m \Delta \textbf{k}$$ - harmonic function, $$M(\textbf{r}) e^-{i 2 \pi  (- m \Delta \textbf{k}) \textbf{r} }$$ is somehow known.  

A low-resolution calibration is used in SMASH to estimate these harmonics.  It determines a set of weights for each channel $$l$$ , $$n^{m}_l$$ , to sum the channels to estimate an $$m$$-order spatial harmonic: 

$$
\begin{align}
\sum_{l=1}^{N_c}[n_l^m Im_{l}^{cal}(\textbf{r})|_{pe}]  = f(\textbf{r})  
\end{align}
$$

where $$|_{pe}$$ indicates the projection, and $$ Im_{l}^{cal}$$ is the low-resolution calibration image.  $$f(\textbf{r})  $$ is some harmonic fitting function.  In the original smash paper, the fitting function was the coil-combined calibration modulated by the harmonic function: $$f(\textbf{r}) = Im_{comb}^{cal}(\textbf{r})|_{pe} e^{-i 2 \pi (-m \Delta \textbf{k} \textbf{r})}$$.  However it is
difficult to find a composite signal profile without avoiding destructive interference from the
multiple channels. Because of this it was later shown that one could use the transverse
magnetization distribution perceived by each channel, 

$$
\begin{align}
\sum_{l=1}^{N_c}[n_l^m Im_{l}^{cal}(\textbf{r})|_{pe}]  =  
Im_{j}^{cal}(\textbf{r})|_{pe} e^{-i 2 \pi (-m \Delta \textbf{k} \textbf{r})}  
\end{align}
$$









................
I will soon upload my simple SENSE implementation and discuss the approach.  Because I do not often use SENSE, my implementation is very simple.  I will then go to the k-space based approaches, starting off with SMASH and then go into GRAPPA, and SPIRiT.  I will then get into low-rank k-space based approaches and demonstrate some code.  I use GRAPPA and low-rank reconstruction methods often, the code I wrote of these methods are mature and I hope they can be useful for the imaging community.  


After these, I will go into SMS imaging, spatial beamforming, and non-Cartesian MRI and upload the code I wrote of these as well. 
