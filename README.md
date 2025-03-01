# reconstruction
MRI reconstruction

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
## Square Root Sum of Squares (Sq. SOS). 
If you treat each voxel across all channels as an Nc-length vector, $$\textbf{Im}(\textbf{r}) $$, the square-root sum of squares of that voxel is simply the magnitude of that vector:  

$$
\begin{align}
\textbf{Im}_{sos}(\textbf{r})  = \sqrt{ \textbf{Im}^T(\textbf{r}) \textbf{Im}(\textbf{r}) }
\end{align}
$$

My func_sqSOS function has two inputs:  multi-channel images (size Nx x Ny x Nz x Nc) and noise (size Nt x Nc).  If the images are already whitened, or you do not want to whiten the data, just place [] in place of noise.  The output is sq. sos. image. I demonstrate this is the file coilCombine/main_demonstrateCoilCombine.m
Below are individual (correlated) channel images and they are followed by a Sq. SOS. recon of whitened channel images.

![](/figures/HipChannelImages.jpg)\

![](/figures/HipSqSOSRecon.jpg)\


## Adaptive Coil Combine (Walsh's Method). 
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


