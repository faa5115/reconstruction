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
![](/figures/ChannelCrossCorrelation.jpg)

