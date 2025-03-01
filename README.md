# reconstruction
MRI reconstruction

I have divided this into different sections:  coil combine, .... 

## Coil Combine
I explain whitening, square root sum of squares and adaptive coil combine (Walsh's method).
### Data Whitening 
I demonstrate this is the file coilCombine/main_demonstrateWhitening.m
Mutual inductance is inevitable for channels in the phased array. This has subtle effects in
the noise power amplitude of the combined image from the $$N_c$$ Channels, leaving each channel its own noise variance. Whitening the data will decorrelate the data
from the channels and result in the same noise variance for each channel.

