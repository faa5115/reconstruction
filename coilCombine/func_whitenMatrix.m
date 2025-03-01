function [W, V, D] = func_whitenMatrix(noise)
% Input :  noise is number of timepoints x Nc
% Output:  W is the whitening matrix.  V and D are the eigenvector and diagonal eigenvalue matrix respectively.   
Rn = noise' * noise;
[V, D] = eig(Rn);% eigen value decomposition of noise correlation matrix
                                % noiseCorrMat * Vnc = Vnc * Dnc;

W = V * diag(diag(D).^(-0.5)) * V'; % works better than diag(diag(Dnc).^(-0.5)) * Vnc'
% both Vnc * diag(diag(Dnc).^(-0.5)) * Vnc' and diag(diag(Dnc).^(-0.5)) *
% Vnc' decorrellate the noise.  But Vnc * diag(diag(Dnc).^(-0.5)) * Vnc'
% preserves the sensitivity profiles.  
