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
