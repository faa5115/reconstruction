function [imRecon, senseMaps] = ...
    func_SENSE(imRaw,  imCalib, R, noise, senseMapsOption)
    % input: 
    % imRaw: multi-channel images of the raw data. 
    %           size Nx x Ny x Nz x Nc
    %           Nx, Ny, and Nz are spatial dimensions.  
    %           Nc is the coil dimension.
    % imCalib:  multi-channel images of the calibration data. 
    %           size Nxc x Ny x Nzc x Nc
    %           Nxc Nyc and Nzc are spatial dimensions
    % R :  reduction/acceleration factor for y and z:  2 x 1 : [Ry Rz]
    % senseMapsOption: value is 1 or 2 for now. 
    %                 1:  compute SENSE map the "traditional" way which is
    %                 taking the imCalib data and dividing by its square
    %                 root sum of squares.  
    %                 2:  Use E-SPIRiT to generate the sensitivity maps.
    % noise: multi-channel noise data.  size Nt x Nc where Nt is the number
    % of samples. 

    % output:  
    % imRecon:  reconstructed image.  has size Nx x Ny x Nz x 1 x
    % Ncomponents.  Ncomponents is usually equal to 1.  Otherwise, it is greater
    % than 1 if we are using E-SPIRiT to generate the sensitivity maps on 
    % the case where even the "full" reconstruction does not capture the
    % full FOV (there are aliased wrap-around artifacts from not prescribing 
    % an FOV the covers the entire signal profile). 

    [Nx , Ny , Nz , Nc] = size(imRaw  );
    [Nxc, Nyc, Nzc, ~ ] = size(imCalib);
    [Nt, ~] = size(noise);

    if ~isempty(noise)
        Rn = (noise'*noise);
        RnPinv = pinv(Rn);
    else
        Rn = eye(Nc, Nc);
        RnPinv = Rn;
    end
    Ry = R(1); Rz = R(2);
   
    % create k-space from calibration image. 
    calib = zeros(Nxc, Nyc, Nzc, Nz);
    for chiter = 1 : Nc
        calib(:, :, :, chiter) = fftnc(imCalib(:, :, :, chiter));
    end
    
    if senseMapsOption == 1 || isempty(senseMapsOption)
        % imCalib / sqSOS(imCalib)
         senseMaps = zeros(Nx, Ny, Nz, Nc);

         % there will be Gibbs ringing from zero padding but that is minor
         % for now. I just have this option to show.  I personally
         % recommend the E-SPIRiT option because the k-space convolution
         % kernel yields smooth sensitivity maps.
         calibPad = padarray(calib,  ...
             [round((Nx - Nxc)/2),...
              round((Ny - Nyc)/2),...
              round((Nz - Nzc)/2)], 'both');
         imCalibPad = zeros(Nx, Ny, Nz, Nc);
         for chiter = 1 : Nc
            imCalibPad(:, :, :, chiter) = ...
                ifftnc(calibPad(:, :, :, chiter));
         end
         imCalibPadsos = func_sqSOS(imCalibPad, noise);
         for chiter = 1 : Nc
            senseMaps(:, :, :, chiter) = ...
                imCalibPad(:, :, :, chiter) ./ imCalibPadsos;
         end
         
         Ncomponents = 1;
        
    elseif senseMapsOption == 2
        % use E-SPIRiT
        % uses a kernel across the calibration k-space.
        Nkx = 5; Nky = 5; Nkz = min(3, Nz);
        kSize = [Nkx, Nky, Nkz];
        [eigValuesM, eigVectorsM] = ...
            func_ESPIRiT_fast(calib, kSize, [], [Nx, Ny, Nz]);
        % eigValuesM: the voxel-by-voxel eigenvalues.  
        %             has size Nx x Ny x Nz x Nc, where the last dimension
        %             refers to number of eigen values instead of "channels", but
        %             their numbers are equivalent. 
        % eigVectorsM: eigenvector for each channel.
        %             has size Nx x Ny x Nz x Nc (channel) x Nc (number of eigen states). 

        eigValueMask =  abs( (eigValuesM(:, :, :, :)))./max(eigValuesM(:)) > 0.97;

        eigValueMaskPowerArr = zeros(Nc, 1);
        for chiter = 1 : Nc
            eigValueMaskiter = eigValueMask(:, :, :, chiter);
            eigValueMaskPowerArr(chiter, 1) = sum(eigValueMaskiter(:));
        end
        Ncomponents = length(find(eigValueMaskPowerArr));
        senseMaps = eigVectorsM(:, :, :, :, 1 : Ncomponents);

    end

    imRecon = zeros(Nx, Ny, Nz, 1, Ncomponents);
    
    xAliased = 0;
    yAliased = [0 : Ry-1] .* (Ny/Ry);
    zAliased = [0 : Rz-1] .* (Nz/Rz);
    aliasedPairs = table2array(combinations(xAliased, yAliased, zAliased));
    aliasedPairsX = aliasedPairs(:, 1);
    aliasedPairsY = aliasedPairs(:, 2);
    aliasedPairsZ = aliasedPairs(:, 3);

    S = zeros(Nc, Ry * Rz * Ncomponents, Nx*round(Ny/Ry)*round(Nz/Rz));
    U = zeros(Ry * Rz * Ncomponents, Nc,  Nx*round(Ny/Ry)*round(Nz/Rz));


    %{ 
     for lociter = 1 : Nx * Ny * Nz
        [xiter, yiter, ziter] = ind2sub([Nx, Ny, Nz], lociter);
        xValues = xiter + aliasedPairsX; % just wrote this to be consistent.
        yValues = yiter + aliasedPairsY;
        zValues = ziter + aliasedPairsZ;
        indicesForS = sub2ind([Nx, Ny, Nz], xValues, yValues, zValues);

        for chiter = 1 : Nc
            startiter = 1;
            for compiter = 1 : Ncomponents
                senseMapiter = senseMaps(:, :, :, chiter, compiter);
                S(chiter, startiter : startiter + length(indicesForS) - 1, lociter) = ...
                    senseMapiter(indicesForS);
                startiter = startiter + length(indicesForS);
            end
        end
        disp(strcat('finished', 32, num2str(lociter),32,'out of',32,num2str(Nx*Ny*Nz)))

    end
    %}


    % have x,y,z indices of the first  Nx * (Ny/Ry)  * (Nz/Rz) elements of
    % a Nx x Ny x Nz matrix.  
    [xind, yind, zind] = ind2sub([Nx, round(Ny/Ry), round(Nz/Rz)], 1 : Nx * round(Ny/Ry)  * round(Nz/Rz));%  Nx*Ny*Nz);

    % create Ry*Rz x Nx * (Ny/Ry)  * (Nz/Rz) matrices for x y and z values.
    % Each columnt tealls you which voxels alias onto each other.  (anywhere  to Ry*Rz)
    xValues = xind + aliasedPairsX; % just wrote this to be consistent.
    yValues = yind + aliasedPairsY;
    zValues = zind + aliasedPairsZ;
    
    riter = 1;
    xValuesriter = xValues(riter, :);
    yValuesriter = yValues(riter, :);
    zValuesriter = zValues(riter, :); 
    
    % indicate the indices that are being aliased on.
    indicesForSGround = sub2ind([Nx, Ny, Nz], xValuesriter, yValuesriter, zValuesriter);

    for riter = 1 : Ry * Rz
        xValuesriter = xValues(riter, :);
        yValuesriter = yValues(riter, :);
        zValuesriter = zValues(riter, :); 
        indicesForS = sub2ind([Nx, Ny, Nz], xValuesriter, yValuesriter, zValuesriter);
        for chiter = 1 : Nc
            % startiter = 1;
            for compiter = 1 : Ncomponents
                senseMapiter = senseMaps(:, :, :, chiter, compiter);

                 % S(chiter, startiter : startiter + Ry*Rz - 1, indicesForS) = ...
                 S(chiter, (compiter-1)*(Ry*Rz) + riter, indicesForSGround) = ...
                    senseMapiter(indicesForS);
                % startiter = startiter + Ry*Rz;

            end
        end
    end
    disp('finished the locations loop')

    %{
    % test:  look at y = 50, x = 50, z = 1.
    % figure, imshow(abs(imRaw(:, :, :, 1)), [])
    % figure, imshow(abs(senseMaps(:, :, :, 1, 1)), [])
    xtest = 50; ytest = 50 + Ny/2; ztest = 1;
    testind = sub2ind([Nx, Ny, Nz], xtest, ytest, ztest);
    abs(S(:, :, testind))
    %}


    % create unfolding matrix.
    % For each voxel, the unfolding matrix is:
    % U = pinv(S'*RnPinv * S) * S' * RnPinv.
    % I will call normMatrixInv = pinv(S'*RnPinv*S);
    normMatrix = pagemtimes(RnPinv, S);
    normMatrix = pagemtimes(S, 'ctranspose',normMatrix,'none');

    % Now computer the p-inverse of normMatrix.  MATLAB 2023b does not have
    % pagepinv.  so i will calculate p-inverse using pagesvd.
    % normMatrix = pagepinv(normMatrix);
    [Upage, Spage, Vpage] = pagesvd(normMatrix);
    SpageRecip = Spage; SpageRecip(find(Spage)) = Spage(find(Spage)).^(-1);
    normMatrixInv = pagemtimes(Vpage, SpageRecip);
    normMatrixInv = pagemtimes(normMatrixInv, 'none', Upage,'ctranspose');
    % This will be post multiplied by S' * RnPinv:
    U = pagemtimes(normMatrixInv, 'none',S,'ctranspose');
    U = pagemtimes(U, RnPinv);

    % Now get the image:
    imRawVect = zeros(Nc, 1, Nx * round(Ny/Ry) * round(Nz/Rz));
    for chiter = 1 : Nc
        imRawiter = imRaw(:, 1:round(Ny/Ry), 1:round(Nz/Rz), chiter);
        imRawVect(chiter, 1, :) = imRawiter(:);
    end
    vectResult = pagemtimes(U, imRawVect);
    
    % figure, plot(squeeze(abs(vectResult(2, 1, :))))
   
    % imRecon =   reshape(vectResult(:, 1,  1 : (Nx)*(Ny/Ry)*(Nz/Rz)), [Nx, Ny, Nz, 1, Ncomponents]);...
    %     reshape(vectResult(:, 1,  (Nx)*(Ny/Ry)*(Nz/Rz)+ 1 : end), [Nx, Ny, Nz, 1, Ncomponents]);
     

    % vectResultSum = sum(vectResult, 1);
    % imRecon =   reshape(vectResultSum, [Nx, Ny, Nz, 1, Ncomponents]);...
    %     reshape(vectResultSum, [Nx, Ny, Nz, 1, Ncomponents]);

    % vectResultTrunc = vectResult(:, :, (Nx*(Ny/Ry)*(Nz/Rz)+1:end) );
    % vectResultTrunc = [vectResultTrunc(1, :), vectResultTrunc(2, :)];
    % imRecon =   reshape(vectResultTrunc, [Nx, Ny, Nz, 1, Ncomponents]);
    
    % vectResultCat = [vectResult(1, :), vectResult(2, :)];
    % imRecon = reshape(vectResultCat, [Nx, Ny, Nz, 1, Ncomponents]);
    
    imRecon = reshape(squeeze(vectResult).', [Nx, Ny, Nz, 1, Ncomponents]);

    % startiter = 1;
    % for compiter = 1 : Ncomponents
    %     % vectResultComp = vectResult(startiter : startiter + Ry*Rz -1, :, :);
    % 
    %     startiter = startiter + Ry*Rz;
    % 
    % end

    disp('test')

    % disp('starting locations loop')
    %{
    for lociter = 1 : Nx * Ny * Nz
        [xiter, yiter, ziter] = ind2sub([Nx, Ny, Nz], lociter);

        imRawVect = squeeze(imRaw(xiter, yiter, ziter, :));

        S = zeros(Nc, Ry * Rz * Ncomponents);
        

        xValues = xiter + aliasedPairsX; % just wrote this to be consistent.
        yValues = yiter + aliasedPairsY;
        zValues = ziter + aliasedPairsZ;
        indicesForS = sub2ind([Nx, Ny, Nz], xValues, yValues, zValues);
        
        
        for chiter = 1 : Nc
            startiter = 1;
            for compiter = 1 : Ncomponents
                senseMapiter = senseMaps(:, :, :, chiter, compiter);
                S(chiter, startiter : startiter + length(indicesForS) - 1) = ...
                    senseMapiter(indicesForS);
                startiter = startiter + length(indicesForS);
            end
        end
        
        % disp('start U matrix')
        % Now create the unfolding matrix, U:
        U = pinv(S'*RnPinv * S) * S' * RnPinv;
        % if the noise whitened, then U simply evaluates to pinv(S).
        vectResult = U * imRawVect;
        startiter = 1;
        for compiter = 1 : Ncomponents
            imRecon(xValues, yValues, zValues, 1, compiter) = ...
                vectResult(startiter : startiter + length(indicesForS) - 1);
        end
        disp(strcat('finished', 32, num2str(lociter),32,'out of',32,num2str(Nx*Ny*Nz)))

    end
    %}


end


