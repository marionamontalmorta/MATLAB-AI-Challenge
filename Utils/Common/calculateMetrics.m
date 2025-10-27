function [psnr_val, ssim_val] = calculateMetrics(image1, image2)
    % calculateMetrics - Calculate PSNR and SSIM between two images
    %
    % Inputs:
    %   image1 - First image (reference or ground truth)
    %   image2 - Second image (predicted or denoised)
    %
    % Outputs:
    %   psnr_val - Peak Signal-to-Noise Ratio (dB)
    %   ssim_val - Structural Similarity Index (0-1)
    
    % Ensure images are in the correct format
    if isa(image1, 'dlarray')
        image1 = extractdata(image1);
    end
    if isa(image2, 'dlarray')
        image2 = extractdata(image2);
    end
    
    % Convert to double and normalize to [0,1] if needed
    if max(image1(:)) > 1
        image1 = double(image1) / 255;
    end
    if max(image2(:)) > 1
        image2 = double(image2) / 255;
    end
    
    % Ensure images are the same size
    if ~isequal(size(image1), size(image2))
        % Resize image2 to match image1
        image2 = imresize(image2, size(image1, [1, 2]));
    end
    
    % Calculate PSNR
    mse = mean((image1(:) - image2(:)).^2);
    if mse == 0
        psnr_val = Inf;
    else
        psnr_val = 10 * log10(1 / mse);
    end
    
    % Calculate SSIM
    if exist('ssim', 'file')
        ssim_val = ssim(image1, image2);
    else
        % Simple SSIM approximation if Image Processing Toolbox is not available
        ssim_val = calculateSimpleSSIM(image1, image2);
    end
end

function ssim_val = calculateSimpleSSIM(img1, img2)
    % Simple SSIM approximation
    % This is a basic implementation for when the Image Processing Toolbox is not available
    
    % Convert to grayscale if needed
    if size(img1, 3) > 1
        img1 = rgb2gray(img1);
    end
    if size(img2, 3) > 1
        img2 = rgb2gray(img2);
    end
    
    % Calculate means
    mu1 = mean(img1(:));
    mu2 = mean(img2(:));
    
    % Calculate variances and covariance
    sigma1_sq = var(img1(:));
    sigma2_sq = var(img2(:));
    sigma12 = mean((img1(:) - mu1) .* (img2(:) - mu2));
    
    % SSIM constants
    C1 = 0.01^2;
    C2 = 0.03^2;
    
    % Calculate SSIM
    numerator = (2*mu1*mu2 + C1) * (2*sigma12 + C2);
    denominator = (mu1^2 + mu2^2 + C1) * (sigma1_sq + sigma2_sq + C2);
    
    ssim_val = numerator / denominator;
end
