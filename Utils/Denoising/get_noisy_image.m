function [img_noisy_pil,img_noisy_mp] = get_noisy_image(img_np,sigma)
    % Add Gaussian noise 
    img_noisy_np = img_np + sigma * randn(size(img_np));
    % Clip the image values between 0 and 1
    img_noisy_np = max(0, min(1, img_noisy_np));
    img_noisy_pil = im2uint8(img_noisy_np);
end