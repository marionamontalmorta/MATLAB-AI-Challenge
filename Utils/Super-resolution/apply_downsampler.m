function y = apply_downsampler(downsampler, input)
    % Extract data if it's a dlarray
    if isa(input, 'dlarray')
        input_data = extractdata(input);
    else
        input_data = input;
    end
    
    % Ensure input is in correct format [H, W, C, N]
    if ndims(input_data) == 3
        input_data = reshape(input_data, [size(input_data,1), size(input_data,2), size(input_data,3), 1]);
    end
    
    if downsampler.preserve_size
        padded_input = padarray(input_data, [downsampler.pad, downsampler.pad], 'replicate', 'both');
    else
        padded_input = input_data;
    end

    % Apply convolution per channel
    [H, W, C, N] = size(padded_input);
    y = zeros(H, W, C, N, 'like', padded_input);
    
    for n = 1:N
        for c = 1:C
            y(:,:,c,n) = conv2(padded_input(:,:,c,n), downsampler.weight(:,:,c), 'same');
        end
    end

    % Downsample
    y = y(1:downsampler.factor:end, 1:downsampler.factor:end, :, :);
    
    % Convert back to dlarray if input was dlarray
    if isa(input, 'dlarray')
        y = dlarray(y, 'SSCB');
    end
end
