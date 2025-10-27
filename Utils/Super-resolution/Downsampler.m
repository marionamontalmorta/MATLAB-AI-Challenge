function downsampler = Downsampler(n_planes, factor, kernel_type, phase, kernel_width, support, sigma, preserve_size)
    % Defaults
    if nargin < 4, phase = 0; end
    if nargin < 8, preserve_size = false; end

    assert(phase == 0 || phase == 0.5, 'phase should be 0 or 0.5');

    % Interpret kernel type
    switch kernel_type
        case 'lanczos2'
            support = 2;
            kernel_width = 4 * factor + 1;
            kernel_type_ = 'lanczos';

        case 'lanczos3'
            support = 3;
            kernel_width = 6 * factor + 1;
            kernel_type_ = 'lanczos';

        case 'gauss12'
            kernel_width = 7;
            sigma = 1/2;
            kernel_type_ = 'gauss';

        case 'gauss1sq2'
            kernel_width = 9;
            sigma = 1/sqrt(2);
            kernel_type_ = 'gauss';

        case {'lanczos', 'gauss', 'box'}
            kernel_type_ = kernel_type;

        otherwise
            error('wrong kernel name');
    end

    % Get kernel
    kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support, sigma);

    % Create convolution layer (depthwise)
    downsampler.weight = zeros([size(kernel), n_planes]);
    for i = 1:n_planes
        downsampler.weight(:,:,i) = kernel;
    end

    % Padding setup
    if preserve_size
        if mod(size(kernel,1), 2) == 1
            pad = (size(kernel,1) - 1) / 2;
        else
            pad = (size(kernel,1) - factor) / 2;
        end
    else
        pad = 0;
    end

    downsampler.kernel = kernel;
    downsampler.factor = factor;
    downsampler.pad = pad;
    downsampler.preserve_size = preserve_size;
end
