function kernel = get_kernel(factor, kernel_type, phase, kernel_width, support, sigma)
    switch kernel_type
        case 'box'
            assert(phase == 0.5, 'Box filter is always half-phased');
            kernel = ones(kernel_width) / (kernel_width^2);

        case 'gauss'
            assert(~isempty(sigma), 'sigma is not specified');
            assert(phase ~= 0.5, 'phase 1/2 for gauss not implemented');

            center = (kernel_width + 1) / 2;
            sigma_sq = sigma^2;
            kernel = zeros(kernel_width);

            for i = 1:kernel_width
                for j = 1:kernel_width
                    di = (i - center) / 2;
                    dj = (j - center) / 2;
                    val = exp(-(di^2 + dj^2)/(2*sigma_sq));
                    val = val / (2 * pi * sigma_sq);
                    kernel(i,j) = val;
                end
            end

        case 'lanczos'
            assert(~isempty(support), 'support is not specified');

            if phase == 0.5
                kernel = zeros(kernel_width-1);
                offset = 0.5;
                center = (kernel_width + 1) / 2;
            else
                kernel = zeros(kernel_width);
                offset = 0;
                center = (kernel_width + 1) / 2;
            end

            for i = 1:size(kernel,1)
                for j = 1:size(kernel,2)
                    di = abs(i + offset - center) / factor;
                    dj = abs(j + offset - center) / factor;

                    val = 1;
                    if di ~= 0
                        val = val * support * sin(pi * di) * sin(pi * di / support) / (pi^2 * di^2);
                    end
                    if dj ~= 0
                        val = val * support * sin(pi * dj) * sin(pi * dj / support) / (pi^2 * dj^2);
                    end

                    kernel(i,j) = val;
                end
            end
        otherwise
            error('wrong method name');
    end

    kernel = kernel / sum(kernel(:));  % Normalize
end
