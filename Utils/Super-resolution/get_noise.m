function net_input = get_noise(input_depth, mode, img_size, var)
if numel(img_size)~=2
    error('img_size must be a 2-element vector for [height,widht].');
end
sz = [img_size, input_depth, 1];

switch lower(mode)
    case 'uniform'
        net_input = rand(sz, 'single');
    case 'normal'
        net_input = randn(sz, 'single');
    case 'meshgrid'
        assert(input_depth == 2, 'Meshgrid mode requires input_depth = 2');
        [X, Y] = meshgrid(linspace(0, 1, img_size(2)), linspace(0, 1, img_size(1)));
        net_input = cat(3, X, Y);                % [H, W, 2]
        net_input = reshape(net_input, [img_size, input_depth, 1]);  % [H, W, 2, 1]
        net_input = dlarray(single(net_input), 'SSCB');
    otherwise
        error('Modo de ruido no soportado.');
end

net_input = net_input * var;
net_input = dlarray(net_input, 'SSCB');

end
