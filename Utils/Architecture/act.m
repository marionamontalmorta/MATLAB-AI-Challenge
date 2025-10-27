function layer = act(act_fun, name)
if nargin < 1
    act_fun = 'LeakyReLU';
end
if nargin < 2
    name = 'activation';
end

if ischar(act_fun) || isstring(act_fun)
    switch char(act_fun)
        case 'LeakyReLU' % MATLAB Leaky ReLU layer with alpha = 0.2
            layer = leakyReluLayer(0.2, 'Name', [name]);
            %layer = leakyReluLayer(0.2, 'Name', [name 'Relu']);
        case 'ELU'
            layer = eluLayer('Name', [name]);
            %layer = eluLayer('Name', [name 'ELU']);
        case 'none' % An empty layer or identity layer
            layer = [];
        otherwise
            error('Unsupported activation function: %s', act_fun);
    end
elseif isa(act_fun, 'function_handle')
        % Call the function handle and return the layer
    layer = act_fun();
     if isprop(layer, 'Name')
        layer.Name = name;
    end
else
    error('Input must be a string or a function handle.');
end

end

