classdef ReflectionPad2dLayer < nnet.layer.Layer
    properties
        PadSize
    end
    methods
        function layer = ReflectionPad2dLayer(padSize, name)
            layer.PadSize = padSize;
            if nargin > 1
                layer.Name = name;
            end
        end
        function Z = predict(layer, X)
            % X: h x w x c x n
            pad = layer.PadSize; 
            Z = padarray(X, pad, 'symmetric', 'both');
        end
    end
end
