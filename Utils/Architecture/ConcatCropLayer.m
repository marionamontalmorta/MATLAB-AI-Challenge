classdef ConcatCropLayer < nnet.layer.Layer
    properties
        Dim % Dimension to concatenate along
        NumInputs % Number of inputs (flexible)
    end

    methods
        function layer = ConcatCropLayer(dim, name, varargin)
            % Constructor
            % ConcatCropLayer(dim, name) - Creates layer with 2 inputs
            % ConcatCropLayer(dim, name, numInputs) - Creates layer with specified number of inputs
            
            layer.Name = name;
            layer.Description = "Crop and concatenate along dimension " + dim;
            layer.Dim = dim;
            
            if nargin > 2
                layer.NumInputs = varargin{1};
            else
                layer.NumInputs = 2; % Default to 2 inputs
            end
        end

        function Z = predict(layer, varargin)
            inputs = varargin;

            % If there's only one input, return it directly
            if length(inputs) == 1
                Z = inputs{1};
                return;
            end

            % Validate inputs
            if isempty(inputs)
                error('ConcatCropLayer: No inputs provided');
            end

            % Get heights and widths of all inputs
            heights = cellfun(@(x)size(x,1), inputs);
            widths  = cellfun(@(x)size(x,2), inputs);

            targetH = min(heights);
            targetW = min(widths);

            % Crop each output to match the smallest height and width
            for i = 1:length(inputs)
                if size(inputs{i}, 1) > targetH || size(inputs{i}, 2) > targetW
                    diffH = floor((size(inputs{i},1) - targetH)/2);
                    diffW = floor((size(inputs{i},2) - targetW)/2);
                    inputs{i} = inputs{i}(diffH+1:diffH+targetH, ...
                                            diffW+1:diffW+targetW, :, :);
                end
            end

            % Concatenate along the specified dimension
            try
                Z = cat(layer.Dim, inputs{:});
            catch ME
                error('ConcatCropLayer: Concatenation failed along dimension %d. Error: %s', layer.Dim, ME.message);
            end
        end
    end
end
