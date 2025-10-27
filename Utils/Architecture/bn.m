function layer = bn(num_features, name)
%layer = batchNormalizationLayer("NumChannels", num_features, "Name", 'batchnorm');
layer = batchNormalizationLayer("Name", [name]);  %matlab troba els valors de bn automaticament
%layer = batchNormalizationLayer("Name", [name 'batchnorm']);  %matlab troba els valors de bn automaticament
end