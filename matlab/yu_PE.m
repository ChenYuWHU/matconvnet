function output = yu_PE(input, patchsize, pad, stride)
% parameter estimation function
%   input:      raw data, width == height
%   patchsize:  the size of patch, width == height == patchsize
%   pad:        padding number, 0 in default
%   stride:     stride number, stride_width == stride_height, 1 in default
%   
%   note: this layer is designed for first layer in Convnet with gray image
%           input.

%%%%%%%%%%%%%% parameters option %%%%%%%%%%%%%%%%
%   type and parameter:   a string, indicate the distribution name. 
%       e.g. {'Lognormal', 'mu'}
para = {%'Lognormal' ,	'mu';...
%         'Lognormal' ,	'sigma';...
%         'Weibull'   ,	'A';...
%         'Weibull'   ,   'B';...
%         'Nakagami'  ,   'mu';...
        'Nakagami'  ,   'omega'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% init parameters
if (nargin <= 3)
    stride = 1;
    if (nargin <= 2)
        pad = 0;
    end
end
input = input - min(min(min(min(input))));
num = size(input,4);
imsize = size(input,1);
paranum = size(para,1);

%% padding
data = zeros(imsize+pad*2, imsize+pad*2, 1, num);
for i = 1:num
    data(:,:,1,i) = [zeros(pad,imsize+pad*2);...
        [zeros(imsize, pad), input(:,:,1,i), zeros(imsize, pad)];...
        zeros(pad,imsize+pad*2)];
end
data = data + 1;
%% parameter estimation
outputsize = (imsize + pad*2 - patchsize) / stride + 1;
output = zeros(outputsize, outputsize, paranum, num);
for n = 1:num
    for p = 1:paranum
        for h = 1:outputsize
            for w = 1:outputsize
                I = data((h-1)*stride+1:h*stride+patchsize-1,...
                    (w-1)*stride+1:w*stride+patchsize-1, 1, n);
                I =  reshape(I,[],1);
                output(h,w,p,n) = getfield(fitdist(I,para{p,1}),para{p,2});
            end
        end
    end
    output(:,:,p,n) = output(:,:,p,n) / max(max(output(:,:,p,n)));
end
output = single(output);
