function [net, info] = cnn_yu(varargin)
% CNN_YU  Demonstrated MatConNet on SAR

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('data','sar') ;
opts.expDir = fullfile('data','yu-baseline') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 2 ;
opts.train.numEpochs = 10 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'para_esti', ...
                           'patchsize', 7, ...
                           'pad', 0, ...
                           'stride', 1);
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,3,1,20, 'single'), ...
                           'biases', zeros(1, 20, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [4 4], ...
                           'stride', 4, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,20,50, 'single'),...
                           'biases', zeros(1,50,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,50,500, 'single'),...
                           'biases', zeros(1,500,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,7, 'single'),...
                           'biases', zeros(1,7,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% Take the mean out and make GPU if needed
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end
% save yu
[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
imgPath = fullfile('data','SAR-baseline');
file=fullfile(imgPath,'Data.mat');


fd=load(file);
im=reshape(permute(fd.data,[2,1]),64,64,[]);
im = single(im);
imdb.images.labels = fd.labels';
imdb.images.data = zeros(64,64,1,numel(fd.labels));
imdb.images.set = ones(1,numel(fd.labels));

% 15% training set for validataion
val_index = randperm(numel(fd.labels));
val_index = val_index(1:ceil(numel(fd.labels)*0.15));

imdb.images.set(val_index) = 3;
imdb.images.data = single(zeros(64,64,1,numel(fd.labels)));

for i = 1 : numel(fd.labels);
    imdb.images.data(:,:,1,i) = im(:,:,i);
end
min(min(imdb.images.data(:,:,1,i)))
imdb.meta.sets = {'train', 'val', 'test'} ;
