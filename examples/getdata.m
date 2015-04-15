function getdata(varargin)
opts.dataDir = fullfile('data','SAR') ;
opts.expDir = fullfile('data','SAR-baseline') ;
opts.dataPath = fullfile(opts.expDir, 'Data.mat');
Data = getMnistImdb(opts);
mkdir(opts.expDir) ;
save(opts.dataPath, '-struct', 'Data') ;


function Data = getMnistImdb(opts)
% --------------------------------------------------------------------
maindir = opts.dataDir;
subdir =  dir( maindir );   % ��ȷ�����ļ���
data_train = cell(length(subdir),1);
labels = cell(length(subdir),1);
for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' ) || ...
        isequal( subdir( i ).name, '..' ) || ...
        ~subdir( i ).isdir )   % �������Ŀ¼����
        continue;
    end
     
    subdirpath = fullfile( maindir, subdir( i ).name, '*.tif' );
    images = dir( subdirpath );   % ��������ļ������Һ�׺Ϊtif���ļ�
     
    % ����ÿ��ͼƬ
    imgdata=cell(160,1);
    labels{i}=(i-2)*ones(160,1);
    for j = 1 : length(images)
        imagepath = fullfile( maindir, subdir( i ).name, images( j ).name);
        imgdata{j} = reshape(imread( imagepath ),1,[]); % ������ж�ȡ����
    end
    data_train{i}=cat(1,imgdata{:});
end
data_train=cat(1,data_train{:});
labels=cat(1,labels{:});
indexes = randperm(numel(labels));
data_train = data_train(indexes,:);
labels = labels(indexes);
Data.data = data_train;
Data.labels = labels;
Data.index = indexes;
