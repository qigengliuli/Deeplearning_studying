close all
clear
clc
%%����ͼ������
DataPath=fullfile('F:\','564218-088536-������Դ','MerchData');
unzip('MerchData.zip');
imds=imageDatastore(DataPath,'IncludeSubfolders',true,'LabelSource','foldernames');
%������֤����ѵ����
[imdsTrain,imdsValidation]=splitEachLabel(imds,0.7,'randomized');
%�����ʾѵ�����еĲ���ͼ��
numTrainImages=numel(imdsTrain.Labels);
idx=randperm(numTrainImages,16);
figure
for i=1:16
    subplot(4,4,i);
    I=readimage(imdsTrain,idx(i));
    imshow(I)
    drawnow
end

%%����Ԥѵ���õ�����
%����alexnet
net=alexnet;

%%��alexnet����
%����alexnet��һ�㵽����������Ľṹ
layersTransfer=net.Layers(1:end-3);
%ȷ��ѵ������������Ҫ���������
numClasses=numel(categories(imdsTrain.Labels));
%�����µ����磬�ڱ�������Alexnet������������
layers=[layersTransfer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%%�������ݼ�
%�鿴���������Ĵ�С��ͨ����
inputSize=net.Layers(1).InputSize;
%������ѵ��ͼ��Ĵ�С����Ϊ�������Ĵ�С��ͬ
augimdsTrain=augmentedImageDatastore(inputSize(1:2),imdsTrain);
%��������֤ͼ��Ĵ�С����Ϊ�������Ĵ�С��ͬ
augimdsValidation=augmentedImageDatastore(inputSize(1:2),imdsValidation);

%%������ѵ��
%��ѵ����������
options=trainingOptions('sgdm',...
    'MiniBatchSize',15,...
    'MaxEpochs',10,...
    'InitialLearnRate',0.00005,...
    'Shuffle','every-epoch',...
    'ValidationData',augimdsValidation,...
    'ValidationFrequency',3,...
    'Verbose',true,...
    'Plots','training-progress');
%��ѵ�������������ѵ��
netTransfer=trainNetwork(augimdsTrain,layers,options);

%%��֤����ʾ���
%��ѵ���õ����������֤��������֤
[YPred,scoress]=classify(netTransfer,augimdsValidation);
%�����ʾ��֤Ч��
idx_Validation=randperm(numel(imdsValidation.Files),4);
figure
for i=1:4
    subplot(2,2,i);
    I=readimage(imdsValidation,idx_Validation(i));
    imshow(I)
    label=YPred(idx_Validation(i));
    title(string(label));
end

%%�������׼ȷ��
YValidation=imdsValidation.Labels;
accuracy=mean(YPred==YValidation)

%%��������ʾ��������
figure
confusionchart(YValidation,YPred)