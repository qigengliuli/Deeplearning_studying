close all
clear
clc
%����matlab�ڲ�����
digitDatasetPath=fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
imds=imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
perm=randperm(10000,20);
for i=1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
%����ѵ�����Ͳ��Լ�
numTrainFiles=750;
[imdsTrain,imdsValidation]=splitEachLabel(imds,numTrainFiles,'randomize');
%��������
layers=[
    imageInputLayer([28 28 1])
    convolution2dLayer([3 3],8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer([3 3],16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%����ѵ��ѡ��
options=trainingOptions('sgdm',...
    'InitialLearnRate',0.01,...
    'MaxEpochs',4,...
    'Shuffle','every-epoch',...
    'ValidationData',imdsValidation,...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');
net=trainNetwork(imdsTrain,layers,options);
%��ѵ���õ�������µ�����ͼ����з��ಢ����׼ȷ��
YPred=classify(net,imdsValidation);
YValidation=imdsValidation.Labels;
accuracy=sum(YPred==YValidation)/numel(YValidation)