close all
clear
clc
%加载matlab内部数据
digitDatasetPath=fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
imds=imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
perm=randperm(10000,20);
for i=1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
%划分训练集和测试集
numTrainFiles=750;
[imdsTrain,imdsValidation]=splitEachLabel(imds,numTrainFiles,'randomize');
%构建网络
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
%配置训练选项
options=trainingOptions('sgdm',...
    'InitialLearnRate',0.01,...
    'MaxEpochs',4,...
    'Shuffle','every-epoch',...
    'ValidationData',imdsValidation,...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');
net=trainNetwork(imdsTrain,layers,options);
%将训练好的网络对新的输入图像进行分类并计算准确率
YPred=classify(net,imdsValidation);
YValidation=imdsValidation.Labels;
accuracy=sum(YPred==YValidation)/numel(YValidation)