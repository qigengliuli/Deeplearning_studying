close all
clear
clc
%%加载图像数据
DataPath=fullfile('F:\','564218-088536-配套资源','MerchData');
unzip('MerchData.zip');
imds=imageDatastore(DataPath,'IncludeSubfolders',true,'LabelSource','foldernames');
%划分验证集和训练集
[imdsTrain,imdsValidation]=splitEachLabel(imds,0.7,'randomized');
%随机显示训练集中的部分图像
numTrainImages=numel(imdsTrain.Labels);
idx=randperm(numTrainImages,16);
figure
for i=1:16
    subplot(4,4,i);
    I=readimage(imdsTrain,idx(i));
    imshow(I)
    drawnow
end

%%加载预训练好的网络
%加载alexnet
net=alexnet;

%%对alexnet改造
%保留alexnet第一层到倒数第三层的结构
layersTransfer=net.Layers(1:end-3);
%确定训练集数据中需要分类的种类
numClasses=numel(categories(imdsTrain.Labels));
%构建新的网络，在保留部分Alexnet网络基础上添加
layers=[layersTransfer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%%调整数据集
%查看网络输入层的大小和通道数
inputSize=net.Layers(1).InputSize;
%将批量训练图像的大小调整为与输入层的大小相同
augimdsTrain=augmentedImageDatastore(inputSize(1:2),imdsTrain);
%将批量验证图像的大小调整为与输入层的大小相同
augimdsValidation=augmentedImageDatastore(inputSize(1:2),imdsValidation);

%%对网络训练
%对训练参数设置
options=trainingOptions('sgdm',...
    'MiniBatchSize',15,...
    'MaxEpochs',10,...
    'InitialLearnRate',0.00005,...
    'Shuffle','every-epoch',...
    'ValidationData',augimdsValidation,...
    'ValidationFrequency',3,...
    'Verbose',true,...
    'Plots','training-progress');
%用训练集对网络进行训练
netTransfer=trainNetwork(augimdsTrain,layers,options);

%%验证并显示结果
%对训练好的网络采用验证集进行验证
[YPred,scoress]=classify(netTransfer,augimdsValidation);
%随机显示验证效果
idx_Validation=randperm(numel(imdsValidation.Files),4);
figure
for i=1:4
    subplot(2,2,i);
    I=readimage(imdsValidation,idx_Validation(i));
    imshow(I)
    label=YPred(idx_Validation(i));
    title(string(label));
end

%%计算分类准确率
YValidation=imdsValidation.Labels;
accuracy=mean(YPred==YValidation)

%%创建并显示混淆矩阵
figure
confusionchart(YValidation,YPred)