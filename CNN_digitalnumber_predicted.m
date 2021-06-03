close all
clear
clc

%加载数据
[XTrain,~,YTrain]=digitTrain4DArrayData;
[XValidation,~,YValidation]=digitTest4DArrayData;
%随机显示20个训练图像
numTrainImages=numel(YTrain);
figure
idx=randperm(numTrainImages,20);
for i=1:numel(idx)
    subplot(4,5,i)
    imshow(XTrain(:,:,:,idx(i)))
    drawnow
end
%构建卷积神经网络
layers=[
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer()
    reluLayer
    averagePooling2dLayer(2,'stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer()
    reluLayer
    
    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    regressionLayer];
%配置训练项
miniBatchSize=128;
validationFrequency=floor(numel(YTrain)/miniBatchSize);
options=trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',30,...
    'InitialLearnRate',0.001,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',20,...
    'Shuffle','every-epoch',...
    'ValidationData',{XValidation,YValidation},...
    'ValidationFrequency',validationFrequency,...
    'Plots','training-progress',...
    'Verbose',true);
%训练网络
net=trainNetwork(XTrain,YTrain,layers,options);
%测试与评估
YPredicted=predict(net,XValidation);
predictionError=YValidation-YPredicted;
%计算准确率
thr=10;
numCorrect=sum(abs(predictionError)<thr);
numValidationImages=numel(YValidation);
Accuracy=numCorrect/numValidationImages
%计算RMSE的值
squares=predictionError.^2;
RMSE=sqrt(mean(squares))
