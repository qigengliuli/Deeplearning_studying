close all
clear
clc

%��������
[XTrain,~,YTrain]=digitTrain4DArrayData;
[XValidation,~,YValidation]=digitTest4DArrayData;
%�����ʾ20��ѵ��ͼ��
numTrainImages=numel(YTrain);
figure
idx=randperm(numTrainImages,20);
for i=1:numel(idx)
    subplot(4,5,i)
    imshow(XTrain(:,:,:,idx(i)))
    drawnow
end
%�������������
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
%����ѵ����
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
%ѵ������
net=trainNetwork(XTrain,YTrain,layers,options);
%����������
YPredicted=predict(net,XValidation);
predictionError=YValidation-YPredicted;
%����׼ȷ��
thr=10;
numCorrect=sum(abs(predictionError)<thr);
numValidationImages=numel(YValidation);
Accuracy=numCorrect/numValidationImages
%����RMSE��ֵ
squares=predictionError.^2;
RMSE=sqrt(mean(squares))
