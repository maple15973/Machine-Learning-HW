clear all; clc; close all;
%% Data preprocessing
filename = 'Irisdat .xls';
[num,txt,raw] = xlsread(filename);
train_SET = [];train_VIR = [];train_VER = [];
test_SET = [];test_VIR = [];test_VER = [];
train_class=zeros(120,1);
test_class=zeros(30,1);
train_data= num(1:120,:);
test_data= num(121:150,:);
for i=2:121
    index=i-1;
    compare_text=txt(i,5);
    if strcmp(compare_text,'SETOSA')
        train_SET = [train_SET;num(index,:)];
        train_class(index)=1;
    elseif strcmp(compare_text,'VIRGINIC')
        train_VIR = [train_VIR;num(index,:)];
        train_class(index)=2;
    elseif strcmp(compare_text,'VERSICOL')
        train_VER = [train_VER;num(index,:)];
        train_class(index)=3;
    end
end   
for i=122:151
    index=i-121;
    compare_text=txt(i,5);
    if strcmp(compare_text,'SETOSA')
        test_SET = [test_SET;num(index,:)];
        test_class(index)=1;
    elseif strcmp(compare_text,'VIRGINIC')
        test_VIR = [test_VIR;num(index,:)];
        test_class(index)=2;
    elseif strcmp(compare_text,'VERSICOL')
        test_VER = [test_VER;num(index,:)];
        test_class(index)=3;
    end
end  
N=zeros(3,1); N(1)= length(train_SET); N(2)= length(train_VIR); N(3)= length(train_VER);
%% (1) Training by generative model
pi=zeros(3,1);
for i=1:3
    pi(i) = N(i)/sum(N);
end
train_mean=zeros(4,3);
train_mean(:,1)=mean(train_SET)';
train_mean(:,2)=mean(train_VIR)';
train_mean(:,3)=mean(train_VER)';
train_s=zeros(4,4,3);
train_s(:,:,1)=(train_SET-(train_mean(:,1)*ones(1,N(1)))')' * (train_SET-(train_mean(:,1)*ones(1,N(1)))')/N(1);
train_s(:,:,2)=(train_VIR-(train_mean(:,2)*ones(1,N(2)))')' * (train_VIR-(train_mean(:,2)*ones(1,N(2)))')/N(2);
train_s(:,:,3)=(train_VER-(train_mean(:,3)*ones(1,N(3)))')' * (train_VER-(train_mean(:,3)*ones(1,N(3)))')/N(3);
S = N(1)/sum(N)*train_s(:,:,1) + N(2)/sum(N)*train_s(:,:,2) + N(3)/sum(N)*train_s(:,:,3);

predict_train_class=zeros(120,1);
predict_test_class=zeros(30,1);
for i=1:150
    data=num(i,:)';
        pdf=zeros(3,1);
        for j=1:3
           pdf(j)  = mvnpdf(data,train_mean(:,j),S);
        end
    if i<121
        [trash predict_train_class(i)]=max(pdf);
    else
        [trash predict_test_class(i-120)]=max(pdf);
    end 
end
train_class_chart = confusionmat(train_class,predict_train_class);
test_class_chart = confusionmat(test_class,predict_test_class);
