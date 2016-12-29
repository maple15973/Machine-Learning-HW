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
%% (2) PCA
[U S V] = svd(num);
test_SET = [];test_VIR = [];test_VER = [];
pca_predict_train_class=zeros(120,3);
pca_predict_test_class=zeros(30,3);
pca_train_class_chart = zeros(3,3,3);
pca_test_class_chart = zeros(3,3,3);
for dim=1:3
    pca_train_SET = [];pca_train_VIR = [];pca_train_VER = [];
    temp = U*S(:,1:dim);
    pca_train_data=temp(1:120,:);   
    pca_test_data=temp(121:150,:);
    for i=1:120
        if train_class(i)==1
            pca_train_SET=[pca_train_SET;pca_train_data(i,:)];
        elseif train_class(i)==2
            pca_train_VIR=[pca_train_VIR;pca_train_data(i,:)];
        elseif train_class(i)==3
            pca_train_VER=[pca_train_VER;pca_train_data(i,:)];
        end
    end
    pca_train_mean=zeros(dim,3);
    pca_train_mean(:,1)=mean(pca_train_SET)';
    pca_train_mean(:,2)=mean(pca_train_VIR)';
    pca_train_mean(:,3)=mean(pca_train_VER)';
    pca_train_s=zeros(dim,dim,3);
    pca_train_s(:,:,1)=(pca_train_SET-(pca_train_mean(:,1)*ones(1,N(1)))')' * (pca_train_SET-(pca_train_mean(:,1)*ones(1,N(1)))')/N(1);
    pca_train_s(:,:,2)=(pca_train_VIR-(pca_train_mean(:,2)*ones(1,N(2)))')' * (pca_train_VIR-(pca_train_mean(:,2)*ones(1,N(2)))')/N(2);
    pca_train_s(:,:,3)=(pca_train_VER-(pca_train_mean(:,3)*ones(1,N(3)))')' * (pca_train_VER-(pca_train_mean(:,3)*ones(1,N(3)))')/N(3);
    pca_S = N(1)/sum(N)*pca_train_s(:,:,1) + N(2)/sum(N)*pca_train_s(:,:,2) + N(3)/sum(N)*pca_train_s(:,:,3);
    
    for i=1:120
        data=pca_train_data(i,:)';
        pdf=zeros(3,1);
        for j=1:3
           pdf(j)= mvnpdf(data,pca_train_mean(:,j),pca_S);
        end
        [trash pca_predict_train_class(i,dim)]=max(pdf);
    end
    pca_train_class_chart(:,:,dim) = confusionmat(train_class,pca_predict_train_class(:,dim));
    for i=1:30
        data=pca_test_data(i,:)';
        pdf=zeros(3,1);
        for j=1:3
           pdf(j)= mvnpdf(data,pca_train_mean(:,j),pca_S);
        end
        [trash pca_predict_test_class(i,dim)]=max(pdf);
    end
    pca_test_class_chart(:,:,dim) = confusionmat(test_class,pca_predict_test_class(:,dim));
end