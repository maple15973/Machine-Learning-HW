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
train_cell={[],[],[]};
for i=2:151
    compare_text=txt(i,5);
    if i<122
        index=i-1;
        if strcmp(compare_text,'SETOSA')
            train_cell{1} = [train_cell{1};num(index,:)];
            train_class(index)=1;
        elseif strcmp(compare_text,'VIRGINIC')
            train_cell{2} = [train_cell{2};num(index,:)];
            train_class(index)=2;
        elseif strcmp(compare_text,'VERSICOL')
            train_cell{3} = [train_cell{3};num(index,:)];
            train_class(index)=3;
        end
    else
        index=i-121;
        if strcmp(compare_text,'SETOSA')
            test_class(index)=1;
        elseif strcmp(compare_text,'VIRGINIC')
            test_class(index)=2;
        elseif strcmp(compare_text,'VERSICOL')
            test_class(index)=3;
        end
    end
end
N=zeros(3,1);
for i=1:3
    N(i)=length(train_cell{i});
end
%% (1) Training by generative model
pi=zeros(3,1);
for i=1:3
    pi(i) = N(i)/sum(N);
end
train_mean=[mean(train_cell{1})',mean(train_cell{2})',mean(train_cell{3})'];
train_s=zeros(4,4,3);
for i=1:3
    train_s(:,:,i)=(train_cell{i}-(train_mean(:,i)*ones(1,N(i)))')' * (train_cell{i}-(train_mean(:,i)*ones(1,N(i)))')/N(i);
end
S = N(1)/sum(N)*train_s(:,:,1) + N(2)/sum(N)*train_s(:,:,2) + N(3)/sum(N)*train_s(:,:,3);

predict_train_class=zeros(120,1);
predict_test_class=zeros(30,1);
pdf=zeros(3,150);
for i=1:150
    data=num(i,:)';
    for j=1:3
        pdf(j,i)  = mvnpdf(data,train_mean(:,j),S);  % multivariate normal distribution
    end
    if i<121
        [trash, predict_train_class(i)]=max(pdf(:,i));
    else
        [trash, predict_test_class(i-120)]=max(pdf(:,i));
    end
end
train_class_chart = confusionmat(train_class,predict_train_class);
test_class_chart = confusionmat(test_class,predict_test_class);
disp('===========Training by generative model===========');
disp('Classification chart of training data ');
disp(train_class_chart);
disp('Classification chart of testing data ');
disp(test_class_chart);
%% (2) PCA
pca_train_class_chart = zeros(3,3,3);
pca_test_class_chart = zeros(3,3,3);
[vector, ~]=eig(S);
for dim=1:3
    pca_train_cell= {[],[],[]};
    pca_test_SET = [];pca_test_VIR = [];pca_test_VER = [];
    pca_train_data=train_data*vector(:,5-dim:4);
    pca_test_data=test_data*vector(:,5-dim:4);
    pca_predict_train_class=zeros(120,3);
    pca_predict_test_class=zeros(30,3);
    pca_train_cell={[],[],[]};
    pca_test_cell={[],[],[]};
    for i=1:120
        if train_class(i)==1
            pca_train_cell{1}=[pca_train_cell{1};pca_train_data(i,:)];
        elseif train_class(i)==2
            pca_train_cell{2}=[pca_train_cell{2};pca_train_data(i,:)];
        elseif train_class(i)==3
            pca_train_cell{3}=[pca_train_cell{3};pca_train_data(i,:)];
        end
    end
    for i=1:30
        if test_class(i)==1
            pca_test_cell{1}=[pca_test_cell{1};pca_test_data(i,:)];
        elseif test_class(i)==2
            pca_test_cell{2}=[pca_test_cell{2};pca_test_data(i,:)];
        elseif test_class(i)==3
            pca_test_cell{3}=[pca_test_cell{3};pca_test_data(i,:)];
        end
    end
    pca_train_mean=[mean(pca_train_cell{1})',mean(pca_train_cell{2})',mean(pca_train_cell{3})'];
    pca_train_s=zeros(dim,dim,3);
    for i=1:3
        pca_train_s(:,:,i)=(pca_train_cell{i}-(pca_train_mean(:,i)*ones(1,N(i)))')' * (pca_train_cell{i}-(pca_train_mean(:,i)*ones(1,N(i)))')/N(i);
    end
    pca_S = N(1)/sum(N)*pca_train_s(:,:,1) + N(2)/sum(N)*pca_train_s(:,:,2) + N(3)/sum(N)*pca_train_s(:,:,3);
    for i=1:150
        t=[pca_train_data;pca_test_data];
        data=t(i,:)';
        pdf=zeros(3,1);
        for j=1:3
            pdf(j)= mvnpdf(data,pca_train_mean(:,j),pca_S);
        end
        if i<121
            [trash, pca_predict_train_class(i,dim)]=max(pdf);
        else
            [trash, pca_predict_test_class(i-120,dim)]=max(pdf);
        end
    end
    pca_train_class_chart(:,:,dim) = confusionmat(train_class,pca_predict_train_class(:,dim));
    pca_test_class_chart(:,:,dim) = confusionmat(test_class,pca_predict_test_class(:,dim));
end
disp('===========Dimension reducation by PCA===========');
disp('Classification chart of training data');
for i=1:3
    disp(['dim ',num2str(i)]);
    disp(pca_train_class_chart(:,:,i));
end
disp('Classification chart of testing data ');
for i=1:3
    disp(['dim ',num2str(i)]);
    disp(pca_test_class_chart(:,:,i));
end
%% (3) LDA
lda_mean=train_mean;
lda_m = 1/120 * lda_mean * N;
Sw=0;Sb=0;
for i=1:3
    Sw=Sw+train_s(:,:,i) * N(i);
    Sb=Sb+N(i)*(lda_mean(:,i)-lda_m)*(lda_mean(:,i)-lda_m)';
end
[vector, value]=eig(inv(Sw)*Sb);
lda_train_class_chart = zeros(3,3,3);
lda_test_class_chart = zeros(3,3,3);
for dim=1:3
    lda_train_cell{1} = [];lda_train_cell{2} = []; lda_train_cell{3} = [];
    temp=num*vector(:,1:dim);
    lda_train_data=temp(1:120,:);
    lda_test_data=temp(121:150,:);
    lda_train_cell={[],[],[]};
    lda_test_cell={[],[],[]};
    for i=1:120
        if train_class(i)==1
            lda_train_cell{1}=[lda_train_cell{1};lda_train_data(i,:)];
        elseif train_class(i)==2
            lda_train_cell{2}=[lda_train_cell{2};lda_train_data(i,:)];
        elseif train_class(i)==3
            lda_train_cell{3}=[lda_train_cell{3};lda_train_data(i,:)];
        end
    end
    for i=1:30
        if test_class(i)==1
            lda_test_cell{1}=[lda_test_cell{1};lda_test_data(i,:)];
        elseif test_class(i)==2
            lda_test_cell{2}=[lda_test_cell{2};lda_test_data(i,:)];
        elseif test_class(i)==3
            lda_test_cell{3}=[lda_test_cell{3};lda_test_data(i,:)];
        end
    end
    lda_train_mean=[mean(lda_train_cell{1})',mean(lda_train_cell{2})',mean(lda_train_cell{3})'];
    lda_train_s=zeros(dim,dim,3);
    for i=1:3
        lda_train_s(:,:,i)=(lda_train_cell{i}-(lda_train_mean(:,i)*ones(1,N(i)))')' * (lda_train_cell{i}-(lda_train_mean(:,i)*ones(1,N(i)))')/N(i);
    end
    lda_S = N(1)/sum(N)*lda_train_s(:,:,1) + N(2)/sum(N)*lda_train_s(:,:,2) + N(3)/sum(N)*lda_train_s(:,:,3);
    for i=1:150
        data=temp(i,:)';
        pdf=zeros(3,1);
        for j=1:3
            pdf(j)= mvnpdf(data,lda_train_mean(:,j),lda_S);
        end
        if i<121
            [trash,lda_predict_train_class(i,dim)]=max(pdf);
        else
            [trash,lda_predict_test_class(i-120,dim)]=max(pdf);
        end
    end
    lda_train_class_chart(:,:,dim) = confusionmat(train_class,lda_predict_train_class(:,dim));
    lda_test_class_chart(:,:,dim) = confusionmat(test_class,lda_predict_test_class(:,dim));
end
disp('===========Dimension reducation by LDA===========');
disp('Classification chart of training data');
for i=1:3
    disp(['dim ',num2str(i)]);
    disp(lda_train_class_chart(:,:,i));
end
disp('Classification chart of testing data ');
for i=1:3
    disp(['dim ',num2str(i)]);
    disp(lda_test_class_chart(:,:,i));
end
%% (4)Plot
%PCA train
figure;
plot3(pca_train_cell{1}(:,1),pca_train_cell{1}(:,2),pca_train_cell{1}(:,3),'ob');
hold on;
plot3(pca_train_cell{2}(:,1),pca_train_cell{2}(:,2),pca_train_cell{2}(:,3),'og');
hold on;
plot3(pca_train_cell{3}(:,1),pca_train_cell{3}(:,2),pca_train_cell{3}(:,3),'or');
title('PCA training data');grid on;
legend('SET','VIR','VER')
%PCA test
figure;
plot3(pca_test_cell{1}(:,1),pca_test_cell{1}(:,2),pca_test_cell{1}(:,3),'ob');
hold on;
plot3(pca_test_cell{2}(:,1),pca_test_cell{2}(:,2),pca_test_cell{2}(:,3),'og');
hold on;
plot3(pca_test_cell{3}(:,1),pca_test_cell{3}(:,2),pca_test_cell{3}(:,3),'or');
title('PCA testing data');grid on;
legend('SET','VIR','VER')
%LDA train
figure;
plot3(lda_train_cell{1}(:,1),lda_train_cell{1}(:,2),lda_train_cell{1}(:,3),'ob');
hold on;
plot3(lda_train_cell{2}(:,1),lda_train_cell{2}(:,2),lda_train_cell{2}(:,3),'og');
hold on;
plot3(lda_train_cell{3}(:,1),lda_train_cell{3}(:,2),lda_train_cell{3}(:,3),'or');
title('LDA training data');grid on;
legend('SET','VIR','VER')
%LDA test
figure;
plot3(lda_test_cell{1}(:,1),lda_test_cell{1}(:,2),lda_test_cell{1}(:,3),'ob');
hold on;
plot3(lda_test_cell{2}(:,1),lda_test_cell{2}(:,2),lda_test_cell{2}(:,3),'og');
hold on;
plot3(lda_test_cell{3}(:,1),lda_test_cell{3}(:,2),lda_test_cell{3}(:,3),'or');
title('LDA testing data');grid on;
legend('SET','VIR','VER')

