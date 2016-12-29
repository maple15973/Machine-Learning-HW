clear all; clc; close all;
%% Data preprocessing
filename1='kdd99_training_data.csv';
filename2='kdd99_testing_data.csv';
[train_num,train_txt,train_raw] = xlsread(filename1);
[test_num,test_txt,test_raw] = xlsread(filename2);
CLASS_NUM=5;
TEST_NUM=50;
TRAIN_NUM=200;
train_x=train_num(:,1:10);
train_y=zeros(CLASS_NUM,TRAIN_NUM);
for i=1:TRAIN_NUM
    train_y(train_num(i,11)+1,i)=1;
end
err=100; %set err>6 to trigger while 
%% (a)gradient descent algorithm
w_init=zeros(10,CLASS_NUM); %Generate initial w in normal distribution N(0,1)
for i=1:10
    for j=1:CLASS_NUM
        w_init(i,j)=normrnd(0,1);
    end
end
learning_rate=5;
w=w_init;
y=zeros(CLASS_NUM,TRAIN_NUM);
iterative_index_gra=0;
error=[];
while err>6
    iterative_index_gra=iterative_index_gra+1;
    a=w'*train_x';
    for j=1:CLASS_NUM
        y(j,:)=1./sum(exp(a-ones(5,1)*a(j,:)));
        w(:,j) = w(:,j)- learning_rate * ((y(j,:)-train_y(j,:)) *train_x)';
    end
    error(iterative_index_gra)=-sum(sum(train_y.*log(y)));
    err=error(iterative_index_gra);
end
%misclassification rate
predict_class=zeros(TEST_NUM,1);
test_x=test_num(:,1:10);
test_y=test_num(:,11);
a=w'*test_x';
predict_y=zeros(CLASS_NUM,TEST_NUM);
for j=1:CLASS_NUM
    predict_y(j,:)=1./sum(exp(a-ones(5,1)*a(j,:)));
end
for i=1:TEST_NUM
    [trash, index]=max(predict_y(:,i));
    predict_class(i)=index-1;
end
test_class_chart= confusionmat(test_y,predict_class);
misclassification_rate= 1- trace(test_class_chart)/sum(sum(test_class_chart));
%plot
figure;
plot(1:1:iterative_index_gra,error);
xlabel('iteration times');ylabel('error');
title('Gradient descent algorithm');
disp(['Misclassification rate of gradient algorithm: ',num2str(misclassification_rate)]);
%% (b)newton algorithm
err=100; %reset err>6
w_init=zeros(10,CLASS_NUM); %Generate initial w in normal distribution N(0,1)
for i=1:10
    for j=1:CLASS_NUM
        w_init(i,j)=normrnd(0,1);
    end
end
learning_rate=5;
w=w_init;
y=zeros(CLASS_NUM,TRAIN_NUM);
R=zeros(TRAIN_NUM,TRAIN_NUM);
iterative_index_new=0;
lambda=exp(-3);
while err>6
    iterative_index_new=iterative_index_new+1;
    a=w'*train_x';
    for j=1:CLASS_NUM
        y(j,:)=1./sum(exp(a-ones(CLASS_NUM,1)*a(j,:)));
        for k=1:TRAIN_NUM
            R(k,k)=y(j,k)*(1-y(j,k));
        end
        H=train_x'*R*train_x;
        w(:,j) = w(:,j)- lambda * inv(H) * ((y(j,:)-train_y(j,:)) *train_x)';
    end
    error_new(iterative_index_new)=-sum(sum(train_y.*log(y)));
    err=error_new(iterative_index_new);
end
%misclassification rate
predict_class_new=zeros(TEST_NUM,1);
a=w'*test_x';
predict_y_new=zeros(CLASS_NUM,TEST_NUM);
for j=1:CLASS_NUM
    predict_y_new(j,:)=1./sum(exp(a-ones(5,1)*a(j,:)));
end
for i=1:TEST_NUM
    [trash, index]=max(predict_y_new(:,i));
    predict_class_new(i)=index-1;
end
test_class_chart_new= confusionmat(test_y,predict_class_new);
misclassification_rate_new= 1- trace(test_class_chart_new)/sum(sum(test_class_chart_new));
%plot
figure;
plot(1:1:iterative_index_new,error_new);
xlabel('iteration times');ylabel('error');
title('Newton algorithm');
disp(['Misclassification rate of newton algorithm: ',num2str(misclassification_rate)]);