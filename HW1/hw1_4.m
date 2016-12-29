clc; clear all; close all;
filename1 = 'x3.mat';
filename2 = 't3.mat';
data1 = load('-mat', filename1);
data2 = load('-mat', filename2);
train_x = data1.x3_v2.train_x;
test_x = data1.x3_v2.test_x;
train_y = data2.t3_v2.train_y;
test_y = data2.t3_v2.test_y;
MAX_ORDER=9;
VALID_NUM=3;
TRAIN_SIZE=10;
VALID_SIZE=5;
TEST_SIZE=10;
valid_err=zeros(MAX_ORDER,VALID_NUM);
valid_err_rms=zeros(MAX_ORDER,VALID_NUM);
test_err=zeros(MAX_ORDER,1);
train_err=zeros(MAX_ORDER,1);
train_err_all=zeros(MAX_ORDER,VALID_NUM);
min_index=zeros(MAX_ORDER,1);

%% (a)
%Split training data into 3 validation set
train_X= zeros(TRAIN_SIZE,VALID_NUM);
train_X(:,1) = train_x(1:10);
train_X(:,2) = [train_x(1:5);train_x(11:15)];
train_X(:,3) = train_x(6:15);
train_Y= zeros(TRAIN_SIZE,VALID_NUM);
train_Y(:,1) = train_y(1:10);
train_Y(:,2) = [train_y(1:5);train_y(11:15)];
train_Y(:,3) = train_y(6:15);
valid_X= zeros(VALID_SIZE,VALID_NUM);
valid_X(:,1) = train_x(11:15);
valid_X(:,2) = train_x(6:10);
valid_X(:,3) = train_x(1:5);
valid_Y= zeros(VALID_SIZE,VALID_NUM);
valid_Y(:,1) = train_y(11:15);
valid_Y(:,2) = train_y(6:10);
valid_Y(:,3) = train_y(1:5);

for order=1:9
    w=zeros(order+1,VALID_NUM);
    for i=1:VALID_NUM
        X =[]; validX=[];
        for j=0:1:order
            X = [X train_X(:,i).^(j)];
            validX= [validX valid_X(:,i).^(j)];
        end
        w(:,i) = (X'*X)\X'*train_Y(:,i);
        valid_err(order,i) = 0.5*sum((valid_Y(:,i) - validX*w(:,i)).^2);
        valid_err_rms(order,i)=sqrt(2*valid_err(order,i)/VALID_SIZE);
        train_err_all(order,i)=0.5*sum((train_Y(:,i) - X*w(:,i)).^2);
    end
    [garbadge min_index(order)] = min(valid_err_rms(order,:));
    index = min_index(order);
    test_X=[];
    for j=0:1:order
        test_X = [test_X test_x.^(j)];
    end
    Wmin = w(:,min_index(order));
    test_err(order) = sum((test_y - test_X*Wmin).^2)/2;
    train_err(order)=train_err_all(order,min_index(order));
end
% % Plot the order vs Erms figure
figure();
order=[1:MAX_ORDER];
train_err_rms = sqrt(2*train_err/TRAIN_SIZE);
test_err_rms = sqrt(2*test_err/TEST_SIZE);
plot(order,train_err_rms,'-or');
hold on;
plot(order,test_err_rms,'-ob');
xlabel('order');ylabel('Erms');
legend('Training','Test')
title('Problem a');
%% (b)
LAMBDA_NUM = 26;
order=9;
regularlized_train_erms= zeros(LAMBDA_NUM,1);
lambda = zeros(LAMBDA_NUM,1);
w=zeros(order+1,LAMBDA_NUM);
for i=1:LAMBDA_NUM  
    I= eye(10);
    lambda(i) = exp(6-i); %e^(-5) ~ exp^(-20)
    X=[];test_X=[];
    for j=0:1:order
        X = [X train_x.^(j)];
        test_X=[test_X test_x.^(j)];
    end
    w(:,i) = (X'*X+lambda(i)*I)\(X'*train_y);
    train_E = 0.5*sum((train_y - X*w(:,i)).^2)+0.5*lambda(i)*sum(w(:,i).^2);
    regularlized_train_erms(i) = sqrt(2*train_E/15);
    test_E = 0.5*sum((test_y - test_X*w(:,i)).^2)+0.5*lambda(i)*sum(w(:,i).^2);
    regularlized_test_erms(i) = sqrt(2*test_E/10);
end  
figure();
plot(log(lambda),regularlized_train_erms,'-r');
hold on
plot(log(lambda),regularlized_test_erms,'-b');
xlabel('ln lambda'); ylabel('Erms');
legend('Training','Test')
title('Problem b');
