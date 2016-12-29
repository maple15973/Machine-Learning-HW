clear all; close all;
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

%order1
order=1;
w=zeros(order+1,VALID_NUM);
for i=1:VALID_NUM
    X = [ones(TRAIN_SIZE,1),train_X(:,i)];
    w(:,i) = (X'*X)\X'*train_Y(:,i);
    validX = [ones(VALID_SIZE,1),valid_X(:,i)];
    valid_err(order,i) = 0.5*sum((valid_Y(:,i) - validX*w(:,i)).^2);
    valid_err_rms(order,i)=sqrt(2*valid_err(order,i)/VALID_SIZE);
    train_err_all(order,i)=0.5*sum((train_Y(:,i) - X*w(:,i)).^2);
end
[garbadge min_index(order)] = min(valid_err_rms(order,:));
test_X = [ones(TRAIN_SIZE,1),test_x];
test_err(order) = 0.5*sum((test_y - test_X*w(:,min_index(order))).^2);
train_err(order)=train_err_all(order,min_index(order));

%order1
figure();
plot(test_x,test_X*w(:,min_index(order)),'-or');
hold on;
plot(test_x,test_y,'ob');
xlabel('x');ylabel('y');
legend('Estimate y','Test y')
title('M=1');

%order2
order=2;
w=zeros(order+1,VALID_NUM);
for i=1:VALID_NUM
    X = [ones(TRAIN_SIZE,1),train_X(:,i),train_X(:,i).^2];
    w(:,i) = (X'*X)\X'*train_Y(:,i);
    validX = [ones(VALID_SIZE,1),valid_X(:,i),valid_X(:,i).^2];
    valid_err(order,i) = 0.5*sum((valid_Y(:,i) - validX*w(:,i)).^2);
    valid_err_rms(order,i)=sqrt(2*valid_err(order,i)/VALID_SIZE);
    train_err_all(order,i)=0.5*sum((train_Y(:,i) - X*w(:,i)).^2);
end
[garbadge min_index(order)] = min(valid_err_rms(order,:));
test_X = [ones(TRAIN_SIZE,1),test_x,test_x.^2];
test_err(order) = 0.5*sum((test_y - test_X*w(:,min_index(order))).^2);
train_err(order)=valid_err(order,min_index(order));
train_err(order)=train_err_all(order,min_index(order));

%plot
figure();
p=[test_x,test_X*w(:,min_index(order))];
p=sortrows(p);
plot(p(:,1),p(:,2),'-or');
hold on;
plot(test_x,test_y,'ob');
xlabel('x');ylabel('y');
legend('Estimate y','Test y')
title('M=2');

%order3
order=3;
w=zeros(order+1,VALID_NUM);
for i=1:VALID_NUM
    X = [ones(TRAIN_SIZE,1),train_X(:,i),train_X(:,i).^2,train_X(:,i).^3];
    w(:,i) = (X'*X)\X'*train_Y(:,i);
    validX = [ones(VALID_SIZE,1),valid_X(:,i),valid_X(:,i).^2,valid_X(:,i).^3];
    valid_err(order,i) = 0.5*sum((valid_Y(:,i) - validX*w(:,i)).^2);
    valid_err_rms(order,i)=sqrt(2*valid_err(order,i)/VALID_SIZE);
    train_err_all(order,i)=0.5*sum((train_Y(:,i) - X*w(:,i)).^2);
end
[garbadge min_index(order)] = min(valid_err_rms(order,:));
test_X = [ones(TRAIN_SIZE,1),test_x,test_x.^2,test_x.^3];
test_err(order) = 0.5*sum((test_y - test_X*w(:,min_index(order))).^2);
train_err(order)=valid_err(order,min_index(order));
train_err(order)=train_err_all(order,min_index(order));
plotesttimatedata(order,test_x,test_X,w,min_index,test_y);

%order4
order=4;
w=zeros(order+1,VALID_NUM);
for i=1:VALID_NUM
    X = [ones(TRAIN_SIZE,1),train_X(:,i),train_X(:,i).^2,train_X(:,i).^3,train_X(:,i).^4];
    w(:,i) = (X'*X)\X'*train_Y(:,i);
    validX = [ones(VALID_SIZE,1),valid_X(:,i),valid_X(:,i).^2,valid_X(:,i).^3,valid_X(:,i).^4];
    valid_err(order,i) = 0.5*sum((valid_Y(:,i) - validX*w(:,i)).^2);
    valid_err_rms(order,i)=sqrt(2*valid_err(order,i)/VALID_SIZE);
    train_err_all(order,i)=0.5*sum((train_Y(:,i) - X*w(:,i)).^2);
end
[garbadge min_index(order)] = min(valid_err_rms(order,:));
test_X = [ones(TRAIN_SIZE,1),test_x,test_x.^2,test_x.^3,test_x.^4];
test_err(order) = 0.5*sum((test_y - test_X*w(:,min_index(order))).^2);
train_err(order)=valid_err(order,min_index(order));
train_err(order)=train_err_all(order,min_index(order));
plotesttimatedata(order,test_x,test_X,w,min_index,test_y);

%order5
order=5;
w=zeros(order+1,VALID_NUM);
for i=1:VALID_NUM
    X = [ones(TRAIN_SIZE,1),train_X(:,i),train_X(:,i).^2,train_X(:,i).^3,train_X(:,i).^4,train_X(:,i).^5];
    w(:,i) = (X'*X)\X'*train_Y(:,i);
    validX = [ones(VALID_SIZE,1),valid_X(:,i),valid_X(:,i).^2,valid_X(:,i).^3,valid_X(:,i).^4,valid_X(:,i).^5];
    valid_err(order,i) = 0.5*sum((valid_Y(:,i) - validX*w(:,i)).^2);
    valid_err_rms(order,i)=sqrt(2*valid_err(order,i)/VALID_SIZE);
    train_err_all(order,i)=0.5*sum((train_Y(:,i) - X*w(:,i)).^2);
end
[garbadge min_index(order)] = min(valid_err_rms(order,:));
test_X = [ones(TRAIN_SIZE,1),test_x,test_x.^2,test_x.^3,test_x.^4,test_x.^5];
test_err(order) = 0.5*sum((test_y - test_X*w(:,min_index(order))).^2);
train_err(order)=valid_err(order,min_index(order));
train_err(order)=train_err_all(order,min_index(order));
plotesttimatedata(order,test_x,test_X,w,min_index,test_y);

%order6
order=6;
w=zeros(order+1,VALID_NUM);
for i=1:VALID_NUM
    X = [ones(TRAIN_SIZE,1),train_X(:,i),train_X(:,i).^2,train_X(:,i).^3,train_X(:,i).^4,train_X(:,i).^5,train_X(:,i).^6];
    w(:,i) = (X'*X)\X'*train_Y(:,i);
    validX = [ones(VALID_SIZE,1),valid_X(:,i),valid_X(:,i).^2,valid_X(:,i).^3,valid_X(:,i).^4,valid_X(:,i).^5,valid_X(:,i).^6];
    valid_err(order,i) = 0.5*sum((valid_Y(:,i) - validX*w(:,i)).^2);
    valid_err_rms(order,i)=sqrt(2*valid_err(order,i)/VALID_SIZE);
    train_err_all(order,i)=0.5*sum((train_Y(:,i) - X*w(:,i)).^2);
end
[garbadge min_index(order)] = min(valid_err_rms(order,:));
test_X = [ones(TRAIN_SIZE,1),test_x,test_x.^2,test_x.^3,test_x.^4,test_x.^5,test_x.^6];
test_err(order) = 0.5*sum((test_y - test_X*w(:,min_index(order))).^2);
train_err(order)=valid_err(order,min_index(order));
train_err(order)=train_err_all(order,min_index(order));
plotesttimatedata(order,test_x,test_X,w,min_index,test_y);

%order7
order=7;
w=zeros(order+1,VALID_NUM);
for i=1:VALID_NUM
    X = [ones(TRAIN_SIZE,1),train_X(:,i),train_X(:,i).^2,train_X(:,i).^3,train_X(:,i).^4,train_X(:,i).^5,train_X(:,i).^6,train_X(:,i).^7];
    w(:,i) = (X'*X)\X'*train_Y(:,i);
    validX = [ones(VALID_SIZE,1),valid_X(:,i),valid_X(:,i).^2,valid_X(:,i).^3,valid_X(:,i).^4,valid_X(:,i).^5,valid_X(:,i).^6,valid_X(:,i).^7];
    valid_err(order,i) = 0.5*sum((valid_Y(:,i) - validX*w(:,i)).^2);
    valid_err_rms(order,i)=sqrt(2*valid_err(order,i)/VALID_SIZE);
    train_err_all(order,i)=0.5*sum((train_Y(:,i) - X*w(:,i)).^2);
end
[garbadge min_index(order)] = min(valid_err_rms(order,:));
test_X = [ones(TRAIN_SIZE,1),test_x,test_x.^2,test_x.^3,test_x.^4,test_x.^5,test_x.^6,test_x.^7];
test_err(order) = 0.5*sum((test_y - test_X*w(:,min_index(order))).^2);
train_err(order)=valid_err(order,min_index(order));
train_err(order)=train_err_all(order,min_index(order));
plotesttimatedata(order,test_x,test_X,w,min_index,test_y);

%order8
order=8;
w=zeros(order+1,VALID_NUM);
for i=1:VALID_NUM
    X = [ones(TRAIN_SIZE,1),train_X(:,i),train_X(:,i).^2,train_X(:,i).^3,train_X(:,i).^4,train_X(:,i).^5,train_X(:,i).^6,train_X(:,i).^7,train_X(:,i).^8];
    w(:,i) = (X'*X)\X'*train_Y(:,i);
    validX = [ones(VALID_SIZE,1),valid_X(:,i),valid_X(:,i).^2,valid_X(:,i).^3,valid_X(:,i).^4,valid_X(:,i).^5,valid_X(:,i).^6,valid_X(:,i).^7,valid_X(:,i).^8];
    valid_err(order,i) = 0.5*sum((valid_Y(:,i) - validX*w(:,i)).^2);
    valid_err_rms(order,i)=sqrt(2*valid_err(order,i)/VALID_SIZE);
    train_err_all(order,i)=0.5*sum((train_Y(:,i) - X*w(:,i)).^2);
end
[garbadge min_index(order)] = min(valid_err_rms(order,:));
test_X = [ones(TRAIN_SIZE,1),test_x,test_x.^2,test_x.^3,test_x.^4,test_x.^5,test_x.^6,test_x.^7,test_x.^8];
test_err(order) = 0.5*sum((test_y - test_X*w(:,min_index(order))).^2);
train_err(order)=valid_err(order,min_index(order));
train_err(order)=train_err_all(order,min_index(order));
plotesttimatedata(order,test_x,test_X,w,min_index,test_y);

%order9
order=9;
w=zeros(order+1,VALID_NUM);
for i=1:VALID_NUM
    X = [ones(TRAIN_SIZE,1),train_X(:,i),train_X(:,i).^2,train_X(:,i).^3,train_X(:,i).^4,train_X(:,i).^5,train_X(:,i).^6,train_X(:,i).^7,train_X(:,i).^8,train_X(:,i).^9];
    w(:,i) = (X'*X)\X'*train_Y(:,i);
    validX = [ones(VALID_SIZE,1),valid_X(:,i),valid_X(:,i).^2,valid_X(:,i).^3,valid_X(:,i).^4,valid_X(:,i).^5,valid_X(:,i).^6,valid_X(:,i).^7,valid_X(:,i).^8,valid_X(:,i).^9];
    valid_err(order,i) = 0.5*sum((valid_Y(:,i) - validX*w(:,i)).^2);
    valid_err_rms(order,i)=sqrt(2*valid_err(order,i)/VALID_SIZE);
    train_err_all(order,i)=0.5*sum((train_Y(:,i) - X*w(:,i)).^2);
end
[garbadge min_index(order)] = min(valid_err_rms(order,:));
test_X = [ones(TRAIN_SIZE,1),test_x,test_x.^2,test_x.^3,test_x.^4,test_x.^5,test_x.^6,test_x.^7,test_x.^8,test_x.^9];
test_err(order) = 0.5*sum((test_y - test_X*w(:,min_index(order))).^2);
train_err(order)=valid_err(order,min_index(order));
train_err(order)=train_err_all(order,min_index(order));
plotesttimatedata(order,test_x,test_X,w,min_index,test_y);

% figure();
% i=min_index(order);
% X = [ones(TRAIN_SIZE,1),train_X(:,i),train_X(:,i).^2,train_X(:,i).^3,train_X(:,i).^4,train_X(:,i).^5,train_X(:,i).^6,train_X(:,i).^7,train_X(:,i).^8,train_X(:,i).^9];
% p=[train_X(:,i),X*w(:,min_index(order))];
% p=sortrows(p);
% plot(p(:,1),p(:,2),'-or');
% hold on;
% plot(train_X(:,i),train_Y(:,i),'ob');
% xlabel('x');ylabel('y');
% legend('Estimate y','Test y')
% title(['M=',num2str(order)]);
% % Plot the order vs Erms figure
% figure();
% order=[1:MAX_ORDER];
% valid_err_rms = sqrt(2*train_err.^2/VALID_SIZE);
% test_err_rms = sqrt(2*test_err.^2/TEST_SIZE);
% plot(order,valid_err_rms,'-or');
% hold on;
% plot(order,test_err_rms,'-ob');
% xlabel('order');ylabel('Erms');
% legend('Training','Test')
% title('Problem a');
%% (b)
LAMBDA_NUM = 21;
regularlized_train_erms= zeros(LAMBDA_NUM,1);
lambda = zeros(LAMBDA_NUM,1);
for i=1:LAMBDA_NUM  
    I= eye(10);
    lambda(i) = 10^(-19-i); %10^(-20) ~ 10^(-40)
    X = [ones(15,1),train_x,train_x.^2,train_x.^3,train_x.^4,train_x.^5,train_x.^6,train_x.^7,train_x.^8,train_x.^9];
    w = pinv(X'*X-lambda(i)*I)*X'*train_y;
    regularlized_train_erms(i) = sqrt(2*(0.5*sum((train_y - X*w).^2)+0.5*lambda(i)*sum(w.^2))^2/15);
    test_X = [ones(10,1),test_x,test_x.^2,test_x.^3,test_x.^4,test_x.^5,test_x.^6,test_x.^7,test_x.^8,test_x.^9];
    regularlized_test_erms(i) = sqrt(2*(0.5*sum((test_y - test_X*w).^2)+0.5*lambda(i)*sum(w.^2))^2/10);
end  
% figure();
% plot(log10(lambda),regularlized_train_erms,'-r');
% hold on
% plot(log10(lambda),regularlized_test_erms,'-b');
% xlabel('ln lambda'); ylabel('Erms');
% legend('Training','Test')
% title('Problem b');