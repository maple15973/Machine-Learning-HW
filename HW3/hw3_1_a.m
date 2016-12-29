clc;close all;
filename = 'gp_data.csv';
[num,txt,raw] = xlsread(filename);
train_length=1200;
test_length=400;
train_data=num(1:train_length,:);
test_data=num(1201:1600,:);
beta = 100;
theta = [1;0.5];
ita = [1;1];

%% Prediction
%Calculate CN
error = Inf;
t = train_data(:,3);
beta_mat = beta^(-1)*eye(train_length);
ker_mat =zeros(train_length,train_length);
for i=1:train_length
    x1= train_data(i,1:2);
    diff = ones(train_length,1)*x1 - train_data(:,1:2);
    diff = diff.^2;
    ker_mat(i,:) = exp(-0.5 * ita' * diff');
end
ones_mat=ones(train_length,train_length);
C = theta(1)*ker_mat+theta(2)*ones_mat+beta_mat;
invC = inv(C); 
estimate = zeros(test_length,1);
%Calculate k
for i=1:test_length
    x1 = test_data(i,1:2);
    diff = ones(train_length,1)*x1 - train_data(:,1:2);
    diff = diff.^2;
    k = (theta(1)*exp(-0.5 * ita' * diff')+theta(2))';
    estimate(i) = k'* invC * t;
end
test_t = test_data(:,3);
Erms =  sqrt(2*sum((test_t-estimate).^2)/test_length)
%% Plot
figure;
plot3(test_data(:,1),test_data(:,2),test_t,'*'); hold on
plot3(test_data(:,1),test_data(:,2),estimate,'ro');grid on
legend('ground truth','regression output');
title('Gaussian process')
xlabel('x1(attribute1)');ylabel('x2(attribute2)');zlabel('y');