clc;close all;
tic;
%% Data preprocessing
filename = 'hiv_data.csv';
[num,txt,raw] = xlsread(filename);
train_num = 2000;
train_x = num(1:train_num,1:8);
tn = num(1:train_num,9);
test_num = 720;
test_x = num(train_num+1:train_num+test_num,1:8);
tn_test = num(train_num+1:train_num+test_num,9);
attribute_num = 8;
%% Training
an = zeros(train_num,1);
mu = 100;
theta = [1;0.5];
ita = ones(attribute_num,1);
%Calculate CN
ker_mat = zeros(train_num,train_num);
for i=1:train_num
    x1= train_x(i,:);
    diff = ones(train_num,1)*x1 - train_x;
    diff = diff.^2;
    ker_mat(i,:) = exp(-0.5 * ita' * diff');
end
ones_mat=ones(train_num,train_num);
ones_vec=ones(train_num,1);
diag_ones = eye(train_num);
C = theta(1)*ker_mat+theta(2)*ones_mat;
anew = Inf*ones(train_num,1);
error = mean(anew - an);
% Train an
%Training Phase
iterate_count=0;
while error>0.01
    iterate_count=iterate_count+1
    %Calculate Wn
    sig=sigmoid(an);
    W = diag_ones .* (sig.*(1 - sig) * ones_vec');
    anew = C*inv(diag_ones+W*C)*(tn - sig + W * an);
    error = sqrt(2*sum((anew - an).^2)/train_num)
    an = anew;
end
%6 1.0867
%7 0.7252
%8 0.3
%% Test
%Calculate k
for i=1:test_num
    x1= test_x(i,:);
    diff = ones(train_num,1)*x1 - train_x;
    diff = diff.^2;
    ker_est_mat(i,:) = exp(-0.5 * ita' * diff');
end
ones_mat_test = ones(test_num,train_num);
K = theta(1)*ker_est_mat+theta(2)*ones_mat_test;
%Estimate test t
sig = sigmoid(an);
mN = K * (tn - sig);
estimate = sigmoid(mN);
estimate = estimate>0.5;
%Calculate accuracy
accuacy = 1 - sum(estimate~=tn_test)/test_num
%accuacy = 0.9034
[result order] = confusionmat(tn_test,+estimate)
toc;