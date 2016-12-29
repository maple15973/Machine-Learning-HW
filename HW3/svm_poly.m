function [w,b,sv] = svm(train_data, label, C, tol)
%polynomial kernel
phi_x = [train_data(:,1).^2,sqrt(train_data(:,1).*train_data(:,2)),train_data(:,2).^2];
K = phi_x*phi_x';
%find a and b
[a,b] = smo(K,label',C,tol)
index = find(a>0);
%a is 1*n, turn it to n*1
w = phi_x'*(a'.*label);
sv = phi_x(index,:);
