function [w,b,sv] = svm(phi_x, label, C, tol)
K = phi_x*phi_x';
%find a and b
[a,b] = smo(K,label',C,tol);
index = find(a>0);
%a is 1*n, turn it to n*1
w = phi_x'*(a'.*label);
sv = phi_x(index,:);
