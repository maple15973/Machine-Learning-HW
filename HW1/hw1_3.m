clear all; close all;
filename = 'data.xlsx';
[num,txt,raw] = xlsread(filename);
train_x = num(1:400,1:4);
train_y = num(1:400,5);
test_x = num(401:500,1:4);
test_y = num(401:500,5);

%% (1)M=2
phi_x = zeros(400,21);  %4*4+4+1
phi_test_x = zeros(100,21);

%train w
for i=1:size(train_x,1)
    order1 = train_x(i,:)';
    order2 = reshape(order1*train_x(i,:),16,1);
    phi_x(i,:) = [1;order1;order2];
end
w= pinv(phi_x'*phi_x)*phi_x'*train_y;
predict=phi_x*w;
%transform test x
for i=1:size(test_x,1)
    test_xx = [1;test_x(i,:)'];
    test_X = test_xx*test_x(i,:);
    phi_test_x(i,:) = [1;reshape(test_X',size(test_X,1)*size(test_X,2),1)];
end
err_train = 0.5*sum((train_y-phi_x*w).^2);
Erms_train_2 = sqrt(2*err_train/400);
err = 0.5*sum((test_y-phi_test_x*w).^2);
Erms_test_2 = sqrt(2*err/100);
%% (1)M=3
phi_x = zeros(400,85);  %4*4*4+4*4+4+1
phi_test_x = zeros(100,85);

%train w
for i=1:size(train_x,1)
    x = [1;train_x(i,:)'];
    X = x*train_x(i,:);
    order1 = train_x(i,:)';
    order2 = reshape(order1*train_x(i,:),16,1);
    order3 = reshape(order2*train_x(i,:),64,1);
    phi_x(i,:) = [1;order1;order2;order3];
end
w= pinv(phi_x'*phi_x)*phi_x'*train_y;

%transform test x
for i=1:size(test_x,1)
    x = [1;test_x(i,:)'];
    X = x*test_x(i,:);
    order1 = test_x(i,:)';
    order2 = reshape(order1*test_x(i,:),16,1);
    order3 = reshape(order2*test_x(i,:),64,1);
    phi_test_x(i,:) = [1;order1;order2;order3];
end
err_train = 0.5*sum((train_y-phi_x*w).^2);
Erms_train_3 = sqrt(2*err_train/400);
err = 0.5*sum((test_y-phi_test_x*w).^2);
Erms_test_3 = sqrt(2*err/100);


%% (2)
error_train=zeros(4,1);
erms_train=zeros(4,1);
error_test=zeros(4,1);
erms_test=zeros(4,1);
for i=1:4   %four columns
    clear separate_train_x separate_test_x;
    separate_train_x = train_x;
    separate_test_x = test_x;
    separate_train_x(:,i)=[]; %delete one column as training data
    separate_test_x(:,i)=[];
    phi_x = zeros(400,40);  %3*3*3+3*3+3+1
    phi_test_x = zeros(100,40);
    %train w
    for j=1:size(separate_train_x,1)
        x = [1;separate_train_x(j,:)'];
        X = x*separate_train_x(j,:);
        order1 = separate_train_x(j,:)';
        order2 = reshape(order1*separate_train_x(j,:),9,1);
        order3 = reshape(order2*separate_train_x(j,:),27,1);
        phi_x(j,:) = [1;order1;order2;order3];
    end
    w= pinv(phi_x'*phi_x)*phi_x'*train_y;
    error_train(i) = 0.5*sum((train_y-phi_x*w).^2);
    erms_train(i) = sqrt(2*error_train(i)/400);
    %transform test x
    for j=1:size(separate_test_x,1)
        x = [1;separate_test_x(j,:)'];
        X = x*separate_test_x(j,:);
        order1 = separate_test_x(j,:)';
        order2 = reshape(order1*separate_test_x(j,:),9,1);
        order3 = reshape(order2*separate_test_x(j,:),27,1);
        phi_test_x(j,:) = [1;order1;order2;order3];
    end
    error_test(i) = 0.5*sum((test_y-phi_test_x*w).^2);
    erms_test(i) = sqrt(2*error_test(i)/100);
end

[garbage index_train]=max(erms_train);
[garbage index_test]=max(erms_test);
%% ANS
disp(['Erms of training set in M=2 is ',num2str(Erms_train_2)]);
disp(['Erms of testing set in M=2 is ',num2str(Erms_test_2)]);
disp(['Erms of training set in M=3 is ',num2str(Erms_train_3)]);
disp(['Erms of testing set in M=3 is ',num2str(Erms_test_3)]);
disp(['The most contributive attribute in training data is ',num2str(index_train)]);
disp(['The most contributive attribute in testing data is ',num2str(index_test)]);


