clc;clear;close all;
load('x3.mat');
load('t3.mat');
xtrain = x3_v2.train_x;
xtraina = xtrain(1:10);xtrainaa = xtrain(11:15);
xtrainb = xtrain(6:15);xtrainbb = xtrain(1:5);
xtrainc = [xtrain(1:5); xtrain(11:15)];xtraincc = xtrain(6:10);

xtest = x3_v2.test_x;

ttrain = t3_v2.train_y;
ttraina = ttrain(1:10);ttrainaa = ttrain(11:15);
ttrainb = ttrain(6:15);ttrainbb = ttrain(1:5);
ttrainc = [ttrain(1:5); ttrain(11:15)];ttraincc = ttrain(6:10);

ttest = t3_v2.test_y;
    
   


xtest1 = [];
for i=0:1:9
    xtest1 = [xtest1 xtest(:).^(i)];
end
for i=1:26
    I = eye(10);
    lambda(i) = exp(6-i); %exp^5~exp-20
    xtrain1 = [];
    for j=0:1:9
        xtrain1 = [xtrain1 xtraina.^(j)];
    end
    w1 = (xtrain1'*xtrain1+lambda(i)*I)\xtrain1'*ttraina;
    E = norm(xtrain1*w1-ttraina)^2/2+0.5*lambda(i)*norm(w1)^2;
    Erms_training(i,1) = sqrt(2*E/10);
    
    xtrain1 = [];
    for j=0:1:9
        xtrain1 = [xtrain1 xtrainaa.^(j)];
    end
    E = norm(xtrain1*w1-ttrainaa)^2/2+0.5*lambda(i)*norm(w1)^2;
    Erms_valid(i,1) = sqrt(2*E/5); %Erms with training setxtrain1 = [];
    
    xtrain2 = [];
    for j=0:1:9
        xtrain2 = [xtrain2 xtrainb.^(j)];
    end
    w2 = (xtrain2'*xtrain2+lambda(i)*I)\xtrain2'*ttrainb;
    E = norm(xtrain2*w2-ttrainb)^2/2+0.5*lambda(i)*norm(w2)^2;
    Erms_training(i,2) = sqrt(2*E/10);
    
    xtrain2 = [];
    for j=0:1:9
        xtrain2 = [xtrain2 xtrainbb.^(j)];
    end
    E = norm(xtrain2*w2-ttrainbb)^2/2+0.5*lambda(i)*norm(w2)^2;
    Erms_valid(i,2) = sqrt(2*E/5); %Erms with training setxtrain1 = [];
    
    xtrain3 = [];
    for j=0:1:9
        xtrain3 = [xtrain3 xtrainc.^(j)];
    end
    w3 = (xtrain3'*xtrain3+lambda(i)*I)\xtrain3'*ttrainc;
    E = norm(xtrain3*w3-ttrainc)^2/2+0.5*lambda(i)*norm(w3)^2;
    Erms_training(i,3) = sqrt(2*E/10);
    
    xtrain3 = [];
    for j=0:1:9
        xtrain3 = [xtrain3 xtraincc.^(j)];
    end
    E = norm(xtrain3*w3-ttraincc)^2/2+0.5*lambda(i)*norm(w3)^2;
    Erms_valid(i,3) = sqrt(2*E/5); %Erms with training set
    
     %% ��̨Ϊ�model
    [trash index]=min(Erms_valid(i,:));
    Erms_train_min_model(27-i) = Erms_training(i,index);
    w = [];
    w = [w1 w2 w3];
    w = w(:,index);
    %% test stage
    E = norm(xtest1*w-ttest)^2/2+0.5*lambda(i)*norm(w)^2;
    Erms_test(i) = sqrt(2*E/length(ttest)); %Erms with test set
end
%% plot training cross validate result
figure(1);
hold on
plot(Erms_train_min_model,'-o','LineWidth',1);
xlabel('ln \lambda');ylabel('Erms');
%% plot testing result
plot(Erms_test,'-ro','LineWidth',1);
legend('Trainning data','Test data');
hold off
