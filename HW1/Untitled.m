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

for M = 1:1:9
    %% training stage
    xtrain1 = [];
    for i=0:1:M
        xtrain1 = [xtrain1 xtraina.^(i)];
    end
    w1 = (xtrain1'*xtrain1)\xtrain1'*ttraina;
    E = norm(xtrain1*w1-ttraina)^2/2;
    Erms_training(M,1) = sqrt(2*E/10);
    
    xtrain1 = [];
    for i=0:1:M
        xtrain1 = [xtrain1 xtrainaa.^(i)];
    end
    E = norm(xtrain1*w1-ttrainaa)^2/2;
    Erms_valid(M,1) = sqrt(2*E/5); %Erms with training setxtrain1 = [];
    
    xtrain2 = [];
    for i=0:1:M
        xtrain2 = [xtrain2 xtrainb.^(i)];
    end
    w2 = (xtrain2'*xtrain2)\xtrain2'*ttrainb;
    E = norm(xtrain2*w2-ttrainb)^2/2;
    Erms_training(M,2) = sqrt(2*E/10);
    
    xtrain2 = [];
    for i=0:1:M
        xtrain2 = [xtrain2 xtrainbb.^(i)];
    end
    E = norm(xtrain2*w2-ttrainbb)^2/2;
    Erms_valid(M,2) = sqrt(2*E/5); %Erms with training setxtrain1 = [];
    
    xtrain3 = [];
    for i=0:1:M
        xtrain3 = [xtrain3 xtrainc.^(i)];
    end
    w3 = (xtrain3'*xtrain3)\xtrain3'*ttrainc;
    E = norm(xtrain3*w3-ttrainc)^2/2;
    Erms_training(M,3) = sqrt(2*E/10);
    
    xtrain3 = [];
    for i=0:1:M
        xtrain3 = [xtrain3 xtraincc.^(i)];
    end
    E = norm(xtrain3*w3-ttraincc)^2/2;
    Erms_valid(M,3) = sqrt(2*E/5); %Erms with training set
    %% §ä³Ì¨Îªºmodel
    [trash index(M)]=min(Erms_valid(M,:));
    Erms_train_min_model(M) = Erms_training(M,index(M));
    w = [];
    w = [w1 w2 w3];
    w = w(:,index(M));
    %Erms_train_avg(M) = (Erms_train(M,1)+Erms_train(M,2)+Erms_train(M,3))/3;
    %% test stage
    xtest1 = [];
    for i=0:1:M
        xtest1 = [xtest1 xtest(:).^(i)];
    end
    E = norm(xtest1*w-ttest)^2/2;
    Erms_test(M) = sqrt(2*E/length(ttest)); %Erms with test set
end
%% plot training regression result
figure(1);
hold on
plot(Erms_train_min_model,'-o','LineWidth',1);
xlabel('M');ylabel('Erms');
%title(['M=',num2str(2),' with training set']);
%% plot testing regression result
plot(Erms_test,'-ro','LineWidth',1);
legend('Trainning data','Test data');
hold off
