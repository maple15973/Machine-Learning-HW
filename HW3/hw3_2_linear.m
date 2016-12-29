clear all; clc; close all;
load Iris
C = 1000;
tol = 0.001;
train_num = length(trainFeature);
test_num = length(testFeature);
%% Training
%Use SVM train w b support vector
[w12,b12,sv12] = svm(trainFeature(1:80,:),[ones(40,1);-1*ones(40,1)],C,tol);
[w23,b23,sv23] = svm(trainFeature(41:120,:),[ones(40,1);-1*ones(40,1)],C,tol);
[w13,b13,sv13] = svm([trainFeature(1:40,:);trainFeature(81:120,:)],[ones(40,1);-1*ones(40,1)],C,tol);
%Get each training y
y12_train = trainFeature*w12+b12;
y23_train = trainFeature*w23+b23;
y13_train = trainFeature*w13+b13;
%% Voting
%Voting training data
vote_train = zeros(train_num,3);
vote_train(:,1) = (y12_train>0)+(y13_train>0);
vote_train(:,2) = (y12_train<0)+(y23_train>0);
vote_train(:,3) = (y13_train<0)+(y23_train<0);
%Training data accuacy
[garbage, index_train] = max(vote_train,[],2);
accuacy_train = 1 - sum(trainLabel~=index_train)/train_num;
%% Testing
%Testing data prediction
y12_test = testFeature*w12+b12;
y23_test = testFeature*w23+b23;
y13_test = testFeature*w13+b13;
%Voting testing data
vote_test = zeros(test_num,3);
vote_test(:,1) = (y12_test>0)+(y13_test>0);
vote_test(:,2) = (y12_test<0)+(y23_test>0);
vote_test(:,3) = (y13_test<0)+(y23_test<0);
%Test data accuacy
[garbage, index_test] = max(vote_test,[],2);
accuacy_test = 1 - sum(testLabel~=index_test)/test_num
%% Plot feature
x_min=4;x_max=8;y_min=2;y_max=5;
plot_num=300;
w1_plot = linspace(x_min,x_max,plot_num)';
w2_plot = linspace(y_min,y_max,plot_num)';
plot_feature=[];
for i=1:plot_num
    plot_feature=[plot_feature;repmat(w1_plot(i),plot_num,1),w2_plot];
end
y12_feature = plot_feature*w12+b12;
y23_feature = plot_feature*w23+b23;
y13_feature = plot_feature*w13+b13;
vote_feature = zeros(plot_num*plot_num,3);
vote_feature(:,1) = (y12_feature>0)+(y13_feature>0);
vote_feature(:,2) = (y12_feature<0)+(y23_feature>0);
vote_feature(:,3) = (y13_feature<0)+(y23_feature<0);
[garbage, index_plot] = max(vote_feature,[],2);

%% Plot
% xrange = linspace(4,7)';
%
% figure;
% test_class1 = testFeature(1:10,:);
% test_class2 = testFeature(11:20,:);
% test_class3 = testFeature(21:30,:);
% plot(test_class1(:,1),test_class1(:,2),'rx'); hold on;
% plot(test_class2(:,1),test_class2(:,2),'g+'); hold on;
% plot(test_class3(:,1),test_class3(:,2),'b*'); hold on;
figure;
plot(plot_feature(find(index_plot==1),1),plot_feature(find(index_plot==1),2),'.','Color',[1 0.7 0.7]); hold on;
plot(plot_feature(find(index_plot==2),1),plot_feature(find(index_plot==2),2),'.','Color',[0.7 1 0.7]); hold on;
plot(plot_feature(find(index_plot==3),1),plot_feature(find(index_plot==3),2),'.','Color',[0.7 0.7 1]); hold on;
train_class1 = trainFeature(1:40,:);
train_class2 = trainFeature(41:80,:);
train_class3 = trainFeature(81:120,:);
plot(train_class1(:,1),train_class1(:,2),'rx'); hold on;
plot(train_class2(:,1),train_class2(:,2),'g+'); hold on;
plot(train_class3(:,1),train_class3(:,2),'b*'); hold on;
plot(sv12(:,1),sv12(:,2),'ko'); hold on;
plot(sv23(:,1),sv23(:,2),'ko'); hold on;
plot(sv13(:,1),sv13(:,2),'ko'); hold on;
axis([x_min,x_max,y_min,y_max]);
legend('Class1','Class2','Class3','Support vector','Location','northoutside','Orientation','horizontal')
title('Classify by SVM(One-versus-one) linear kernel')
xlabel('sepal length');ylabel('sepal width');