clear all; clc; close all;
load Iris
C = 1000;
tol = 0.001;
train_num = length(trainFeature);
test_num = length(testFeature);
class_num =40;
%% Training
%Use SVM train w b support vector
[w1,b1,sv1] = svm(trainFeature,[ones(40,1);-1*ones(80,1)],C,tol);
[w2,b2,sv2] = svm(trainFeature,[-1*ones(40,1);ones(40,1);-1*ones(40,1)],C,tol);
[w3,b3,sv3] = svm(trainFeature,[-1*ones(80,1);ones(40,1)],C,tol);
%Get each training y
y1_train = trainFeature*w1+b1;
y2_train = trainFeature*w2+b2;
y3_train = trainFeature*w3+b3;
%% Classify
train_y = [y1_train, y2_train, y3_train];
%Training data accuacy
[garbage, index_train] = max(train_y,[],2);
error = trainLabel~=index_train;
accuacy_train_class1 = 1 - sum(error(1:40))/class_num;
accuacy_train_class2 = 1 - sum(error(41:80))/class_num;
accuacy_train_class3 = 1 - sum(error(81:120))/class_num;
accuacy_train = 1 - sum(error)/train_num
%% Testing
%Testing data prediction
y1_test = testFeature*w1+b1;
y2_test = testFeature*w2+b2;
y3_test = testFeature*w3+b3;
%% Classify
test_y = [y1_test, y2_test, y3_test];
%Test data accuacy
[garbage, index_test] = max(test_y,[],2);
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
y1_feature = plot_feature*w1+b1;
y2_feature = plot_feature*w2+b2;
y3_feature = plot_feature*w3+b3;
y_feature = [y1_feature, y2_feature, y3_feature];
[garbage, index_plot] = max(y_feature,[],2);
%% Plot
figure;
x_min=4;x_max=8;y_min=2;y_max=5;
train_class1 = trainFeature(1:40,:); 
train_class2 = trainFeature(41:80,:);
train_class3 = trainFeature(81:120,:);
plot(plot_feature(find(index_plot==1),1),plot_feature(find(index_plot==1),2),'.','Color',[1 0.7 0.7]); hold on;
plot(plot_feature(find(index_plot==2),1),plot_feature(find(index_plot==2),2),'.','Color',[0.7 1 0.7]); hold on;
plot(plot_feature(find(index_plot==3),1),plot_feature(find(index_plot==3),2),'.','Color',[0.7 0.7 1]); hold on;
plot(train_class1(:,1),train_class1(:,2),'rx'); hold on;
plot(train_class2(:,1),train_class2(:,2),'g+'); hold on;
plot(train_class3(:,1),train_class3(:,2),'b*'); hold on;
 
plot(sv1(:,1),sv1(:,2),'ko'); hold on;
plot(sv2(:,1),sv2(:,2),'ko'); hold on;
plot(sv3(:,1),sv3(:,2),'ko'); hold on;
% %find cross point
% mid_point1213=inv([w12';w13'])*[-b12;-b13];
% mid_point1223=inv([w12';w23'])*[-b12;-b23];
% mid_point1323=inv([w13';w23'])*[-b13;-b23];
% plot(mid_point1213(1),mid_point1213(2),'rs'); hold on;
% plot(mid_point1223(1),mid_point1223(2),'gv'); hold on;
% plot(mid_point1323(1),mid_point1323(2),'b^'); hold on;
% line([mid_point1213(1),x_max],[(-b12-w12(1)*mid_point1213(1))/w12(2),(-b12-w12(1)*x_max)/w12(2)],'Color',[0 0 0]);
% line([mid_point1223(1),x_max],[(-b23-w23(1)*mid_point1223(1))/w23(2),(-b23-w23(1)*x_max)/w23(2)],'Color',[0 0 0]);
% line([x_min,mid_point1323(1)],[(-b13-w13(1)*x_min)/w13(2),(-b13-w13(1)*mid_point1323(1))/w13(2)],'Color',[0 0 0]);
% line([x_min,x_max],[(-b1-w1(1)*x_min)/w1(2),(-b1-w1(1)*x_max)/w1(2)],'Color',[0 0 0]);
% line([x_min,x_max],[(-b2-w2(1)*x_min)/w2(2),(-b2-w2(1)*x_max)/w2(2)],'Color',[0 0 0]);
% line([x_min,x_max],[(-b3-w3(1)*x_min)/w3(2),(-b3-w3(1)*x_max)/w3(2)],'Color',[0 0 0]);
axis([x_min,x_max,y_min,y_max]);
legend('Class1','Class2','Class3','Support vector','Location','northoutside','Orientation','horizontal')
title('Classify by SVM(One-versus-all) linear kernel')
xlabel('sepal length');ylabel('sepal width');