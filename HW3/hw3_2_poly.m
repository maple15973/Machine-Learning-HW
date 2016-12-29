clear all; clc; close all;
load Iris
C = 10;
tol = 0.001;
train_num = length(trainFeature);
test_num = length(testFeature);
%% Training
%Turn trainFeature and testFeature to polynomial feature space
phi_train_x = [trainFeature(:,1).^2,sqrt(2)*trainFeature(:,1).*trainFeature(:,2),trainFeature(:,2).^2];
phi_test_x = [testFeature(:,1).^2,sqrt(2)*testFeature(:,1).*testFeature(:,2),testFeature(:,2).^2];
%Use SVM train w b support vector
[w12,b12,sv12] = svm(phi_train_x(1:80,:),[ones(40,1);-1*ones(40,1)],C,tol);
[w23,b23,sv23] = svm(phi_train_x(41:120,:),[ones(40,1);-1*ones(40,1)],C,tol);
[w13,b13,sv13] = svm([phi_train_x(1:40,:);phi_train_x(81:120,:)],[ones(40,1);-1*ones(40,1)],C,tol);
w = [w12,w23,w13];
b = [b12,b23,b13];
%Turn sv to origin feature space
supv1 = sqrt([sv12(:,1),sv12(:,3)]);
supv2 = sqrt([sv23(:,1),sv23(:,3)]);
supv3 = sqrt([sv13(:,1),sv13(:,3)]);
%Get each training y
y12_train = phi_train_x*w12+b12;
y23_train = phi_train_x*w23+b23;
y13_train = phi_train_x*w13+b13;
%% Voting
%Voting training data
vote_train = zeros(train_num,3);
vote_train(:,1) = (y12_train>0)+(y13_train>0);
vote_train(:,2) = (y12_train<0)+(y23_train>0);
vote_train(:,3) = (y13_train<0)+(y23_train<0);
%Training data accuacy
[garbage, index] = max(vote_train,[],2);
accuacy_train = 1 - sum(trainLabel~=index)/train_num
%% Testing
%Testing data prediction
y12_test = phi_test_x*w12+b12;
y23_test = phi_test_x*w23+b23;
y13_test = phi_test_x*w13+b13;
%Voting testing data
vote_test = zeros(test_num,3);
vote_test(:,1) = (y12_test>0)+(y13_test>0);
vote_test(:,2) = (y12_test<0)+(y23_test>0);
vote_test(:,3) = (y13_test<0)+(y23_test<0);
%Test data accuacy
[garbage, index] = max(vote_test,[],2);
accuacy_test = 1 - sum(testLabel~=index)/test_num
%% Plot feature
x_min=4;x_max=8;y_min=2;y_max=5;
plot_num=300;
w1_plot = linspace(x_min,x_max,plot_num)';
w2_plot = linspace(y_min,y_max,plot_num)';
plot_feature=[];
for i=1:plot_num
    plot_feature=[plot_feature;repmat(w1_plot(i),plot_num,1),w2_plot];
end
phi_feature = [plot_feature(:,1).^2,sqrt(2)*plot_feature(:,1).*plot_feature(:,2),plot_feature(:,2).^2];
y12_feature = phi_feature*w12+b12;
y23_feature = phi_feature*w23+b23;
y13_feature = phi_feature*w13+b13;
vote_feature = zeros(plot_num*plot_num,3);
vote_feature(:,1) = (y12_feature>0)+(y13_feature>0);
vote_feature(:,2) = (y12_feature<0)+(y23_feature>0);
vote_feature(:,3) = (y13_feature<0)+(y23_feature<0);
[garbage, index_plot] = max(vote_feature,[],2);

%% Plot
% figure;
% test_class1 = phi_test_x(1:10,:); 
% test_class2 = phi_test_x(11:20,:);
% test_class3 = phi_test_x(21:30,:);
% plot(test_class1(:,1),test_class1(:,2),'rx'); hold on;
% plot(test_class2(:,1),test_class2(:,2),'g+'); hold on;
% plot(test_class3(:,1),test_class3(:,2),'b*'); hold on;
% 
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
plot(supv1(:,1),supv1(:,2),'ko'); hold on;
plot(supv2(:,1),supv2(:,2),'ko'); hold on;
plot(supv3(:,1),supv3(:,2),'ko'); hold on;
%find cross point
% mid_point1213=inv([w12';w13'])*[-b12;-b13];
% mid_point1223=inv([w12';w23'])*[-b12;-b23];
% mid_point1323=inv([w13';w23'])*[-b13;-b23];
% plot(mid_point1213(1),mid_point1213(2),'rs'); hold on;
% plot(mid_point1223(1),mid_point1223(2),'gv'); hold on;
% plot(mid_point1323(1),mid_point1323(2),'b^'); hold on;
% line([mid_point1213(1),x_max],[(-b12-w12(1)*mid_point1213(1))/w12(2),(-b12-w12(1)*x_max)/w12(2)],'Color',[0 0 0]);
% line([mid_point1223(1),x_max],[(-b23-w23(1)*mid_point1223(1))/w23(2),(-b23-w23(1)*x_max)/w23(2)],'Color',[0 0 0]);
% line([x_min,mid_point1323(1)],[(-b13-w13(1)*x_min)/w13(2),(-b13-w13(1)*mid_point1323(1))/w13(2)],'Color',[0 0 0]);
% figure;
% x_min=4;x_max=8;y_min=2;y_max=4.5;
% train_class1 = trainFeature(1:40,:); 
% train_class2 = trainFeature(41:80,:);
% train_class3 = trainFeature(81:120,:);
% plot(train_class1(:,1),train_class1(:,2),'rx'); hold on;
% plot(train_class2(:,1),train_class2(:,2),'g+'); hold on;
% plot(train_class3(:,1),train_class3(:,2),'b*'); hold on;
% plot(supv1(:,1),supv1(:,2),'ko'); hold on;
% plot(supv2(:,1),supv2(:,2),'ko'); hold on;
% plot(supv3(:,1),supv3(:,2),'ko'); hold on;
% x = linspace(x_min,x_max)';
% y = linspace(y_min,y_max)';
% I=eye(3);
% for i=1:3
%     cal_min_a =  w(3,i);
%     cal_min_b =  sqrt(2)*x_min*w(2,i);
%     cal_min_c =  x_min^2*w(1,i)+b(i);
%     cal_max_a =  w(3,i);
%     cal_max_b =  sqrt(2)*x_max*w(2,i);
%     cal_max_c =  x_max^2*w(1,i)+b(i);
%     r_min = roots([cal_min_a cal_min_b cal_min_c]);
%     r_max = roots([cal_max_a cal_max_b cal_max_c]);
%     line([x_min,x_max],[max(r_min),max(r_max)],'Color',[0,0,0]);
% end
% axis([x_min,x_max,y_min,y_max]);
% legend('Class1','Class2','Class3','Support vector','Location','northoutside','Orientation','horizontal')
% title('Classify by SVM(One-versus-one) with polynomial kernel')
% xlabel('sepal length');ylabel('sepal width');
% legend('Class1','Class2','Class3','Support vector','Location','northoutside','Orientation','horizontal')
% title('Classify by SVM(One-versus-one) with polynomial kernel')
% xlabel('sepal length');ylabel('sepal width');