function plotesttimatedata(order,test_x,test_X,w,min_index,test_y)
    figure();
    max= max(test_x);
    min= min(test_x);
    lin
    p=[test_x,test_X*w(:,min_index(order))];
    p=sortrows(p);
    plot(p(:,1),p(:,2),'-or');
    hold on;
    plot(test_x,test_y,'ob');
    xlabel('x');ylabel('y');
    legend('Estimate y','Test y')
    title(['M=',num2str(order)]);
end