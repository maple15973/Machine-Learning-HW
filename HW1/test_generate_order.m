clear all;
x=[1;2;3;4];
%X=zeros(:,:,3);
% for i=1:3
%     t=x;
%     for j=1:i
%         t=[t,t.^i];
%     end
%     X(:,:,i)=t;
% end
t=x;
for j=2:5
    t=[t,x.^j];
end