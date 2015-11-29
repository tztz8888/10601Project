function [l2dist]= L2Distance(f1,f2)

len= length(f1);
tmp=0;
for i=1: len
    tmp=tmp+ (f1(i)-f2(i))^2;
end
l2dist=sqrt(tmp);
