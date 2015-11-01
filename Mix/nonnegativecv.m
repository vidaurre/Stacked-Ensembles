function [predictedY,beta,my] = nonnegativecv(Y,X,Xtest)
my = mean(Y);
Y = Y - mean(Y);
beta = lsqnonneg(X,Y);
Ntest = size(Xtest,1);
if Ntest>0
    predictedY = Xtest * beta + repmat(my,Ntest,1);
else
    predictedY = [];
end
end
