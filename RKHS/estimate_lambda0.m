function lambda0 = estimate_lambda0(K, Y, nfolds,lambda)
% tune lambda0.
if length(lambda)==1, 
    lambda0 = lambda; 
    return
end
folds = cvfolds(Y,'gaussian',nfolds);
N = length(Y);
cvlambda = zeros(length(lambda),length(folds));
for ifold = 1:length(folds)
    J = folds{ifold}; ji=setdiff(1:N,J);
    Ytest = Y(J);
    Ytrain = Y(ji);
    Ntrain = length(Ytrain);
    Ktrain = K(ji,ji);
    for j = 1:length(lambda)
        lambda0 = lambda(j);
        %lambda0 = 2^( - j);
        bigKtrain = [Ktrain + lambda0 * eye(Ntrain), ones(Ntrain,1); ones(1,Ntrain), 0];
        cb = bigKtrain\[Ytrain;0];
        Kpred = K(J,ji);
        prediction = Kpred * cb(1:Ntrain, 1) + cb(Ntrain + 1);
        cvlambda(j,ifold) = (norm(Ytest - prediction))^2;
    end
end
meancvlambda = cvlambda * ones(length(folds), 1) / N;
[~, cviter] = min(meancvlambda);
lambda0 = lambda(cviter); %2^( - cviter);




