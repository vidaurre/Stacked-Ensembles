function [Yhat,usedPred,models] = TreeEnsemble(X,Y,Xtest,options,prevfit)
% Computes an ensemble of trees

[N,p] = size(X);
Ntest = size(Xtest,1);

if nargin<5 || isempty(prevfit), warmstart = 0;
else warmstart = size(prevfit.Yhat,2);
end
if ~isfield(options,'nlearn'), options.nlearn = 100; end
if ~isfield(options,'splits'), options.splits = 1; end
if ~isfield(options,'alpha'), options.alpha = min(p,4); end % proportion of predictors within weak learners
if any(options.alpha>=1), % expressed in terms of absolute no. of predictors
    options.alpha(options.alpha>=1) = options.alpha(options.alpha>=1) / p; 
end 
if any(options.alpha==0), options.alpha(options.alpha==0) = 1; end % all predictors

mx = mean(X);
X = bsxfun(@minus,X,mx);
Xtest = bsxfun(@minus,Xtest,mx);
nlearn = options.nlearn(end); 
Nalpha = length(options.alpha); 
Nsplits = length(options.splits);
npredtosample = round(options.alpha * p);

usedPred = cell(1,Nalpha);
models = cell(Nalpha);
Yhat = zeros(Ntest,nlearn,Nsplits,Nalpha);

for j=1:Nalpha

    samplePredictors;
    
    models{j} = cell(nlearn,Nsplits);
    
    if warmstart>0
        Yhat(:,1:warmstart,:,j) = prevfit.Yhat(:,:,:,j);
        for s=1:warmstart
            for r=1:Nsplits
                models{j}{s,r} = prevfit.models{s,r};
            end
        end
    end
    
    for s=warmstart+1:nlearn
        Xs = [ones(N,1) X(:,usedPred{j}(s,:))];
        Xstest = [ones(Ntest,1) Xtest(:,usedPred{j}(s,:))];
        for r=1:Nsplits  % Nsplits has no effect unless you use fitrtree
            %spl = options.splits(r);
            % only if matlab version is 2015 (maybe also 2014)
            %models{j}{s,r} = fitrtree(Xs,Y,'MaxNumSplits',spl,'CrossVal','off','Surrogate','off','Prune','off');
            %Yhat(:,s,r,j) = predict(models{j}{s,r},Xstest);
            models{j}{s,r} = RegressionTree.fit(Xs,Y,'MinParent',N,'crossVal','off','Surrogate','off','Prune','off');
            Yhat(:,s,r,j) = predict(models{j}{s,r},Xstest);
        end        
    end   
end
                         