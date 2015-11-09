function [Yhat,usedPred,models] = LinearEnsemble(X,Y,Xtest,options,interactions,prevfit)
% Computes an ensemble of linear regressors
 
[N,p] = size(X);
Ntest = size(Xtest,1);

if nargin<6 || isempty(prevfit), warmstart = 0; % when path is being computer (only 1 alpha)
else warmstart = size(prevfit.Yhat,2);
end
if ~isfield(options,'nlearn'), options.nlearn = 100; end
if ~isfield(options,'kernel'), options.kernel = 'tricube'; end
if ~isfield(options,'alpha'), options.alpha = min(p,4); end % proportion of predictors within weak learners
if any(options.alpha>=1), % expressed in terms of absolute no. of predictors
    options.alpha(options.alpha>=1) = options.alpha(options.alpha>=1) / p; 
end 
if any(options.alpha==0), options.alpha(options.alpha==0) = 1; end % all predictors
if ~isfield(options,'maxN'), options.maxN = N; end

mx = mean(X);
X = bsxfun(@minus,X,mx);
Xtest = bsxfun(@minus,Xtest,mx);
nlearn = options.nlearn(end); 
Nalpha = length(options.alpha); 
npredtosample = round(options.alpha * p);
NN = round(N/options.maxN); 
if NN>1, segments = cvfolds(Y,'gaussian',NN); 
else segments = cell(1); segments{1} = 1:N;
end

usedPred = cell(1,Nalpha);
if interactions
    quad_terms = (npredtosample.*(npredtosample+1))/2;
    if Nalpha == 1, models = zeros(nlearn,npredtosample+quad_terms+1);
    else models = cell(1,Nalpha);
    end
else
    quad_terms = zeros(Nalpha,1);
    if Nalpha == 1, models = zeros(nlearn,npredtosample+1);
    else models = cell(1,Nalpha);
    end
end
Yhat = zeros(Ntest,nlearn,Nalpha);


for j=1:Nalpha

    samplePredictors;
    
    if Nalpha>1, models{j} = zeros(nlearn,npredtosample(j)+quad_terms(j)+1); end
    
    if warmstart>0
        Yhat(:,1:warmstart,:,j) = prevfit.Yhat(:,:,:,j);
        if Nalpha == 1, models(1:warmstart,:) = prevfit.models;
        else models{j}(1:warmstart,:) = prevfit.models{j};
        end
    end
    
    for s=warmstart+1:nlearn
        Xs = [ones(N,1) X(:,usedPred{j}(s,:))];
        Xstest = [ones(Ntest,1) Xtest(:,usedPred{j}(s,:))];
        if interactions
            Xs2way = zeros(N,quad_terms(j)); 
            Xstest2way = zeros(Ntest,quad_terms(j)); 
            ind = triu(ones(npredtosample(j)),0)==1;
            for n=1:N 
                mat = X(n,usedPred{j}(s,:))' * X(n,usedPred{j}(s,:));
                Xs2way(n,:) = mat(ind)';
            end
            for n=1:Ntest
                mat = Xtest(n,usedPred{j}(s,:))' * Xtest(n,usedPred{j}(s,:));
                Xstest2way(n,:) = mat(ind)';
            end
            Xs = [Xs Xs2way];
            Xstest = [Xstest Xstest2way];
        end  
        R = 1e-10 * diag([0 (ones(1,size(Xs,2)-1))]);
        Yhat_segm = zeros(Ntest,NN);
        models_segm = zeros(NN,npredtosample(j)+quad_terms(j)+1);
        for nn=1:NN
            Nsegment = length(segments{nn});
            b = (Xs(segments{nn},:)' * Xs(segments{nn},:) + R) \ Xs(segments{nn},:)' * Y(segments{nn});
            models_segm(nn,:) = b';
            Yhat_segm(:,nn) = Xstest * b;
        end
        Yhat(:,s,j) = mean(Yhat_segm,2);
        if Nalpha>1, models{j}(s,:) = mean(models_segm,1);
        else models(s,:) = mean(models_segm,1);
        end
    end
    
end
                         