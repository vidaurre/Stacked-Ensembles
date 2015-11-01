function [Yhat,usedPred] = SmootherEnsemble(X,Y,Xtest,options,LOESS,prevfit)
% Computes an ensemble of local regressors, where locality is understood
% on the basis of the distance of X to Xtest

[N,p] = size(X);
Ntest = size(Xtest,1);

if nargin<6 || isempty(prevfit), warmstart = 0;
else warmstart = size(prevfit.Yhat,2);
end
if ~isfield(options,'nlearn'), options.nlearn = 100; end
if ~isfield(options,'kernel'), options.kernel = 'tricube'; end
if ~isfield(options,'radius'), options.radius = [0.1 0.5 0.99]; end % kernel width
if ~isfield(options,'alpha'), options.alpha = min(p,4); end % proportion of predictors within weak learners
if any(options.alpha>=1), % expressed in terms of absolute no. of predictors
    options.alpha(options.alpha>=1) = options.alpha(options.alpha>=1) / p; 
end 
if any(options.alpha==0), options.alpha(options.alpha==0) = 1; end % all predictors

mx = mean(X);
X = bsxfun(@minus,X,mx);
Xtest = bsxfun(@minus,Xtest,mx);
nlearn = options.nlearn(end); % regardless the path of learners, get'em all
Nalpha = length(options.alpha); 
Nradius = length(options.radius); 
npredtosample = round(options.alpha * p);

usedPred = cell(1,Nalpha); 
Yhat = zeros(Ntest,nlearn,Nradius,Nalpha);


for j=1:Nalpha

    samplePredictors;
    
    if LOESS
        R = 1e-12 * eye(npredtosample(j)+1); R(1,1) = 0; 
    end

    if warmstart>0
        Yhat(:,1:warmstart,:,j) = prevfit.Yhat(:,:,:,j);
    end
    
    for s=warmstart+1:nlearn
        
        Xs = X(:,usedPred{j}(s,:));
        Xstest = Xtest(:,usedPred{j}(s,:));
        D = pdist2(Xs,Xstest); %quickdist

        for r=1:Nradius
            
            if strcmp(options.kernel,'uniform') && options.radius(r)==1 % save time
                wd = ones(N,Ntest);
            else
                wd = kernelf(D,options.kernel,options.radius(r));
            end
            for n=1:Ntest
                if LOESS
                    wd = sqrt(wd);
                    Xws = [ones(N,1) bsxfun(@minus,Xs,Xstest(n,:))];
                    Xws = Xws .* repmat(wd(:,n),1,size(Xws,2));
                    Yw = Y .* wd(:,n);
                    b = (Xws' * Xws + R) \ Xws' * Yw;
                    Yhat(n,s,r,j) = b(1);
                else % Nadaraya-Watson estimator
                    Yhat(n,s,r,j) = sum(Y .* wd(:,n)) / sum(wd(:,n)); 
                end
            end
            
        end
        
    end
    
end
                         