function [YhatTrain,beta,my,usedPred,cvparameters,meanCVError] = ...
    TreeEnsembleCV(X,Y,options,path)
% beta is the mixing coefficients
% alpha is the preferred no. of predictors, one per value of options.nlearn
% nlearn is the number of learners eventually used, one per value of options.nlearn
% usedPred are the sampled predictors, per learner and per value of
%    options.nlearn, for the CV-estiamted value of alpha 

[N,p]=size(X);
cvparameters = struct('splits',[],'alpha',[],'nlearn',[]);

if ~isfield(options,'CVscheme'), CVscheme=10;
else CVscheme = options.CVscheme; end
if ~isfield(options,'alpha'), % no. of predictors within weak learners
    options.alpha = min(p,4); 
end 
if nargin<4, path = 0; end

nlearn = options.nlearn; 
Nlearn = length(nlearn);
Nalpha = length(options.alpha); 
Nsplits = length(options.splits);

Qfolds = cvfolds(Y,'gaussian',CVscheme);
Yhat = zeros(N,nlearn(end),Nsplits,Nalpha);

% Obtain cross-validated learner predictions
for Qifold = 1:length(Qfolds)
    QJ = Qfolds{Qifold}; Qji=setdiff(1:N,QJ);
    QX=X(Qji,:);  QY=Y(Qji,:); QXJ=X(QJ,:);
    if Qifold==1
        [Yhat(QJ,:,:),usedPred] = TreeEnsemble(QX,QY,QXJ,options);
        options.usedPred = usedPred;
    else
        Yhat(QJ,:,:) = TreeEnsemble(QX,QY,QXJ,options);
    end
end

CVError = zeros(N,Nsplits,Nalpha,Nlearn); 
%Beta = zeros(nlearn(end),Nalpha,Nlearn);

% Cross-validate alpha and no. of learners
for Qifold = 1:length(Qfolds)
    QJ = Qfolds{Qifold}; Qji=setdiff(1:N,QJ);
    QY = Y(Qji,:); QYJ = Y(QJ,:);
    for is=1:Nlearn
        s = nlearn(is);
        for j=1:Nalpha
            for r=1:Nsplits
                QFX = Yhat(Qji,1:s,r,j); QFXJ = Yhat(QJ,1:s,r,j);
                %[QYhat,Beta(1:s,r,j,is),Mu(r,j,is)] = elasticnetcv(QY,QFX,QFXJ,'gaussian',optionsEN);
                switch options.mix
                    case 'elasticnet'
                        if ~isfield(options,'alphaEN'), options.alphaEN = 0.1; end
                        if ~isfield(options,'CVEN'), options.CVEN = 10; end
                        optionsEN = struct('alpha',options.alphaEN,'CVscheme',options.CVEN);
                        QYhat = elasticnetcv(QY,QFX,QFXJ,'gaussian',optionsEN);
                    case 'nonnegative'
                        QYhat = nonnegativecv(QY,QFX,QFXJ);
                end
                CVError(QJ,r,j,is) = (QYhat - QYJ).^2;
            end
        end
    end
end

meanCVError = permute(mean(CVError,1),[2 3 4 1]);
[~,I] = min(meanCVError(:));
[isplit,ialph,is] = ind2sub(size(meanCVError),I);
cvparameters.alpha = options.alpha(ialph);
cvparameters.splits = options.splits(isplit);

if path==1 % return the entire path of nlearn
    
    YhatTrain = zeros(N,Nlearn);
    beta = cell(Nlearn,1); my = zeros(Nlearn,1);
    usedPred = usedPred{ialph};
    for is=1:Nlearn
        switch options.mix
            case 'elasticnet'
                [~,beta{is},my(is)] = ...
                    elasticnetcv(Y,Yhat(:,1:nlearn(is),isplit,ialph),[],'gaussian',optionsEN);
            case 'nonnegative'
                [~,beta{is},my(is)] = nonnegativecv(Y,Yhat(:,1:nlearn(is),isplit,ialph),[]);
        end
        cvparameters.nlearn(is) = nlearn(is); % We can't sparsify if path==1
        YhatTrain(:,is) = PredictEnsemble(Yhat(:,1:nlearn(is),isplit,ialph),beta{is},my(is));       
    end

else % cross-validates nlearn and returns the best

    if nlearn(is)>1
        switch options.mix
            case 'elasticnet'
                [~,beta,my] = elasticnetcv(Y,Yhat(:,1:nlearn(is),isplit,ialph),[],'gaussian',optionsEN);
            case 'nonnegative'
                [~,beta,my] = nonnegativecv(Y,Yhat(:,1:nlearn(is),isplit,ialph),[]);
        end
    else
        beta = 1; my = 0;
    end
    ind = beta>0; cvparameters.nlearn = sum(ind);
    usedPred = usedPred{ialph}(1:nlearn(is),:);
    usedPred = usedPred(ind,:);
    beta = beta(ind);
    YhatTrain = PredictEnsemble(Yhat(:,ind,isplit,ialph),beta,my); 
    
end


