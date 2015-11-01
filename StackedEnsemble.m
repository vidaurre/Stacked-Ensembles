function [Yhat,YhatTrain,beta,my,Yhatlearners,usedPred,cvparameters,...
    models,selected,meanCVError,sqerrorTrain,sqerrorTest] = ...
    StackedEnsemble(Y,X,Xtest,options,path,Ytest)

models = [];
if nargin<5, path = 0; end % return the entire path of nlearns? 
if nargin<6, Ytest = []; end
[Ntest,p] = size(Xtest);

if ~isfield(options,'nlearn') && path, options.nlearn = [2:10 20:20:100]; 
elseif ~isfield(options,'nlearn'), options.nlearn = 100;
end
if ~isfield(options,'mix'), options.mix = 'nonnegative'; end
if ~isfield(options,'Nfeatures'), Nfeatures=p;
else Nfeatures = options.Nfeatures; end

if Nfeatures<p && Nfeatures>0,
    sxtr = std(X); dev = zeros(1,p);
    dev(sxtr>0) = abs(Y'*X(:,sxtr>0)) ./ sxtr(sxtr>0);
    [~,selected]=sort(dev);
    selected=selected(end-Nfeatures+1:end);
    X = X(:,selected); Xtest = Xtest(:,selected);
else
    selected = 1:p;
end

Nlearn = length(options.nlearn);

Yhat = zeros(Ntest,Nlearn);

if (any(X(:)<0) || any(X(:)>1) || any(Xtest(:)<0) || any(Xtest(:)>1)) && ...
    (strcmp(options.Learners,'RKHSadd') || strcmp(options.Learners,'RKHS2way'))
    warning('Predictors are to be rescaled to range between 0 and 1'); 
    N = size(X,1);
    Z = [X; Xtest];
    Z = Z - repmat(min(Z),N+Ntest,1);
    Z = Z ./ repmat(max(Z),N+Ntest,1);
    X = Z(1:N,:); Xtest = Z(N+1:N+Ntest,:); clear Z
end

switch options.Learners
    case {'LOESS','NW'}, 
        loess = strcmp(options.Learners,'LOESS');
        [YhatTrain,beta,my,usedPred,cvparameters,meanCVError] = ...
            SmootherEnsembleCV(X,Y,options,path,loess); 
    case 'Linear', 
        [YhatTrain,beta,my,usedPred,cvparameters,meanCVError] = ...
            LinearEnsembleCV(X,Y,options,path); 
    case {'RKHSadd','RKHS2way'},
        interactions = strcmp(options.Learners,'RKHS2way');
        [YhatTrain,beta,my,usedPred,cvparameters,meanCVError] = ...
            RKHSEnsembleCV(X,Y,options,path,interactions);
    case 'Tree',
        [YhatTrain,beta,my,usedPred,cvparameters,meanCVError] = ...
            TreeEnsembleCV(X,Y,options,path); 
end

if path % work out the entire path of nlearn
    if ~isempty(Ytest), sqerrorTest = zeros(Ntest,Nlearn); end
    Yhatlearners = cell(Nlearn,1);    
    path_options = options;
    path_options.alpha = cvparameters.alpha;
    if strcmp(options.Learners,'Linear')
        if cvparameters.alpha>=1
            models = zeros(options.nlearn(end),cvparameters.alpha+1);
        else
            models = zeros(options.nlearn(end),round(cvparameters.alpha * p)+1);
        end
    elseif strcmp(options.Learners,'Tree')
        models = cell(options.nlearn(end),1);
    end
    prevfit = [];
    for is=1:length(options.nlearn)
        path_options.nlearn = options.nlearn(is);  
        switch options.Learners
            case {'LOESS','NW'}
                path_options.radius = cvparameters.radius;
                path_options.usedPred = usedPred;
                Yhatlearners{is} = ...
                    SmootherEnsemble(X,Y,Xtest,path_options,loess,prevfit);
                prevfit = struct('Yhat',Yhatlearners{is});
            case 'Linear'
                path_options.usedPred = usedPred;
                [Yhatlearners{is},~,m] = LinearEnsemble(X,Y,Xtest,path_options,prevfit);
                models(1:options.nlearn(is),:) = m;
                prevfit = struct('Yhat',Yhatlearners{is},'models',m);
            case {'RKHSadd','RKHS2way'}
                path_options.M = cvparameters.M; path_options.lambda0 = cvparameters.lambda0;
                path_options.usedPred = usedPred;
                Yhatlearners{is} = RKHSEnsemble(X,Y,Xtest,path_options,interactions,prevfit);
                prevfit = struct('Yhat',Yhatlearners{is});
            case 'Tree'
                path_options.splits = cvparameters.splits;  
                path_options.usedPred = usedPred;
                [Yhatlearners{is},~,models] = TreeEnsemble(X,Y,Xtest,path_options,prevfit);
                prevfit = struct('Yhat',Yhatlearners{is},'models',models);
        end
                
        Yhat(:,is) = PredictEnsemble(Yhatlearners{is},beta{is},my(is));
        if ~isempty(Ytest), sqerrorTest(:,is) = (Yhat(:,is) - Ytest).^2; end
    end
    sqerrorTrain = (repmat(Y,1,length(options.nlearn)) - YhatTrain).^2;

else % just the best nlearn
   
    options.alpha = cvparameters.alpha; options.nlearn = cvparameters.nlearn;
    switch options.Learners
        case {'LOESS','NW'}
            options.radius = cvparameters.radius;
            options.usedPred = {}; options.usedPred{1} = usedPred;
            Yhatlearners = SmootherEnsemble(X,Y,Xtest,options,loess);
        case 'Linear'
            options.usedPred = {}; options.usedPred{1} = usedPred;
            [Yhatlearners,~,models] = LinearEnsemble(X,Y,Xtest,options);
        case {'RKHSadd','RKHS2way'}
            options.M = cvparameters.M; %options.lambda0 = cvparameters.lambda0;
            options.usedPred = {}; options.usedPred{1} = usedPred;
            Yhatlearners = RKHSEnsemble(X,Y,Xtest,options,interactions);
        case 'Tree'
            options.splits = cvparameters.splits;  
            options.usedPred = {}; options.usedPred{1} = usedPred;
            [Yhatlearners,~,models] = TreeEnsemble(X,Y,Xtest,options);
            models = models{1};
    end
    
    Yhat = PredictEnsemble(Yhatlearners,beta,my);
    if ~isempty(Ytest), sqerrorTest = (Yhat - Ytest).^2; end
    sqerrorTrain = (Y - YhatTrain).^2;
    
end
