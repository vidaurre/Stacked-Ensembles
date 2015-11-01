function [predictedY,beta,my,Dev,fit,selected,alpha,lambda] = ...
    elasticnetcv(Y,X,Xtest,family,parameters)

[N,p]=size(X);
Ntest = size(Xtest,1);

if nargin<3, family = 'gaussian'; end; options.family = family;
if nargin<4, parameters = {}; end
if ~isfield(parameters,'Nfeatures'), Nfeatures=p;
else Nfeatures = parameters.Nfeatures; end
if ~isfield(parameters,'alpha'), alpha = [0.1 0.4 0.6 0.99];
else alpha = parameters.alpha; end
if ~isfield(parameters,'CVscheme'), CVscheme=10;
else CVscheme = parameters.CVscheme; end
if ~isfield(parameters,'nlambda'), nlambda=1000;
else nlambda = parameters.nlambda; end
if ~isfield(parameters,'normalise'), normalise=1;
else normalise = parameters.normalise; end

tmpnm = tempname; mkdir(tmpnm); mkdir(strcat(tmpnm,'/out')); mkdir(strcat(tmpnm,'/params'));

classes = [];
if strcmp(family,'multinomial') % for SubspaceRegr, if Y is vector it has to be {-1,1}
    if size(Y,2)==1,
        classes = unique(Y);
        Y = nets_class_vectomat(Y,classes);
        q = size(Y,2);
    else
        q = size(Y,2);
        classes = 1:q;
    end
    if q>9, error('Too many classes!'); end
end

% standardising
if normalise % do it particularly if method=='ElasticNet'
    sx = std([X; Xtest]); mx=mean([X; Xtest]);
    X = X - repmat(mx,N,1); X = X ./ repmat(sx,N,1);
    if Ntest>0
        Xtest = Xtest - repmat(mx,Ntest,1); Xtest = Xtest ./ repmat(sx,Ntest,1);
    end
end

% Impute missing value with KNN
if any(isnan(X(:))) || any(isnan(Xtest(:))),
    X2 = knnimpute([X;Xtest]);
    X = X2(1:N,:); Xtest = X2(N+1:N+Ntest,:);
    clear X2
end

% pre-kill features  
% they have their own stuff
if Nfeatures<p && Nfeatures>0,
    if strcmp(family,'gaussian')
        sxtr = std(X); dev = zeros(1,p);
        dev(sxtr>0) = abs(Y'*X(sxtr>0)) ./ sx2(sxtr>0);
        [~,selected]=sort(dev);
    else % multinomial
        [~,t] = ttest2classes(x,y);
        [~,selected]=sort(t);
    end
    selected=selected(end-Nfeatures+1:end);
else
    selected = 1:p;
end


if length(alpha)>1 % Do we need to cross-validate the parameters?
    % create the inner CV structure - stratified for family=multinomial
    Qfolds = cvfolds(Y,family,CVscheme);
    Dev = Inf(nlambda,length(alpha));
    Lambda = {};
    
    for ialph = 1:length(alpha)
        
        options.alpha = alpha(ialph);
        if strcmp(family,'gaussian') , QpredictedYp = Inf(N,nlambda);
        elseif strcmp(method,'ElasticNet') && strcmp(family,'multinomial') , QpredictedYp = Inf(N,q,nlambda);
        end
        options.nlambda = nlambda;

        % Inner CV loop
        for Qifold = 1:length(Qfolds)
            QJ = Qfolds{Qifold}; Qji=setdiff(1:N,QJ);
            QX=X(Qji,:);  QY=Y(Qji,:); QXJ=X(QJ,:);
            % center response
            if strcmp(family,'gaussian') 
                Qmy=mean(QY);  QY=QY-Qmy;
            end
            
            options.standardize = false;
            if strcmp(family,'gaussian'), options.intr = false; end
            if Qifold>1, options.lambda = Lambda{ialph};
            elseif isfield(options,'lambda'), options = rmfield(options,'lambda'); end
            fit = nets_glmnet(QX(:,selected),QY,family,0,tmpnm,options);
            if Qifold == 1,
                Lambda{ialph} = fit.lambda;
                options = rmfield(options,'nlambda');
            end
            
            if strcmp(family,'gaussian'),
                QpredictedYp(QJ,1:length(fit.lambda)) = QXJ(:,selected) * fit.beta + repmat(Qmy,length(QJ),length(fit.lambda));
            else % multinomial
                QpredictedYp(QJ,:,1:length(fit.lambda)) = nets_glmnetpredict(fit,QXJ(:,selected),fit.lambda,'response');
            end
        end
        
        % Pick the one with the lowest deviance (=quadratic error for family="gaussian")
        if strcmp(family,'gaussian'), % it's actually N*log(sum.... but it doesn't matter
            err = sum(( QpredictedYp - repmat(Y,1,size(QpredictedYp,2))).^2) / N;
            Dev(1:length(err),ialph) = err';
        else %if strcmp(family,'multinomial'),
            for i=1:size(QpredictedYp,3),
                Dev(i,ialph,irad) = 1 - mean(sum(Y .* QpredictedYp(:,:,i,irad),2)); % accuracy actually
            end
        end
    end
    [~,I] = min(Dev(:));
    [ilamb, ialph] = ind2sub(size(Dev),I);
    alpha = alpha(ialph); options.alpha = alpha;
    lambda = (2:-.1:1)' * Lambda{ialph}(ilamb); % it doesn't like just 1 lambda
    options.lambda = lambda; lambda = lambda(end);
else
    Dev = []; 
    options.alpha = alpha;
end

if strcmp(family,'gaussian'), my = mean(Y); Y=Y-my; end
% run again on the whole data
fit = nets_glmnet(X(:,selected),Y,family,0,tmpnm,options);
beta = fit.beta(:,end);
if Ntest>0
    if strcmp(family,'gaussian'),
        predictedY = Xtest * beta + repmat(my,Ntest,1);
    elseif strcmp(family,'multinomial')
        predictedY = nets_glmnetpredict(fit,Xtest(:,selected),fit.lambda(end),'response');
        predictedY = nets_class_mattovec(predictedY,classes);
    end
else
    predictedY = [];
end

end


