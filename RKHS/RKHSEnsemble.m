function [Yhat,usedPred,Lambda0] = RKHSEnsemble(X,Y,Xtest,options,interactions,prevfit)
% Computes an ensemble of splines

[N,p] = size(X);
Ntest = size(Xtest,1);

if nargin<6 || isempty(prevfit), warmstart = 0;
else warmstart = size(prevfit.Yhat,2);
end
if ~isfield(options,'nlearn'), options.nlearn = 100; end
if ~isfield(options,'alpha'), options.alpha = min(p,4); end % no. of predictors per learners
if any(options.alpha>=1), % expressed in terms of absolute no. of predictors
    options.alpha(options.alpha>=1) = options.alpha(options.alpha>=1) / p; 
end 
if any(options.alpha==0), options.alpha(options.alpha==0) = 1; end % all predictors
if ~isfield(options,'M'), options.M = [10, 25, 35]; end % smoothness parameter
if ~isfield(options,'CVRKHS'), options.CVRKHS = 5; end % CVscheme for RKHS stuff
if ~isfield(options,'lambda0'), options.lambda0 = 2.^(-(1:2:14)); end % CVscheme for RKHS stuff

mx = mean(X); my = mean(Y); Y = Y - my;
X = bsxfun(@minus,X,mx);
Xtest = bsxfun(@minus,Xtest,mx);
nlearn = options.nlearn(end);
Nalpha = length(options.alpha);
NM = length(options.M);
npredtosample = round(options.alpha * p);

usedPred = cell(1,Nalpha);
%models = cell(1,Nalpha);
Yhat = zeros(Ntest,nlearn,NM,Nalpha);
Lambda0 = zeros(nlearn,Nalpha);

for j=1:Nalpha
    
    samplePredictors;
    d = npredtosample(j);
        
    if warmstart>0
        Yhat(:,1:warmstart,:,j) = prevfit.Yhat(:,:,:,j);
        if isfield(prevfit,'Lambda0')
            Lambda0(1:warmstart,j) = prevfit.Lambda0(:,j);
        end
    end
    
    for s=warmstart+1:nlearn
        Xs = X(:,usedPred{j}(s,:));
        Xstest = Xtest(:,usedPred{j}(s,:));
        
        % kernel
        K3darray = zeros(N+Ntest,N+Ntest,d);
        for i = 1:d
            K3darray(:,:,i) = compute_kernel([Xs(:,i);Xstest(:,i)], [Xs(:,i);Xstest(:,i)]);
        end
        if interactions
            D = d * (d+1)/2;
            theta = ones(D,1);
            Kth0 = zeros(N+Ntest,N+Ntest);
            for i = 1:d
                Kth0 = Kth0 + theta(i) * K3darray(:,:, i);
            end
            index = d;
            for i1 = 1:(d-1)
                for i2 = (i1+1):d
                    index = index + 1;
                    Kth0 = Kth0 + theta(index) * (K3darray(:,:, i1) .* K3darray(:,:, i2));
                end
            end
        else
            D = d;
            theta = ones(D,1);
            Kth0 = zeros(N+Ntest,N+Ntest);
            for i = 1:d
                Kth0 = Kth0 + theta(i) * K3darray(:,:, i);
            end
        end
        
        % lambda0 (kind of scaling factor for the penalization)
        if length(options.lambda0)>1
            Lambda0(s,j) = estimate_lambda0(Kth0(1:N,1:N), Y, options.CVRKHS, options.lambda0);
        else
            Lambda0(s,j) = options.lambda0;
        end

        % some stuff you'll need
        bigKth0 = [Kth0(1:N,1:N) + Lambda0(s,j) * eye(N), ones(N,1); ones(1,N), 0];
        cb0 = bigKth0 \ [Y;0];
        c0 = cb0(1:N);
        b0 = cb0(N+1);
        G0 = zeros(N,d);
        for i = 1:d, 
            G0(:, i) = K3darray(1:N,1:N,i) * c0; 
        end
        if interactions
            index = d;
            for i1 = 1:(d-1)
                for i2 = (i1+1):d
                    index = index + 1;
                    G0(:, index) = (K3darray(1:N,1:N,i1) .* K3darray(1:N,1:N,i2)) * c0;
                end
            end
        end
        
        % compute splines for each level of smoothness (regularization)
        for im = 1:NM;
            M = options.M(im);
            if interactions
                fit = estimate_2way_spline(K3darray(1:N,1:N,:),Y,d,G0,c0,b0,Lambda0(s,j),M);
            else
                fit = estimate_add_spline(K3darray(1:N,1:N,:),Y,d,G0,c0,b0,Lambda0(s,j),M);
            end
            Kpred = zeros(Ntest, N);
            theta = fit(N+1, 1:D);
            for i = 1:d
                Kpred = Kpred + theta(i) * K3darray(N+1:N+Ntest, 1:N,i);
            end
            if interactions
                index = d;
                for i1 = 1:(d-1)
                    for i2 = (i1+1):d
                        index = index + 1;
                        Kpred = Kpred + theta(index) *  ...
                            (K3darray(N+1:N+Ntest,1:N,i1) .* K3darray(N+1:N+Ntest,1:N,i2));
                    end
                end
            end
            Yhat(:,s,im,j) = Kpred * fit(1:N, D+1) + fit(N+1, D+1);
        end
            
    end 
                
end
                         