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
if ~isfield(options,'maxN'), options.maxN = N; end

mx = mean(X); my = mean(Y); Y = Y - my;
X = bsxfun(@minus,X,mx);
Xtest = bsxfun(@minus,Xtest,mx);
nlearn = options.nlearn(end);
Nalpha = length(options.alpha);
NM = length(options.M);
npredtosample = round(options.alpha * p);
NN = round(N/options.maxN); 
if NN>1, segments = cvfolds(Y,'gaussian',NN); 
else segments = cell(1); segments{1} = 1:N;
end

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
            Lambda0vec = zeros(NN,1);
            for nn=1:NN
                Lambda0vec(nn) = estimate_lambda0(Kth0(segments{nn},segments{nn}), Y(segments{nn}), ...
                    options.CVRKHS, options.lambda0); % N x N matrix inversion (CVRKHS times)
            end
            Lambda0(s,j) = median(Lambda0vec);
        else
            Lambda0(s,j) = options.lambda0;
        end
        
        Yhat_segm = zeros(Ntest,NN,NM);
        
        for nn=1:NN
            Nsegment = length(segments{nn});
            bigKth0 = [Kth0(segments{nn},segments{nn}) + Lambda0(s,j) * ...
                eye(Nsegment), ones(Nsegment,1); ones(1,Nsegment), 0];
            cb0 = bigKth0 \ [Y(segments{nn});0]; % N x N matrix inversion
            c0 = cb0(1:Nsegment);
            b0 = cb0(Nsegment+1);
            G0 = zeros(Nsegment,d);
            for i = 1:d,
                G0(:, i) = K3darray(segments{nn},segments{nn},i) * c0;
            end
            if interactions
                index = d;
                for i1 = 1:(d-1)
                    for i2 = (i1+1):d
                        index = index + 1;
                        G0(:, index) = (K3darray(segments{nn},segments{nn},i1) .* ...
                            K3darray(segments{nn},segments{nn},i2)) * c0;
                    end
                end
            end
            
            % compute splines for each level of smoothness (regularization)
            for im = 1:NM;
                M = options.M(im);
                if interactions % N x N matrix inversions (NM times)
                    fit = estimate_2way_spline(K3darray(segments{nn},segments{nn},:),Y(segments{nn}),d,G0,c0,b0,Lambda0(s,j),M);
                else
                    fit = estimate_add_spline(K3darray(segments{nn},segments{nn},:),Y(segments{nn}),d,G0,c0,b0,Lambda0(s,j),M);
                end
                Kpred = zeros(Ntest, Nsegment);
                theta = fit(Nsegment+1, 1:D);
                for i = 1:d
                    Kpred = Kpred + theta(i) * K3darray(N+1:N+Ntest, segments{nn},i);
                end
                if interactions
                    index = d;
                    for i1 = 1:(d-1)
                        for i2 = (i1+1):d
                            index = index + 1;
                            Kpred = Kpred + theta(index) *  ...
                                (K3darray(N+1:N+Ntest,segments{nn},i1) .* K3darray(N+1:N+Ntest,segments{nn},i2));
                        end
                    end
                end
                Yhat_segm(:,nn,im) = Kpred * fit(1:Nsegment, D+1) + fit(Nsegment+1, D+1);    
            end
        end
        
        for im = 1:NM;
            Yhat(:,s,im,j) = mean(Yhat_segm(:,:,im),2);
        end
            
    end 
                
end

end
                         