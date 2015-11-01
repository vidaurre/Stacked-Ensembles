% macro called from xxxEnsemble.m
samplePreds = ~isfield(options,'usedPred');
if ~isfield(options,'weights'), options.weights = ones(1,p)/p; end % weight for each predictor
if ~exist('j','var'), j = 1; end
if ~exist('npredtosample','var'), npredtosample = round(options.alpha * p); end

if samplePreds
    usedPred{j} = false(nlearn,p);
    for s=1:nlearn
        if options.alpha(j)>0
            if s>2
                nUsed = sum(usedPred{j}(1:s-1,:));
                nUsed = nUsed - min(nUsed) + 1;
            else
                nUsed = zeros(1,p);
            end
            % Sample a random subset of predictors without replacement
            sampling_weights = (options.weights).^nUsed;
            usedPred{j}(s,datasample(1:p,npredtosample(j),'Replace',false,'Weights',sampling_weights)) = true;
        else
            usedPred{j}(s,:) = true;
        end
    end
else
    if iscell(options.usedPred)
        usedPred{j} = options.usedPred{j};
    else
        usedPred{1} = options.usedPred;
    end
end