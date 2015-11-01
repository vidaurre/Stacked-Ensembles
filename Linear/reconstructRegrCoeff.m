function b = reconstructRegrCoeff(beta,models,usedPred)

[nlearn,p] = size(usedPred);
b = zeros(1,p+1);
for s = 1:nlearn
   b([ true usedPred(s,:)]) = b([ true usedPred(s,:)]) + beta(s) * models(s,:);
end
b = b';

end