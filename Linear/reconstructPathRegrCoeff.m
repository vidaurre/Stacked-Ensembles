function B = reconstructPathRegrCoeff(beta,my,models,usedPred)

nlearn = length(usedPred);
p = size(usedPred{1},2);
B = zeros(p+1,nlearn);

for is = 1:nlearn
    B(:,is) = reconstructRegrCoeff(beta{is},models{is},usedPred{is});
end
B(1,:) = B(1,:) + my';

end