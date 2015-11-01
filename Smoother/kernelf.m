function W = kernelf(D,kernel,radius)
if nargin<2, kernel='tricube'; end
if strcmp(kernel,'uniform') && radius == 1; 
    W = ones(size(D)); return;
end
r = zeros(1,size(D,2));
for j=1:size(D,2)  
    if radius==1,
        r(j) = max(D(:,j));
    else
        d = sort(D(:,j)); r(j) = d(round(radius*length(d))); % quicker than quantile
    end
end
D = D ./ repmat(r,size(D,1),1); 
W = zeros(size(D));
C = D<=1;
if strcmp(kernel,'uniform')
    W(C) = 1;
elseif strcmp(kernel,'tricube')
    W(C) = (1-D(C).^3).^3;
elseif strcmp(kernel,'gaussian')
    W(C) = exp(-(D(C).^2)/2);
elseif strcmp(kernel,'epanechnikov')
    W(C) = (1 - D(C).^2);
elseif strcmp(kernel,'susan')
    W(C) = exp((-D(C)).^6);
end
end
