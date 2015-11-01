function solution = estimate_2way_spline(K3d, y, d, G0, c0, b0, lambda0, M)

     % Given the data, lambda0, and M, solve for the estimate. G0, c0, b0 is the starting G, c, b. The solution is arranged as a (n+1) by (D+1) matrix.

n = length(y);
D = d * (d+1) /2;
ycb0 = y - lambda0 * c0 /2 - b0;

warning off;
%options = optimoptions('lsqlin','Display','off');
%theta = lsqlin(G0, ycb0, ones(1,D), M, [], [], zeros(D,1));
[~,theta] = evalc('lsqlin(G0, ycb0, ones(1,D), M, [], [], zeros(D,1));');

Kth = zeros(n,n);
for i = 1:d
   Kth = Kth + theta(i) * K3d(:,:, i);
end
index = d;
for i = 1:(d-1)
     for j = (i+1):d
        index = index + 1;
        Kth = Kth + theta(index) * (K3d(:,:, i) .* K3d(:,:, j));
     end
end

bigKth = [Kth + lambda0 * eye(n), ones(n,1); ones(1,n), 0];
cb = bigKth \ [y;0];
c = cb(1:n);
b = cb(n+1);
for i = 1:d
   G0(:, i) = K3d(:,:,i) * c;
end
index = d;
for i = 1:(d-1)
    for j = (i+1):d
           index = index + 1;
           G0(:, index) = (K3d(:,:,i) .* K3d(:,:,j)) * c;
    end
end

solution = [[G0; theta'], [c; b]];
warning on




