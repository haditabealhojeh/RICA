
function res = logmm(X)

[V, D] = eig(X);
res = V * diag(log(diag(D))) * V';

end