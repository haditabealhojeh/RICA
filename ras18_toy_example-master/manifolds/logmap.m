function U = logmap(X,S)
% Logarithm map (SPD manifold)
N = size(X,3);
for n = 1:N
	U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
end

end
