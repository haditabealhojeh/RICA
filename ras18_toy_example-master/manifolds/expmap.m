function X = expmap(U,S)
% Exponential map (SPD manifold)
N = size(U,3);
for n = 1:N
	X(:,:,n) = S^.5 * expm(S^-.5 * U(:,:,n) * S^-.5) * S^.5;
end

end
