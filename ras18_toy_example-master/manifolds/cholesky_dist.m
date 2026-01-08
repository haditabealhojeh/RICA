function res = cholesky_dist(A,B)
% Affine invariant
% Geodesic: No

% Reference:
% I. L. Dryden, A. Koloydenko, and D. Zhou, "Non-Euclidean Statistics for 
% Covariance Matrices, with Applications to Diffusion Tensor Imaging", The 
% Annals of Ap- plied Statistics, 2009.

res = norm(chol(A) - chol(B),'fro');

end