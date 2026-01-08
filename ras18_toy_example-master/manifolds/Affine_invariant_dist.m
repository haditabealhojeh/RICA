function res = Affine_invariant_dist(A,B)
% Affine invariant
% Geodesic: Yes

% Reference:
% X. Pennec, P. Fillard, and N. Ayache, "A Riemannian Framework for Tensor 
% Computing", IJCV, 2006.

res = norm(logm(A^-.5 * B * A^-.5),'fro');

end