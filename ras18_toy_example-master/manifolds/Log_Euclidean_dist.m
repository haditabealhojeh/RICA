function res = Log_Euclidean_dist(A,B)
% Log-Frobenius
% Geodesic: Yes

% Reference:
% V. Arsigny, P. Fillard, X. Pennec, and N. Ayache, “Log-EuclideanMetrics 
% for Fast and Simple Calculus on Diffusion Tensors”, Magnetic Resonance 
% in Medicine, 2006.

%res = norm((logm(B)-logm(A)),'fro')^2;
res = norm((logm(B)-logm(A)),'fro');

end