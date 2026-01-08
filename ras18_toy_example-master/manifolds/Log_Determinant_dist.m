function res = Log_Determinant_dist(A,B)
% Jensen-Bregman Log Determinant Divergence
% Geodesic: No

% Reference:
% A. Cherian, S. Sra, A. Banerjee, and N. Papanikolopoulos, “Efficient 
% Similarity Search for Covariance Matrices via the Jensen-Bregman LogDet 
% Divergence”, In ICCV, 2011.

res = sqrt(log(det((A+B)/2))-0.5*log(det(A*B)));

end