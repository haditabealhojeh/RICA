function g = geodesic(U,S,t)
Shp = S^.5;
Smp = S^-.5;
for i=1:length(t)
	g(:,:,i) = Shp * expm(t(i) .* Smp * U * Smp) * Shp;
end

end