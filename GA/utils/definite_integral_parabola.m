function integral_val = definite_integral_parabola(Q, a, b)

a = a(:);
b = b(:);

assert(size(Q,1)==size(Q,2) && size(Q,1)==size(a,1) && size(Q,1)==size(b,1), ...
       'Check input sizes!');

n = length(a);
if (n==1)
    integral_val = Q / 3 * (b^3 - a^3);
    return;
end

diff = (b-a);
if (any(diff==0))
    integral_val = 0;
    return;
end
pdiff = prod(diff);
diff2 = (b.^2 - a.^2);
diff3 = (b.^3 - a.^3);

integral_val = (diag(Q)' / 3) * (pdiff ./ diff .* diff3);

Qpairs = nchoosek(1:1:n, 2);
Qindices = (Qpairs(:,1)-1)*n + Qpairs(:,2);

integral_val = integral_val ...
                + 0.5 * Q(Qindices)' * (diff2(Qpairs(:,1)) .* diff2(Qpairs(:,2)) ...
                                        .* (pdiff ./ (diff(Qpairs(:,1)) .* diff(Qpairs(:,2)))));

end
