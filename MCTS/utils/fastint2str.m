function [ o ] = fastint2str( x )
% Converts row vectors of integers to strings
% Works much faster than num2str

maxvalue = max(x(:));
required_digits = ceil(log(double(maxvalue+1))/log(10));
o=zeros(required_digits, size(x,2));%initialize array of required size
for c = size(o,1):-1:1
   o(c,:) = mod(x,10);
   x = (x-o(c,:))/10;
end
o = char(o+'0');
end