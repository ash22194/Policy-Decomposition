function e = fastsetequal(a, b)
    % Assumes a and b are arrays without repititions
    if (numel(a)~=numel(b))
        e = false;
        return;
    else
        a = a(:);
        b = b(:);
        e = a==b';
        e = sum(e, 'all')==numel(a);
    end
end