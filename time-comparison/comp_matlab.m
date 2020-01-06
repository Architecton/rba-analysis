res = zeros(1, 50);
idx_res = 1;
for idx = 100:100:5000
    data = rand(100, idx);
    target = data(:, 1) > data(:, 2);
    t = cputime;
    relieff(data, target, 5);
    e = cputime-t;
    res(idx_res) = e;
    idx_res = idx_res + 1;
    fprintf("done %d/%d\n", idx_res, length(res));
end
