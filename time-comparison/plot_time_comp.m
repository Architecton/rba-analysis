function [] = plot_time_comp(rba_name)
    % Plot results of comparing running times for two different
    % implementations of Relief-based algorithms. The parameter rba_name
    % specifies the name of the Relief-based algorithm being evaluated.

    % Parse data.
    loaded_time1 = load(sprintf('results/time1_%s.mat', rba_name));
    time1 = loaded_time1.time1;
    loaded_time2 = load(sprintf('results/time2_%s.mat', rba_name));
    time2 = loaded_time2.time2;

    % Plot running times.
    figure(1); hold on;
    plot(1:size(time1), time1, 'LineWidth', 2);
    plot(1:size(time2), time2, 'LineWidth', 2);
    xlabel('sample dimensionality');
    ylabel('time');

    % Set legend.
    legend(sprintf('%s (skrelief)', rba_name), sprintf('%s (scikit-rebate)', rba_name));
    
end