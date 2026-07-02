% onhw_projection.m
% -------------------------------------------------------------------------
% IMU smart-pen (ballpoint, regular paper) — writer-independent accuracy
% projection. Reads the measured WI learning curve produced by
% make_learning_curve.py and extrapolates expected accuracy as the number of
% enrolled writers grows toward full-dataset scale.
%
% Model (logistic, our own — matches the empirical S-shape of WI scaling: a
% shallow start with few writers, a steep middle, then saturation):
%     acc(W) = L / (1 + exp(-a * (W - w0)))
%   L  = asymptotic accuracy ceiling for this architecture/feature set
%   a  = growth rate; w0 = writer count at the steepest (mid) point
% Parameters are fit by least squares with base fminsearch (no toolboxes),
% with a soft bound keeping L in a sane [60, 95]% range, so this runs unchanged
% in MATLAB or GNU Octave.
%
% Usage:  run onhw_projection   (after make_learning_curve.py has written the CSV)
% -------------------------------------------------------------------------

clear; clc;

csv_path = fullfile('results', 'learning_curve.csv');
if ~exist(csv_path, 'file')
    error('Missing %s — run: python make_learning_curve.py', csv_path);
end

T = csvread(csv_path, 1, 0);          % skip header row
W   = T(:,1);                         % n_writers
N   = T(:,2);                         % n_samples
ACC = T(:,3);                         % WI test accuracy (%)

% ---- fit acc(W) = L/(1+exp(-a*(W-w0))) by least squares ----------------
model = @(p, w) p(1) ./ (1 + exp(-max(p(2),1e-6) .* (w - p(3))));
% objective: SSE + soft penalty keeping the ceiling L within [60, 95]%
penalty = @(p) 1e4 * (max(0, 60 - p(1))^2 + max(0, p(1) - 95)^2);
sse   = @(p) sum((model(p, W) - ACC).^2) + penalty(p);
p0    = [75, 0.25, 17];               % init: ceiling ~75%, moderate rate, mid ~17 writers
opts  = optimset('Display', 'off', 'MaxFunEvals', 20000, 'MaxIter', 20000);
p     = fminsearch(sse, p0, opts);
L = p(1); a = p(2); w0 = p(3);

% ---- projection targets -------------------------------------------------
% Full OnHW-chars: 119 writers total; with a 60/20/20 writer split ~71 train.
W_full_train = 71;
samples_per_writer = mean(N ./ W);
acc_current = model(p, max(W));
acc_full    = model(p, W_full_train);
acc_ceiling = L;

% published reference: 52-class WI baseline (Ott et al., IMWUT 2020 ~64%)
ref_baseline = 64.0;

% ---- plot ---------------------------------------------------------------
ww = linspace(0, max(W_full_train*1.1, 80), 300);
figure('Color', 'w', 'Position', [100 100 820 520]); hold on; grid on;

ylim([0 100]); xlim([0 max(ww)]);            % set limits first so refs span fully
plot(ww, model(p, ww), 'b-', 'LineWidth', 2);                       % fitted curve
plot(W, ACC, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', [0 0.6 0]);  % measured pts
plot(W_full_train, acc_full, 'rp', 'MarkerSize', 16, ...
     'MarkerFaceColor', 'r');                                       % projection
yline_compat(ref_baseline, '--', ...
    sprintf('OnHW 52-class baseline (~%.0f%%)', ref_baseline));
yline_compat(acc_ceiling, ':', ...
    sprintf('fitted ceiling L = %.1f%%', acc_ceiling));

xlabel('Number of enrolled (training) writers');
ylabel('Writer-independent character accuracy (%)');
title({'IMU Smart-Pen — Writer-Independent Accuracy Projection', ...
       'CNN+BiLSTM on regular-paper ballpoint IMU data'});
legend({'fitted saturating model', 'measured (this repo subset)', ...
        sprintf('projected @ %d writers = %.1f%%', W_full_train, acc_full)}, ...
       'Location', 'southeast');

% ---- console summary ----------------------------------------------------
fprintf('\n=== IMU Smart-Pen WI Accuracy Projection ===\n');
fprintf('Fitted model: acc(W) = %.1f / (1 + exp(-%.3f * (W - %.1f)))\n', L, a, w0);
fprintf('Samples / writer (measured): %.0f\n', samples_per_writer);
fprintf('Current   (%2d writers): %.1f%%\n', max(W), acc_current);
fprintf('Projected (%2d writers, full-scale split): %.1f%%\n', W_full_train, acc_full);
fprintf('Asymptotic ceiling L      : %.1f%%\n', acc_ceiling);
fprintf('Reference 52-class baseline: %.1f%%\n', ref_baseline);

if ~exist('results', 'dir'); mkdir('results'); end
print(gcf, fullfile('results', 'onhw_projection.png'), '-dpng', '-r150');
fprintf('\nSaved results/onhw_projection.png\n');


function yline_compat(yval, style, label)
% Draw a horizontal reference line + text that works in MATLAB and Octave.
    xl = xlim;
    plot(xl, [yval yval], style, 'Color', [0.4 0.4 0.4], 'LineWidth', 1.2);
    text(xl(1) + 0.02*(xl(2)-xl(1)), yval + 2, label, ...
         'Color', [0.3 0.3 0.3], 'FontSize', 9);
end
