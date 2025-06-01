clc; clear all;
img1 = imread("C:\Users\ashwi\Downloads\Lena (1).TIF");
[h w] = size(img1);

img = im2double((imread("C:\Users\ashwi\Downloads\Lena (1).TIF")));
img = padarray(img, [64 - mod(size(img,1),64), 64 - mod(size(img,2),64)], 'replicate', 'post');
[H, W] = size(img);
recon_img = zeros(H, W);

% Parameters
Qstep = 10;
lambda = 0.85 * 2^((22 - 12)/3);

% Process each 64x64 CTU
for i = 1:64:H
    for j = 1:64:W
        block = img(i:i+63, j:j+63);
        recon_ctu = partition_ctu_recursive(block, img, i, j, 64, Qstep, lambda);
        recon_img(i:i+63, j:j+63) = recon_ctu;
    end
end

recon_img = recon_img(1:h, 1:w);

imshow(recon_img);
title('Reconstructed Intra Image (Clipped)');










function [pred_blocks] = hevc_intra_predict(top_ref, left_ref, blockSize)
% top_ref: 1x(2*blockSize+1) including top-left
% left_ref: (2*blockSize+1)x1 including top-left
% blockSize: block size (e.g., 4, 8)

% Output: pred_blocks(:,:,1:35)

pred_blocks = zeros(blockSize, blockSize, 35);

% Planar mode (0)
for i = 1:blockSize
    for j = 1:blockSize
        hor = (blockSize - j) * left_ref(i+1) + j * left_ref(blockSize+1);
        ver = (blockSize - i) * top_ref(j+1) + i * top_ref(blockSize+1);
        pred_blocks(i,j,1) = (hor + ver + blockSize) / (2 * blockSize);
    end
end

% DC mode (1)
dc_val = mean([top_ref(2:blockSize+1), left_ref(2:blockSize+1)']);
pred_blocks(:,:,2) = repmat(dc_val, blockSize, blockSize);

% Angular offsets as per HEVC spec (Table 8-2)
angles = [ ...
    32, 26, 21, 17, 13, 9, 5, 2, 0, ...
    178, 175, 171, 167, 163, 159, 154, ...
    148, 144, 139, 135, 130, 125, 120, ...
    115, 110, 104, 98, 90, 82, 74, 63, 51, 30];

% Modes 3 to 18: vertical prediction (uses top_ref)
for mode = 3:18
    angIdx = mode - 2;
    angle = angles(angIdx);
    for i = 1:blockSize
        for j = 1:blockSize
            ref_idx = j + floor((i * angle + 32) / 64);
            ref_frac = mod(i * angle + 32, 64);

            idx1 = max(1, min(length(top_ref), ref_idx + 1));
            idx2 = max(1, min(length(top_ref), ref_idx + 2));

            ref_val1 = top_ref(idx1);
            ref_val2 = top_ref(idx2);
            pred_blocks(i,j,mode) = ((64 - ref_frac) * ref_val1 + ref_frac * ref_val2) / 64;
        end
    end
end

% Modes 19 to 34: horizontal prediction (uses left_ref)
for mode = 19:34
    angIdx = mode - 18;
    angle = angles(angIdx);
    for i = 1:blockSize
        for j = 1:blockSize
            ref_idx = i + floor((j * angle + 32) / 64);
            ref_frac = mod(j * angle + 32, 64);

            idx1 = max(1, min(length(left_ref), ref_idx + 1));
            idx2 = max(1, min(length(left_ref), ref_idx + 2));

            ref_val1 = left_ref(idx1);
            ref_val2 = left_ref(idx2);
            pred_blocks(i,j,mode) = ((64 - ref_frac) * ref_val1 + ref_frac * ref_val2) / 64;
        end
    end
end
end









function recon_block = partition_ctu_recursive(block, img, x, y, blockSize, Qstep, lambda)

    % Base case
    if blockSize == 4
        recon_block = predict_block(block, img, x, y, blockSize, Qstep);
        return;
    end

    % Compute variance
    if var(block(:)) < 0.002 || blockSize == 4
        recon_block = predict_block(block, img, x, y, blockSize, Qstep);
    else
        half = blockSize / 2;
        recon_block = zeros(blockSize);

        recon_block(1:half, 1:half) = partition_ctu_recursive( ...
            block(1:half, 1:half), img, x, y, half, Qstep, lambda);

        recon_block(1:half, half+1:end) = partition_ctu_recursive( ...
            block(1:half, half+1:end), img, x, y+half, half, Qstep, lambda);

        recon_block(half+1:end, 1:half) = partition_ctu_recursive( ...
            block(half+1:end, 1:half), img, x+half, y, half, Qstep, lambda);

        recon_block(half+1:end, half+1:end) = partition_ctu_recursive( ...
            block(half+1:end, half+1:end), img, x+half, y+half, half, Qstep, lambda);
    end
end







function recon = predict_block(block, img, x, y, N, Qstep)

    [H, W] = size(img);

    % --- TOP reference (1x(2N+1)) ---
    if x > 1
        % Top-left corner
        if y > 1
            top_left = img(x-1, y-1);
        else
            top_left = 128;
        end

        % Top pixels (clamped if right edge exceeds image)
        top_range = y : min(y + 2*N - 1, W);
        top_ref_main = img(x-1, top_range);
        missing = 2*N - length(top_ref_main);
        top_ref_main = [top_ref_main, repmat(top_ref_main(end), 1, missing)];  % replicate if needed

        top_ref = [top_left, top_ref_main];
    else
        top_ref = 128 * ones(1, 2*N + 1);
    end

    % --- LEFT reference ((2N+1)x1) ---
    if y > 1
        if x > 1
            left_top = img(x-1, y-1);
        else
            left_top = 128;
        end

        % Left pixels (clamped if bottom edge exceeds image)
        left_range = x : min(x + 2*N - 1, H);
        left_ref_main = img(left_range, y-1);
        missing = 2*N - length(left_ref_main);
        left_ref_main = [left_ref_main; repmat(left_ref_main(end), missing, 1)];  % replicate if needed

        left_ref = [left_top; left_ref_main];
    else
        left_ref = 128 * ones(2*N + 1, 1);
    end

    % --- Intra Prediction ---
    preds = hevc_intra_predict(top_ref, left_ref, N);
    bestCost = inf;

    for mode = 1:35
        pred = preds(:,:,mode);
        residual = block - pred;
        dct_block = dct2(residual);
        quant = round(dct_block / Qstep);
        recon_residual = idct2(quant * Qstep);
        recon_block = pred + recon_residual;

        cost = sum((block - recon_block).^2, 'all'); % SSD only

        if cost < bestCost
            bestCost = cost;
            recon = recon_block;
        end
    end
end

