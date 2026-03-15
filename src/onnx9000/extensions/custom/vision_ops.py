"""Module providing core logic and structural definitions."""

from typing import List, Any
import math


def rotated_nms(
    boxes: List[List[float]],
    scores: List[float],
    iou_threshold: float,
    score_threshold: float = 0.0,
) -> List[int]:
    """
    Very simplified rotated NMS. Real rotated NMS requires polygon intersection.
    For this native implementation, we'll bound the rotated boxes and apply standard NMS.
    """
    from onnx9000.extensions.custom.ops import nms

    bounded_boxes = []
    for cx, cy, w, h, angle in boxes:
        # compute bounding box
        cos_a = abs(math.cos(angle))
        sin_a = abs(math.sin(angle))

        bw = w * cos_a + h * sin_a
        bh = w * sin_a + h * cos_a

        bounded_boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])

    return nms(bounded_boxes, scores, iou_threshold, score_threshold)


def grid_sample(
    input_tensor: List[List[List[List[float]]]],
    grid: List[List[List[List[float]]]],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> List[List[List[List[float]]]]:
    """
    N, C, H, W input and N, H_out, W_out, 2 grid
    """
    N = len(input_tensor)
    C = len(input_tensor[0]) if N > 0 else 0
    H = len(input_tensor[0][0]) if C > 0 else 0
    W = len(input_tensor[0][0][0]) if H > 0 else 0

    H_out = len(grid[0]) if len(grid) > 0 else 0
    W_out = len(grid[0][0]) if H_out > 0 else 0

    out = [
        [[[0.0 for _ in range(W_out)] for _ in range(H_out)] for _ in range(C)]
        for _ in range(N)
    ]

    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                x, y = grid[n][h_out][w_out]

                if align_corners:
                    ix = (x + 1) / 2 * (W - 1)
                    iy = (y + 1) / 2 * (H - 1)
                else:
                    ix = ((x + 1) * W - 1) / 2
                    iy = ((y + 1) * H - 1) / 2

                if padding_mode == "border":
                    ix = max(0.0, min(W - 1.0, ix))
                    iy = max(0.0, min(H - 1.0, iy))
                elif padding_mode == "reflection":
                    # Simplified reflection
                    if ix < 0:
                        ix = -ix
                    if ix >= W:
                        ix = W - 1 - (ix - W)
                    if iy < 0:
                        iy = -iy
                    if iy >= H:
                        iy = H - 1 - (iy - H)

                if mode == "nearest":
                    ix_n = int(round(ix))
                    iy_n = int(round(iy))
                    for c in range(C):
                        if 0 <= ix_n < W and 0 <= iy_n < H:
                            out[n][c][h_out][w_out] = input_tensor[n][c][iy_n][ix_n]
                else:  # bilinear
                    ix_nw = int(math.floor(ix))
                    iy_nw = int(math.floor(iy))
                    ix_ne = ix_nw + 1
                    iy_sw = iy_nw + 1

                    dx = ix - ix_nw
                    dy = iy - iy_nw

                    for c in range(C):
                        v_nw = (
                            input_tensor[n][c][iy_nw][ix_nw]
                            if (0 <= ix_nw < W and 0 <= iy_nw < H)
                            else 0.0
                        )
                        v_ne = (
                            input_tensor[n][c][iy_nw][ix_ne]
                            if (0 <= ix_ne < W and 0 <= iy_nw < H)
                            else 0.0
                        )
                        v_sw = (
                            input_tensor[n][c][iy_sw][ix_nw]
                            if (0 <= ix_nw < W and 0 <= iy_sw < H)
                            else 0.0
                        )
                        v_se = (
                            input_tensor[n][c][iy_sw][ix_ne]
                            if (0 <= ix_ne < W and 0 <= iy_sw < H)
                            else 0.0
                        )

                        out[n][c][h_out][w_out] = (
                            v_nw * (1 - dx) * (1 - dy)
                            + v_ne * dx * (1 - dy)
                            + v_sw * (1 - dx) * dy
                            + v_se * dx * dy
                        )
    return out


def roi_align(
    input_tensor: List[List[List[List[float]]]],
    rois: List[List[float]],
    batch_indices: List[int],
    output_height: int,
    output_width: int,
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool = True,
) -> List[List[List[List[float]]]]:
    """Provides semantic functionality and verification."""
    # Simplified RoIAlign
    num_rois = len(rois)
    C = len(input_tensor[0])
    H = len(input_tensor[0][0])
    W = len(input_tensor[0][0][0])

    out = [
        [
            [[0.0 for _ in range(output_width)] for _ in range(output_height)]
            for _ in range(C)
        ]
        for _ in range(num_rois)
    ]

    offset = 0.5 if aligned else 0.0

    for r_idx in range(num_rois):
        b_idx = batch_indices[r_idx]
        x1, y1, x2, y2 = rois[r_idx]

        x1 = x1 * spatial_scale - offset
        y1 = y1 * spatial_scale - offset
        x2 = x2 * spatial_scale - offset
        y2 = y2 * spatial_scale - offset

        roi_w = max(x2 - x1, 1.0)
        roi_h = max(y2 - y1, 1.0)

        bin_size_h = roi_h / output_height
        bin_size_w = roi_w / output_width

        grid_h = (
            sampling_ratio
            if sampling_ratio > 0
            else int(math.ceil(roi_h / output_height))
        )
        grid_w = (
            sampling_ratio
            if sampling_ratio > 0
            else int(math.ceil(roi_w / output_width))
        )

        for c in range(C):
            for ph in range(output_height):
                for pw in range(output_width):
                    val = 0.0
                    for iy in range(grid_h):
                        for ix in range(grid_w):
                            y = y1 + ph * bin_size_h + (iy + 0.5) * bin_size_h / grid_h
                            x = x1 + pw * bin_size_w + (ix + 0.5) * bin_size_w / grid_w

                            if y < -1.0 or y > H or x < -1.0 or x > W:
                                val += 0.0
                                continue

                            y = max(0.0, min(H - 1.0, y))
                            x = max(0.0, min(W - 1.0, x))

                            y_low = int(y)
                            x_low = int(x)
                            y_high = min(y_low + 1, H - 1)
                            x_high = min(x_low + 1, W - 1)

                            dy = y - y_low
                            dx = x - x_low

                            w1 = (1 - dx) * (1 - dy)
                            w2 = dx * (1 - dy)
                            w3 = (1 - dx) * dy
                            w4 = dx * dy

                            val += (
                                w1 * input_tensor[b_idx][c][y_low][x_low]
                                + w2 * input_tensor[b_idx][c][y_low][x_high]
                                + w3 * input_tensor[b_idx][c][y_high][x_low]
                                + w4 * input_tensor[b_idx][c][y_high][x_high]
                            )

                    out[r_idx][c][ph][pw] = val / (grid_h * grid_w)

    return out


def deform_conv2d(
    x: Any,
    weight: Any,
    offset: Any,
    mask: Any = None,
    bias: Any = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Any:
    """
    Naive DeformConv2D fallback.
    """
    N = len(x)
    in_C = len(x[0])
    in_H = len(x[0][0])
    in_W = len(x[0][0][0])

    out_C = len(weight)
    kH = len(weight[0][0])
    kW = len(weight[0][0][0])

    out_H = (in_H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    out_W = (in_W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1

    out = [
        [[[0.0 for _ in range(out_W)] for _ in range(out_H)] for _ in range(out_C)]
        for _ in range(N)
    ]

    for n in range(N):
        for oc in range(out_C):
            for oh in range(out_H):
                for ow in range(out_W):
                    val = bias[oc] if bias else 0.0
                    for ic in range(in_C):
                        for kh in range(kH):
                            for kw in range(kW):
                                # standard conv index
                                ih = oh * stride - padding + kh * dilation
                                iw = ow * stride - padding + kw * dilation

                                # add offset
                                # offset shape: N, 2*kH*kW, out_H, out_W
                                offset_idx_y = 2 * (kh * kW + kw)
                                offset_idx_x = offset_idx_y + 1

                                dy = offset[n][offset_idx_y][oh][ow]
                                dx = offset[n][offset_idx_x][oh][ow]

                                sampled_y = ih + dy
                                sampled_x = iw + dx

                                # apply mask if DCNv2
                                m_val = 1.0
                                if mask is not None:
                                    mask_idx = kh * kW + kw
                                    m_val = mask[n][mask_idx][oh][ow]

                                # bilinear sample
                                if 0 <= sampled_y < in_H and 0 <= sampled_x < in_W:
                                    y0 = int(math.floor(sampled_y))
                                    x0 = int(math.floor(sampled_x))
                                    y1 = min(y0 + 1, in_H - 1)
                                    x1 = min(x0 + 1, in_W - 1)

                                    wy = sampled_y - y0
                                    wx = sampled_x - x0

                                    v00 = x[n][ic][y0][x0]
                                    v01 = x[n][ic][y0][x1]
                                    v10 = x[n][ic][y1][x0]
                                    v11 = x[n][ic][y1][x1]

                                    s_val = (
                                        v00 * (1 - wy) * (1 - wx)
                                        + v01 * (1 - wy) * wx
                                        + v10 * wy * (1 - wx)
                                        + v11 * wy * wx
                                    )

                                    val += s_val * weight[oc][ic][kh][kw] * m_val
                    out[n][oc][oh][ow] = val

    return out
