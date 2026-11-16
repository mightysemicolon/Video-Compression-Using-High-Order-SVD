# Video Compression Using Higher-Order SVD

Compressing a short video with the Higher-Order SVD (HOSVD). A video is naturally a tensor (frames x height x width), so instead of compressing every frame separately with a matrix SVD, the whole clip is decomposed at once and truncated in all three directions.

The tensor functions (unfold, fold, mode-n product) are my own NumPy implementation in `All.py`. It is the same file I used in my face recognition project.

## Data

`original.mp4`: 132 frames, 720 x 1280, RGB. Each color channel is treated as a separate tensor of shape 132 x 720 x 1280.

## Method

For each channel `A`:

1. Unfold `A` along the three modes. The unfoldings have shapes 132 x 921600, 720 x 168960 and 1280 x 95040.
2. Take the SVD of each unfolding and keep the left singular vectors `U1`, `U2`, `U3`.
3. Compute the core tensor `S = A x1 U1^T x2 U2^T x3 U3^T`.
4. To compress, keep only the first `r1`, `r2`, `r3` columns of `U1`, `U2`, `U3` and the corner `S[:r1, :r2, :r3]` of the core.
5. Reconstruct with `A_hat = S_r x1 U1_r x2 U2_r x3 U3_r`, put the three channels back together and write the video.

`r1` is the rank in the time direction, `r2` and `r3` are the ranks in the height and width directions.

## Settings

I tried three truncations. The table shows how many numbers have to be stored per channel (core + three factor matrices) compared to the original 132 x 720 x 1280 = 121,651,200 values.

| (r1, r2, r3) | Stored values | Percent of original |
|---|---|---|
| (20, 100, 200) | 730,640 | 0.6% |
| (40, 250, 400) | 4,697,280 | 3.9% |
| (80, 400, 700) | 23,594,560 | 19.4% |

The notebook writes one output video for each setting (`outputvideo_20_100_200.mp4` and so on), so the quality can be compared by eye. The time rank can be cut a lot more than the spatial ranks, because neighboring frames are very similar.

## Files

- `Video_Compression.ipynb` : the notebook
- `All.py` : `unfold`, `fold`, `Mul_1`, `Mul_2`, `Mul_3`
- `original.mp4` : input video

## Requirements

numpy, scikit-video (needs ffmpeg installed)

Note that the SVDs of the unfoldings are heavy. The full tensor in float64 is about 1 GB per channel, so this needs a machine with enough RAM.
