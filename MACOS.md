# macOS notes

On Apple Silicon, use PyTorch 2.13.0 or later (tested: 2.14.1 and 2.15.0). Older versions compute attention wrongly on the Mac GPU, which makes the Policy-DROID actions jittery, and they are 4-5x slower.

```bash
pip install torch==2.15.0 torchvision==0.29.1
```

## Wrong attention on the Mac GPU

In PyTorch 2.7.1 to 2.12.x, `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True` returns wrong values on the Mac GPU in bfloat16 and float16, with no error. float32 and an explicit causal mask are fine. Reported upstream: [pytorch/pytorch#195910](https://github.com/pytorch/pytorch/issues/195910).

Error of this function in bfloat16, against a float64 reference:

| PyTorch | 2.7.1 | 2.12.1 | 2.13.0 | 2.14.1 |
|---|---|---|---|---|
| Error | 6.5% | 6.4% | 0.5% | 0.3% |

Cosmos3 runs in bfloat16 and uses this function, so its actions came out wrong.

## Speed

Seconds per Edge-Policy-DROID call on an M5 Max (40-core GPU):

| | PyTorch 2.7.1 | PyTorch 2.14.1 | PyTorch 2.15.0, our VAE change | NVIDIA's policy server (RTX 4500) |
|---|---|---|---|---|
| Without the predicted video | 14 | 3.6 | 1.5 | 0.7 |
| With `--decode-video` | 60 | 12 | 8.4 | 3.6 |

Newer PyTorch speeds up the VAE on the Mac. Diffusers also VAE-encodes the placeholder frames for the future video; we encode only the camera image, which matches NVIDIA's server speed on CUDA with the same actions ([Implementation](README.md#implementation)).

## Mac against CUDA

Same commit, input and starting noise. Mean (max) action difference of the Mac in bfloat16: joint angles in rad for Policy-DROID, the model's raw actions for the base model.

| Model | Against CUDA bfloat16 | Against Mac float32 |
|---|---|---|
| Cosmos3-Edge-Policy-DROID | 0.015 (0.11) | 0.013 (0.07) |
| Cosmos3-Edge | 0.005 (0.03) | 0.002 (0.02) |

Both are about the same size, so the Mac's difference from CUDA is bfloat16 rounding. Against the official code, see [Numerical differences](README.md#numerical-differences).

## Other notes

- Each script sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, so the few operations the Mac GPU lacks run on the CPU.
