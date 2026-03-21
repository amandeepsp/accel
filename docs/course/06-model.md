# Unit 6: End-to-End — Run a Real Model

> **Course:** [00-architecture](00-architecture.md) > [01-compute](01-compute.md) > [02-vertical-slice](02-vertical-slice.md) > [03-fusion](03-fusion.md) > [04-engine](04-engine.md) > [05-compiler](05-compiler.md) > **[06-model](06-model.md)** > [07-systolic](07-systolic.md)
> **Reference:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

---

You've built compute (MAC4), fused post-processing (SRDHM+RDBPOT), an autonomous execution engine, and a compiler backend. Now run a real model.

Everything so far has been exercised on synthetic data -- hand-crafted weight matrices, known inputs, expected outputs. This unit connects the full stack to a production-grade quantized neural network and asks one question: **does the top-1 prediction match the reference?**

Performance is not the goal. Correctness is.

---

## 6.1  Why MobileNet v2 0.25 INT8

You need a model that is:
- **Small enough** to fit in your constrained memory hierarchy (~8 KiB activation BSRAM, ~4 KiB weight BSRAM)
- **Representative enough** to exercise real quantization, multiple layer types, and nontrivial data flow
- **Well-supported** with freely available pre-trained INT8 checkpoints in TFLite format

MobileNet v2 with width multiplier 0.25 hits all three. It is a real ImageNet classifier, not a toy.

> **MLSys Connection:** Width multipliers are a standard model-architecture knob for deployment on edge devices. When NVIDIA publishes "Tensor Core performance" numbers, they benchmark on full-size models. When you target a 20K-LUT FPGA, you need models designed for the constraint envelope. The width multiplier is the same idea as choosing a GPU instance size in the cloud -- match compute to budget.

---

## 6.2  MobileNet v2 Architecture Overview

MobileNet v2 is built from **inverted residual blocks** (also called "bottleneck blocks"). Each block has three layers:

```
  Inverted Residual Block:

  Input (low-dim) ──► 1x1 Pointwise Conv (expand)
                          │
                          ▼
                      3x3 Depthwise Conv (spatial filtering)
                          │
                          ▼
                      1x1 Pointwise Conv (project back to low-dim)
                          │
                          ▼
                      + residual ──► Output
```

The "inverted" part: standard residual blocks compress then expand. MobileNet v2 expands *first* (to a higher channel count), does spatial filtering cheaply in the expanded space (depthwise), then projects back down.

With width multiplier 0.25, channel counts are 1/4 of the full model. The first conv layer produces 8 channels instead of 32. The largest internal expansion produces ~24 channels instead of ~96. This keeps everything small.

> **MLSys Connection:** Depthwise separable convolutions are the key efficiency trick in mobile inference. A standard 3x3 conv on C channels costs `C * C * 9` MACs per spatial position. A depthwise separable conv costs `C * 9 + C * C` -- roughly 9x fewer MACs. This is why MobileNets dominate edge deployment. The same factorization idea appears in GPU kernel design: decompose expensive ops into cheaper sequences that better match the hardware.

---

## 6.3  INT8 Quantization and TFLite Parameters

Every conv layer in a TFLite INT8 model carries quantization metadata:

```
  Per-layer:
    input_zero_point    (int8)   -- offset to make the input distribution symmetric-ish
    output_zero_point   (int8)   -- offset for the output
    weight_zero_point   (int8)   -- 0 for symmetric quantization (standard)

  Per-channel (one value per output channel):
    multiplier          (int32)  -- fixed-point scale factor for requantization
    shift               (int)    -- right-shift amount for requantization
```

These per-channel multiplier and shift values are exactly what your SRDHM+RDBPOT hardware consumes. The hardware does:

```
  acc = MAC(input, weights)             // your MAC4 or systolic array
  acc = SRDHM(acc, multiplier[ch])      // your fused post-processing
  acc = RDBPOT(acc, shift[ch])          // your fused post-processing
  acc += output_zero_point              // simple add
  output[ch] = clamp(acc, -128, 127)    // saturate to INT8
```

The multiplier encodes the ratio of (input_scale * weight_scale) / output_scale as a fixed-point number in [0.5, 1.0), and the shift compensates. Together they implement arbitrary scale conversion without floating point.

> **MLSys Connection:** Per-channel quantization (as opposed to per-tensor) is standard practice because different output channels can have very different dynamic ranges. Google's quantization-aware training (QAT) and TFLite's post-training quantization both produce per-channel params. When NVIDIA's TensorRT quantizes a model, it solves the same problem: find scale factors that minimize accuracy loss while fitting into INT8 arithmetic. Your hardware consumes these params directly.

---

## 6.4  Exercise: Extract Quantization Parameters

Parse a TFLite flatbuffer and extract the quantization parameters for every conv layer.

**Input:** A MobileNet v2 0.25 INT8 `.tflite` file (available from TensorFlow Hub or the TFLite model zoo).

**Output:** For each conv layer, print:
- Layer name and type (CONV_2D or DEPTHWISE_CONV_2D)
- Input/output tensor shapes
- Input zero point, output zero point
- Per-channel multiplier array (int32[])
- Per-channel shift array (int[])
- Weight tensor shape and a few sample values

<details><summary>Hint 1</summary>

The TFLite flatbuffer schema defines `QuantizationParameters` on each tensor, containing `scale` (float[]) and `zero_point` (int64[]). The per-channel multiplier and shift are *not* stored directly -- you must compute them from the float scales using `quantize_multiplier_smaller_than_one()`.

</details>

<details><summary>Hint 2</summary>

Use `tflite_runtime.interpreter.Interpreter` to load the model and inspect tensors. Or use the `flatbuffers` library directly with the TFLite schema. The key function to implement:

```
def quantize_multiplier(double_multiplier):
    """Convert a float scale ratio to (int32 multiplier, int shift) pair.

    The multiplier is in the range [0.5, 1.0) represented as a fixed-point
    int32 (so the actual int32 value is in [2^30, 2^31)).
    The shift is a non-negative right-shift amount.
    """
```

Reference: TFLite's `QuantizeMultiplierSmallerThanOne` in `tensorflow/lite/kernels/kernel_util.cc`.

</details>

<details><summary>Solution</summary>See solutions/06-model/extract_quant_params.py</details>

---

## 6.5  The Ops Your Hardware Handles

Your hardware accelerates **pointwise (1x1) convolutions**. These are matrix multiplications:

```
  1x1 Conv with C_in input channels, C_out output channels, H x W spatial:

  Reshape activations: [H, W, C_in] -> [H*W, C_in]
  Weight matrix:       [C_out, C_in]
  Output:              [H*W, C_out]

  This is a standard matmul: Output = Activations @ Weights.T
```

Pointwise convolutions are the **heavy compute ops** in MobileNet v2. They contain the vast majority of multiply-accumulate operations because they operate across all channels.

---

## 6.6  The Ops Your Hardware Does NOT Handle

These run on the host CPU (or in firmware on the RISC-V):

| Op | Why not hardware? |
|---|---|
| **Depthwise 3x3 conv** | Different data access pattern -- each output channel depends on only one input channel. Not a matmul. |
| **ReLU6** | `min(max(x, 0), 6)` -- a trivial clamp. Not worth the hardware complexity to offload. |
| **Average pooling** | Simple reduction. Few MACs relative to convolutions. |
| **Fully connected** | Only the final classifier layer. Small enough (e.g., 16 x 1001) that CPU handles it fine. |
| **Add (residual)** | Element-wise addition for skip connections. Memory-bound, not compute-bound. |

### Exercise: Where's the 80/20?

For each layer in MobileNet v2 0.25, calculate the MAC count:

```
  Pointwise 1x1 conv: H * W * C_in * C_out
  Depthwise 3x3 conv: H * W * C * 9
```

Sum the MACs for all pointwise layers and all depthwise layers separately. What fraction of total compute is in the ops your hardware handles?

<details><summary>Hint 1</summary>

For MobileNet v2, the pointwise convolutions typically account for 75-85% of total MACs, even though they are "just" 1x1 kernels. The channel dimension dominates.

</details>

<details><summary>Hint 2</summary>

Use your parameter extraction script from 6.4 to get the tensor shapes, then compute MACs directly from the shapes. No need to run the model.

</details>

<details><summary>Solution</summary>See solutions/06-model/mac_breakdown.py</details>

> **MLSys Connection:** This exercise is exactly what ML compiler engineers do when deciding which ops to lower to a hardware backend vs. leave on CPU. In XLA (Google's ML compiler), the "cost model" estimates how expensive each op is and routes it accordingly. In TVM, the `relay.transform.AnnotateTarget` pass marks ops for offloading. You are doing manual operator placement -- the same problem, at a smaller scale.

---

## 6.7  Tiling for BSRAM Capacity

Your on-chip memory is limited:
- ~8 KiB for activations (4 BSRAM banks)
- ~4 KiB for weights (2 BSRAM filter stores)

Most layers in MobileNet v2 0.25 are small enough to fit. But the early layers have large spatial dimensions:

```
  First pointwise conv: 48x48 spatial, 16 input channels, 8 output channels
  Activation tensor:    48 * 48 * 16 = 36,864 bytes  (does NOT fit in 8 KiB)
  Weight tensor:        8 * 16 = 128 bytes            (fits easily)
```

When activations don't fit, you **tile**: break the spatial dimension into strips that fit in BSRAM, process each strip, and assemble the results.

### Exercise: Tiling Strategy

For MobileNet v2 0.25's largest pointwise conv (48x48 spatial, 16 input channels):

1. How many rows of activations fit in 8 KiB? (`rows * 48 * 16 <= 8192`)
2. How many tiles does that give you? (`ceil(48 / rows_per_tile)`)
3. What is the weight reload cost per tile? (Weights are the same for every spatial tile -- do you need to reload them?)
4. Write down the tiling loop structure.

<details><summary>Hint 1</summary>

Each row is `48 * 16 = 768` bytes. You can fit `floor(8192 / 768) = 10` rows per tile. That gives `ceil(48 / 10) = 5` tiles. Weights don't change across spatial tiles, so you load them once.

</details>

<details><summary>Hint 2</summary>

The tiling loop looks like:

```
load weights into filter BSRAM (once)
load requant params (once)
for tile_start in range(0, H, rows_per_tile):
    tile_end = min(tile_start + rows_per_tile, H)
    load activation rows [tile_start:tile_end] into act BSRAM
    execute hardware matmul
    read output FIFO -> store results for rows [tile_start:tile_end]
```

The key insight: weight loading is amortized across all spatial tiles.

</details>

<details><summary>Solution</summary>See solutions/06-model/tiling.py</details>

> **MLSys Connection:** Tiling is the fundamental technique in every ML compiler. NVIDIA's cuDNN tiles convolutions to fit in shared memory (48 KiB on Ampere). Google's XLA tiles matmuls to fit in TPU HBM scratch. TVM's `schedule` primitives (`tile`, `split`, `reorder`) exist entirely to control tiling. The constraint is always the same: fast memory is small, so you must break the problem into pieces that fit. Your 8 KiB BSRAM is a miniature version of a GPU's shared memory.

---

## 6.8  Putting It Together: The Inference Runner

### Exercise: End-to-End Inference

Write an inference runner that classifies an image using your FPGA accelerator. The structure:

```
  1. Load the TFLite model
     - Extract weights, biases, and quantization params for each layer
     - Identify which layers are pointwise (hardware) vs. other (CPU)

  2. Preprocess the input image
     - Resize to model input size (96x96 or as specified)
     - Quantize to INT8 using the input tensor's scale and zero point

  3. For each layer in order:
     If pointwise conv (hardware):
         a. Determine tiling (does the activation fit in BSRAM?)
         b. Send weights and requant params to device
         c. For each spatial tile:
            - Send activation tile to device
            - Execute
            - Read results
         d. Assemble tiled outputs
     If other op (CPU):
         - Run with numpy (depthwise conv, relu6, add, pool, etc.)

  4. Final output: argmax of the last layer = predicted class index

  5. Validate: compare your prediction against the reference TFLite interpreter
```

<details><summary>Hint 1</summary>

Start by getting the reference answer first. Load the model in `tflite_runtime`, run inference, get the predicted class. That is your ground truth. Then implement your runner layer by layer, comparing intermediate activations against the reference at each layer boundary.

</details>

<details><summary>Hint 2</summary>

For CPU ops, use numpy with careful attention to quantization:

```python
# Depthwise conv (3x3, per-channel quantized)
# - Pad input (zero_point padding, not zero padding!)
# - For each channel independently: conv2d with 3x3 kernel
# - Requantize output using per-channel multiplier/shift

# ReLU6 in quantized domain:
# relu6_output = clamp(input, quantized_zero, quantized_six)
# where quantized_six = round(6.0 / scale) + zero_point
```

The most common bug: forgetting that "zero" in quantized INT8 is `zero_point`, not `0`.

</details>

<details><summary>Hint 3</summary>

For hardware ops, reuse the protocol you built in earlier units. The host-side Python should look roughly like:

```python
device.load_weights(layer_weights)
device.load_requant_params(multipliers, shifts, output_zero_point)
for tile in tiles:
    device.load_activations(tile)
    device.execute(input_depth=C_in, output_channels=C_out)
    tile_result = device.read_output()
```

If you don't have hardware yet, write a software simulator that does the same math your hardware would do (MAC4 + SRDHM + RDBPOT), to validate the data flow before touching the FPGA.

</details>

<details><summary>Solution</summary>See solutions/06-model/inference_runner.py</details>

---

## 6.9  Debugging Quantization Mismatches

When your output doesn't match the reference, the bug is almost always in quantization arithmetic, not in the matmul itself. Common failure modes:

| Symptom | Likely cause |
|---|---|
| Output is entirely zeros | Wrong zero point subtraction (input or weight) |
| Output is close but off by 1-2 | Rounding mode mismatch in SRDHM or RDBPOT |
| Output is wildly wrong for some channels | Per-channel multiplier/shift extracted incorrectly |
| Output shape is correct but values are garbage | Transposed weight matrix (C_in vs C_out axes swapped) |
| First layer works, second layer fails | Output zero point from layer N not used as input zero point for layer N+1 |

**Debugging strategy:** Compare your intermediate activations against the TFLite reference after every layer. The first layer where they diverge tells you exactly where to look.

> **MLSys Connection:** This is the same debugging process used when bringing up a new GPU backend in a compiler. NVIDIA's cuDNN has extensive numerical validation against reference implementations. When TVM compiles a model to a new target, the `relay.testing` infrastructure compares against a known-good NumPy reference layer by layer. Bitwise reproducibility is hard; "close enough" is defined by tolerance thresholds. For INT8, you should expect exact match (not approximate) because the arithmetic is integer.

---

## 6.10  What "Correct" Means

If your inference runner produces the same top-1 class as the TFLite reference interpreter for a given image, you have proven:

1. Your MAC hardware computes correctly
2. Your SRDHM+RDBPOT fusion matches the TFLite requantization spec
3. Your autonomous sequencer iterates correctly over channels and spatial positions
4. Your tiling logic preserves output correctness across tile boundaries
5. Your compiler/host code correctly extracts and transmits model parameters
6. Your UART/bus protocol delivers data without corruption

That is the **entire stack**, from Python framework down to gate-level hardware, producing a correct answer on a real model. Slow is fine. It will be slow -- UART bandwidth limits you to seconds per inference. That's okay. Performance comes from:
- Unit 7 (systolic array -- 4x compute throughput)
- Higher clock speeds (2-4x)
- Better bus interfaces (SPI/USB -- 100-1000x over UART)

Correctness comes first. Always.

---

## 6.11  Checkpoint

- [ ] I can extract quantization parameters from a TFLite flatbuffer
- [ ] I can calculate MAC counts per layer and identify the hardware/CPU split
- [ ] I have a tiling strategy for layers that exceed BSRAM capacity
- [ ] I have an inference runner that processes all layer types (hardware + CPU fallback)
- [ ] My top-1 prediction matches the TFLite reference interpreter
- [ ] I can debug quantization mismatches using layer-by-layer comparison

---

**Previous:** [Unit 5 -- Compiler Backend](05-compiler.md)
**Next:** [Unit 7 -- Spatial Parallelism: Systolic Arrays](07-systolic.md)
**Reference:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

## Side Quests

- **Different model architectures.** Try [MCUNet](https://arxiv.org/abs/2007.10319) (designed for microcontrollers with <1 MB SRAM) or [SqueezeNet v1.1](https://arxiv.org/abs/1602.07360) (50x fewer parameters than AlexNet). How do their layer shapes differ from MobileNet? Which map better to your 4-wide MAC?
- **INT4 quantization.** Quantize weights to 4 bits using [GPTQ](https://arxiv.org/abs/2210.17323) or simple min/max scaling. This halves weight storage and BSRAM bandwidth. You'll need a new MAC variant that unpacks two INT4 values from each byte — 8 lanes instead of 4.
- **Accuracy vs. speed Pareto curve.** Run MobileNet v2 at width multipliers 0.1, 0.25, 0.5, 0.75, and 1.0. Plot top-1 accuracy vs. estimated inference time on your hardware. Where does the knee of the curve fall? This is the fundamental design tradeoff for edge ML.
- **Layer-by-layer profiling.** For each layer in MobileNet v2 0.25, measure (or calculate): MAC count, weight bytes, activation bytes, and estimated cycles. Plot as a stacked bar chart. Which layers dominate? This tells you where to focus optimization effort.
- **Person detection.** Run the same [person detection model](https://github.com/google/CFU-Playground/tree/main/common/src/models/pdti8) that CFU-Playground uses as its benchmark. Direct comparison of your numbers against Google's published results.

---

## Suggested Readings

| Topic | Source |
|---|---|
| MobileNet v2 paper | [arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381) -- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" |
| MobileNet v1 paper | [arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861) -- Howard et al., "MobileNets: Efficient CNNs for Mobile Vision Applications." Read this first — v2 makes more sense with v1 context |
| Quantization white paper | [arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877) -- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" |
| TFLite quantization spec | [tensorflow.org/lite/performance/quantization_spec](https://www.tensorflow.org/lite/performance/quantization_spec) |
| TFLite flatbuffer schema | [github.com/tensorflow/tensorflow/.../schema.fbs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs) |
| Per-channel vs per-tensor quantization | [arxiv.org/abs/2004.09602](https://arxiv.org/abs/2004.09602) -- Nagel et al., "Up or Down? Adaptive Rounding for Post-Training Quantization" |
| CFU-Playground model runner | [github.com/google/CFU-Playground](https://github.com/google/CFU-Playground) -- see `common/src/models/` for TFLite integration patterns |
| MCUNet (tiny models for MCUs) | [arxiv.org/abs/2007.10319](https://arxiv.org/abs/2007.10319) -- Lin et al., "MCUNet: Tiny Deep Learning on IoT Devices." NAS-optimized models for <1 MB SRAM — directly applicable to your 8 KiB budget |
| TinyML book | Pete Warden & Daniel Situnayake, *TinyML* (O'Reilly, 2019) — practical guide to running ML on microcontrollers. Ch. 7–9 cover quantization, model optimization, and deployment on constrained devices |
