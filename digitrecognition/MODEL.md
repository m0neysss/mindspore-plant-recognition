# Handwritten Digit Model

The app loads `src/main/assets/model/digit_lenet_quant.ms`.

Expected model contract:

- Format: MindSpore Lite 1.1.0 `.ms`
- Task: MNIST/LeNet handwritten digit recognition
- Input: one `float32` tensor shaped `[1, 1, 28, 28]`
- Input range: `0..1`, black background and white foreground
- Output: one `float32` tensor shaped `[1, 10]`
- Output order: digits `0..9`

The checked-in file is a placeholder so the Android integration can compile without downloading model artifacts. Replace it with the quantized `.ms` file before final device demonstration.
