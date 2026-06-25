package com.mindspore.flowerrecognition;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;
import android.util.Log;

import com.mindspore.lite.DataType;
import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.Model;
import com.mindspore.lite.Version;
import com.mindspore.lite.config.CpuBindMode;
import com.mindspore.lite.config.DeviceType;
import com.mindspore.lite.config.MSConfig;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class FlowerRecognitionExecutor {
    private static final String TAG = "FlowerRecognition";
    private static final String MODEL_PATH = "model/train.ms";
    private static final String INPUT_NAME = "6170_6169_4140_construct_wrapper:x";
    private static final String OUTPUT_NAME = "Default/Softmax-op27";
    private static final int INPUT_SIZE = 224;
    private static final int CHANNELS = 3;
    private static final int CLASS_COUNT = 16;
    private static final int INPUT_ELEMENTS = INPUT_SIZE * INPUT_SIZE * CHANNELS;
    private static final long INPUT_BYTES = INPUT_ELEMENTS * 4L;
    private static final float MIN_CONFIDENCE = 0.45f;
    private static final float MIN_MARGIN = 0.10f;
    private static final int NUM_THREADS = 4;

    private final Context context;
    private Model model;
    private LiteSession session;
    private boolean ready;

    public FlowerRecognitionExecutor(Context context) {
        this.context = context.getApplicationContext();
    }

    public synchronized LoadResult loadModel() {
        releaseLocked();
        long start = SystemClock.elapsedRealtime();
        String runtimeVersion;
        try {
            runtimeVersion = Version.version();
        } catch (Throwable error) {
            runtimeVersion = "unavailable: " + error.getMessage();
        }
        Log.i(TAG, "Runtime version: " + runtimeVersion);

        try (AssetFileDescriptor descriptor = context.getAssets().openFd(MODEL_PATH)) {
            Log.i(TAG, "Model asset: " + MODEL_PATH + ", size=" + descriptor.getLength() + " bytes");
        } catch (IOException error) {
            return loadFailure("模型文件不存在或无法读取: " + MODEL_PATH, error, start, runtimeVersion);
        }

        try {
            model = new Model();
            boolean loaded = model.loadModel(context, MODEL_PATH);
            Log.i(TAG, "Model.loadModel result: " + loaded);
            if (!loaded) {
                return loadFailure("MindSpore Lite 无法解析 train.ms，可能是模型格式或 Runtime 版本不兼容",
                        null, start, runtimeVersion);
            }

            MSConfig config = new MSConfig();
            boolean configReady = config.init(DeviceType.DT_CPU, NUM_THREADS, CpuBindMode.MID_CPU);
            Log.i(TAG, "MSConfig.init result: " + configReady);
            if (!configReady) {
                config.free();
                return loadFailure("MindSpore Lite CPU Runtime 初始化失败", null, start, runtimeVersion);
            }

            session = new LiteSession();
            boolean sessionReady = session.init(config);
            config.free();
            Log.i(TAG, "LiteSession.init result: " + sessionReady);
            if (!sessionReady) {
                return loadFailure("MindSpore Lite Session 初始化失败", null, start, runtimeVersion);
            }

            boolean compiled = session.compileGraph(model);
            Log.i(TAG, "CompileGraph result: " + compiled);
            if (!compiled) {
                return loadFailure("CompileGraph 失败，train.ms 与当前 Runtime 或算子不兼容",
                        null, start, runtimeVersion);
            }
            model.freeBuffer();

            String interfaceError = inspectAndValidateTensors();
            if (interfaceError != null) {
                return loadFailure(interfaceError, null, start, runtimeVersion);
            }
            ready = true;
            long elapsed = SystemClock.elapsedRealtime() - start;
            Log.i(TAG, "Model load completed in " + elapsed + " ms");
            return LoadResult.success(runtimeVersion, elapsed);
        } catch (Throwable error) {
            return loadFailure("模型加载异常: " + error.getClass().getSimpleName() + ": "
                    + error.getMessage(), error, start, runtimeVersion);
        }
    }

    private String inspectAndValidateTensors() {
        List<MSTensor> inputs = session.getInputs();
        int inputCount = inputs == null ? 0 : inputs.size();
        Log.i(TAG, "Input tensor count: " + inputCount);
        if (inputCount != 1) {
            return "输入 Tensor 数量不匹配，期望 1，实际 " + inputCount;
        }
        MSTensor input = inputs.get(0);
        MSTensor namedInput = session.getInputsByTensorName(INPUT_NAME);
        Log.i(TAG, "Input tensor name: " + INPUT_NAME + ", lookup=" + (namedInput != null));
        logTensor("Input", INPUT_NAME, input);
        if (namedInput == null) {
            return "找不到预期输入 Tensor: " + INPUT_NAME;
        }
        String inputError = validateTensor("输入", input, new int[]{1, INPUT_SIZE, INPUT_SIZE, CHANNELS},
                INPUT_ELEMENTS, INPUT_BYTES);
        if (inputError != null) {
            return inputError;
        }

        List<String> outputNames = session.getOutputTensorNames();
        Map<String, MSTensor> outputs = session.getOutputMapByTensor();
        int outputCount = outputNames == null ? 0 : outputNames.size();
        Log.i(TAG, "Output tensor count: " + outputCount);
        if (outputCount != 1 || outputs == null) {
            return "输出 Tensor 数量不匹配，期望 1，实际 " + outputCount;
        }
        String actualOutputName = outputNames.get(0);
        MSTensor output = outputs.get(actualOutputName);
        logTensor("Output", actualOutputName, output);
        if (!OUTPUT_NAME.equals(actualOutputName)) {
            return "输出 Tensor 名称不匹配，期望 " + OUTPUT_NAME + "，实际 " + actualOutputName;
        }
        if (output == null) {
            return "无法取得输出 Tensor: " + actualOutputName;
        }
        return validateTensor("输出", output, new int[]{1, CLASS_COUNT}, CLASS_COUNT,
                CLASS_COUNT * 4L);
    }

    private void logTensor(String kind, String name, MSTensor tensor) {
        if (tensor == null) {
            Log.e(TAG, kind + " tensor is null: " + name);
            return;
        }
        Log.i(TAG, String.format(Locale.US,
                "%s tensor: name=%s, shape=%s, dtype=%d, elements=%d, bytes=%d",
                kind, name, Arrays.toString(tensor.getShape()), tensor.getDataType(),
                tensor.elementsNum(), tensor.size()));
    }

    private String validateTensor(String kind, MSTensor tensor, int[] shape, int elements, long bytes) {
        if (!Arrays.equals(shape, tensor.getShape())) {
            return kind + " Tensor shape 不匹配，期望 " + Arrays.toString(shape)
                    + "，实际 " + Arrays.toString(tensor.getShape());
        }
        if (tensor.getDataType() != DataType.kNumberTypeFloat32) {
            return kind + " Tensor dtype 不匹配，期望 FLOAT32(" + DataType.kNumberTypeFloat32
                    + ")，实际 " + tensor.getDataType();
        }
        if (tensor.elementsNum() != elements || tensor.size() != bytes) {
            return kind + " Tensor 大小不匹配，期望 elements=" + elements + ", bytes=" + bytes
                    + "，实际 elements=" + tensor.elementsNum() + ", bytes=" + tensor.size();
        }
        return null;
    }

    public synchronized InferenceResult recognize(Bitmap source) throws RecognitionException {
        if (!ready || session == null) {
            throw new RecognitionException("模型尚未加载成功");
        }
        long preprocessStart = SystemClock.elapsedRealtime();
        ByteBuffer input = preprocess(source);
        long preprocessMs = SystemClock.elapsedRealtime() - preprocessStart;

        List<MSTensor> inputs = session.getInputs();
        if (inputs == null || inputs.size() != 1) {
            throw new RecognitionException("推理前输入 Tensor 数量发生变化");
        }
        inputs.get(0).setData(input);

        long inferenceStart = SystemClock.elapsedRealtime();
        boolean runResult = session.runGraph();
        long inferenceMs = SystemClock.elapsedRealtime() - inferenceStart;
        Log.i(TAG, "runGraph result: " + runResult + ", inference=" + inferenceMs + " ms");
        if (!runResult) {
            throw new RecognitionException("MindSpore Lite 推理返回 false");
        }

        MSTensor output = session.getOutputByTensorName(OUTPUT_NAME);
        if (output == null) {
            throw new RecognitionException("推理后找不到输出 Tensor: " + OUTPUT_NAME);
        }
        float[] probabilities = output.getFloatData();
        if (probabilities == null || probabilities.length != CLASS_COUNT) {
            throw new RecognitionException("输出长度错误，期望 16，实际 "
                    + (probabilities == null ? 0 : probabilities.length));
        }
        float sum = 0f;
        for (float value : probabilities) {
            if (Float.isNaN(value) || Float.isInfinite(value)) {
                throw new RecognitionException("模型输出包含 NaN 或 Infinity");
            }
            sum += value;
        }
        Log.i(TAG, "Raw probabilities: " + Arrays.toString(probabilities));
        Log.i(TAG, String.format(Locale.US, "Probability sum: %.8f", sum));

        List<Prediction> ranked = new ArrayList<>(CLASS_COUNT);
        for (int i = 0; i < CLASS_COUNT; i++) {
            ranked.add(new Prediction(i, FlowerLabels.ALL.get(i), probabilities[i]));
        }
        Collections.sort(ranked, (left, right) -> Float.compare(right.confidence, left.confidence));
        List<Prediction> topThree = Collections.unmodifiableList(new ArrayList<>(ranked.subList(0, 3)));
        boolean unreliable = topThree.get(0).confidence < MIN_CONFIDENCE
                || topThree.get(0).confidence - topThree.get(1).confidence < MIN_MARGIN;
        return new InferenceResult(topThree, unreliable, preprocessMs, inferenceMs, sum);
    }

    private ByteBuffer preprocess(Bitmap source) throws RecognitionException {
        if (source == null || source.isRecycled()) {
            throw new RecognitionException("待识别 Bitmap 无效");
        }
        Bitmap square = null;
        Bitmap scaled = null;
        try {
            int side = Math.min(source.getWidth(), source.getHeight());
            int left = (source.getWidth() - side) / 2;
            int top = (source.getHeight() - side) / 2;
            square = Bitmap.createBitmap(source, left, top, side, side);
            scaled = Bitmap.createScaledBitmap(square, INPUT_SIZE, INPUT_SIZE, true);

            int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
            scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
            ByteBuffer buffer = ByteBuffer.allocateDirect((int) INPUT_BYTES);
            buffer.order(ByteOrder.nativeOrder());

            // NHWC: iterate y, x, then write normalized RGB floats exactly once.
            for (int y = 0; y < INPUT_SIZE; y++) {
                for (int x = 0; x < INPUT_SIZE; x++) {
                    int pixel = pixels[y * INPUT_SIZE + x];
                    buffer.putFloat((Color.red(pixel) / 255.0f - 0.485f) / 0.229f);
                    buffer.putFloat((Color.green(pixel) / 255.0f - 0.456f) / 0.224f);
                    buffer.putFloat((Color.blue(pixel) / 255.0f - 0.406f) / 0.225f);
                }
            }
            buffer.rewind();
            return buffer;
        } catch (OutOfMemoryError error) {
            throw new RecognitionException("图片预处理内存不足", error);
        } catch (RuntimeException error) {
            throw new RecognitionException("图片预处理失败: " + error.getMessage(), error);
        } finally {
            if (scaled != null && scaled != source && scaled != square) {
                scaled.recycle();
            }
            if (square != null && square != source) {
                square.recycle();
            }
        }
    }

    public synchronized boolean isReady() {
        return ready;
    }

    public synchronized void release() {
        releaseLocked();
    }

    private void releaseLocked() {
        ready = false;
        if (session != null) {
            session.free();
            session = null;
        }
        if (model != null) {
            model.free();
            model = null;
        }
    }

    private LoadResult loadFailure(String message, Throwable error, long start, String runtimeVersion) {
        long elapsed = SystemClock.elapsedRealtime() - start;
        if (error == null) {
            Log.e(TAG, message + ", elapsed=" + elapsed + " ms");
        } else {
            Log.e(TAG, message + ", elapsed=" + elapsed + " ms", error);
        }
        releaseLocked();
        return LoadResult.failure(runtimeVersion, elapsed, message);
    }

    public static final class LoadResult {
        public final boolean success;
        public final String runtimeVersion;
        public final long loadTimeMs;
        public final String error;

        private LoadResult(boolean success, String runtimeVersion, long loadTimeMs, String error) {
            this.success = success;
            this.runtimeVersion = runtimeVersion;
            this.loadTimeMs = loadTimeMs;
            this.error = error;
        }

        private static LoadResult success(String version, long timeMs) {
            return new LoadResult(true, version, timeMs, null);
        }

        private static LoadResult failure(String version, long timeMs, String error) {
            return new LoadResult(false, version, timeMs, error);
        }
    }

    public static final class Prediction {
        public final int index;
        public final FlowerLabels.FlowerLabel label;
        public final float confidence;

        private Prediction(int index, FlowerLabels.FlowerLabel label, float confidence) {
            this.index = index;
            this.label = label;
            this.confidence = confidence;
        }
    }

    public static final class InferenceResult {
        public final List<Prediction> topThree;
        public final boolean unreliable;
        public final long preprocessTimeMs;
        public final long inferenceTimeMs;
        public final float probabilitySum;

        private InferenceResult(List<Prediction> topThree, boolean unreliable,
                                long preprocessTimeMs, long inferenceTimeMs, float probabilitySum) {
            this.topThree = topThree;
            this.unreliable = unreliable;
            this.preprocessTimeMs = preprocessTimeMs;
            this.inferenceTimeMs = inferenceTimeMs;
            this.probabilitySum = probabilitySum;
        }
    }

    public static final class RecognitionException extends Exception {
        private RecognitionException(String message) {
            super(message);
        }

        private RecognitionException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
