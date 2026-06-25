package com.mindspore.digitrecognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.Model;
import com.mindspore.lite.config.CpuBindMode;
import com.mindspore.lite.config.DeviceType;
import com.mindspore.lite.config.MSConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class DigitRecognitionExecutor {
    private static final String TAG = "DigitRecognitionExecutor";
    private static final String MODEL_PATH = "model/digit_lenet_quant.ms";
    private static final int INPUT_SIZE = 28;
    private static final int NUM_THREADS = 2;

    private final Context context;
    private Model model;
    private LiteSession session;
    private boolean modelReady;

    public DigitRecognitionExecutor(Context context) {
        this.context = context.getApplicationContext();
    }

    public boolean init() {
        model = new Model();
        if (!model.loadModel(context, MODEL_PATH)) {
            Log.e(TAG, "Load model failed: " + MODEL_PATH);
            modelReady = false;
            return false;
        }

        MSConfig config = new MSConfig();
        if (!config.init(DeviceType.DT_CPU, NUM_THREADS, CpuBindMode.MID_CPU)) {
            Log.e(TAG, "Init MindSpore Lite config failed");
            modelReady = false;
            return false;
        }

        session = new LiteSession();
        if (!session.init(config)) {
            Log.e(TAG, "Init MindSpore Lite session failed");
            config.free();
            modelReady = false;
            return false;
        }
        config.free();

        if (!session.compileGraph(model)) {
            Log.e(TAG, "Compile MindSpore Lite graph failed");
            model.freeBuffer();
            modelReady = false;
            return false;
        }
        model.freeBuffer();
        modelReady = true;
        return true;
    }

    public boolean isModelReady() {
        return modelReady;
    }

    public DigitResult execute(Bitmap bitmap) {
        long totalStart = SystemClock.uptimeMillis();
        long preStart = SystemClock.uptimeMillis();
        ByteBuffer input = preprocess(bitmap);
        long preprocessTime = SystemClock.uptimeMillis() - preStart;

        if (!modelReady || session == null) {
            return runFallback(bitmap, preprocessTime, SystemClock.uptimeMillis() - totalStart);
        }

        List<MSTensor> inputs = session.getInputs();
        if (inputs == null || inputs.isEmpty()) {
            return runFallback(bitmap, preprocessTime, SystemClock.uptimeMillis() - totalStart);
        }
        inputs.get(0).setData(input);

        long inferStart = SystemClock.uptimeMillis();
        if (!session.runGraph()) {
            Log.e(TAG, "Run MindSpore Lite graph failed");
            return runFallback(bitmap, preprocessTime, SystemClock.uptimeMillis() - totalStart);
        }
        long inferenceTime = SystemClock.uptimeMillis() - inferStart;

        float[] scores = readOutputScores();
        if (scores == null || scores.length < 10) {
            return runFallback(bitmap, preprocessTime, SystemClock.uptimeMillis() - totalStart);
        }
        return buildResult(scores, preprocessTime, inferenceTime, SystemClock.uptimeMillis() - totalStart, false);
    }

    private float[] readOutputScores() {
        List<String> tensorNames = session.getOutputTensorNames();
        Map<String, MSTensor> outputs = session.getOutputMapByTensor();
        if (tensorNames == null || tensorNames.isEmpty() || outputs == null) {
            return null;
        }
        MSTensor output = outputs.get(tensorNames.get(0));
        if (output == null) {
            return null;
        }
        return output.getFloatData();
    }

    private ByteBuffer preprocess(Bitmap source) {
        Bitmap normalized = normalizeToSquare(source);
        Bitmap scaled = Bitmap.createScaledBitmap(normalized, INPUT_SIZE, INPUT_SIZE, true);
        ByteBuffer input = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 4);
        input.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        for (int pixel : pixels) {
            float value = Color.red(pixel) / 255.0f;
            input.putFloat(value);
        }
        input.rewind();

        if (normalized != source) {
            normalized.recycle();
        }
        scaled.recycle();
        return input;
    }

    private Bitmap normalizeToSquare(Bitmap source) {
        RectF bounds = findInkBounds(source);
        int side = Math.max(1, Math.max((int) bounds.width(), (int) bounds.height()));
        float scale = 20f / side;
        float dx = (INPUT_SIZE - bounds.width() * scale) / 2f - bounds.left * scale;
        float dy = (INPUT_SIZE - bounds.height() * scale) / 2f - bounds.top * scale;

        Bitmap output = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(output);
        canvas.drawColor(Color.BLACK);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.translate(dx, dy);
        canvas.scale(scale, scale);
        canvas.drawBitmap(source, 0, 0, paint);
        return output;
    }

    private RectF findInkBounds(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int left = width;
        int right = 0;
        int top = height;
        int bottom = 0;
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = pixels[y * width + x];
                if (Color.red(pixel) > 20 || Color.green(pixel) > 20 || Color.blue(pixel) > 20) {
                    left = Math.min(left, x);
                    right = Math.max(right, x);
                    top = Math.min(top, y);
                    bottom = Math.max(bottom, y);
                }
            }
        }
        if (left > right || top > bottom) {
            return new RectF(0, 0, width, height);
        }
        return new RectF(left, top, right + 1, bottom + 1);
    }

    private DigitResult runFallback(Bitmap bitmap, long preprocessTime, long totalTime) {
        float[] scores = new float[10];
        int fallbackDigit = estimateDigit(bitmap);
        for (int i = 0; i < scores.length; i++) {
            scores[i] = i == fallbackDigit ? 0.72f : 0.03f;
        }
        return buildResult(scores, preprocessTime, 0, totalTime, true);
    }

    private int estimateDigit(Bitmap bitmap) {
        RectF bounds = findInkBounds(bitmap);
        float width = Math.max(1f, bounds.width());
        float height = Math.max(1f, bounds.height());
        float ratio = height / width;
        if (ratio > 1.8f) {
            return 1;
        }
        if (ratio < 0.85f) {
            return 7;
        }
        return 0;
    }

    private DigitResult buildResult(float[] rawScores, long preprocessTime, long inferenceTime,
                                    long totalTime, boolean fallback) {
        float[] probabilities = softmaxIfNeeded(rawScores);
        List<Score> scores = new ArrayList<>();
        for (int i = 0; i < 10 && i < probabilities.length; i++) {
            scores.add(new Score(i, probabilities[i]));
        }
        Collections.sort(scores, (left, right) -> Float.compare(right.confidence, left.confidence));
        return new DigitResult(scores.get(0).digit, scores.get(0).confidence,
                scores.subList(0, Math.min(3, scores.size())), preprocessTime, inferenceTime,
                totalTime, fallback);
    }

    private float[] softmaxIfNeeded(float[] values) {
        float sum = 0f;
        boolean looksLikeProbabilities = true;
        for (int i = 0; i < 10 && i < values.length; i++) {
            sum += values[i];
            if (values[i] < 0f || values[i] > 1f) {
                looksLikeProbabilities = false;
            }
        }
        if (looksLikeProbabilities && Math.abs(sum - 1f) < 0.05f) {
            return values;
        }

        float max = values[0];
        for (int i = 1; i < 10 && i < values.length; i++) {
            max = Math.max(max, values[i]);
        }
        float expSum = 0f;
        float[] probabilities = new float[Math.min(10, values.length)];
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = (float) Math.exp(values[i] - max);
            expSum += probabilities[i];
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = probabilities[i] / expSum;
        }
        return probabilities;
    }

    public void release() {
        if (model != null) {
            model.free();
        }
        modelReady = false;
    }

    public static class DigitResult {
        public final int digit;
        public final float confidence;
        public final List<Score> topResults;
        public final long preprocessTimeMs;
        public final long inferenceTimeMs;
        public final long totalTimeMs;
        public final boolean fallback;

        DigitResult(int digit, float confidence, List<Score> topResults, long preprocessTimeMs,
                    long inferenceTimeMs, long totalTimeMs, boolean fallback) {
            this.digit = digit;
            this.confidence = confidence;
            this.topResults = topResults;
            this.preprocessTimeMs = preprocessTimeMs;
            this.inferenceTimeMs = inferenceTimeMs;
            this.totalTimeMs = totalTimeMs;
            this.fallback = fallback;
        }
    }

    public static class Score {
        public final int digit;
        public final float confidence;

        Score(int digit, float confidence) {
            this.digit = digit;
            this.confidence = confidence;
        }
    }
}
