package com.mindspore.digitrecognition;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;

import java.util.Locale;

@Route(path = "/digitrecognition/DigitRecognitionActivity")
public class DigitRecognitionActivity extends AppCompatActivity {
    private DigitCanvasView digitCanvas;
    private TextView resultText;
    private DigitRecognitionExecutor executor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ARouter.getInstance().inject(this);
        setContentView(R.layout.activity_digit_recognition);
        initView();
        initModel();
    }

    private void initView() {
        Toolbar toolbar = findViewById(R.id.digit_toolbar);
        toolbar.setTitle(getString(R.string.digit_title));
        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(view -> finish());

        digitCanvas = findViewById(R.id.digit_canvas);
        resultText = findViewById(R.id.tv_digit_result);
        Button recognizeButton = findViewById(R.id.btn_digit_recognize);
        Button clearButton = findViewById(R.id.btn_digit_clear);

        recognizeButton.setOnClickListener(view -> recognize());
        clearButton.setOnClickListener(view -> {
            digitCanvas.clear();
            resultText.setText(R.string.digit_result_placeholder);
        });
    }

    private void initModel() {
        executor = new DigitRecognitionExecutor(this);
        if (!executor.init()) {
            Toast.makeText(this, R.string.digit_model_loading_failed, Toast.LENGTH_SHORT).show();
        }
    }

    private void recognize() {
        if (!digitCanvas.hasInk()) {
            Toast.makeText(this, R.string.digit_empty_canvas, Toast.LENGTH_SHORT).show();
            return;
        }

        Bitmap bitmap = digitCanvas.exportBitmap();
        DigitRecognitionExecutor.DigitResult result = executor.execute(bitmap);
        bitmap.recycle();
        resultText.setText(formatResult(result));
    }

    private String formatResult(DigitRecognitionExecutor.DigitResult result) {
        StringBuilder builder = new StringBuilder();
        if (result.fallback) {
            builder.append(getString(R.string.digit_model_fallback)).append("\n\n");
        }
        builder.append(String.format(Locale.US, "Digit: %d\nConfidence: %.2f%%\n\n",
                result.digit, result.confidence * 100f));
        builder.append("Top 3:\n");
        for (DigitRecognitionExecutor.Score score : result.topResults) {
            builder.append(String.format(Locale.US, "%d  %.2f%%\n", score.digit,
                    score.confidence * 100f));
        }
        builder.append(String.format(Locale.US,
                "\nPreprocess: %d ms\nInference: %d ms\nTotal: %d ms",
                result.preprocessTimeMs, result.inferenceTimeMs, result.totalTimeMs));
        return builder.toString();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor != null) {
            executor.release();
        }
    }
}
