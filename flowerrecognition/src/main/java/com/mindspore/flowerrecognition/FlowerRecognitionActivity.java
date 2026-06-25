package com.mindspore.flowerrecognition;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.alibaba.android.arouter.facade.annotation.Route;

import java.io.File;
import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Route(path = "/flowerrecognition/FlowerRecognitionActivity")
public class FlowerRecognitionActivity extends AppCompatActivity {
    private static final int REQUEST_GALLERY = 1001;
    private static final int REQUEST_CAMERA = 1002;
    private static final int REQUEST_CAMERA_PERMISSION = 1003;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private FlowerRecognitionExecutor executor;
    private ImageView preview;
    private TextView modelStatus;
    private TextView resultView;
    private TextView errorView;
    private TextView timingView;
    private ProgressBar progress;
    private Button recognizeButton;
    private Button retryButton;
    private Button galleryButton;
    private Button cameraButton;
    private Bitmap selectedBitmap;
    private Uri pendingCameraUri;
    private volatile boolean destroyed;
    private boolean modelLoading;
    private boolean modelReady;
    private boolean decoding;
    private boolean inferencing;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_flower_recognition);
        bindViews();
        executor = new FlowerRecognitionExecutor(this);
        loadModel();
    }

    private void bindViews() {
        Toolbar toolbar = findViewById(R.id.flower_toolbar);
        toolbar.setTitle(R.string.flower_title);
        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(view -> finish());

        preview = findViewById(R.id.flower_preview);
        modelStatus = findViewById(R.id.flower_model_status);
        resultView = findViewById(R.id.flower_result);
        errorView = findViewById(R.id.flower_error);
        timingView = findViewById(R.id.flower_timing);
        progress = findViewById(R.id.flower_progress);
        recognizeButton = findViewById(R.id.flower_recognize);
        retryButton = findViewById(R.id.flower_retry_model);
        galleryButton = findViewById(R.id.flower_gallery);
        cameraButton = findViewById(R.id.flower_camera);

        galleryButton.setOnClickListener(view -> openGallery());
        cameraButton.setOnClickListener(view -> requestCamera());
        recognizeButton.setOnClickListener(view -> recognize());
        retryButton.setOnClickListener(view -> loadModel());
        updateControls();
    }

    private void loadModel() {
        if (modelLoading || inferencing || destroyed) {
            return;
        }
        modelLoading = true;
        modelReady = false;
        clearError();
        modelStatus.setText(R.string.flower_model_loading);
        retryButton.setVisibility(View.GONE);
        updateControls();
        worker.execute(() -> {
            FlowerRecognitionExecutor.LoadResult result = executor.loadModel();
            postToUi(() -> {
                modelLoading = false;
                modelReady = result.success;
                if (result.success) {
                    modelStatus.setText(getString(R.string.flower_model_ready,
                            result.runtimeVersion, result.loadTimeMs));
                } else {
                    modelStatus.setText(getString(R.string.flower_model_failed,
                            result.runtimeVersion));
                    showError(result.error);
                    retryButton.setVisibility(View.VISIBLE);
                }
                updateControls();
            });
        });
    }

    private void openGallery() {
        if (decoding || inferencing) {
            return;
        }
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, REQUEST_GALLERY);
    }

    private void requestCamera() {
        if (decoding || inferencing) {
            return;
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA_PERMISSION);
            return;
        }
        launchCamera();
    }

    private void launchCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) == null) {
            showError(getString(R.string.flower_no_camera));
            return;
        }
        try {
            File root = getExternalCacheDir() != null ? getExternalCacheDir() : getCacheDir();
            File directory = new File(root, "flower_images");
            if (!directory.exists() && !directory.mkdirs()) {
                throw new IOException("无法创建相机缓存目录");
            }
            File photo = File.createTempFile("flower_camera_", ".jpg", directory);
            pendingCameraUri = FileProvider.getUriForFile(this,
                    getPackageName() + ".flowerrecognition.fileprovider", photo);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, pendingCameraUri);
            intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivityForResult(intent, REQUEST_CAMERA);
        } catch (IOException | IllegalArgumentException error) {
            showError(getString(R.string.flower_camera_failed, error.getMessage()));
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK) {
            return;
        }
        Uri uri;
        if (requestCode == REQUEST_GALLERY) {
            uri = data == null ? null : data.getData();
            if (uri != null) {
                try {
                    getContentResolver().takePersistableUriPermission(uri,
                            Intent.FLAG_GRANT_READ_URI_PERMISSION);
                } catch (SecurityException ignored) {
                    // The current provider may only grant temporary read access.
                }
            }
        } else if (requestCode == REQUEST_CAMERA) {
            uri = pendingCameraUri;
        } else {
            return;
        }
        decodeImage(uri);
    }

    private void decodeImage(Uri uri) {
        if (uri == null) {
            showError(getString(R.string.flower_empty_uri));
            return;
        }
        decoding = true;
        clearError();
        resultView.setText(R.string.flower_result_placeholder);
        timingView.setText("");
        updateControls();
        worker.execute(() -> {
            try {
                Bitmap bitmap = FlowerImageLoader.decode(this, uri);
                if (destroyed) {
                    bitmap.recycle();
                    return;
                }
                runOnUiThread(() -> {
                    if (destroyed || isFinishing()) {
                        bitmap.recycle();
                        return;
                    }
                    replaceSelectedBitmap(bitmap);
                    decoding = false;
                    updateControls();
                });
            } catch (FlowerImageLoader.ImageLoadException error) {
                postToUi(() -> {
                    decoding = false;
                    showError(error.getMessage());
                    updateControls();
                });
            }
        });
    }

    private void replaceSelectedBitmap(Bitmap bitmap) {
        if (selectedBitmap != null && selectedBitmap != bitmap && !selectedBitmap.isRecycled()) {
            selectedBitmap.recycle();
        }
        selectedBitmap = bitmap;
        preview.setImageBitmap(bitmap);
    }

    private void recognize() {
        if (selectedBitmap == null) {
            showError(getString(R.string.flower_select_first));
            return;
        }
        if (!modelReady) {
            showError(getString(R.string.flower_model_not_ready));
            return;
        }
        if (inferencing || decoding) {
            return;
        }
        inferencing = true;
        clearError();
        resultView.setText(R.string.flower_inferencing);
        timingView.setText("");
        updateControls();
        Bitmap bitmap = selectedBitmap;
        worker.execute(() -> {
            try {
                FlowerRecognitionExecutor.InferenceResult result = executor.recognize(bitmap);
                postToUi(() -> {
                    inferencing = false;
                    showResult(result);
                    updateControls();
                });
            } catch (FlowerRecognitionExecutor.RecognitionException | RuntimeException error) {
                postToUi(() -> {
                    inferencing = false;
                    resultView.setText(R.string.flower_result_placeholder);
                    showError(error.getMessage());
                    updateControls();
                });
            }
        });
    }

    private void showResult(FlowerRecognitionExecutor.InferenceResult result) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < result.topThree.size(); i++) {
            FlowerRecognitionExecutor.Prediction item = result.topThree.get(i);
            text.append(String.format(Locale.US, "%d. %s  %s\n%.2f%%",
                    i + 1, item.label.getChineseName(), item.label.getEnglishName(),
                    item.confidence * 100f));
            if (i < result.topThree.size() - 1) {
                text.append("\n\n");
            }
        }
        resultView.setText(text.toString());
        timingView.setText(getString(R.string.flower_timing, result.preprocessTimeMs,
                result.inferenceTimeMs, result.probabilitySum));
        if (result.unreliable) {
            showError(getString(R.string.flower_unreliable));
        }
    }

    private void clearError() {
        errorView.setText("");
        errorView.setVisibility(View.GONE);
    }

    private void showError(String message) {
        errorView.setText(message == null ? getString(R.string.flower_unknown_error) : message);
        errorView.setVisibility(View.VISIBLE);
    }

    private void updateControls() {
        boolean busy = modelLoading || decoding || inferencing;
        recognizeButton.setEnabled(modelReady && selectedBitmap != null && !busy);
        galleryButton.setEnabled(!decoding && !inferencing);
        cameraButton.setEnabled(!decoding && !inferencing);
        retryButton.setEnabled(!busy);
        progress.setVisibility(busy ? View.VISIBLE : View.GONE);
    }

    private void postToUi(Runnable action) {
        runOnUiThread(() -> {
            if (!destroyed && !isFinishing()) {
                action.run();
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                launchCamera();
            } else {
                Toast.makeText(this, R.string.flower_camera_permission_denied,
                        Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        preview.setImageDrawable(null);
        Bitmap bitmapToRecycle = selectedBitmap;
        selectedBitmap = null;
        worker.execute(() -> {
            executor.release();
            if (bitmapToRecycle != null && !bitmapToRecycle.isRecycled()) {
                bitmapToRecycle.recycle();
            }
        });
        worker.shutdown();
        super.onDestroy();
    }
}
