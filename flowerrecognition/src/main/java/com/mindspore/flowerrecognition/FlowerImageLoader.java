package com.mindspore.flowerrecognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

final class FlowerImageLoader {
    private static final int MAX_DECODE_DIMENSION = 2048;

    private FlowerImageLoader() {
    }

    static Bitmap decode(Context context, Uri uri) throws ImageLoadException {
        if (uri == null) {
            throw new ImageLoadException("相册或相机返回了空 URI");
        }
        File cacheFile = null;
        try {
            cacheFile = copyToCache(context, uri);
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(cacheFile.getAbsolutePath(), bounds);
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
                throw new ImageLoadException("无法读取图片尺寸");
            }

            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inSampleSize = calculateSampleSize(bounds.outWidth, bounds.outHeight);
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            Bitmap decoded = BitmapFactory.decodeFile(cacheFile.getAbsolutePath(), options);
            if (decoded == null) {
                throw new ImageLoadException("Bitmap 解码失败");
            }
            return applyExifOrientation(decoded, cacheFile.getAbsolutePath());
        } catch (OutOfMemoryError error) {
            throw new ImageLoadException("图片解码内存不足，请选择分辨率较低的图片", error);
        } catch (IOException | SecurityException error) {
            throw new ImageLoadException("图片读取失败: " + error.getMessage(), error);
        } finally {
            if (cacheFile != null && !cacheFile.delete()) {
                cacheFile.deleteOnExit();
            }
        }
    }

    private static File copyToCache(Context context, Uri uri) throws IOException, ImageLoadException {
        File directory = new File(context.getCacheDir(), "flower_decode");
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("无法创建图片缓存目录");
        }
        File target = File.createTempFile("flower_", ".image", directory);
        try {
            try (InputStream input = context.getContentResolver().openInputStream(uri);
                 OutputStream output = new FileOutputStream(target)) {
                if (input == null) {
                    throw new ImageLoadException("无法打开图片 URI");
                }
                byte[] block = new byte[16 * 1024];
                int count;
                while ((count = input.read(block)) != -1) {
                    output.write(block, 0, count);
                }
            }
            return target;
        } catch (IOException | ImageLoadException | RuntimeException error) {
            if (!target.delete()) {
                target.deleteOnExit();
            }
            throw error;
        }
    }

    private static int calculateSampleSize(int width, int height) {
        int sample = 1;
        while (width / sample > MAX_DECODE_DIMENSION || height / sample > MAX_DECODE_DIMENSION) {
            sample *= 2;
        }
        return sample;
    }

    private static Bitmap applyExifOrientation(Bitmap source, String path) throws IOException {
        ExifInterface exif = new ExifInterface(path);
        int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL);
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180f);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setScale(1f, -1f);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90f);
                matrix.postScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90f);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90f);
                matrix.postScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90f);
                break;
            default:
                return source;
        }

        Bitmap oriented;
        try {
            oriented = Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(),
                    matrix, true);
        } catch (OutOfMemoryError error) {
            source.recycle();
            throw error;
        }
        if (oriented != source) {
            source.recycle();
        }
        return oriented;
    }

    static final class ImageLoadException extends Exception {
        private ImageLoadException(String message) {
            super(message);
        }

        private ImageLoadException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
