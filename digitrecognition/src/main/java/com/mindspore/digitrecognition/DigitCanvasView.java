package com.mindspore.digitrecognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class DigitCanvasView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path path = new Path();
    private boolean hasInk;
    private float lastX;
    private float lastY;

    public DigitCanvasView(Context context) {
        super(context);
        init();
    }

    public DigitCanvasView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        setBackgroundColor(Color.BLACK);
        paint.setColor(Color.WHITE);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeWidth(36f);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawPath(path, paint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y);
                lastX = x;
                lastY = y;
                hasInk = true;
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                path.quadTo(lastX, lastY, (x + lastX) / 2f, (y + lastY) / 2f);
                lastX = x;
                lastY = y;
                invalidate();
                return true;
            case MotionEvent.ACTION_UP:
                path.lineTo(x, y);
                invalidate();
                return true;
            default:
                return true;
        }
    }

    public void clear() {
        path.reset();
        hasInk = false;
        invalidate();
    }

    public boolean hasInk() {
        return hasInk;
    }

    public Bitmap exportBitmap() {
        Bitmap bitmap = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        canvas.drawColor(Color.BLACK);
        canvas.drawPath(path, paint);
        return bitmap;
    }
}
