package com.olsc.removemosaic;

import android.graphics.Bitmap;
import android.graphics.RectF;

import androidx.lifecycle.ViewModel;

import java.util.ArrayList;
import java.util.List;

/**
 * Retains the in-progress editing session (source image, current selections
 * and the loaded AI model) across configuration changes such as screen
 * rotation. Everything is released in {@link #onCleared()}.
 */
public class MainViewModel extends ViewModel {

    public final ImageProcessor processor = new ImageProcessor();

    /** Source image as imported by the user (EXIF rotation already applied). */
    public Bitmap original;

    /** Latest selections in image coordinates; null until an image is loaded. */
    public List<RectF> rectangles = new ArrayList<>();

    /** Result of the last successful AI restoration; null if none yet. */
    public Bitmap processed;

    /**
     * Post-restoration sharpening strength, 0..1. Controlled by the slider
     * on the result screen and kept here so it survives rotation.
     */
    public float sharpenStrength = 0.6f;

    @Override
    protected void onCleared() {
        processor.close();
        if (original != null) {
            original.recycle();
            original = null;
        }
        if (processed != null) {
            processed.recycle();
            processed = null;
        }
        rectangles = null;
    }
}
