package com.olsc.removemosaic;

import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.exifinterface.media.ExifInterface;
import androidx.lifecycle.ViewModelProvider;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends AppCompatActivity {

    /** Images larger than this are downsampled on import to avoid OOM. */
    private static final int MAX_IMPORT_DIMENSION = 2560;

    private MosaicView mosaicView;
    private MainViewModel viewModel;

    private View step1Container, step2Container, step3Container;
    private View loadingOverlay;
    private TextView txtLoadingStatus;
    private SeekBar seekSharpen;
    private TextView txtSharpenValue;
    private com.github.chrisbanes.photoview.PhotoView imgResult;
    private View btnProcessAction;
    private View btnSave;

    /** Single background worker; replaces ad-hoc threads so repeated taps
     *  cannot start concurrent inference runs. */
    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    /** Guard so the restore button can only start one job at a time. */
    private final AtomicBoolean processing = new AtomicBoolean(false);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Full-screen immersive experience.
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        // Keep the toolbar below the status bar.
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.app_bar), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(0, systemBars.top, 0, 0);
            return insets;
        });
        // Keep the bottom control cards out of the gesture navigation bar
        // (required on Android 15+ where edge-to-edge is enforced).
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.content_frame), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(0, 0, 0, systemBars.bottom);
            return insets;
        });

        viewModel = new ViewModelProvider(this).get(MainViewModel.class);

        mosaicView = findViewById(R.id.mosaic_view);

        step1Container = findViewById(R.id.step_1_container);
        step2Container = findViewById(R.id.step_2_container);
        step3Container = findViewById(R.id.step_3_container);
        loadingOverlay = findViewById(R.id.loading_overlay);
        txtLoadingStatus = findViewById(R.id.txt_loading_status);
        imgResult = findViewById(R.id.img_result);
        btnProcessAction = findViewById(R.id.btn_process_action);
        btnSave = findViewById(R.id.btn_save);

        // Step 1: import.
        findViewById(R.id.btn_main_pick).setOnClickListener(v -> pickImage());

        // Top toolbar.
        findViewById(R.id.btn_toolbar_pick).setOnClickListener(v -> pickImage());

        // Step 2: area selection.
        findViewById(R.id.btn_undo).setOnClickListener(v -> {
            mosaicView.undoLast();
            syncRectanglesToViewModel();
        });
        findViewById(R.id.btn_clear).setOnClickListener(v -> {
            mosaicView.clearSelection();
            syncRectanglesToViewModel();
        });
        findViewById(R.id.btn_to_step3).setOnClickListener(v -> showStep3());

        // Step 3: preview and processing.
        findViewById(R.id.btn_back_to_selection).setOnClickListener(v -> showStep2());
        btnProcessAction.setOnClickListener(v -> processImage());
        btnSave.setOnClickListener(v -> saveImage());

        // Sharpening strength slider (0-100%). The chosen value is applied by
        // the unsharp mask on the next restoration run; it lives in the
        // ViewModel so it survives rotation.
        seekSharpen = findViewById(R.id.seek_sharpen);
        txtSharpenValue = findViewById(R.id.txt_sharpen_value);
        seekSharpen.setMax(100);
        int initial = Math.round(viewModel.sharpenStrength * 100f);
        seekSharpen.setProgress(initial);
        txtSharpenValue.setText(initial + "%");
        seekSharpen.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                viewModel.sharpenStrength = progress / 100f;
                txtSharpenValue.setText(progress + "%");
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        // Restore an in-progress session after a configuration change
        // (e.g. screen rotation) instead of dropping the user's work.
        if (viewModel.original != null) {
            mosaicView.setImage(viewModel.original);
            mosaicView.restoreRectangles(viewModel.rectangles);
            if (viewModel.processed != null) {
                imgResult.setImageBitmap(viewModel.processed);
                showStep3();
            } else {
                showStep2();
            }
        } else {
            showStep1();
        }

        // Show the disclaimer on first launch.
        checkDisclaimer();
    }

    private void checkDisclaimer() {
        android.content.SharedPreferences prefs = getSharedPreferences("app_prefs", MODE_PRIVATE);
        boolean agreed = prefs.getBoolean("disclaimer_agreed", false);

        if (!agreed) {
            new com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
                    .setTitle(R.string.disclaimer_title)
                    .setMessage(R.string.disclaimer_content)
                    .setCancelable(false)
                    .setPositiveButton(R.string.btn_agree, (dialog, which) ->
                            prefs.edit().putBoolean("disclaimer_agreed", true).apply())
                    .setNegativeButton(R.string.btn_exit, (dialog, which) ->
                            finish())
                    .show();
        }
    }

    // Modern photo picker (no storage permission needed; falls back to the
    // system document picker on devices without the photo picker).
    private final ActivityResultLauncher<PickVisualMediaRequest> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
                if (uri != null) {
                    loadImage(uri);
                }
            });

    private void pickImage() {
        pickImageLauncher.launch(new PickVisualMediaRequest.Builder()
                .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                .build());
    }

    /**
     * Decodes the picked image, applies EXIF orientation so the picture is
     * never sideways, downsamples huge photos, and hands it to the selection
     * view. The stream is fully managed via try-with-resources.
     */
    private void loadImage(Uri uri) {
        Bitmap bitmap;
        try {
            // First pass: read the dimensions only.
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                BitmapFactory.decodeStream(is, null, bounds);
            }

            // Pick a sample size so the longest edge stays within the cap.
            int sampleSize = 1;
            while (bounds.outWidth / sampleSize > MAX_IMPORT_DIMENSION
                    || bounds.outHeight / sampleSize > MAX_IMPORT_DIMENSION) {
                sampleSize *= 2;
            }

            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inSampleSize = sampleSize;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                bitmap = BitmapFactory.decodeStream(is, null, opts);
            }

            if (bitmap == null) {
                Toast.makeText(this, R.string.error_decode_image, Toast.LENGTH_SHORT).show();
                return;
            }

            // Respect the EXIF orientation flag of the source file.
            int rotation = 0;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                ExifInterface exif = new ExifInterface(is);
                switch (exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL)) {
                    case ExifInterface.ORIENTATION_ROTATE_90:  rotation = 90; break;
                    case ExifInterface.ORIENTATION_ROTATE_180: rotation = 180; break;
                    case ExifInterface.ORIENTATION_ROTATE_270: rotation = 270; break;
                    default: break;
                }
            } catch (IOException ignored) {
                // EXIF is optional; decode without rotation when unavailable.
            }
            if (rotation != 0) {
                Matrix m = new Matrix();
                m.postRotate(rotation);
                Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0,
                        bitmap.getWidth(), bitmap.getHeight(), m, true);
                if (rotated != bitmap) {
                    bitmap.recycle();
                    bitmap = rotated;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, R.string.error_load_image, Toast.LENGTH_SHORT).show();
            return;
        }

        if (viewModel.processed != null) {
            viewModel.processed.recycle();
            viewModel.processed = null;
        }
        viewModel.original = bitmap;
        viewModel.rectangles = new ArrayList<>();

        mosaicView.setImage(bitmap);
        showStep2();
    }

    private void syncRectanglesToViewModel() {
        if (viewModel.original != null) {
            viewModel.rectangles = new ArrayList<>(mosaicView.getRectangles());
        }
    }

    // Step 1: import.
    private void showStep1() {
        step1Container.setVisibility(View.VISIBLE);
        step2Container.setVisibility(View.GONE);
        step3Container.setVisibility(View.GONE);
        findViewById(R.id.btn_toolbar_pick).setVisibility(View.GONE);
    }

    // Step 2: area selection.
    private void showStep2() {
        syncRectanglesToViewModel();
        step1Container.setVisibility(View.GONE);
        step2Container.setVisibility(View.VISIBLE);
        step3Container.setVisibility(View.GONE);
        findViewById(R.id.btn_toolbar_pick).setVisibility(View.VISIBLE);
    }

    // Step 3: result preview.
    private void showStep3() {
        if (mosaicView.getRectangles().isEmpty()) {
            Toast.makeText(this, R.string.msg_select_area, Toast.LENGTH_SHORT).show();
            return;
        }
        syncRectanglesToViewModel();
        Bitmap shown = (viewModel.processed != null) ? viewModel.processed : viewModel.original;
        imgResult.setImageBitmap(shown);
        step1Container.setVisibility(View.GONE);
        step2Container.setVisibility(View.GONE);
        step3Container.setVisibility(View.VISIBLE);
        findViewById(R.id.btn_toolbar_pick).setVisibility(View.VISIBLE);
    }

    // Core AI restoration flow.
    private void processImage() {
        if (viewModel.original == null) return;
        if (!processing.compareAndSet(false, true)) return;

        loadingOverlay.setVisibility(View.VISIBLE);
        btnProcessAction.setEnabled(false);
        btnSave.setEnabled(false);
        txtLoadingStatus.setText(getString(R.string.status_waking_ai));

        // Snapshot the selection on the UI thread; the worker must never
        // touch the view's live, mutable list. Fall back to the ViewModel's
        // copy when the selection view is empty (e.g. after rotation while
        // already on the result screen).
        final Bitmap source = viewModel.original;
        final ArrayList<android.graphics.RectF> rects =
                new ArrayList<>(mosaicView.getRectangles());
        if (rects.isEmpty() && viewModel.rectangles != null) {
            rects.addAll(viewModel.rectangles);
        }
        final float sharpen = viewModel.sharpenStrength;

        worker.execute(() -> {
            Bitmap result = null;
            Throwable failure = null;
            try {
                runOnUiThread(() ->
                        txtLoadingStatus.setText(getString(R.string.status_loading_weights)));
                viewModel.processor.loadModel(this);

                runOnUiThread(() ->
                        txtLoadingStatus.setText(getString(R.string.status_performing_magic)));
                result = viewModel.processor.process(source, rects, sharpen);
            } catch (Exception | OutOfMemoryError e) {
                // OutOfMemoryError is an Error, not an Exception: leaving it
                // uncaught killed the process (fatal crash in the field).
                failure = e;
            }

            final Bitmap finalResult = result;
            final Throwable finalFailure = failure;
            runOnUiThread(() -> {
                if (isDestroyed() || isFinishing()) return;
                loadingOverlay.setVisibility(View.GONE);
                btnProcessAction.setEnabled(true);
                btnSave.setEnabled(true);
                processing.set(false);
                if (finalFailure != null) {
                    int msgRes = (finalFailure instanceof FileNotFoundException)
                            ? R.string.error_model_missing
                            : (finalFailure instanceof OutOfMemoryError)
                            ? R.string.error_oom
                            : R.string.error_process_fail_ext;
                    String errorMsg = (finalFailure instanceof FileNotFoundException
                            || finalFailure instanceof OutOfMemoryError)
                            ? getString(msgRes)
                            : getString(msgRes, finalFailure.getMessage());
                    Toast.makeText(MainActivity.this, errorMsg, Toast.LENGTH_LONG).show();
                    return;
                }
                if (finalResult != null) {
                    if (viewModel.processed != null && viewModel.processed != finalResult) {
                        viewModel.processed.recycle();
                    }
                    viewModel.processed = finalResult;
                    imgResult.setImageBitmap(finalResult);
                    Toast.makeText(this, R.string.status_complete, Toast.LENGTH_SHORT).show();
                }
            });
        });
    }

    // Saves the restored image to the gallery.
    private void saveImage() {
        Bitmap toSave = (viewModel.processed != null) ? viewModel.processed : viewModel.original;
        if (toSave == null) return;

        String filename = "Unpixelated_" + System.currentTimeMillis() + ".jpg";

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                // Android 10+: MediaStore insert into Pictures (no permission).
                ContentValues contentValues = new ContentValues();
                contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, filename);
                contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg");
                contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES);
                Uri imageUri = getContentResolver().insert(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
                if (imageUri == null) {
                    throw new IOException("MediaStore rejected the insert");
                }
                try (OutputStream fos = getContentResolver().openOutputStream(imageUri)) {
                    writeJpeg(toSave, fos);
                }
            } else {
                // Android 7-9: WRITE_EXTERNAL_STORAGE covers the shared
                // Pictures folder, which the app does not request. Save into
                // the app's own external Pictures directory (permission-free)
                // and ask the media scanner to index it so it shows in the
                // gallery.
                File dir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
                if (dir == null) dir = getFilesDir();
                File image = new File(dir, filename);
                try (OutputStream fos = new java.io.FileOutputStream(image)) {
                    writeJpeg(toSave, fos);
                }
                MediaScannerConnection.scanFile(this,
                        new String[]{image.getAbsolutePath()},
                        new String[]{"image/jpeg"}, null);
            }
            Toast.makeText(this, R.string.msg_save_success, Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            e.printStackTrace();
            String errorMsg = getString(R.string.error_save_fail, e.getMessage());
            Toast.makeText(this, errorMsg, Toast.LENGTH_SHORT).show();
        }
    }

    private void writeJpeg(Bitmap bitmap, OutputStream os) throws IOException {
        if (os == null) throw new IOException("Output stream unavailable");
        if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 95, os)) {
            throw new IOException("JPEG compression failed");
        }
        os.flush();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Always stop accepting new work when this instance goes away; a
        // plain Executors.newSingleThreadExecutor() keeps its core thread
        // alive forever, which would leak one thread per rotation. The
        // bitmaps and the AI session live in the ViewModel and survive.
        worker.shutdownNow();
    }
}
