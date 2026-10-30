package com.olsc.removemosaic;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapShader;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Shader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * ImageProcessor runs the ONNX inference pipeline that restores the selected
 * region of an image.
 *
 * The model (big-lama ONNX export) has a fixed 1x3x512x512 input, so the
 * pipeline works in one of two modes depending on the size of the selection:
 *
 * <ul>
 * <li><b>Compact selections</b> (context window up to {@link #TILE_TRIGGER}
 * source pixels): the bounding box of all selections plus padding is expanded
 * to a square, resampled once onto the 512x512 model canvas (an upscale for
 * tiny selections — harmless; at most a 1.25x downscale at the trigger point),
 * a single inference runs, and the crop's sub-rect is scaled back, sharpened,
 * and alpha-blended through a feathered selection mask.</li>
 * <li><b>Large selections</b>: a single 512x512 pass would clamp the whole
 * area to a hard 512-pixel ceiling and the upscaled result would read as
 * blur (the on-device "worse blurry" reports). Instead the area is covered by
 * a grid of overlapping 512x512 tiles sampled 1:1 from the ORIGINAL image, so
 * the restoration resolves at native resolution. Tiles that do not touch the
 * selection (plus the blend feather) are skipped; the remaining tiles are
 * blended with smooth cosine weights so their seams cancel out.</li>
 * </ul>
 *
 * Model loading never materializes the ~200 MB model as a Java byte[]: the
 * asset is streamed once into filesDir and the ONNX session is created
 * directly from that path. (Loading through a ByteArrayOutputStream proved
 * fatal on devices with a 256 MB heap: its growth-by-doubling needs a
 * 256 MB + 16 B array, which is larger than the entire heap.)
 */
public class ImageProcessor {

    /**
     * Fixed input side required by the bundled ONNX model (big-lama ONNX
     * export: 1x3x512x512 float [0,1] "image" + 1x1x512x512 {0,1} "mask",
     * 1 marks the restore zone; output arrives at 255 scale).
     */
    private static final int MODEL_INPUT_SIZE = 512;
    private static final int CROP_PADDING = 32;

    /** Asset name of the packaged ONNX model; also the extracted file name. */
    private static final String MODEL_ASSET = "model.onnx";
    /** Chunk size for asset extraction; small on purpose, never heap-sized. */
    private static final int COPY_BUFFER = 64 * 1024;
    /** Free internal storage required before extracting the ~200 MB model. */
    private static final long MIN_FREE_FOR_EXTRACT = 220L * 1024 * 1024;

    /**
     * Side of the intermediate square canvas the model input is composed on
     * (single-pass mode). Equal to the model resolution, so the compose step
     * feeds the network 1:1 with no additional lossy resampling in between.
     */
    private static final int WORK_SIZE = 512;

    /** Sharpening strength used when the caller does not specify one. */
    public static final float DEFAULT_SHARPEN = 0.6f;

    // -----------------------------------------------------------------
    // Tiled-mode constants (large selections).
    // -----------------------------------------------------------------

    /** Tile side; identical to the model input side so tiles feed the
     *  network 1:1 at native image resolution. */
    private static final int TILE_SIZE = 512;

    /** Overlap between neighbouring tiles. Smooth cosine blending across the
     *  overlap hides the seams; 128px (25%) also guarantees every restored
     *  pixel is inferred with plenty of surrounding context. */
    private static final int TILE_OVERLAP = 128;

    /** Extra context around the selection bbox covered by the tile grid, so
     *  hole pixels near the selection border never sit next to a tile edge. */
    private static final int TILE_CONTEXT_MARGIN = 64;

    /**
     * Selections whose square context window exceeds this side are processed
     * in tiled mode. Below it the single-pass resample loses at most ~1.25x
     * linear resolution, which the sharpening pass fully compensates.
     */
    private static final int TILE_TRIGGER = 640;

    /**
     * Tiles whose window misses every selection by more than this margin are
     * skipped. The blend feather is at most 6px; 8px covers it with slack, so
     * a skipped tile can never be needed for a pixel the blend mask touches.
     */
    private static final int TILE_FEATHER_MARGIN = 8;

    private OrtSession session;
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();

    /** Per-axis tile blending weights, indexed by pixel offset inside the
     *  tile: cosine ramp across the overlap zones at both ends, 1 in the
     *  core. Shared by every tile. */
    private static final float[] TILE_WEIGHT_1D = buildTileWeight1D();

    private static float[] buildTileWeight1D() {
        float[] w = new float[TILE_SIZE];
        for (int rel = 0; rel < TILE_SIZE; rel++) {
            w[rel] = tileAxisWeight(rel);
        }
        return w;
    }

    private static float tileAxisWeight(int rel) {
        if (rel < 0 || rel >= TILE_SIZE) return 0f;
        final int o = TILE_OVERLAP;
        if (rel < o) {
            return 0.5f * (1f - (float) Math.cos(Math.PI * rel / o));
        }
        if (rel > TILE_SIZE - o) {
            return 0.5f * (1f - (float) Math.cos(Math.PI * (TILE_SIZE - rel) / o));
        }
        return 1f;
    }

    /** Callback for coarse progress reporting during long restorations. */
    public interface ProgressListener {
        /**
         * Called after each model inference finishes (and once with
         * {@code done == 0} before the first). {@code total} is the number of
         * inference passes the current run will perform; 1 in single-pass
         * mode. Invoked on the calling (worker) thread.
         */
        void onProgress(int done, int total);
    }

    // -----------------------------------------------------------------
    // Model lifecycle
    // -----------------------------------------------------------------

    /** True when the model asset has already been extracted for the current
     *  install, so the UI can tell a one-time first-run extraction from a
     *  fast cached load. */
    public static boolean isModelCached(Context context) {
        final File dir = context.getFilesDir();
        final File target = new File(dir, MODEL_ASSET);
        final File stamp = new File(dir, MODEL_ASSET + ".stamp");
        try {
            return target.isFile() && target.length() > 0
                    && stamp.isFile()
                    && installStamp(context).equals(readStampFile(stamp));
        } catch (IOException unreadableStamp) {
            return false; // unreadable stamp == not cached -> will re-extract
        }
    }

    /**
     * Loads the ONNX model for inference.
     *
     * The packaged asset (about 200 MB for big-lama) is streamed out of the
     * APK to filesDir with a small fixed buffer and the session is created
     * from the resulting FILE PATH. The model is never buffered in the Java
     * heap: holding a 200+ MB model in a byte[] needs a heap several times
     * the file size (ByteArrayOutputStream doubling plus the toByteArray
     * copy), which is impossible on devices with a 256 MB heap cap — it died
     * with OutOfMemoryError in the field. ORT keeps the model in native
     * memory instead, which is not subject to the Java heap limit.
     *
     * The extracted copy is stamped with the package's lastUpdateTime, so an
     * app update always re-extracts (a model swap must never serve a stale
     * file) while a normal launch reuses the copy at zero cost. Extraction
     * is crash-safe: write to a temp file, fsync, rename; the stamp is only
     * written after the target file is complete.
     *
     * Idempotent and thread-safe.
     */
    public synchronized void loadModel(Context context) throws IOException {
        if (session != null) return;

        final File dir = context.getFilesDir();
        final File target = new File(dir, MODEL_ASSET);
        final File stamp = new File(dir, MODEL_ASSET + ".stamp");

        // Self-healing: if a previously extracted copy fails to produce a
        // working session (corrupt file), purge it and retry once with a
        // fresh extraction instead of failing on a bad cache forever.
        for (int attempt = 0; attempt < 2; attempt++) {
            final boolean fresh = ensureModelExtracted(context, dir, target, stamp);
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                session = env.createSession(target.getAbsolutePath(), options);
                return;
            } catch (Exception e) {
                if (fresh || attempt == 1) {
                    // Wrap and preserve the root cause instead of swallowing it.
                    throw new IOException("Failed to create ONNX session: "
                            + e.getMessage(), e);
                }
            }
        }
    }

    /**
     * Makes sure {@code target} holds the packaged model asset. Returns true
     * when this call extracted it, false when an existing stamped copy from
     * the current install was reused. Throws FileNotFoundException when the
     * APK ships no model at all.
     */
    private static boolean ensureModelExtracted(Context context, File dir,
                                                 File target, File stamp)
            throws IOException {
        final String expect = installStamp(context);
        boolean cached = false;
        try {
            cached = target.isFile() && target.length() > 0
                    && stamp.isFile() && expect.equals(readStampFile(stamp));
        } catch (IOException unreadableStamp) {
            cached = false; // unreadable stamp == not stamped -> re-extract
        }
        if (cached) return false;

        // The model is roughly 200 MB on disk; fail fast with a clear
        // message instead of filling the flash halfway into a copy.
        if (dir.getUsableSpace() < MIN_FREE_FOR_EXTRACT) {
            throw new IOException("Not enough free storage to extract the AI model "
                    + "(about " + (MIN_FREE_FOR_EXTRACT / (1024 * 1024)) + " MB needed)");
        }

        final File tmp = new File(dir, MODEL_ASSET + ".tmp");
        try {
            streamAssetToFile(context, MODEL_ASSET, tmp);
            if (tmp.length() == 0) {
                throw new IOException("model.onnx is empty");
            }
            if (!tmp.renameTo(target)) {
                throw new IOException("Failed to move the extracted model into place");
            }
            writeStampFile(stamp, expect);
        } catch (FileNotFoundException e) {
            throw new FileNotFoundException("model.onnx is missing from app assets");
        } finally {
            // No-op after a successful rename; cleans up the half-written
            // temp file when the copy failed midway.
            tmp.delete();
        }
        return true;
    }

    /** Copies an asset to a file with a small fixed buffer (never heap-sized). */
    private static void streamAssetToFile(Context context, String asset, File out)
            throws IOException {
        final byte[] buf = new byte[COPY_BUFFER];
        try (InputStream is = context.getAssets().open(asset);
             FileOutputStream fos = new FileOutputStream(out, false)) {
            int n;
            while ((n = is.read(buf)) > 0) {
                fos.write(buf, 0, n);
            }
            fos.flush();
            fos.getFD().sync();
        }
    }

    /** Package update stamp; changes on every install or update of the APK. */
    private static String installStamp(Context context) {
        try {
            final android.content.pm.PackageManager pm = context.getPackageManager();
            final long t;
            if (android.os.Build.VERSION.SDK_INT >= 33) {
                t = pm.getPackageInfo(context.getPackageName(),
                        android.content.pm.PackageManager.PackageInfoFlags.of(0)).lastUpdateTime;
            } else {
                t = pm.getPackageInfo(context.getPackageName(), 0).lastUpdateTime;
            }
            return Long.toString(t);
        } catch (Exception e) {
            return "unknown"; // fall back to always-extract behaviour
        }
    }

    private static String readStampFile(File f) throws IOException {
        try (FileInputStream in = new FileInputStream(f)) {
            final StringBuilder sb = new StringBuilder(32);
            final byte[] b = new byte[32];
            int n;
            while ((n = in.read(b)) > 0) {
                sb.append(new String(b, 0, n, StandardCharsets.US_ASCII));
            }
            return sb.toString().trim();
        }
    }

    private static void writeStampFile(File f, String value) throws IOException {
        try (FileOutputStream out = new FileOutputStream(f, false)) {
            out.write(value.getBytes(StandardCharsets.US_ASCII));
            out.flush();
            out.getFD().sync();
        }
    }

    /** Releases the native ONNX session. Safe to call multiple times. */
    public synchronized void close() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                // Native release failed; dropping the reference is still the
                // best we can do from a cleanup path.
            }
            session = null;
        }
    }

    // -----------------------------------------------------------------
    // Restoration entry points
    // -----------------------------------------------------------------

    /** Compatibility overload: restores with the default sharpening strength. */
    public Bitmap process(Bitmap origImage, List<RectF> rectangles) throws Exception {
        return process(origImage, rectangles, DEFAULT_SHARPEN, null);
    }

    /** Compatibility overload: restores with an explicit sharpening strength. */
    public Bitmap process(Bitmap origImage, List<RectF> rectangles,
                          float sharpenStrength) throws Exception {
        return process(origImage, rectangles, sharpenStrength, null);
    }

    /**
     * Core restoration logic. {@code sharpenStrength} is a 0..1 user control
     * (0 = raw model output, 1 = maximum unsharp mask); values outside the
     * range are clamped. {@code progress} is optional.
     */
    public synchronized Bitmap process(Bitmap origImage, List<RectF> rectangles,
                                       float sharpenStrength,
                                       ProgressListener progress) throws Exception {
        if (session == null) {
            throw new IllegalStateException("ONNX model not loaded");
        }
        if (origImage == null || rectangles == null || rectangles.isEmpty()) {
            return origImage;
        }
        final float strength = Math.max(0f, Math.min(1f, sharpenStrength));

        final int width = origImage.getWidth();
        final int height = origImage.getHeight();

        // Clamp every rectangle to the image bounds and compute the bounding
        // box of the usable selections. Selections made in the letterbox area
        // outside the picture are ignored instead of crashing the crop step.
        List<RectF> valid = new ArrayList<>();
        int minX = width, minY = height, maxX = 0, maxY = 0;
        for (RectF rect : rectangles) {
            float l = Math.max(0f, rect.left);
            float t = Math.max(0f, rect.top);
            float r = Math.min((float) width, rect.right);
            float b = Math.min((float) height, rect.bottom);
            if (r - l <= 0f || b - t <= 0f) continue;
            valid.add(new RectF(l, t, r, b));
            minX = Math.min(minX, (int) l);
            minY = Math.min(minY, (int) t);
            maxX = Math.max(maxX, (int) Math.ceil(r));
            maxY = Math.max(maxY, (int) Math.ceil(b));
        }
        if (minX >= maxX || minY >= maxY) {
            return origImage;
        }

        // Expand the box so the network can perceive its surroundings.
        minX = Math.max(0, minX - CROP_PADDING);
        minY = Math.max(0, minY - CROP_PADDING);
        maxX = Math.min(width, maxX + CROP_PADDING);
        maxY = Math.min(height, maxY + CROP_PADDING);

        final int cropW = maxX - minX;
        final int cropH = maxY - minY;

        if (Math.max(cropW, cropH) > TILE_TRIGGER) {
            return processTiled(origImage, valid, minX, minY, maxX, maxY,
                    strength, progress);
        }
        return processSinglePass(origImage, valid, minX, minY, maxX, maxY,
                strength, progress);
    }

    // -----------------------------------------------------------------
    // Single-pass mode (compact selections)
    // -----------------------------------------------------------------

    private Bitmap processSinglePass(Bitmap origImage, List<RectF> valid,
                                     int minX, int minY, int maxX, int maxY,
                                     float strength, ProgressListener progress) throws Exception {
        final int width = origImage.getWidth();
        final int height = origImage.getHeight();
        final int cropW = maxX - minX;
        final int cropH = maxY - minY;

        // ------------------------------------------------------------------
        // Square context window. The model input is square; feeding
        // it a stretched non-square crop makes it reconstruct distorted
        // geometry, which reads as blur and warping after scaling back. The
        // window is centered on the crop, shifted to stay inside the image,
        // and whatever still sticks out (image smaller than the window) is
        // edge-replicated below — standard clamp-to-edge padding for CNNs.
        // ------------------------------------------------------------------
        final int side = Math.max(cropW, cropH);
        int sqX = minX - (side - cropW) / 2;
        int sqY = minY - (side - cropH) / 2;
        if (sqX < 0) sqX = 0;
        if (sqX + side > width) sqX = width - side; // may stay negative when side > width
        if (sqY < 0) sqY = 0;
        if (sqY + side > height) sqY = height - side;

        final int srcX = Math.max(0, sqX);
        final int srcY = Math.max(0, sqY);
        final int srcW = Math.min(width, sqX + side) - srcX;
        final int srcH = Math.min(height, sqY + side) - srcY;
        final int leftPad = srcX - sqX;   // >= 0
        final int topPad = srcY - sqY;    // >= 0

        // ------------------------------------------------------------------
        // Compose the WORK_SIZE x WORK_SIZE working square: downscale the
        // real pixels with progressive halving, then draw them through a
        // CLAMP shader so out-of-image strips replicate the edge pixels.
        // ------------------------------------------------------------------
        final float k = WORK_SIZE / (float) side;
        final int interW = Math.max(1, Math.round(srcW * k));
        final int interH = Math.max(1, Math.round(srcH * k));
        final Bitmap inter = downscaleRegion(origImage, srcX, srcY, srcW, srcH, interW, interH);

        final Bitmap squareWork = Bitmap.createBitmap(WORK_SIZE, WORK_SIZE, Bitmap.Config.ARGB_8888);
        final Canvas squareCanvas = new Canvas(squareWork);
        final Matrix shaderMatrix = new Matrix();
        shaderMatrix.setScale(interW / (float) srcW, interH / (float) srcH);
        shaderMatrix.preTranslate(-leftPad, -topPad);
        shaderMatrix.preScale(side / (float) WORK_SIZE, side / (float) WORK_SIZE);
        final BitmapShader shader = new BitmapShader(inter,
                Shader.TileMode.CLAMP, Shader.TileMode.CLAMP);
        shader.setLocalMatrix(shaderMatrix);
        final Paint shaderPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
        shaderPaint.setShader(shader);
        squareCanvas.drawRect(0, 0, WORK_SIZE, WORK_SIZE, shaderPaint);

        // Model mask on the same working square (white = restore). Kept
        // binary: the blend mask is feathered separately further down.
        final Bitmap maskWork = Bitmap.createBitmap(WORK_SIZE, WORK_SIZE, Bitmap.Config.ARGB_8888);
        final Canvas maskCanvas = new Canvas(maskWork);
        maskCanvas.drawColor(Color.BLACK);
        final Paint whitePaint = new Paint();
        whitePaint.setColor(Color.WHITE);
        whitePaint.setStyle(Paint.Style.FILL);
        whitePaint.setAntiAlias(false);
        maskCanvas.save();
        maskCanvas.scale(k, k);
        maskCanvas.translate(-sqX, -sqY);
        for (RectF rect : valid) {
            maskCanvas.drawRect(rect, whitePaint);
        }
        maskCanvas.restore();

        // Final step to the model's resolution (1:1 when WORK_SIZE equals
        // MODEL_INPUT_SIZE; createScaledBitmap would still copy, so skip it).
        final Bitmap modelImg = (squareWork.getWidth() == MODEL_INPUT_SIZE)
                ? squareWork
                : Bitmap.createScaledBitmap(squareWork, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);
        final Bitmap modelMask = (maskWork.getWidth() == MODEL_INPUT_SIZE)
                ? maskWork
                : Bitmap.createScaledBitmap(maskWork, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);

        if (progress != null) progress.onProgress(0, 1);
        final Bitmap outBitmap = runInference(modelImg, modelMask);
        if (progress != null) progress.onProgress(1, 1);

        // ------------------------------------------------------------------
        // Cut the crop's sub-rect out of the model-resolution output. Because the
        // square mapping is uniform, the sub-rect has the crop's aspect
        // ratio — scaling it back to cropW x cropH involves no distortion.
        // ------------------------------------------------------------------
        int subX = Math.round((minX - sqX) * MODEL_INPUT_SIZE / (float) side);
        int subY = Math.round((minY - sqY) * MODEL_INPUT_SIZE / (float) side);
        int subW = Math.round(cropW * MODEL_INPUT_SIZE / (float) side);
        int subH = Math.round(cropH * MODEL_INPUT_SIZE / (float) side);
        subW = Math.max(1, Math.min(subW, MODEL_INPUT_SIZE));
        subH = Math.max(1, Math.min(subH, MODEL_INPUT_SIZE));
        // Derive the smaller dimension from the larger one so the sub-rect
        // keeps the crop's aspect ratio; rounding both independently can
        // compound up to several percent on extreme sliver selections.
        if (cropW >= cropH) {
            subH = Math.max(1, Math.min(MODEL_INPUT_SIZE,
                    Math.round(subW * cropH / (float) cropW)));
        } else {
            subW = Math.max(1, Math.min(MODEL_INPUT_SIZE,
                    Math.round(subH * cropW / (float) cropH)));
        }
        subX = Math.min(Math.max(0, subX), MODEL_INPUT_SIZE - subW);
        subY = Math.min(Math.max(0, subY), MODEL_INPUT_SIZE - subH);

        final Bitmap cropOut = Bitmap.createBitmap(outBitmap, subX, subY, subW, subH);
        final Bitmap restored = Bitmap.createScaledBitmap(cropOut, cropW, cropH, true);

        // ------------------------------------------------------------------
        // Blend mask at crop resolution, feathered a few pixels so the seam
        // between original and restored content is a smooth transition
        // instead of a hard step that emphasizes the restored region.
        // ------------------------------------------------------------------
        final Paint whitePaint2 = new Paint();
        whitePaint2.setColor(Color.WHITE);
        whitePaint2.setStyle(Paint.Style.FILL);
        whitePaint2.setAntiAlias(false);
        final int[] maskPixels = buildFeatheredMask(valid, minX, minY, cropW, cropH, whitePaint2);

        // ------------------------------------------------------------------
        // Unsharp mask on the restored pixels, at the FINAL resolution so it
        // puts back the acutance the model-resolution ceiling and the bilinear
        // upscale took away. Runs channel-by-channel to keep the memory
        // footprint at a single extra int[] regardless of crop size.
        // ------------------------------------------------------------------
        int[] outPixels = new int[cropW * cropH];
        restored.getPixels(outPixels, 0, cropW, 0, 0, cropW, cropH);
        if (strength > 0.01f) {
            final float upscale = side / (float) MODEL_INPUT_SIZE;
            final int radius = Math.max(1, Math.min(4, Math.round(upscale * 0.75f)));
            unsharpMask(outPixels, cropW, cropH, 1.2f * strength, radius, 3);
        }

        // Recycle intermediates. createScaledBitmap()/createBitmap() may
        // return the source object itself when no scaling/subsetting is
        // needed, so identity is checked to avoid double-recycling.
        inter.recycle();
        squareWork.recycle();
        maskWork.recycle();
        if (modelImg != squareWork) modelImg.recycle();
        if (modelMask != maskWork) modelMask.recycle();
        if (cropOut != outBitmap) cropOut.recycle();
        if (restored != cropOut) restored.recycle();
        outBitmap.recycle();

        return blendIntoOriginal(origImage, outPixels, maskPixels,
                minX, minY, cropW, cropH);
    }

    // -----------------------------------------------------------------
    // Tiled mode (large selections) — native-resolution restoration
    // -----------------------------------------------------------------

    private Bitmap processTiled(Bitmap origImage, List<RectF> valid,
                                int minX, int minY, int maxX, int maxY,
                                float strength, ProgressListener progress) throws Exception {
        final int cropW = maxX - minX;
        final int cropH = maxY - minY;

        // Tile grid over the crop expanded by a context margin. Tiles at the
        // grid border stick out past the image on edge selections; the CLAMP
        // shader below edge-replicates those strips.
        final int gx0 = minX - TILE_CONTEXT_MARGIN;
        final int gy0 = minY - TILE_CONTEXT_MARGIN;
        final int[] xs = tileStarts(gx0, (maxX + TILE_CONTEXT_MARGIN) - gx0);
        final int[] ys = tileStarts(gy0, (maxY + TILE_CONTEXT_MARGIN) - gy0);

        // Feathered blend mask first — it does not depend on the restored
        // content, and the skip test below leans on the guarantee that only
        // pixels under (selection + feather) ever take model output.
        final Paint whitePaint = new Paint();
        whitePaint.setColor(Color.WHITE);
        whitePaint.setStyle(Paint.Style.FILL);
        whitePaint.setAntiAlias(false);
        final int[] maskPixels = buildFeatheredMask(valid, minX, minY, cropW, cropH, whitePaint);

        // Only tiles whose window comes within the feather margin of a
        // selection can contribute a pixel the blend mask touches; the rest
        // would run a full inference for output that is blended away.
        final List<int[]> runTiles = new ArrayList<>();
        for (int ty : ys) {
            for (int tx : xs) {
                if (tileTouchesSelection(tx, ty, valid)) {
                    runTiles.add(new int[]{tx, ty});
                }
            }
        }
        final int totalTiles = runTiles.size();
        if (progress != null) progress.onProgress(0, totalTiles);

        final int n = cropW * cropH;
        final float[] accR = new float[n];
        final float[] accG = new float[n];
        final float[] accB = new float[n];
        final float[] accW = new float[n];

        final int[] tilePixels = new int[TILE_SIZE * TILE_SIZE];
        final Paint shaderPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
        final Paint maskPaint = new Paint();
        maskPaint.setColor(Color.WHITE);
        maskPaint.setStyle(Paint.Style.FILL);
        maskPaint.setAntiAlias(false);

        Bitmap outTile = null;
        try {
            for (int t = 0; t < totalTiles; t++) {
                final int tx = runTiles.get(t)[0];
                final int ty = runTiles.get(t)[1];

                // 1:1 tile of the original image, edge-replicated wherever
                // the tile crosses the image border.
                final Bitmap tileImg = Bitmap.createBitmap(TILE_SIZE, TILE_SIZE, Bitmap.Config.ARGB_8888);
                final BitmapShader shader = new BitmapShader(origImage,
                        Shader.TileMode.CLAMP, Shader.TileMode.CLAMP);
                final Matrix m = new Matrix();
                m.setTranslate(-tx, -ty);
                shader.setLocalMatrix(m);
                shaderPaint.setShader(shader);
                final Canvas imgCanvas = new Canvas(tileImg);
                imgCanvas.drawRect(0, 0, TILE_SIZE, TILE_SIZE, shaderPaint);
                shaderPaint.setShader(null);

                // Binary hole mask for this tile (white = restore).
                final Bitmap tileMask = Bitmap.createBitmap(TILE_SIZE, TILE_SIZE, Bitmap.Config.ARGB_8888);
                final Canvas maskCanvas = new Canvas(tileMask);
                maskCanvas.drawColor(Color.BLACK);
                maskCanvas.save();
                maskCanvas.translate(-tx, -ty);
                for (RectF rect : valid) {
                    maskCanvas.drawRect(rect, maskPaint);
                }
                maskCanvas.restore();

                outTile = runInference(tileImg, tileMask);
                tileImg.recycle();
                tileMask.recycle();

                // Accumulate the tile's output into the crop buffers with
                // its blending weights. Rows/columns outside the crop cannot
                // receive output and are skipped wholesale.
                outTile.getPixels(tilePixels, 0, TILE_SIZE, 0, 0, TILE_SIZE, TILE_SIZE);
                final int py0 = Math.max(ty, minY);
                final int py1 = Math.min(ty + TILE_SIZE, maxY);
                final int px0 = Math.max(tx, minX);
                final int px1 = Math.min(tx + TILE_SIZE, maxX);
                for (int py = py0; py < py1; py++) {
                    final float wy = TILE_WEIGHT_1D[py - ty];
                    if (wy <= 0f) continue;
                    final int rowBase = (py - minY) * cropW;
                    for (int px = px0; px < px1; px++) {
                        final float w = wy * TILE_WEIGHT_1D[px - tx];
                        if (w <= 0f) continue;
                        final int i = rowBase + (px - minX);
                        final int c = tilePixels[(py - ty) * TILE_SIZE + (px - tx)];
                        accR[i] += w * Color.red(c);
                        accG[i] += w * Color.green(c);
                        accB[i] += w * Color.blue(c);
                        accW[i] += w;
                    }
                }
                outTile.recycle();
                outTile = null;
                if (progress != null) progress.onProgress(t + 1, totalTiles);
            }
        } finally {
            if (outTile != null) outTile.recycle();
        }

        // Normalize the weighted sums into restored pixels. Pixels no tile
        // covered (cannot happen for selection pixels — the grid provably
        // covers the whole crop — but be defensive) keep the original.
        int[] outPixels = new int[n];
        int[] cropOrigPixels = new int[n];
        origImage.getPixels(cropOrigPixels, 0, cropW, minX, minY, cropW, cropH);
        for (int i = 0; i < n; i++) {
            final float w = accW[i];
            if (w > 1e-3f) {
                int r = clamp255(Math.round(accR[i] / w));
                int g = clamp255(Math.round(accG[i] / w));
                int b = clamp255(Math.round(accB[i] / w));
                outPixels[i] = Color.argb(255, r, g, b);
            } else {
                outPixels[i] = cropOrigPixels[i];
            }
        }

        // Unsharp mask at native resolution: radius 1 is the right scale —
        // there is no upscale to compensate for, only the model's natural
        // softness.
        if (strength > 0.01f) {
            unsharpMask(outPixels, cropW, cropH, 1.2f * strength, 1, 3);
        }

        return blendIntoOriginal(origImage, outPixels, maskPixels,
                minX, minY, cropW, cropH);
    }

    /**
     * Start coordinates of TILE_SIZE-wide tiles covering [start,
     * start+span): the first tile starts at {@code start}, the last ends
     * exactly at {@code start+span}, and everything in between is spaced
     * uniformly with at least {@link #TILE_OVERLAP} of overlap (exactly
     * uniform when the span is a multiple of the stride).
     */
    private static int[] tileStarts(int start, int span) {
        if (span <= TILE_SIZE) {
            return new int[]{start};
        }
        final int stride = TILE_SIZE - TILE_OVERLAP;
        final int n = (int) Math.ceil((span - TILE_SIZE) / (double) stride) + 1;
        final int[] starts = new int[n];
        final double step = (span - TILE_SIZE) / (double) (n - 1);
        for (int i = 0; i < n; i++) {
            starts[i] = start + (int) Math.round(step * i);
        }
        return starts;
    }

    /** True when the tile's window comes within {@link #TILE_FEATHER_MARGIN}
     *  of any selection, i.e. the tile can feed pixels the feathered blend
     *  mask might touch. */
    private static boolean tileTouchesSelection(int tx, int ty, List<RectF> valid) {
        final float l = tx - TILE_FEATHER_MARGIN;
        final float t = ty - TILE_FEATHER_MARGIN;
        final float r = tx + TILE_SIZE + TILE_FEATHER_MARGIN;
        final float b = ty + TILE_SIZE + TILE_FEATHER_MARGIN;
        for (RectF rect : valid) {
            if (rect.right > l && rect.left < r && rect.bottom > t && rect.top < b) {
                return true;
            }
        }
        return false;
    }

    // -----------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------

    /**
     * Runs one 512x512 inference: packs the (already model-sized) image and
     * binary mask into NCHW float tensors, executes the session, and unpacks
     * the 255-scale output back into a bitmap. Non-finite outputs fall back
     * to the input pixel.
     */
    private Bitmap runInference(Bitmap image, Bitmap mask) throws OrtException {
        final int pixelCount = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
        FloatBuffer imgBuffer = FloatBuffer.allocate(3 * pixelCount);
        FloatBuffer maskBuffer = FloatBuffer.allocate(pixelCount);
        int[] pixelsImg = new int[pixelCount];
        int[] pixelsMask = new int[pixelCount];
        image.getPixels(pixelsImg, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
        mask.getPixels(pixelsMask, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < pixelCount; i++) {
                int color = pixelsImg[i];
                float value;
                if (c == 0) value = Color.red(color) / 255.0f;
                else if (c == 1) value = Color.green(color) / 255.0f;
                else value = Color.blue(color) / 255.0f;
                imgBuffer.put(c * pixelCount + i, value);
            }
        }
        for (int i = 0; i < pixelCount; i++) {
            maskBuffer.put(i, Color.red(pixelsMask[i]) / 255.0f > 0.5f ? 1.0f : 0.0f);
        }

        try (OnnxTensor tImage = OnnxTensor.createTensor(env, imgBuffer,
                new long[]{1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE});
             OnnxTensor tMask = OnnxTensor.createTensor(env, maskBuffer,
                     new long[]{1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE})) {

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("image", tImage);
            inputs.put("mask", tMask);

            try (OrtSession.Result result = session.run(inputs)) {
                float[][][][] outputArr = (float[][][][]) result.get(0).getValue();

                int[] outPixels = new int[pixelCount];
                for (int i = 0; i < pixelCount; i++) {
                    int row = i / MODEL_INPUT_SIZE;
                    int col = i % MODEL_INPUT_SIZE;
                    // The bundled big-lama ONNX export emits its output at 255
                    // scale (0-255 floats) even though the input is [0,1];
                    // normalize back before the 0-255 pixel quantization below.
                    float rf = outputArr[0][0][row][col] / 255.0f;
                    float gf = outputArr[0][1][row][col] / 255.0f;
                    float bf = outputArr[0][2][row][col] / 255.0f;
                    // Guard against non-finite values some models emit:
                    // NaN would otherwise become a black pixel.
                    if (Float.isNaN(rf)) rf = Color.red(pixelsImg[i]) / 255.0f;
                    if (Float.isNaN(gf)) gf = Color.green(pixelsImg[i]) / 255.0f;
                    if (Float.isNaN(bf)) bf = Color.blue(pixelsImg[i]) / 255.0f;

                    int r = Math.min(255, Math.max(0, (int) (rf * 255)));
                    int g = Math.min(255, Math.max(0, (int) (gf * 255)));
                    int b = Math.min(255, Math.max(0, (int) (bf * 255)));
                    outPixels[i] = Color.argb(255, r, g, b);
                }

                Bitmap outBitmap = Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
                        Bitmap.Config.ARGB_8888);
                outBitmap.setPixels(outPixels, 0, MODEL_INPUT_SIZE, 0, 0,
                        MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
                return outBitmap;
            }
        }
    }

    /**
     * Builds the feathered per-pixel blend mask for the crop: white over the
     * selections, black outside, smoothed by a couple of box-blur rounds over
     * a few pixels so the seam between original and restored content is a
     * gradient instead of a hard step.
     */
    private static int[] buildFeatheredMask(List<RectF> valid, int minX, int minY,
                                            int cropW, int cropH, Paint whitePaint) {
        final Bitmap blendMask = Bitmap.createBitmap(cropW, cropH, Bitmap.Config.ARGB_8888);
        final Canvas blendCanvas = new Canvas(blendMask);
        blendCanvas.drawColor(Color.BLACK);
        for (RectF rect : valid) {
            blendCanvas.drawRect(rect.left - minX, rect.top - minY,
                    rect.right - minX, rect.bottom - minY, whitePaint);
        }
        int[] maskPixels = new int[cropW * cropH];
        blendMask.getPixels(maskPixels, 0, cropW, 0, 0, cropW, cropH);
        blendMask.recycle();
        final int feather = Math.max(1, Math.min(6, Math.round(Math.max(cropW, cropH) * 0.006f)));
        for (int i = 0; i < 2; i++) {
            boxBlurH(maskPixels, cropW, cropH, feather);
            boxBlurV(maskPixels, cropW, cropH, feather);
        }
        return maskPixels;
    }

    /**
     * Alpha-blends the restored crop pixels into a copy of the original
     * through the feathered mask and returns the full-size result.
     */
    private static Bitmap blendIntoOriginal(Bitmap origImage, int[] outPixels, int[] maskPixels,
                                            int minX, int minY, int cropW, int cropH) {
        int[] cropOrigPixels = new int[cropW * cropH];
        origImage.getPixels(cropOrigPixels, 0, cropW, minX, minY, cropW, cropH);

        for (int i = 0; i < cropW * cropH; i++) {
            float m = Color.red(maskPixels[i]) / 255.0f;
            int origC = cropOrigPixels[i];
            int outC = outPixels[i];

            int r = Math.round(Color.red(origC) * (1 - m) + Color.red(outC) * m);
            int g = Math.round(Color.green(origC) * (1 - m) + Color.green(outC) * m);
            int b = Math.round(Color.blue(origC) * (1 - m) + Color.blue(outC) * m);

            cropOrigPixels[i] = Color.argb(255, r, g, b);
        }

        Bitmap finalResult = origImage.copy(Bitmap.Config.ARGB_8888, true);
        finalResult.setPixels(cropOrigPixels, 0, cropW, minX, minY, cropW, cropH);
        return finalResult;
    }

    private static int clamp255(int v) {
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }

    /**
     * Downscales a rectangular region of {@code src} to {@code dstW} x
     * {@code dstH} using progressive halving. Plain bilinear skips source
     * pixels when minifying by more than ~2x, which aliases fine line art;
     * halving first keeps every resampling step inside the bilinear-safe
     * window. Always returns a new bitmap (never aliases {@code src}), so
     * the caller can recycle the result unconditionally.
     */
    private static Bitmap downscaleRegion(Bitmap src, int sx, int sy, int sw, int sh,
                                          int dstW, int dstH) {
        final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        Bitmap cur = null;
        Rect curRect = new Rect(sx, sy, sx + sw, sy + sh);
        int curW = sw;
        int curH = sh;

        while (curW > dstW * 2 || curH > dstH * 2) {
            final int hw = Math.max(1, curW / 2);
            final int hh = Math.max(1, curH / 2);
            Bitmap next = Bitmap.createBitmap(hw, hh, Bitmap.Config.ARGB_8888);
            new Canvas(next).drawBitmap(cur != null ? cur : src, curRect,
                    new RectF(0, 0, hw, hh), paint);
            if (cur != null) cur.recycle();
            cur = next;
            curRect = new Rect(0, 0, hw, hh);
            curW = hw;
            curH = hh;
        }

        Bitmap out = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888);
        new Canvas(out).drawBitmap(cur != null ? cur : src, curRect,
                new RectF(0, 0, dstW, dstH), paint);
        if (cur != null) cur.recycle();
        return out;
    }

    /**
     * In-place unsharp mask: out = px + amount * (px - blur(px)). Three
     * box-blur rounds approximate a Gaussian with sigma ~= radius + 0.5.
     * The threshold stops the filter from amplifying noise and JPEG banding
     * in flat areas (anime cel shading has plenty of those). Works one
     * channel at a time; original values are re-read from the packed pixel
     * array so only one scratch buffer is needed.
     */
    private static void unsharpMask(int[] px, int w, int h, float amount, int radius, int threshold) {
        final int n = w * h;
        int[] chan = new int[n];
        for (int ch = 0; ch < 3; ch++) {
            final int shift = 16 - 8 * ch;
            for (int i = 0; i < n; i++) {
                chan[i] = (px[i] >> shift) & 0xFF;
            }
            for (int round = 0; round < 3; round++) {
                boxBlurH(chan, w, h, radius);
                boxBlurV(chan, w, h, radius);
            }
            for (int i = 0; i < n; i++) {
                final int v = (px[i] >> shift) & 0xFF;
                final int d = v - chan[i];
                if (Math.abs(d) >= threshold) {
                    int nv = v + (int) (d * amount);
                    if (nv < 0) nv = 0;
                    else if (nv > 255) nv = 255;
                    px[i] = (px[i] & ~(0xFF << shift)) | (nv << shift);
                }
            }
        }
    }

    /** Horizontal box blur with clamped (edge-extended) borders, in place. */
    private static void boxBlurH(int[] c, int w, int h, int r) {
        final int[] tmp = new int[w];
        final int window = r * 2 + 1;
        for (int y = 0; y < h; y++) {
            final int base = y * w;
            int sum = 0;
            // Prime the sliding window with the clamped left edge.
            for (int i = -r; i <= r; i++) {
                sum += c[base + Math.min(w - 1, Math.max(0, i))];
            }
            for (int x = 0; x < w; x++) {
                tmp[x] = sum / window;
                final int add = Math.min(w - 1, x + r + 1);
                final int sub = Math.max(0, x - r);
                sum += c[base + add] - c[base + sub];
            }
            System.arraycopy(tmp, 0, c, base, w);
        }
    }

    /** Vertical box blur with clamped (edge-extended) borders, in place. */
    private static void boxBlurV(int[] c, int w, int h, int r) {
        final int[] tmp = new int[h];
        final int window = r * 2 + 1;
        for (int x = 0; x < w; x++) {
            int sum = 0;
            for (int i = -r; i <= r; i++) {
                sum += c[Math.min(h - 1, Math.max(0, i)) * w + x];
            }
            for (int y = 0; y < h; y++) {
                tmp[y] = sum / window;
                final int add = Math.min(h - 1, y + r + 1);
                final int sub = Math.max(0, y - r);
                sum += c[add * w + x] - c[sub * w + x];
            }
            for (int y = 0; y < h; y++) {
                c[y * w + x] = tmp[y];
            }
        }
    }
}
