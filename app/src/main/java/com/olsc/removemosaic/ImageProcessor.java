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
 * Pipeline: compute the bounding box of all selections (plus padding so the
 * network sees surrounding context) -> expand it to a SQUARE context window
 * (stretching a non-square crop to the model's square input used to distort the
 * geometry the model perceives and blur the result) -> progressively halve
 * the region down to the working resolution (single-step bilinear skips
 * source pixels when minifying more than ~2x and shimmers line art) -> run
 * inference -> extract exactly the crop's sub-rect from the model-resolution output
 * -> scale it back -> sharpen it (unsharp mask; the model regresses to the
 * mean so its output is systematically soft) -> alpha-blend it into the
 * original through a feathered selection mask.
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
     * Side of the intermediate square canvas the model input is composed on.
     * Equal to the model resolution, so the compose step feeds the network
     * 1:1 with no additional lossy resampling in between.
     */
    private static final int WORK_SIZE = 512;

    /** Sharpening strength used when the caller does not specify one. */
    public static final float DEFAULT_SHARPEN = 0.6f;

    private OrtSession session;
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();

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

    /** Compatibility overload: restores with the default sharpening strength. */
    public Bitmap process(Bitmap origImage, List<RectF> rectangles) throws Exception {
        return process(origImage, rectangles, DEFAULT_SHARPEN);
    }

    /**
     * Core restoration logic. {@code sharpenStrength} is a 0..1 user control
     * (0 = raw model output, 1 = maximum unsharp mask); values outside the
     * range are clamped.
     */
    public synchronized Bitmap process(Bitmap origImage, List<RectF> rectangles,
                                       float sharpenStrength) throws Exception {
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

        // Prepare tensor data (NCHW float, values normalized to [0, 1]).
        final int pixelCount = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
        FloatBuffer imgBuffer = FloatBuffer.allocate(3 * pixelCount);
        FloatBuffer maskBuffer = FloatBuffer.allocate(pixelCount);
        int[] pixelsImg = new int[pixelCount];
        int[] pixelsMask = new int[pixelCount];
        modelImg.getPixels(pixelsImg, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
        modelMask.getPixels(pixelsMask, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

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

        // Run inference. Tensors and the result are AutoCloseable; the
        // try-with-resources block guarantees they are freed even if the
        // session throws mid-flight.
        Bitmap outBitmap;
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

                outBitmap = Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888);
                outBitmap.setPixels(outPixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
            }
        }

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
        final Bitmap blendMask = Bitmap.createBitmap(cropW, cropH, Bitmap.Config.ARGB_8888);
        final Canvas blendCanvas = new Canvas(blendMask);
        blendCanvas.drawColor(Color.BLACK);
        for (RectF rect : valid) {
            blendCanvas.drawRect(rect.left - minX, rect.top - minY,
                    rect.right - minX, rect.bottom - minY, whitePaint);
        }
        int[] maskPixels = new int[cropW * cropH];
        blendMask.getPixels(maskPixels, 0, cropW, 0, 0, cropW, cropH);
        final int feather = Math.max(1, Math.min(6, Math.round(side * 0.006f)));
        for (int i = 0; i < 2; i++) {
            boxBlurH(maskPixels, cropW, cropH, feather);
            boxBlurV(maskPixels, cropW, cropH, feather);
        }

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

        // Seamless alpha blend: linearly mix original and restored pixels
        // according to the (feathered) mask value of each pixel.
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

        // Recycle intermediates. createScaledBitmap()/createBitmap() may
        // return the source object itself when no scaling/subsetting is
        // needed, so identity is checked to avoid double-recycling.
        inter.recycle();
        squareWork.recycle();
        maskWork.recycle();
        if (modelImg != squareWork) modelImg.recycle();
        if (modelMask != maskWork) modelMask.recycle();
        blendMask.recycle();
        if (cropOut != outBitmap) cropOut.recycle();
        if (restored != cropOut) restored.recycle();
        outBitmap.recycle();

        return finalResult;
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
