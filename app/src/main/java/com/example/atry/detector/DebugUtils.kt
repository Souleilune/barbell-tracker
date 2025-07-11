package com.example.atry.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream

/**
 * Debug utilities for troubleshooting YOLOv11 detection issues
 */
object DebugUtils {
    private const val TAG = "DebugUtils"

    /**
     * Save bitmap to internal storage for debugging
     */
    fun saveBitmapForDebugging(context: Context, bitmap: Bitmap, filename: String) {
        try {
            val file = File(context.filesDir, filename)
            val outputStream = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.PNG, 90, outputStream)
            outputStream.close()
            Log.d(TAG, "Saved debug bitmap to: ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save debug bitmap: ${e.message}", e)
        }
    }

    /**
     * Create a test bitmap with known objects for testing detection
     */
    fun createTestBitmap(): Bitmap {
        val bitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint().apply {
            isAntiAlias = true
            strokeWidth = 5f
            style = Paint.Style.STROKE
        }

        // Fill background
        canvas.drawColor(Color.BLACK)

        // Draw a simple barbell-like shape for testing
        paint.color = Color.WHITE
        paint.style = Paint.Style.FILL

        // Bar (horizontal line)
        canvas.drawRect(100f, 315f, 540f, 325f, paint)

        // Weights (circles on ends)
        canvas.drawCircle(100f, 320f, 30f, paint)
        canvas.drawCircle(540f, 320f, 30f, paint)

        Log.d(TAG, "Created test bitmap with synthetic barbell")
        return bitmap
    }

    /**
     * Log detailed model information
     */
    fun debugModelInfo(interpreter: Interpreter) {
        try {
            Log.d(TAG, "=== MODEL DEBUG INFO ===")

            // Input tensor info
            for (i in 0 until interpreter.inputTensorCount) {
                val tensor = interpreter.getInputTensor(i)
                Log.d(TAG, "Input $i: shape=${tensor.shape().contentToString()}, " +
                        "dataType=${tensor.dataType()}, " +
                        "quantizationParams=${tensor.quantizationParams()}")
            }

            // Output tensor info
            for (i in 0 until interpreter.outputTensorCount) {
                val tensor = interpreter.getOutputTensor(i)
                Log.d(TAG, "Output $i: shape=${tensor.shape().contentToString()}, " +
                        "dataType=${tensor.dataType()}, " +
                        "quantizationParams=${tensor.quantizationParams()}")
            }

            Log.d(TAG, "=== END MODEL DEBUG INFO ===")
        } catch (e: Exception) {
            Log.e(TAG, "Error getting model info: ${e.message}", e)
        }
    }

    /**
     * Log raw model output for debugging
     */
    fun logRawModelOutput(outputBuffer: Array<Array<FloatArray>>, maxSamples: Int = 10) {
        try {
            Log.d(TAG, "=== RAW MODEL OUTPUT ===")
            Log.d(TAG, "Output buffer dimensions: [${outputBuffer.size}][${outputBuffer[0].size}][${outputBuffer[0][0].size}]")

            // Log first few values from each channel
            for (batch in 0 until minOf(outputBuffer.size, 1)) {
                for (channel in 0 until minOf(outputBuffer[batch].size, 5)) {
                    val channelData = outputBuffer[batch][channel]
                    val samples = channelData.take(maxSamples)
                    Log.d(TAG, "Batch $batch, Channel $channel (first $maxSamples values): $samples")

                    // Log min/max/mean for this channel
                    val min = channelData.minOrNull() ?: 0f
                    val max = channelData.maxOrNull() ?: 0f
                    val mean = channelData.average().toFloat()
                    Log.d(TAG, "Channel $channel stats - min: $min, max: $max, mean: $mean")
                }
            }
            Log.d(TAG, "=== END RAW MODEL OUTPUT ===")
        } catch (e: Exception) {
            Log.e(TAG, "Error logging model output: ${e.message}", e)
        }
    }

    /**
     * Verify detection coordinates are reasonable
     */
    fun validateDetection(detection: Detection): Boolean {
        val bbox = detection.bbox

        // Check if coordinates are in valid range [0, 1]
        val validCoords = bbox.left >= 0f && bbox.left <= 1f &&
                bbox.top >= 0f && bbox.top <= 1f &&
                bbox.right >= 0f && bbox.right <= 1f &&
                bbox.bottom >= 0f && bbox.bottom <= 1f

        // Check if bbox has positive area
        val validArea = bbox.right > bbox.left && bbox.bottom > bbox.top

        // Check if confidence is reasonable
        val validConfidence = detection.score >= 0f && detection.score <= 1f

        val isValid = validCoords && validArea && validConfidence

        if (!isValid) {
            Log.w(TAG, "Invalid detection: bbox=$bbox, score=${detection.score}, classId=${detection.classId}")
        }

        return isValid
    }

    /**
     * Test model with a synthetic image
     */
    fun testModelWithSyntheticImage(detector: YOLOv8ObjectDetector, context: Context): List<Detection> {
        val testBitmap = createTestBitmap()
        saveBitmapForDebugging(context, testBitmap, "test_input.png")

        Log.d(TAG, "Testing model with synthetic barbell image...")
        val detections = detector.detect(testBitmap)

        Log.d(TAG, "Synthetic test results: ${detections.size} detections")
        detections.forEachIndexed { index, detection ->
            Log.d(TAG, "Detection $index: bbox=${detection.bbox}, score=${detection.score}, class=${detection.classId}")
        }

        return detections
    }

    /**
     * Enhanced logging for detection pipeline
     */
    fun logDetectionPipeline(
        inputBitmap: Bitmap,
        rawDetections: List<Detection>,
        filteredDetections: List<Detection>,
        finalDetections: List<Detection>
    ) {
        Log.d(TAG, "=== DETECTION PIPELINE ===")
        Log.d(TAG, "Input bitmap: ${inputBitmap.width}x${inputBitmap.height}")
        Log.d(TAG, "Raw detections: ${rawDetections.size}")
        Log.d(TAG, "After filtering: ${filteredDetections.size}")
        Log.d(TAG, "Final detections: ${finalDetections.size}")

        finalDetections.forEachIndexed { index, detection ->
            Log.d(TAG, "Final detection $index: " +
                    "bbox=[${String.format("%.3f", detection.bbox.left)}, " +
                    "${String.format("%.3f", detection.bbox.top)}, " +
                    "${String.format("%.3f", detection.bbox.right)}, " +
                    "${String.format("%.3f", detection.bbox.bottom)}], " +
                    "score=${String.format("%.3f", detection.score)}, " +
                    "class=${detection.classId}")
        }
        Log.d(TAG, "=== END DETECTION PIPELINE ===")
    }
}