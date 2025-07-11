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
 * Debug utilities for troubleshooting Generic TFLite detection issues
 * Updated to work with any TensorFlow Lite model
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
     * Log raw model output for debugging - Works with any output format
     */
    fun logRawModelOutput(outputBuffer: Any, maxSamples: Int = 10) {
        try {
            Log.d(TAG, "=== RAW MODEL OUTPUT ===")

            when (outputBuffer) {
                is Array<*> -> {
                    when {
                        outputBuffer.isNotEmpty() && outputBuffer[0] is Array<*> -> {
                            // 3D array: [batch][detections][features]
                            val output3D = outputBuffer as Array<Array<FloatArray>>
                            Log.d(TAG, "3D Output buffer dimensions: [${output3D.size}][${output3D[0].size}][${output3D[0][0].size}]")

                            // Log first few values from each channel
                            for (batch in 0 until minOf(output3D.size, 1)) {
                                for (detection in 0 until minOf(output3D[batch].size, 5)) {
                                    val detectionData = output3D[batch][detection]
                                    val samples = detectionData.take(maxSamples)
                                    Log.d(TAG, "Batch $batch, Detection $detection (first $maxSamples values): $samples")

                                    // Log min/max/mean for this detection
                                    val min = detectionData.minOrNull() ?: 0f
                                    val max = detectionData.maxOrNull() ?: 0f
                                    val mean = detectionData.average().toFloat()
                                    Log.d(TAG, "Detection $detection stats - min: $min, max: $max, mean: $mean")
                                }
                            }
                        }
                        outputBuffer.isNotEmpty() && outputBuffer[0] is FloatArray -> {
                            // 2D array: [batch][features] or [detections][features]
                            val output2D = outputBuffer as Array<FloatArray>
                            Log.d(TAG, "2D Output buffer dimensions: [${output2D.size}][${output2D[0].size}]")

                            for (i in 0 until minOf(output2D.size, 5)) {
                                val rowData = output2D[i]
                                val samples = rowData.take(maxSamples)
                                Log.d(TAG, "Row $i (first $maxSamples values): $samples")

                                val min = rowData.minOrNull() ?: 0f
                                val max = rowData.maxOrNull() ?: 0f
                                val mean = rowData.average().toFloat()
                                Log.d(TAG, "Row $i stats - min: $min, max: $max, mean: $mean")
                            }
                        }
                        else -> {
                            Log.d(TAG, "Unknown array format: ${outputBuffer.javaClass.simpleName}")
                        }
                    }
                }
                is FloatArray -> {
                    // 1D array
                    val output1D = outputBuffer as FloatArray
                    Log.d(TAG, "1D Output buffer size: ${output1D.size}")

                    val samples = output1D.take(maxSamples)
                    Log.d(TAG, "First $maxSamples values: $samples")

                    val min = output1D.minOrNull() ?: 0f
                    val max = output1D.maxOrNull() ?: 0f
                    val mean = output1D.average().toFloat()
                    Log.d(TAG, "Overall stats - min: $min, max: $max, mean: $mean")
                }
                else -> {
                    Log.d(TAG, "Unknown output buffer type: ${outputBuffer.javaClass.simpleName}")
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
     * Test model with a synthetic image - Works with Generic TFLite Detector
     */
    fun testModelWithSyntheticImage(detector: GenericTFLiteDetector, context: Context): List<Detection> {
        val testBitmap = createTestBitmap()
        saveBitmapForDebugging(context, testBitmap, "test_input.png")

        Log.d(TAG, "Testing Generic TFLite model with synthetic barbell image...")
        val detections = detector.detect(testBitmap)

        Log.d(TAG, "Synthetic test results: ${detections.size} detections")
        detections.forEachIndexed { index, detection ->
            Log.d(TAG, "Detection $index: bbox=${detection.bbox}, score=${detection.score}, class=${detection.classId}")

            // Validate each detection
            val isValid = validateDetection(detection)
            Log.d(TAG, "Detection $index validation: ${if (isValid) "VALID" else "INVALID"}")
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

            // Validate each final detection
            val isValid = validateDetection(detection)
            Log.d(TAG, "Final detection $index validation: ${if (isValid) "VALID" else "INVALID"}")
        }
        Log.d(TAG, "=== END DETECTION PIPELINE ===")
    }

    /**
     * Log model architecture analysis results
     */
    fun logModelArchitectureAnalysis(
        modelType: String,
        inputShape: IntArray,
        outputShape: IntArray,
        outputTensorCount: Int
    ) {
        Log.d(TAG, "=== MODEL ARCHITECTURE ANALYSIS ===")
        Log.d(TAG, "Detected model type: $modelType")
        Log.d(TAG, "Input shape: ${inputShape.contentToString()}")
        Log.d(TAG, "Primary output shape: ${outputShape.contentToString()}")
        Log.d(TAG, "Total output tensors: $outputTensorCount")

        // Provide architecture-specific insights
        when (modelType) {
            "YOLO" -> {
                Log.d(TAG, "YOLO Analysis:")
                Log.d(TAG, "  - Expected format: [batch, detections, features]")
                Log.d(TAG, "  - Features typically: [x, y, w, h, conf, class_scores...]")
                Log.d(TAG, "  - Number of detections: ${if (outputShape.size > 1) outputShape[1] else "unknown"}")
            }
            "EFFICIENTDET" -> {
                Log.d(TAG, "EfficientDet Analysis:")
                Log.d(TAG, "  - Multiple output tensors expected")
                Log.d(TAG, "  - Typically: boxes, scores, classes, num_detections")
                Log.d(TAG, "  - Complex post-processing required")
            }
            "MOBILENET" -> {
                Log.d(TAG, "MobileNet SSD Analysis:")
                Log.d(TAG, "  - Expected: detection boxes + classification scores")
                Log.d(TAG, "  - Standard SSD format")
            }
            else -> {
                Log.d(TAG, "Unknown/Custom Model:")
                Log.d(TAG, "  - Using generic processing approach")
                Log.d(TAG, "  - May require custom post-processing")
            }
        }
        Log.d(TAG, "=== END MODEL ARCHITECTURE ANALYSIS ===")
    }

    /**
     * Performance monitoring utilities
     */
    fun logPerformanceMetrics(
        modelName: String,
        inferenceTimeMs: Long,
        preprocessTimeMs: Long,
        postprocessTimeMs: Long,
        totalDetections: Int,
        validDetections: Int
    ) {
        Log.d(TAG, "=== PERFORMANCE METRICS ===")
        Log.d(TAG, "Model: $modelName")
        Log.d(TAG, "Inference time: ${inferenceTimeMs}ms")
        Log.d(TAG, "Preprocessing time: ${preprocessTimeMs}ms")
        Log.d(TAG, "Post-processing time: ${postprocessTimeMs}ms")
        Log.d(TAG, "Total pipeline time: ${inferenceTimeMs + preprocessTimeMs + postprocessTimeMs}ms")
        Log.d(TAG, "Raw detections: $totalDetections")
        Log.d(TAG, "Valid detections: $validDetections")
        Log.d(TAG, "Detection efficiency: ${if (totalDetections > 0) (validDetections * 100 / totalDetections) else 0}%")
        Log.d(TAG, "=== END PERFORMANCE METRICS ===")
    }

    /**
     * Memory usage monitoring
     */
    fun logMemoryUsage(context: Context, tag: String = "Generic") {
        try {
            val runtime = Runtime.getRuntime()
            val usedMemory = runtime.totalMemory() - runtime.freeMemory()
            val maxMemory = runtime.maxMemory()
            val availableMemory = maxMemory - usedMemory

            Log.d(TAG, "=== MEMORY USAGE ($tag) ===")
            Log.d(TAG, "Used memory: ${usedMemory / 1024 / 1024}MB")
            Log.d(TAG, "Max memory: ${maxMemory / 1024 / 1024}MB")
            Log.d(TAG, "Available memory: ${availableMemory / 1024 / 1024}MB")
            Log.d(TAG, "Memory usage: ${(usedMemory * 100 / maxMemory)}%")
            Log.d(TAG, "=== END MEMORY USAGE ===")
        } catch (e: Exception) {
            Log.e(TAG, "Error logging memory usage: ${e.message}")
        }
    }
}