// 1. Create OptimizedBarbellDetector.kt in your detector package
// This is the main file your MainActivity expects

package com.example.atry.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.*

/**
 * Optimized EfficientDet-Lite2 Detector specifically for simonskina.tflite
 *
 * Model Specifications:
 * - Architecture: EfficientDet-Lite2
 * - Input: 448×448×3 Float32
 * - Classes: 1 (barbells only)
 * - Preprocessing: Resize + normalize [0-1]
 * - Recommended threshold: 0.3-0.5
 */
class OptimizedBarbellDetector(
    private val context: Context,
    private val modelPath: String = "simonskina.tflite",
    private val confThreshold: Float = 0.4f, // Optimal for this model
    private val iouThreshold: Float = 0.5f,
    private val maxDetections: Int = 10
) {

    private lateinit var interpreter: Interpreter
    private var isInitialized = false

    // Model-specific constants (now we know exactly what we're dealing with)
    private val INPUT_SIZE = 448
    private val INPUT_CHANNELS = 3
    private val NUM_CLASSES = 1 // Only barbells
    private val INPUT_DATA_TYPE = DataType.FLOAT32

    // Optimized buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputLocations: Array<Array<FloatArray>> // Bounding boxes
    private lateinit var outputClasses: Array<Array<FloatArray>>   // Class scores (will be all 0 since only 1 class)
    private lateinit var outputScores: Array<FloatArray>           // Detection confidence scores
    private lateinit var outputNumDetections: FloatArray          // Number of valid detections

    companion object {
        private const val TAG = "OptimizedBarbellDetector"
        private const val BARBELL_CLASS_ID = 0 // Only class in the model
    }

    init {
        try {
            Log.d(TAG, "🚀 Initializing Optimized Barbell Detector (EfficientDet-Lite2)")
            Log.d(TAG, "📊 Model specs: 448×448 Float32, 1 class (barbells), threshold: $confThreshold")

            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
                setAllowFp16PrecisionForFp32(true) // Enable FP16 optimization
            }

            val modelBuffer = loadModelFile(context, modelPath)
            interpreter = Interpreter(modelBuffer, options)

            verifyModelSpecs()
            initializeOptimizedBuffers()

            isInitialized = true
            Log.d(TAG, "✅ Optimized barbell detector initialized successfully")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to initialize optimized detector: ${e.message}", e)
            throw RuntimeException("Failed to initialize optimized barbell detector", e)
        }
    }

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(assetPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun verifyModelSpecs() {
        try {
            Log.d(TAG, "🔍 Verifying model specifications...")

            // Verify input tensor
            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor.shape()
            val inputDataType = inputTensor.dataType()

            Log.d(TAG, "📥 Input verification:")
            Log.d(TAG, "  Expected: [1, 448, 448, 3] Float32")
            Log.d(TAG, "  Actual: ${inputShape.contentToString()} $inputDataType")

            // Verify our assumptions
            if (inputShape.contentEquals(intArrayOf(1, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS)) &&
                inputDataType == INPUT_DATA_TYPE) {
                Log.d(TAG, "✅ Input specifications match perfectly!")
            } else {
                Log.w(TAG, "⚠️ Input specifications don't match expected values")
            }

            // Verify output tensors
            Log.d(TAG, "📤 Output verification:")
            for (i in 0 until interpreter.outputTensorCount) {
                val outputTensor = interpreter.getOutputTensor(i)
                val shape = outputTensor.shape()
                val dataType = outputTensor.dataType()
                Log.d(TAG, "  Output $i: ${shape.contentToString()} $dataType")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error verifying model specs: ${e.message}", e)
        }
    }

    private fun initializeOptimizedBuffers() {
        try {
            Log.d(TAG, "🔧 Initializing optimized buffers...")

            // Input buffer: 448×448×3×4 bytes = 2,408,448 bytes
            val inputBytes = INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS * 4
            inputBuffer = ByteBuffer.allocateDirect(inputBytes)
            inputBuffer.order(ByteOrder.nativeOrder())

            Log.d(TAG, "📥 Input buffer: ${INPUT_SIZE}×${INPUT_SIZE}×${INPUT_CHANNELS} = $inputBytes bytes")

            // Initialize output buffers based on EfficientDet-Lite2 standard format
            when (interpreter.outputTensorCount) {
                4 -> {
                    // Standard EfficientDet format: [boxes, classes, scores, num_detections]
                    val boxShape = interpreter.getOutputTensor(0).shape()      // [1, max_detections, 4]
                    val classShape = interpreter.getOutputTensor(1).shape()    // [1, max_detections, num_classes]
                    val scoreShape = interpreter.getOutputTensor(2).shape()    // [1, max_detections]
                    val numShape = interpreter.getOutputTensor(3).shape()      // [1]

                    outputLocations = Array(boxShape[0]) { Array(boxShape[1]) { FloatArray(boxShape[2]) } }
                    outputClasses = Array(classShape[0]) { Array(classShape[1]) { FloatArray(classShape[2]) } }
                    outputScores = Array(scoreShape[0]) { FloatArray(scoreShape[1]) }
                    outputNumDetections = FloatArray(numShape[0])

                    Log.d(TAG, "✅ Standard EfficientDet format detected:")
                    Log.d(TAG, "  Boxes: ${boxShape.contentToString()}")
                    Log.d(TAG, "  Classes: ${classShape.contentToString()}")
                    Log.d(TAG, "  Scores: ${scoreShape.contentToString()}")
                    Log.d(TAG, "  Num detections: ${numShape.contentToString()}")
                }
                1 -> {
                    // Single output format - common in optimized models
                    val outputShape = interpreter.getOutputTensor(0).shape()
                    Log.d(TAG, "Single output detected: ${outputShape.contentToString()}")

                    // Assume format: [1, max_detections, 6] where 6 = [x1, y1, x2, y2, score, class]
                    val maxDetections = if (outputShape.size >= 2) outputShape[1] else 100
                    outputLocations = Array(1) { Array(maxDetections) { FloatArray(4) } }
                    outputClasses = Array(1) { Array(maxDetections) { FloatArray(1) } }
                    outputScores = Array(1) { FloatArray(maxDetections) }
                    outputNumDetections = FloatArray(1)

                    Log.d(TAG, "✅ Single output format configured for $maxDetections max detections")
                }
                else -> {
                    throw RuntimeException("Unexpected output tensor count: ${interpreter.outputTensorCount}")
                }
            }

            Log.d(TAG, "✅ Optimized buffers initialized successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing optimized buffers: ${e.message}", e)
            throw e
        }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        if (!isInitialized) {
            Log.w(TAG, "Optimized detector not initialized")
            return emptyList()
        }

        return try {
            Log.d(TAG, "🔍 Optimized barbell detection on ${bitmap.width}×${bitmap.height} image")

            // Optimized preprocessing for 448×448 Float32
            preprocessOptimized(bitmap)

            // Run optimized inference
            val inferenceStart = System.currentTimeMillis()
            runOptimizedInference()
            val inferenceTime = System.currentTimeMillis() - inferenceStart

            Log.d(TAG, "✅ Optimized inference completed in ${inferenceTime}ms")

            // Post-process with barbell-specific logic
            val detections = postprocessBarbellDetections()

            Log.d(TAG, "🎯 Found ${detections.size} barbell detections")

            // Apply NMS with barbell-optimized parameters
            val filteredDetections = applyOptimizedNMS(detections)

            Log.d(TAG, "🎯 Final barbell detections after NMS: ${filteredDetections.size}")

            // Log detailed results for debugging
            filteredDetections.forEachIndexed { index, detection ->
                Log.d(TAG, "Barbell $index: conf=${String.format("%.3f", detection.score)}, " +
                        "bbox=[${String.format("%.3f", detection.bbox.left)}, " +
                        "${String.format("%.3f", detection.bbox.top)}, " +
                        "${String.format("%.3f", detection.bbox.right)}, " +
                        "${String.format("%.3f", detection.bbox.bottom)}], " +
                        "size=${String.format("%.3f", (detection.bbox.right - detection.bbox.left) * (detection.bbox.bottom - detection.bbox.top))}")
            }

            filteredDetections

        } catch (e: Exception) {
            Log.e(TAG, "❌ Optimized detection error: ${e.message}", e)
            emptyList()
        }
    }

    private fun preprocessOptimized(bitmap: Bitmap) {
        Log.d(TAG, "🔄 Optimized preprocessing: ${bitmap.width}×${bitmap.height} → ${INPUT_SIZE}×${INPUT_SIZE}")

        // Resize to exact model input size
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        inputBuffer.clear()

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaledBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Optimized normalization to [0, 1] range as specified
        for (pixel in pixels) {
            // Extract RGB channels and normalize to [0, 1]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            // Write in NHWC format: [batch, height, width, channels]
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        scaledBitmap.recycle()
        inputBuffer.rewind()

        Log.d(TAG, "✅ Optimized preprocessing complete")
    }

    private fun runOptimizedInference() {
        try {
            when (interpreter.outputTensorCount) {
                4 -> {
                    // Multi-output inference
                    interpreter.runForMultipleInputsOutputs(
                        arrayOf(inputBuffer),
                        mapOf(
                            0 to outputLocations,
                            1 to outputClasses,
                            2 to outputScores,
                            3 to outputNumDetections
                        )
                    )
                }
                1 -> {
                    // Single output inference
                    val singleOutput = Array(1) { Array(100) { FloatArray(6) } }
                    interpreter.run(inputBuffer, singleOutput)
                    parseSingleOutput(singleOutput)
                }
                else -> {
                    throw RuntimeException("Unsupported output format")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Optimized inference error: ${e.message}", e)
            throw e
        }
    }

    private fun parseSingleOutput(singleOutput: Array<Array<FloatArray>>) {
        try {
            val detections = singleOutput[0]
            var validDetections = 0

            for (i in detections.indices) {
                if (detections[i].size >= 6) {
                    // Format: [x1, y1, x2, y2, score, class]
                    outputLocations[0][i][0] = detections[i][0] // x1
                    outputLocations[0][i][1] = detections[i][1] // y1
                    outputLocations[0][i][2] = detections[i][2] // x2
                    outputLocations[0][i][3] = detections[i][3] // y2
                    outputScores[0][i] = detections[i][4]       // confidence
                    outputClasses[0][i][0] = detections[i][5]   // class (should be 0 for barbells)

                    if (detections[i][4] >= confThreshold) {
                        validDetections++
                    }
                }
            }

            outputNumDetections[0] = validDetections.toFloat()
            Log.d(TAG, "Parsed single output: $validDetections valid detections")

        } catch (e: Exception) {
            Log.e(TAG, "Error parsing single output: ${e.message}", e)
        }
    }

    private fun postprocessBarbellDetections(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            val numValidDetections = outputNumDetections[0].toInt()
            val detectionsToProcess = minOf(numValidDetections, outputScores[0].size, maxDetections)

            Log.d(TAG, "Processing $detectionsToProcess barbell detections (valid: $numValidDetections)")

            for (i in 0 until detectionsToProcess) {
                val score = outputScores[0][i]

                // Apply confidence threshold
                if (score >= confThreshold) {
                    // Get bounding box coordinates (should be normalized [0, 1])
                    val x1 = outputLocations[0][i][0]
                    val y1 = outputLocations[0][i][1]
                    val x2 = outputLocations[0][i][2]
                    val y2 = outputLocations[0][i][3]

                    // Ensure proper coordinate order and normalization
                    val left = minOf(x1, x2).coerceIn(0f, 1f)
                    val top = minOf(y1, y2).coerceIn(0f, 1f)
                    val right = maxOf(x1, x2).coerceIn(0f, 1f)
                    val bottom = maxOf(y1, y2).coerceIn(0f, 1f)

                    // Validate bounding box dimensions
                    val width = right - left
                    val height = bottom - top
                    val area = width * height

                    // Barbell-specific validation
                    if (width > 0.01f && height > 0.01f && area > 0.0001f && area < 0.9f) {
                        // Check aspect ratio - barbells are typically horizontal
                        val aspectRatio = width / height

                        // Accept reasonable aspect ratios for barbells (can be vertical or horizontal)
                        if (aspectRatio > 0.2f && aspectRatio < 10f) {
                            // Get class (should always be 0 for barbells, but let's verify)
                            val classId = if (outputClasses[0][i].isNotEmpty()) {
                                outputClasses[0][i][0].toInt()
                            } else {
                                BARBELL_CLASS_ID
                            }

                            detections.add(Detection(
                                bbox = RectF(left, top, right, bottom),
                                score = score,
                                classId = classId
                            ))

                            Log.d(TAG, "Valid barbell detection $i: score=$score, " +
                                    "dimensions=${String.format("%.3f", width)}×${String.format("%.3f", height)}, " +
                                    "aspect_ratio=${String.format("%.2f", aspectRatio)}, class=$classId")
                        } else {
                            Log.d(TAG, "Filtered detection $i: invalid aspect ratio $aspectRatio")
                        }
                    } else {
                        Log.d(TAG, "Filtered detection $i: invalid dimensions ${String.format("%.3f", width)}×${String.format("%.3f", height)}")
                    }
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error post-processing barbell detections: ${e.message}", e)
        }

        return detections
    }

    private fun applyOptimizedNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        // Sort by confidence score (highest first)
        val sortedDetections = detections.sortedByDescending { it.score }.toMutableList()
        val finalDetections = mutableListOf<Detection>()

        while (sortedDetections.isNotEmpty() && finalDetections.size < maxDetections) {
            val best = sortedDetections.removeAt(0)
            finalDetections.add(best)

            // Remove overlapping detections
            sortedDetections.removeAll { detection ->
                calculateIoU(best.bbox, detection.bbox) > iouThreshold
            }
        }

        Log.d(TAG, "NMS: ${detections.size} → ${finalDetections.size} detections")
        return finalDetections
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectionLeft = max(box1.left, box2.left)
        val intersectionTop = max(box1.top, box2.top)
        val intersectionRight = min(box1.right, box2.right)
        val intersectionBottom = min(box1.bottom, box2.bottom)

        val intersectionWidth = max(0f, intersectionRight - intersectionLeft)
        val intersectionHeight = max(0f, intersectionBottom - intersectionTop)
        val intersectionArea = intersectionWidth * intersectionHeight

        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val unionArea = box1Area + box2Area - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    // Public interface methods
    fun getClassLabel(classId: Int): String = "Barbell" // Only one class

    fun getDetectionCenter(detection: Detection): Pair<Float, Float> = Pair(
        (detection.bbox.left + detection.bbox.right) / 2f,
        (detection.bbox.top + detection.bbox.bottom) / 2f
    )

    fun getDetectionQuality(detection: Detection): DetectionQuality {
        val bbox = detection.bbox
        val width = bbox.right - bbox.left
        val height = bbox.bottom - bbox.top
        val area = width * height
        val aspectRatio = if (height > 0) width / height else 1f

        // Barbell-specific quality assessment
        val sizeScore = when {
            area > 0.05f -> 1.0f  // Large barbell
            area > 0.02f -> 0.8f  // Medium barbell
            area > 0.01f -> 0.6f  // Small barbell
            else -> 0.4f          // Very small
        }

        val aspectScore = when {
            aspectRatio > 2f || aspectRatio < 0.5f -> 1.0f // Good horizontal/vertical barbell
            aspectRatio > 1.5f || aspectRatio < 0.7f -> 0.8f // Decent aspect ratio
            else -> 0.6f // Square-ish (unusual for barbells)
        }

        return DetectionQuality(
            confidence = detection.score,
            size = area,
            aspectRatio = aspectRatio,
            stability = (sizeScore + aspectScore) / 2f
        )
    }

    fun isUsingGPU(): Boolean = false

    fun getPerformanceInfo(): String = "Optimized EfficientDet-Lite2 (448×448, Float32, Barbells Only)"

    fun cleanup() {
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
        Log.d(TAG, "🧹 Optimized barbell detector cleaned up")
    }

    fun close() = cleanup()
}

/**
 * Detection quality assessment for barbell-specific use case
 */
data class DetectionQuality(
    val confidence: Float,
    val size: Float,
    val aspectRatio: Float,
    val stability: Float
) {
    fun getOverallQuality(): Float {
        return (confidence * 0.6f +      // Confidence is most important
                minOf(size * 15f, 1f) * 0.2f +  // Size matters for barbells
                stability * 0.2f)        // Aspect ratio and size consistency
    }

    fun getQualityGrade(): String {
        val quality = getOverallQuality()
        return when {
            quality >= 0.9f -> "Excellent"
            quality >= 0.8f -> "Very Good"
            quality >= 0.7f -> "Good"
            quality >= 0.6f -> "Fair"
            quality >= 0.5f -> "Poor"
            else -> "Very Poor"
        }
    }
}