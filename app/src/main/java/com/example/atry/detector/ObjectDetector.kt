package com.example.atry.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Dynamic Threshold Barbell Detector - Automatically finds barbells by adjusting threshold
 */
class YOLOv8ObjectDetector(
    private val context: Context,
    modelPath: String = "yolofinal.tflite",
    private val inputSize: Int = 320,
    private var confThreshold: Float = 0.003f, // Start lower to find barbells
    private val iouThreshold: Float = 0.45f,
    private val maxDetections: Int = 3
) {

    private val classLabels = arrayOf("Barbell")
    private lateinit var interpreter: Interpreter

    private lateinit var outputBuffer: Array<Array<FloatArray>>
    private var actualOutputShape: IntArray = intArrayOf()
    private var isInitialized = false

    private val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).apply {
        order(ByteOrder.nativeOrder())
    }

    private var frameCount = 0
    private var realDetectionCount = 0
    private var lastThresholdAdjustment = 0L
    private var recentMaxConfidences = mutableListOf<Float>()

    companion object {
        private const val TAG = "DynamicBarbell"

        // Edge filtering parameters (same as before)
        private const val EDGE_MARGIN = 0.1f
        private const val MIN_CENTER_AREA = 0.03f   // Reduced for smaller barbells
        private const val MAX_CENTER_AREA = 0.7f
        private const val MIN_ASPECT_RATIO = 0.2f   // More flexible for barbells
        private const val MAX_ASPECT_RATIO = 5.0f

        // Dynamic threshold parameters
        private const val THRESHOLD_STEP = 0.001f
        private const val MIN_THRESHOLD = 0.001f
        private const val MAX_THRESHOLD = 0.02f
        private const val ADJUSTMENT_INTERVAL = 3000L // 3 seconds
    }

    init {
        try {
            Log.e(TAG, "🚀 STARTING DYNAMIC BARBELL DETECTOR...")
            showToast("Dynamic Barbell Detector Starting...")

            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }

            val modelBuffer = loadModelFile(context, modelPath)
            interpreter = Interpreter(modelBuffer, options)

            initializeBuffers()
            isInitialized = true

            Log.e(TAG, "✅ DYNAMIC BARBELL DETECTOR READY")
            Log.e(TAG, "🎯 Will automatically adjust threshold to find barbells")
            Log.e(TAG, "📍 Starting threshold: $confThreshold")

            showToast("Ready! Point camera at barbell - threshold: $confThreshold")

        } catch (e: Exception) {
            Log.e(TAG, "❌ INITIALIZATION FAILED: ${e.message}", e)
            showToast("Init failed: ${e.message}")
            throw RuntimeException("Initialization failed", e)
        }
    }

    private fun showToast(message: String) {
        try {
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            // Ignore
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

    private fun initializeBuffers() {
        val outputTensor = interpreter.getOutputTensor(0)
        actualOutputShape = outputTensor.shape()

        val batchSize = actualOutputShape[0]
        val features = actualOutputShape[1]
        val detections = actualOutputShape[2]

        outputBuffer = Array(batchSize) { Array(features) { FloatArray(detections) } }

        Log.e(TAG, "✅ Buffers: [$batchSize, $features, $detections]")
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        inputBuffer.clear()

        val pixels = IntArray(inputSize * inputSize)
        scaledBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        scaledBitmap.recycle()
        inputBuffer.rewind()

        return inputBuffer
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        if (!isInitialized) return emptyList()

        frameCount++

        return try {
            val input = preprocessImage(bitmap)
            interpreter.run(input, outputBuffer)

            // Dynamic threshold adjustment
            dynamicallyAdjustThreshold()

            val detections = processDetectionsWithFiltering()

            // Report real barbell detections
            if (detections.isNotEmpty()) {
                realDetectionCount++

                Log.e(TAG, "🏋️ BARBELL DETECTED! (Frame $frameCount, Total: $realDetectionCount)")
                detections.forEachIndexed { index, detection ->
                    val centerX = (detection.bbox.left + detection.bbox.right) / 2f
                    val centerY = (detection.bbox.top + detection.bbox.bottom) / 2f
                    val width = detection.bbox.right - detection.bbox.left
                    val height = detection.bbox.bottom - detection.bbox.top
                    val area = width * height
                    val aspectRatio = if (height > 0) width / height else 1f

                    Log.e(TAG, "   Barbell ${index + 1}:")
                    Log.e(TAG, "     🎯 Confidence: ${String.format("%.4f", detection.score)}")
                    Log.e(TAG, "     📍 Center: (${String.format("%.2f", centerX)}, ${String.format("%.2f", centerY)})")
                    Log.e(TAG, "     📏 Size: ${String.format("%.1f", width * 100)}% x ${String.format("%.1f", height * 100)}%")
                    Log.e(TAG, "     📐 Aspect: ${String.format("%.2f", aspectRatio)} (${if (aspectRatio > 1.5f) "horizontal" else if (aspectRatio < 0.7f) "vertical" else "square"})")
                    Log.e(TAG, "     📊 Area: ${String.format("%.1f", area * 100)}%")
                }

                showToast("🏋️ ${detections.size} barbell(s) detected!")
            }

            // Periodic stats with threshold info
            if (frameCount % 30 == 0) {
                val currentMaxConf = outputBuffer[0][4].maxOrNull() ?: 0f
                val rawDetectionCount = countRawDetections()
                val centerDetectionCount = countCenterDetections()
                val filteredCount = centerDetectionCount - detections.size

                Log.e(TAG, "📊 DETECTION STATS (Frame $frameCount):")
                Log.e(TAG, "   🎛️ Current threshold: ${String.format("%.4f", confThreshold)}")
                Log.e(TAG, "   📈 Max confidence: ${String.format("%.4f", currentMaxConf)}")
                Log.e(TAG, "   🔍 Raw detections: $rawDetectionCount")
                Log.e(TAG, "   🎯 Center detections: $centerDetectionCount")
                Log.e(TAG, "   ❌ Edge filtered: ${rawDetectionCount - centerDetectionCount}")
                Log.e(TAG, "   ✅ Valid barbells: ${detections.size}")
                Log.e(TAG, "   🏋️ Total barbells found: $realDetectionCount")

                if (rawDetectionCount == 0 && currentMaxConf > 0.001f) {
                    Log.e(TAG, "   💡 Threshold may be too high - consider lowering")
                }
            }

            detections

        } catch (e: Exception) {
            Log.e(TAG, "❌ Detection error: ${e.message}", e)
            emptyList()
        }
    }

    private fun dynamicallyAdjustThreshold() {
        val currentTime = System.currentTimeMillis()

        // Adjust threshold every 3 seconds
        if (currentTime - lastThresholdAdjustment > ADJUSTMENT_INTERVAL) {
            val currentMaxConf = outputBuffer[0][4].maxOrNull() ?: 0f
            recentMaxConfidences.add(currentMaxConf)

            // Keep only recent confidences
            if (recentMaxConfidences.size > 10) {
                recentMaxConfidences.removeAt(0)
            }

            val avgMaxConf = if (recentMaxConfidences.isNotEmpty()) {
                recentMaxConfidences.average().toFloat()
            } else currentMaxConf

            val rawDetectionCount = countRawDetections()
            val centerDetectionCount = countCenterDetections()

            val oldThreshold = confThreshold

            // Threshold adjustment logic
            when {
                // No raw detections but high confidence available - lower threshold
                rawDetectionCount == 0 && avgMaxConf > confThreshold * 2f -> {
                    confThreshold = maxOf(avgMaxConf * 0.7f, confThreshold - THRESHOLD_STEP)
                    Log.e(TAG, "🔽 Lowering threshold to find barbells: $oldThreshold -> $confThreshold")
                }

                // Too many raw detections but no center ones - edges detected, raise threshold slightly
                rawDetectionCount > 10 && centerDetectionCount == 0 -> {
                    confThreshold = minOf(confThreshold + THRESHOLD_STEP, avgMaxConf * 0.8f)
                    Log.e(TAG, "🔼 Raising threshold to reduce edge noise: $oldThreshold -> $confThreshold")
                }

                // Good balance - center detections found
                centerDetectionCount in 1..5 -> {
                    // Threshold is working well, no change
                    Log.e(TAG, "✅ Threshold working well: $confThreshold")
                }

                // Too many center detections - might be noise
                centerDetectionCount > 5 -> {
                    confThreshold = minOf(confThreshold + THRESHOLD_STEP * 0.5f, MAX_THRESHOLD)
                    Log.e(TAG, "🔼 Too many detections, raising threshold: $oldThreshold -> $confThreshold")
                }
            }

            // Apply bounds
            confThreshold = confThreshold.coerceIn(MIN_THRESHOLD, MAX_THRESHOLD)

            if (abs(confThreshold - oldThreshold) > 0.0005f) {
                showToast("Threshold: ${String.format("%.4f", confThreshold)}")
            }

            lastThresholdAdjustment = currentTime
        }
    }

    private fun countRawDetections(): Int {
        return try {
            val confArray = outputBuffer[0][4]
            confArray.count { it >= confThreshold }
        } catch (e: Exception) {
            0
        }
    }

    private fun countCenterDetections(): Int {
        val centerDetections = mutableListOf<Detection>()

        try {
            val numDetections = actualOutputShape[2]
            val xArray = outputBuffer[0][0]
            val yArray = outputBuffer[0][1]
            val wArray = outputBuffer[0][2]
            val hArray = outputBuffer[0][3]
            val confArray = outputBuffer[0][4]

            for (i in 0 until numDetections) {
                val conf = confArray[i]

                if (conf >= confThreshold) {
                    val x = xArray[i]
                    val y = yArray[i]
                    val w = wArray[i]
                    val h = hArray[i]

                    val detection = createDetection(x, y, w, h, conf)
                    if (detection != null && isValidCenterDetection(detection)) {
                        centerDetections.add(detection)
                    }
                }
            }
        } catch (e: Exception) {
            // Ignore
        }

        return centerDetections.size
    }

    private fun processDetectionsWithFiltering(): List<Detection> {
        val centerDetections = mutableListOf<Detection>()

        try {
            val numDetections = actualOutputShape[2]
            val xArray = outputBuffer[0][0]
            val yArray = outputBuffer[0][1]
            val wArray = outputBuffer[0][2]
            val hArray = outputBuffer[0][3]
            val confArray = outputBuffer[0][4]

            for (i in 0 until numDetections) {
                val conf = confArray[i]

                if (conf >= confThreshold) {
                    val x = xArray[i]
                    val y = yArray[i]
                    val w = wArray[i]
                    val h = hArray[i]

                    val detection = createDetection(x, y, w, h, conf)
                    if (detection != null && isValidCenterDetection(detection)) {
                        centerDetections.add(detection)
                    }
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing detections: ${e.message}")
        }

        return applyNMS(centerDetections)
    }

    private fun isValidCenterDetection(detection: Detection): Boolean {
        val bbox = detection.bbox

        val left = bbox.left
        val top = bbox.top
        val right = bbox.right
        val bottom = bbox.bottom
        val width = right - left
        val height = bottom - top
        val area = width * height
        val centerX = (left + right) / 2f
        val centerY = (top + bottom) / 2f
        val aspectRatio = if (height > 0) width / height else 1f

        // Filter 1: Remove edge detections
        val tooCloseToEdge = left < EDGE_MARGIN || top < EDGE_MARGIN ||
                right > (1f - EDGE_MARGIN) || bottom > (1f - EDGE_MARGIN)

        if (tooCloseToEdge) return false

        // Filter 2: Area constraints
        if (area < MIN_CENTER_AREA || area > MAX_CENTER_AREA) return false

        // Filter 3: Aspect ratio constraints
        if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) return false

        // Filter 4: Center region preference
        val distanceFromCenter = kotlin.math.sqrt(
            (centerX - 0.5f) * (centerX - 0.5f) + (centerY - 0.5f) * (centerY - 0.5f)
        )

        if (distanceFromCenter > 0.45f) return false

        return true
    }

    private fun createDetection(x: Float, y: Float, w: Float, h: Float, conf: Float): Detection? {
        return try {
            var centerX = x
            var centerY = y
            var width = w
            var height = h

            if (x > 1f || y > 1f || w > 1f || h > 1f) {
                centerX = x / inputSize
                centerY = y / inputSize
                width = w / inputSize
                height = h / inputSize
            }

            val left = (centerX - width / 2f).coerceIn(0f, 1f)
            val top = (centerY - height / 2f).coerceIn(0f, 1f)
            val right = (centerX + width / 2f).coerceIn(0f, 1f)
            val bottom = (centerY + height / 2f).coerceIn(0f, 1f)

            if (right > left && bottom > top) {
                Detection(RectF(left, top, right, bottom), conf, 0)
            } else null

        } catch (e: Exception) {
            null
        }
    }

    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sortedDetections = detections.sortedByDescending { it.score }.toMutableList()
        val finalDetections = mutableListOf<Detection>()

        while (sortedDetections.isNotEmpty() && finalDetections.size < maxDetections) {
            val best = sortedDetections.removeAt(0)
            finalDetections.add(best)

            sortedDetections.removeAll { detection ->
                calculateIoU(best.bbox, detection.bbox) > iouThreshold
            }
        }

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

    // Interface methods
    fun getClassLabel(classId: Int): String = classLabels.getOrElse(classId) { "Unknown" }

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

        return DetectionQuality(
            confidence = detection.score,
            size = area,
            aspectRatio = aspectRatio,
            stability = 0.8f
        )
    }

    fun isUsingGPU(): Boolean = false

    fun getPerformanceInfo(): String = "Dynamic (${String.format("%.4f", confThreshold)}) - ${realDetectionCount} barbells"

    fun cleanup() {}

    fun close() {
        cleanup()
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
    }
}

data class DetectionQuality(
    val confidence: Float,
    val size: Float,
    val aspectRatio: Float,
    val stability: Float
) {
    fun getOverallQuality(): Float {
        return (confidence * 0.5f +
                minOf(size * 20f, 1f) * 0.2f +
                (1f - kotlin.math.abs(aspectRatio - 1.5f) / 1.5f).coerceIn(0f, 1f) * 0.2f +
                stability * 0.1f)
    }
}