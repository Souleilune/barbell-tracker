package com.example.atry.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.*

/**
 * FIXED Hybrid Detector - Resolves BufferOverflowException
 * Combines classification model with basic tracking for simonskina.tflite
 */
class HybridClassificationDetector(
    private val context: Context,
    private val modelPath: String = "simonskina.tflite",
    private val confThreshold: Float = 0.05f,
    private val maxDetections: Int = 3
) {

    private lateinit var interpreter: Interpreter
    private var isInitialized = false

    // FIXED: Model input/output info
    private var realInputSize = 224
    private var inputChannels = 3
    private var outputClasses = 25

    // FIXED: Classification model buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: Array<FloatArray>

    // Basic tracking state
    private var lastDetectionTime = 0L
    private var lastDetectionCenter = Pair(0.5f, 0.5f)
    private var trackingActive = false
    private val trackingTimeout = 2000L // 2 seconds

    // Simple motion detection
    private var previousBitmap: Bitmap? = null
    private var motionAreas = mutableListOf<RectF>()

    companion object {
        private const val TAG = "HybridClassificationDetector"
    }

    init {
        try {
            Log.d(TAG, "🚀 Initializing FIXED Hybrid Classification Detector")

            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }

            val modelBuffer = loadModelFile(context, modelPath)
            interpreter = Interpreter(modelBuffer, options)

            // FIXED: Proper buffer initialization
            initializeBuffersFixed()
            isInitialized = true

            Log.d(TAG, "✅ FIXED Hybrid detector initialized successfully")
            Log.d(TAG, "📊 Input: ${realInputSize}x${realInputSize}x${inputChannels}")
            Log.d(TAG, "📊 Output: ${outputClasses} classes")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to initialize FIXED Hybrid Detector: ${e.message}", e)
            throw RuntimeException("Failed to initialize detector", e)
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

    private fun initializeBuffersFixed() {
        try {
            // FIXED: Get actual tensor information
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)

            val inputShape = inputTensor.shape()
            val outputShape = outputTensor.shape()

            Log.d(TAG, "🔍 FIXED Input tensor shape: ${inputShape.contentToString()}")
            Log.d(TAG, "🔍 FIXED Output tensor shape: ${outputShape.contentToString()}")
            Log.d(TAG, "🔍 FIXED Input tensor bytes: ${inputTensor.numBytes()}")
            Log.d(TAG, "🔍 FIXED Output tensor bytes: ${outputTensor.numBytes()}")

            // FIXED: Extract real dimensions from tensor shape
            when {
                inputShape.size == 4 -> {
                    // Format: [batch, height, width, channels] or [batch, channels, height, width]
                    if (inputShape[1] == 3 || inputShape[1] == 1) {
                        // NCHW format: [batch, channels, height, width]
                        inputChannels = inputShape[1]
                        realInputSize = inputShape[2]
                        Log.d(TAG, "✅ FIXED Detected NCHW format")
                    } else {
                        // NHWC format: [batch, height, width, channels]
                        realInputSize = inputShape[1]
                        inputChannels = inputShape[3]
                        Log.d(TAG, "✅ FIXED Detected NHWC format")
                    }
                }
                inputShape.size == 3 -> {
                    // Format: [channels, height, width] or [height, width, channels]
                    if (inputShape[0] == 3 || inputShape[0] == 1) {
                        // CHW format
                        inputChannels = inputShape[0]
                        realInputSize = inputShape[1]
                        Log.d(TAG, "✅ FIXED Detected CHW format")
                    } else {
                        // HWC format
                        realInputSize = inputShape[0]
                        inputChannels = inputShape[2]
                        Log.d(TAG, "✅ FIXED Detected HWC format")
                    }
                }
                else -> {
                    Log.w(TAG, "⚠️ FIXED Unknown input shape, using defaults")
                    realInputSize = 224
                    inputChannels = 3
                }
            }

            // FIXED: Extract output classes
            outputClasses = when {
                outputShape.size == 2 -> outputShape[1] // [batch, classes]
                outputShape.size == 1 -> outputShape[0] // [classes]
                else -> 25 // Default fallback
            }

            // FIXED: Use exact tensor size from TensorFlow Lite
            val exactInputBytes = inputTensor.numBytes()
            val exactOutputBytes = outputTensor.numBytes()

            Log.d(TAG, "✅ FIXED Using exact tensor sizes:")
            Log.d(TAG, "  📥 Input: $exactInputBytes bytes")
            Log.d(TAG, "  📤 Output: $exactOutputBytes bytes")
            Log.d(TAG, "  🖼️ Calculated input size: ${realInputSize}x${realInputSize}x${inputChannels}")
            Log.d(TAG, "  🎯 Output classes: $outputClasses")

            // FIXED: Initialize buffers with exact sizes
            inputBuffer = ByteBuffer.allocateDirect(exactInputBytes)
            inputBuffer.order(ByteOrder.nativeOrder())

            // FIXED: Initialize output buffer based on actual output shape
            outputBuffer = when {
                outputShape.size == 2 -> Array(outputShape[0]) { FloatArray(outputShape[1]) }
                outputShape.size == 1 -> Array(1) { FloatArray(outputShape[0]) }
                else -> Array(1) { FloatArray(outputClasses) }
            }

            Log.d(TAG, "✅ FIXED Buffers initialized successfully")
            Log.d(TAG, "  📥 Input buffer: ${inputBuffer.capacity()} bytes")
            Log.d(TAG, "  📤 Output buffer: ${outputBuffer.size} x ${outputBuffer[0].size}")

        } catch (e: Exception) {
            Log.e(TAG, "❌ FIXED Error initializing buffers: ${e.message}", e)
            // FIXED: Safe fallback initialization
            realInputSize = 224
            inputChannels = 3
            outputClasses = 25
            inputBuffer = ByteBuffer.allocateDirect(224 * 224 * 3 * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            outputBuffer = Array(1) { FloatArray(25) }
            Log.d(TAG, "⚠️ FIXED Using fallback buffer initialization")
        }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        if (!isInitialized) return emptyList()

        return try {
            // Step 1: Run classification model
            val classificationScore = runClassificationFixed(bitmap)

            if (classificationScore < confThreshold) {
                // No object detected, clear tracking
                trackingActive = false
                motionAreas.clear()
                return emptyList()
            }

            // Step 2: Detect motion areas if classification is positive
            val motionDetections = detectMotionAreas(bitmap)

            // Step 3: Combine classification + motion detection
            val finalDetections = combineClassificationAndMotion(
                classificationScore,
                motionDetections,
                bitmap.width,
                bitmap.height
            )

            if (finalDetections.isNotEmpty()) {
                trackingActive = true
                lastDetectionTime = System.currentTimeMillis()
                // Update last detection center
                val firstDetection = finalDetections.first()
                lastDetectionCenter = Pair(
                    (firstDetection.bbox.left + firstDetection.bbox.right) / 2f,
                    (firstDetection.bbox.top + firstDetection.bbox.bottom) / 2f
                )
            }

            Log.d(TAG, "🎯 FIXED Hybrid detection: classification=$classificationScore, motion=${motionDetections.size}, final=${finalDetections.size}")

            finalDetections

        } catch (e: Exception) {
            Log.e(TAG, "❌ FIXED Hybrid detection error: ${e.message}", e)
            emptyList()
        }
    }

    private fun runClassificationFixed(bitmap: Bitmap): Float {
        return try {
            // FIXED: Preprocess image with proper buffer management
            preprocessImageFixed(bitmap)

            // Run inference
            interpreter.run(inputBuffer, outputBuffer)

            // Get maximum confidence from output
            val maxConfidence = when {
                outputBuffer.isNotEmpty() && outputBuffer[0].isNotEmpty() -> {
                    outputBuffer[0].maxOrNull() ?: 0f
                }
                else -> 0f
            }

            val maxClassIndex = if (outputBuffer.isNotEmpty() && outputBuffer[0].isNotEmpty()) {
                outputBuffer[0].indexOfFirst { it == maxConfidence }
            } else {
                -1
            }

            Log.d(TAG, "✅ FIXED Classification: max_conf=$maxConfidence at class $maxClassIndex")

            maxConfidence

        } catch (e: Exception) {
            Log.e(TAG, "❌ FIXED Classification error: ${e.message}", e)
            return 0f
        }
    }

    private fun preprocessImageFixed(bitmap: Bitmap) {
        try {
            Log.d(TAG, "🔄 FIXED Preprocessing: ${bitmap.width}x${bitmap.height} -> ${realInputSize}x${realInputSize}")

            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, realInputSize, realInputSize, true)

            // FIXED: Clear buffer and verify capacity
            inputBuffer.clear()
            val requiredBytes = realInputSize * realInputSize * inputChannels * 4 // 4 bytes per float

            Log.d(TAG, "🔍 FIXED Buffer check:")
            Log.d(TAG, "  📊 Required bytes: $requiredBytes")
            Log.d(TAG, "  📊 Buffer capacity: ${inputBuffer.capacity()}")
            Log.d(TAG, "  📊 Buffer remaining: ${inputBuffer.remaining()}")

            if (inputBuffer.capacity() < requiredBytes) {
                Log.e(TAG, "❌ FIXED Buffer too small! Required: $requiredBytes, Available: ${inputBuffer.capacity()}")
                return
            }

            val pixels = IntArray(realInputSize * realInputSize)
            scaledBitmap.getPixels(pixels, 0, realInputSize, 0, 0, realInputSize, realInputSize)

            // FIXED: Careful pixel processing with bounds checking
            var pixelsProcessed = 0
            val maxPixels = realInputSize * realInputSize

            for (i in 0 until minOf(pixels.size, maxPixels)) {
                val pixel = pixels[i]
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                // FIXED: Check buffer space before each put operation
                if (inputBuffer.remaining() >= 12) { // 3 floats = 12 bytes
                    inputBuffer.putFloat(r)
                    inputBuffer.putFloat(g)
                    inputBuffer.putFloat(b)
                    pixelsProcessed++
                } else {
                    Log.w(TAG, "⚠️ FIXED Buffer full after $pixelsProcessed pixels")
                    break
                }
            }

            scaledBitmap.recycle()
            inputBuffer.rewind()

            Log.d(TAG, "✅ FIXED Preprocessing complete: $pixelsProcessed pixels processed")
            Log.d(TAG, "  📊 Buffer position: ${inputBuffer.position()}")
            Log.d(TAG, "  📊 Buffer limit: ${inputBuffer.limit()}")

        } catch (e: Exception) {
            Log.e(TAG, "❌ FIXED Preprocessing error: ${e.message}", e)
            throw e
        }
    }

    private fun detectMotionAreas(bitmap: Bitmap): List<RectF> {
        val currentTime = System.currentTimeMillis()

        // Skip motion detection if too recent
        if (currentTime - lastDetectionTime < 100) {
            return motionAreas.toList()
        }

        val newMotionAreas = mutableListOf<RectF>()

        try {
            // Simple motion detection using bitmap comparison
            val smallBitmap = Bitmap.createScaledBitmap(bitmap, 160, 120, true)

            previousBitmap?.let { prevBitmap ->
                if (prevBitmap.width == smallBitmap.width && prevBitmap.height == smallBitmap.height) {
                    val motionRegions = findMotionRegions(prevBitmap, smallBitmap)
                    newMotionAreas.addAll(motionRegions)
                }
            }

            // Update previous bitmap
            previousBitmap?.recycle()
            previousBitmap = smallBitmap

        } catch (e: Exception) {
            Log.w(TAG, "FIXED Motion detection error: ${e.message}")
        }

        motionAreas = newMotionAreas
        return newMotionAreas
    }

    private fun findMotionRegions(prev: Bitmap, curr: Bitmap): List<RectF> {
        val motionRegions = mutableListOf<RectF>()

        try {
            val width = prev.width
            val height = prev.height
            val blockSize = 20 // Process in 20x20 pixel blocks
            val threshold = 30 // Motion threshold

            for (y in 0 until height step blockSize) {
                for (x in 0 until width step blockSize) {
                    val endX = minOf(x + blockSize, width)
                    val endY = minOf(y + blockSize, height)

                    var totalDiff = 0
                    var pixelCount = 0

                    for (py in y until endY) {
                        for (px in x until endX) {
                            val prevPixel = prev.getPixel(px, py)
                            val currPixel = curr.getPixel(px, py)

                            val prevGray = (android.graphics.Color.red(prevPixel) +
                                    android.graphics.Color.green(prevPixel) +
                                    android.graphics.Color.blue(prevPixel)) / 3
                            val currGray = (android.graphics.Color.red(currPixel) +
                                    android.graphics.Color.green(currPixel) +
                                    android.graphics.Color.blue(currPixel)) / 3

                            totalDiff += abs(currGray - prevGray)
                            pixelCount++
                        }
                    }

                    val avgDiff = if (pixelCount > 0) totalDiff / pixelCount else 0

                    if (avgDiff > threshold) {
                        // Convert back to normalized coordinates
                        val left = x.toFloat() / width
                        val top = y.toFloat() / height
                        val right = endX.toFloat() / width
                        val bottom = endY.toFloat() / height

                        motionRegions.add(RectF(left, top, right, bottom))
                    }
                }
            }

        } catch (e: Exception) {
            Log.w(TAG, "FIXED Motion region detection error: ${e.message}")
        }

        return motionRegions
    }

    private fun combineClassificationAndMotion(
        classificationScore: Float,
        motionAreas: List<RectF>,
        imageWidth: Int,
        imageHeight: Int
    ): List<Detection> {

        val detections = mutableListOf<Detection>()

        when {
            motionAreas.isNotEmpty() -> {
                // Use motion areas as bounding boxes
                motionAreas.take(maxDetections).forEach { motionArea ->
                    // Expand motion area slightly for better visibility
                    val expandedArea = expandBoundingBox(motionArea, 0.1f)

                    detections.add(Detection(
                        bbox = expandedArea,
                        score = classificationScore,
                        classId = 0
                    ))
                }
                Log.d(TAG, "✅ FIXED Using ${motionAreas.size} motion areas as detections")
            }

            trackingActive && (System.currentTimeMillis() - lastDetectionTime) < trackingTimeout -> {
                // Continue tracking at last known position
                val trackingBox = RectF(
                    (lastDetectionCenter.first - 0.15f).coerceIn(0f, 1f),
                    (lastDetectionCenter.second - 0.15f).coerceIn(0f, 1f),
                    (lastDetectionCenter.first + 0.15f).coerceIn(0f, 1f),
                    (lastDetectionCenter.second + 0.15f).coerceIn(0f, 1f)
                )

                detections.add(Detection(
                    bbox = trackingBox,
                    score = classificationScore * 0.8f, // Reduce confidence for tracking
                    classId = 0
                ))
                Log.d(TAG, "✅ FIXED Using tracking continuation at last position")
            }

            else -> {
                // Fallback: create detection in center area
                val centerBox = RectF(0.3f, 0.3f, 0.7f, 0.7f)
                detections.add(Detection(
                    bbox = centerBox,
                    score = classificationScore,
                    classId = 0
                ))
                Log.d(TAG, "✅ FIXED Using fallback center detection")
            }
        }

        return detections
    }

    private fun expandBoundingBox(bbox: RectF, factor: Float): RectF {
        val centerX = (bbox.left + bbox.right) / 2f
        val centerY = (bbox.top + bbox.bottom) / 2f
        val width = bbox.right - bbox.left
        val height = bbox.bottom - bbox.top

        val newWidth = width * (1f + factor)
        val newHeight = height * (1f + factor)

        return RectF(
            (centerX - newWidth / 2f).coerceIn(0f, 1f),
            (centerY - newHeight / 2f).coerceIn(0f, 1f),
            (centerX + newWidth / 2f).coerceIn(0f, 1f),
            (centerY + newHeight / 2f).coerceIn(0f, 1f)
        )
    }

    // Interface methods for compatibility
    fun getClassLabel(classId: Int): String = "Barbell"

    fun getDetectionCenter(detection: Detection): Pair<Float, Float> = Pair(
        (detection.bbox.left + detection.bbox.right) / 2f,
        (detection.bbox.top + detection.bbox.bottom) / 2f
    )

    fun getDetectionQuality(detection: Detection): DetectionQuality {
        return DetectionQuality(
            confidence = detection.score,
            size = (detection.bbox.right - detection.bbox.left) * (detection.bbox.bottom - detection.bbox.top),
            aspectRatio = (detection.bbox.right - detection.bbox.left) / (detection.bbox.bottom - detection.bbox.top),
            stability = if (trackingActive) 0.8f else 0.6f
        )
    }

    fun isUsingGPU(): Boolean = false

    fun getPerformanceInfo(): String = "FIXED Hybrid Classification+Motion (simonskina.tflite)"

    fun cleanup() {
        previousBitmap?.recycle()
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
    }

    fun close() = cleanup()
}

/**
 * Detection quality data class
 */
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