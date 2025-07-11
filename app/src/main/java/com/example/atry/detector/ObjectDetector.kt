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
 * Fixed TensorFlow Lite Object Detector
 * Addresses detection coordinate and preprocessing issues
 */
class GenericTFLiteDetector(
    private val context: Context,
    private val modelPath: String = "simonskina.tflite",
    private val inputSize: Int = 320, // Will be overridden by actual model
    private val confThreshold: Float = 0.1f, // Lowered threshold
    private val iouThreshold: Float = 0.5f,
    private val maxDetections: Int = 10
) {

    private lateinit var interpreter: Interpreter
    private var isInitialized = false

    // Model info - Fixed
    private var inputTensorIndex = 0
    private var outputTensorIndex = 0
    private var inputShape = intArrayOf()
    private var outputShape = intArrayOf()
    private var modelType = ModelType.UNKNOWN
    private var actualInputSize = 320 // Actual model input size

    // Buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: Any

    companion object {
        private const val TAG = "FixedTFLiteDetector"

        enum class ModelType {
            YOLO_V5,        // YOLOv5 format
            YOLO_V8,        // YOLOv8 format
            CLASSIFICATION, // Classification only
            DETECTION,      // Generic detection
            UNKNOWN
        }
    }

    init {
        try {
            Log.d(TAG, "🚀 Initializing Fixed TFLite Detector: $modelPath")

            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }

            val modelBuffer = loadModelFile(context, modelPath)
            interpreter = Interpreter(modelBuffer, options)

            analyzeModelFixed()
            initializeBuffersFixed()

            isInitialized = true
            Log.d(TAG, "✅ Fixed detector initialized successfully")
            Log.d(TAG, "📊 Model type: $modelType")
            Log.d(TAG, "📥 Actual input size: ${actualInputSize}x$actualInputSize")
            Log.d(TAG, "📤 Output shape: ${outputShape.contentToString()}")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to initialize Fixed TFLite Detector: ${e.message}", e)
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

    private fun analyzeModelFixed() {
        try {
            Log.d(TAG, "📊 Analyzing model architecture (Fixed)...")

            // Get input tensor info
            val inputTensor = interpreter.getInputTensor(inputTensorIndex)
            inputShape = inputTensor.shape()

            Log.d(TAG, "Input tensor details:")
            Log.d(TAG, "  Shape: ${inputShape.contentToString()}")
            Log.d(TAG, "  Data type: ${inputTensor.dataType()}")
            Log.d(TAG, "  Name: ${inputTensor.name()}")

            // Extract actual input size from shape
            // Common formats: [1, height, width, channels] or [1, channels, height, width]
            actualInputSize = when {
                inputShape.size == 4 -> {
                    // Determine if NHWC or NCHW format
                    if (inputShape[1] == 3 || inputShape[1] == 1) {
                        // NCHW format: [batch, channels, height, width]
                        inputShape[2] // height (assuming square input)
                    } else {
                        // NHWC format: [batch, height, width, channels]
                        inputShape[1] // height
                    }
                }
                inputShape.size == 3 -> {
                    // CHW or HWC format
                    if (inputShape[0] == 3 || inputShape[0] == 1) {
                        inputShape[1] // HW
                    } else {
                        inputShape[0] // HW
                    }
                }
                else -> 320 // Default fallback
            }

            Log.d(TAG, "✅ Extracted input size: ${actualInputSize}x$actualInputSize")

            // Analyze all output tensors
            Log.d(TAG, "Output tensors:")
            for (i in 0 until interpreter.outputTensorCount) {
                val outputTensor = interpreter.getOutputTensor(i)
                val shape = outputTensor.shape()
                Log.d(TAG, "  Output $i: ${shape.contentToString()} - ${outputTensor.dataType()}")

                if (i == 0) {
                    outputShape = shape
                }
            }

            // Determine model type based on output analysis
            modelType = determineModelType()
            Log.d(TAG, "🎯 Detected model type: $modelType")

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing model: ${e.message}", e)
            modelType = ModelType.UNKNOWN
        }
    }

    private fun determineModelType(): ModelType {
        return when {
            // YOLOv5: [1, 25200, 85] or similar
            outputShape.size == 3 && outputShape[2] > 80 -> ModelType.YOLO_V5

            // YOLOv8: [1, 84, 8400] - transposed format
            outputShape.size == 3 && outputShape[1] > 80 && outputShape[2] > 1000 -> ModelType.YOLO_V8

            // Classification: [1, num_classes]
            outputShape.size == 2 && outputShape[1] < 1000 -> ModelType.CLASSIFICATION

            // Generic detection
            outputShape.size == 3 -> ModelType.DETECTION

            else -> ModelType.UNKNOWN
        }
    }

    private fun initializeBuffersFixed() {
        // Calculate exact buffer size needed
        val channels = if (inputShape[1] == 3 || inputShape[1] == 1) inputShape[1] else 3
        val bufferSize = actualInputSize * actualInputSize * channels * 4 // 4 bytes per float

        Log.d(TAG, "Initializing buffers:")
        Log.d(TAG, "  Input size: ${actualInputSize}x$actualInputSize")
        Log.d(TAG, "  Channels: $channels")
        Log.d(TAG, "  Buffer size: $bufferSize bytes")

        inputBuffer = ByteBuffer.allocateDirect(bufferSize)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Initialize output buffer based on model type
        outputBuffer = when (modelType) {
            ModelType.YOLO_V5, ModelType.YOLO_V8, ModelType.DETECTION -> {
                Array(outputShape[0]) {
                    Array(outputShape[1]) {
                        FloatArray(outputShape[2])
                    }
                }
            }
            ModelType.CLASSIFICATION -> {
                Array(outputShape[0]) { FloatArray(outputShape[1]) }
            }
            else -> {
                // Generic fallback
                Array(outputShape[0]) { FloatArray(outputShape[1]) }
            }
        }

        Log.d(TAG, "✅ Buffers initialized successfully")
    }

    private fun preprocessImageFixed(bitmap: Bitmap): ByteBuffer {
        Log.d(TAG, "Preprocessing image: ${bitmap.width}x${bitmap.height} -> ${actualInputSize}x$actualInputSize")

        // Create scaled bitmap
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, actualInputSize, actualInputSize, true)

        inputBuffer.clear()

        val pixels = IntArray(actualInputSize * actualInputSize)
        scaledBitmap.getPixels(pixels, 0, actualInputSize, 0, 0, actualInputSize, actualInputSize)

        // Determine input format from shape
        val isNHWC = inputShape.size == 4 && inputShape[3] == 3
        val isNHCW = inputShape.size == 4 && inputShape[1] == 3

        Log.d(TAG, "Input format: ${if (isNHWC) "NHWC" else if (isNHCW) "NCHW" else "Unknown"}")

        if (isNHCW) {
            // NCHW format: need to arrange as [channels][height][width]
            val rChannel = FloatArray(actualInputSize * actualInputSize)
            val gChannel = FloatArray(actualInputSize * actualInputSize)
            val bChannel = FloatArray(actualInputSize * actualInputSize)

            for (i in pixels.indices) {
                val pixel = pixels[i]
                rChannel[i] = ((pixel shr 16) and 0xFF) / 255.0f
                gChannel[i] = ((pixel shr 8) and 0xFF) / 255.0f
                bChannel[i] = (pixel and 0xFF) / 255.0f
            }

            // Write channels in order
            rChannel.forEach { inputBuffer.putFloat(it) }
            gChannel.forEach { inputBuffer.putFloat(it) }
            bChannel.forEach { inputBuffer.putFloat(it) }
        } else {
            // NHWC format or default: [height][width][channels]
            for (pixel in pixels) {
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
            }
        }

        scaledBitmap.recycle()
        inputBuffer.rewind()

        Log.d(TAG, "✅ Image preprocessed successfully")
        return inputBuffer
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        if (!isInitialized) {
            Log.w(TAG, "Detector not initialized")
            return emptyList()
        }

        return try {
            Log.d(TAG, "🔍 Starting detection on ${bitmap.width}x${bitmap.height} bitmap")

            val input = preprocessImageFixed(bitmap)

            // Run inference
            val inferenceStart = System.currentTimeMillis()
            interpreter.run(input, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - inferenceStart

            Log.d(TAG, "✅ Inference completed in ${inferenceTime}ms")

            // Process output based on model type
            val detections = when (modelType) {
                ModelType.YOLO_V5 -> processYOLOv5Output()
                ModelType.YOLO_V8 -> processYOLOv8Output()
                ModelType.CLASSIFICATION -> processClassificationOutput()
                ModelType.DETECTION -> processGenericDetectionOutput()
                else -> processUnknownOutput()
            }

            Log.d(TAG, "🎯 Raw detections found: ${detections.size}")

            // Apply NMS and return results
            val filteredDetections = applyNMS(detections)
            Log.d(TAG, "🎯 Final detections after NMS: ${filteredDetections.size}")

            if (filteredDetections.isNotEmpty()) {
                filteredDetections.forEachIndexed { index, detection ->
                    Log.d(TAG, "Detection $index: conf=${String.format("%.3f", detection.score)}, " +
                            "bbox=[${String.format("%.3f", detection.bbox.left)}, " +
                            "${String.format("%.3f", detection.bbox.top)}, " +
                            "${String.format("%.3f", detection.bbox.right)}, " +
                            "${String.format("%.3f", detection.bbox.bottom)}]")
                }
            }

            filteredDetections

        } catch (e: Exception) {
            Log.e(TAG, "❌ Detection error: ${e.message}", e)
            emptyList()
        }
    }

    private fun processYOLOv5Output(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            val output = outputBuffer as Array<Array<FloatArray>>
            // YOLOv5 format: [1, 25200, 85] -> [x, y, w, h, conf, class_scores...]

            val numDetections = output[0].size
            Log.d(TAG, "Processing YOLOv5 output: $numDetections detections")

            for (i in 0 until numDetections) {
                val detection = output[0][i]

                if (detection.size >= 5) {
                    // YOLOv5 format: center_x, center_y, width, height, confidence, class_scores...
                    val centerX = detection[0]
                    val centerY = detection[1]
                    val width = detection[2]
                    val height = detection[3]
                    val confidence = detection[4]

                    if (confidence >= confThreshold) {
                        // Convert from center format to corner format
                        val left = (centerX - width / 2f).coerceIn(0f, 1f)
                        val top = (centerY - height / 2f).coerceIn(0f, 1f)
                        val right = (centerX + width / 2f).coerceIn(0f, 1f)
                        val bottom = (centerY + height / 2f).coerceIn(0f, 1f)

                        // Ensure valid bounding box
                        if (right > left && bottom > top &&
                            width > 0.01f && height > 0.01f) { // Minimum size check

                            detections.add(Detection(
                                bbox = RectF(left, top, right, bottom),
                                score = confidence,
                                classId = 0 // Assume barbell class
                            ))
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing YOLOv5 output: ${e.message}")
        }

        return detections
    }

    private fun processYOLOv8Output(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            val output = outputBuffer as Array<Array<FloatArray>>
            // YOLOv8 format: [1, 84, 8400] - transposed format

            val numFeatures = output[0].size // 84
            val numDetections = output[0][0].size // 8400

            Log.d(TAG, "Processing YOLOv8 output: $numDetections detections, $numFeatures features")

            for (i in 0 until numDetections) {
                // Extract values for this detection
                val centerX = output[0][0][i]
                val centerY = output[0][1][i]
                val width = output[0][2][i]
                val height = output[0][3][i]

                // Find max class confidence (features 4 to end are class scores)
                var maxConfidence = 0f
                var maxClassId = 0
                for (classIdx in 4 until numFeatures) {
                    val classConf = output[0][classIdx][i]
                    if (classConf > maxConfidence) {
                        maxConfidence = classConf
                        maxClassId = classIdx - 4
                    }
                }

                if (maxConfidence >= confThreshold) {
                    // Convert from center format to corner format
                    val left = (centerX - width / 2f).coerceIn(0f, 1f)
                    val top = (centerY - height / 2f).coerceIn(0f, 1f)
                    val right = (centerX + width / 2f).coerceIn(0f, 1f)
                    val bottom = (centerY + height / 2f).coerceIn(0f, 1f)

                    // Ensure valid bounding box
                    if (right > left && bottom > top &&
                        width > 0.01f && height > 0.01f) {

                        detections.add(Detection(
                            bbox = RectF(left, top, right, bottom),
                            score = maxConfidence,
                            classId = maxClassId
                        ))
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing YOLOv8 output: ${e.message}")
        }

        return detections
    }

    private fun processClassificationOutput(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            val output = outputBuffer as Array<FloatArray>
            val classScores = output[0]

            val maxConfidence = classScores.maxOrNull() ?: 0f
            val maxClassIndex = classScores.indexOfFirst { it == maxConfidence }

            Log.d(TAG, "Classification output: max confidence = $maxConfidence at class $maxClassIndex")

            if (maxConfidence > confThreshold) {
                // For classification, create a detection covering center area
                detections.add(Detection(
                    bbox = RectF(0.2f, 0.2f, 0.8f, 0.8f), // Center 60% area
                    score = maxConfidence,
                    classId = maxClassIndex
                ))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing classification output: ${e.message}")
        }

        return detections
    }

    private fun processGenericDetectionOutput(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            val output = outputBuffer as Array<Array<FloatArray>>

            Log.d(TAG, "Processing generic detection output")

            for (i in output[0].indices) {
                val detection = output[0][i]
                if (detection.size >= 5) {
                    val confidence = detection[4]

                    if (confidence >= confThreshold) {
                        // Assume format: [x1, y1, x2, y2, conf] or [cx, cy, w, h, conf]
                        val left = detection[0].coerceIn(0f, 1f)
                        val top = detection[1].coerceIn(0f, 1f)
                        val right = detection[2].coerceIn(0f, 1f)
                        val bottom = detection[3].coerceIn(0f, 1f)

                        // Check if this might be center format
                        if (right < left || bottom < top) {
                            // Probably center format: convert
                            val centerX = detection[0]
                            val centerY = detection[1]
                            val width = detection[2]
                            val height = detection[3]

                            val newLeft = (centerX - width / 2f).coerceIn(0f, 1f)
                            val newTop = (centerY - height / 2f).coerceIn(0f, 1f)
                            val newRight = (centerX + width / 2f).coerceIn(0f, 1f)
                            val newBottom = (centerY + height / 2f).coerceIn(0f, 1f)

                            if (newRight > newLeft && newBottom > newTop) {
                                detections.add(Detection(
                                    bbox = RectF(newLeft, newTop, newRight, newBottom),
                                    score = confidence,
                                    classId = 0
                                ))
                            }
                        } else if (right > left && bottom > top) {
                            detections.add(Detection(
                                bbox = RectF(left, top, right, bottom),
                                score = confidence,
                                classId = 0
                            ))
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing generic detection output: ${e.message}")
        }

        return detections
    }

    private fun processUnknownOutput(): List<Detection> {
        Log.w(TAG, "Unknown model format - attempting basic processing")

        // Try to extract any meaningful detections from unknown format
        val detections = mutableListOf<Detection>()

        try {
            when (outputBuffer) {
                is Array<*> -> {
                    val output = outputBuffer as Array<*>
                    if (output.isNotEmpty() && output[0] is FloatArray) {
                        val floatOutput = output as Array<FloatArray>

                        // Look for values that might be detections
                        for (i in floatOutput.indices) {
                            val row = floatOutput[i]
                            if (row.size >= 5) {
                                // Try to interpret as detection
                                val confidence = row.maxOrNull() ?: 0f
                                if (confidence > confThreshold) {
                                    // Create a generic detection
                                    detections.add(Detection(
                                        bbox = RectF(0.3f, 0.3f, 0.7f, 0.7f),
                                        score = confidence,
                                        classId = 0
                                    ))
                                    break // Only add one for unknown format
                                }
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing unknown output: ${e.message}")
        }

        return detections
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

    // Compatibility methods
    fun getClassLabel(classId: Int): String = "Barbell"

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
    fun getPerformanceInfo(): String = "Fixed TFLite ($modelType)"

    fun cleanup() {
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
    }

    fun close() = cleanup()
}