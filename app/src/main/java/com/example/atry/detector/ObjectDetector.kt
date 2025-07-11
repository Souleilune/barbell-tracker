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
 * Generic TensorFlow Lite Object Detector
 * Works with different model architectures including EfficientDet, MobileNet, etc.
 * Automatically detects model format and adapts accordingly
 */
class GenericTFLiteDetector(
    private val context: Context,
    private val modelPath: String = "simonskina.tflite",
    private val inputSize: Int = 320,
    private val confThreshold: Float = 0.05f, // Much lower threshold for custom models
    private val iouThreshold: Float = 0.5f,
    private val maxDetections: Int = 10
) {

    private lateinit var interpreter: Interpreter
    private var isInitialized = false

    // Model info
    private var inputTensorIndex = 0
    private var outputTensorIndex = 0
    private var inputShape = intArrayOf()
    private var outputShape = intArrayOf()
    private var modelType = ModelType.UNKNOWN
    private var realInputSize = 224 // Will be calculated from actual tensor

    // Buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: Any

    // Class labels - generic for barbell detection
    private val classLabels = arrayOf("barbell", "weight", "plate", "object")

    companion object {
        private const val TAG = "GenericTFLiteDetector"

        enum class ModelType {
            YOLO,           // YOLO format: [batch, num_detections, 85] or similar
            EFFICIENTDET,   // EfficientDet format: Multiple outputs
            MOBILENET,      // MobileNet SSD format
            UNKNOWN
        }
    }

    init {
        try {
            Log.d(TAG, "🚀 Initializing Generic TFLite Detector for: $modelPath")

            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
                // Don't use GPU delegate initially - try CPU first
            }

            val modelBuffer = loadModelFile(context, modelPath)
            interpreter = Interpreter(modelBuffer, options)

            analyzeModel()
            initializeBuffers()

            isInitialized = true
            Log.d(TAG, "✅ Generic detector initialized successfully")
            Log.d(TAG, "📊 Model type: $modelType")
            Log.d(TAG, "📥 Input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "📤 Output shape: ${outputShape.contentToString()}")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to initialize Generic TFLite Detector: ${e.message}", e)
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

    private fun analyzeModel() {
        try {
            // Analyze input tensor
            val inputTensor = interpreter.getInputTensor(inputTensorIndex)
            inputShape = inputTensor.shape()

            Log.d(TAG, "📊 Analyzing model architecture...")
            Log.d(TAG, "Input tensor count: ${interpreter.inputTensorCount}")
            Log.d(TAG, "Output tensor count: ${interpreter.outputTensorCount}")

            // Log input details for debugging
            Log.d(TAG, "Input shape from tensor: ${inputShape.contentToString()}")
            Log.d(TAG, "Input data type: ${inputTensor.dataType()}")
            Log.d(TAG, "Input tensor name: ${inputTensor.name()}")

            // CRITICAL: Get the actual tensor size in bytes, not calculated
            val actualTensorBytes = try {
                inputTensor.numBytes()
            } catch (e: Exception) {
                Log.w(TAG, "Could not get tensor numBytes, calculating manually")
                inputShape.fold(1) { acc, dim -> acc * dim } * 4
            }

            Log.d(TAG, "🔍 Actual tensor expects: $actualTensorBytes bytes")

            // Calculate what size this actually corresponds to
            val actualInputSize = kotlin.math.sqrt((actualTensorBytes / 4 / 3).toDouble()).toInt() // bytes / 4 (float) / 3 (channels)
            Log.d(TAG, "🎯 Real input size should be: ${actualInputSize}x$actualInputSize")

            // Analyze all output tensors to determine model type
            for (i in 0 until interpreter.outputTensorCount) {
                val outputTensor = interpreter.getOutputTensor(i)
                val shape = outputTensor.shape()
                Log.d(TAG, "Output $i shape: ${shape.contentToString()}")
                Log.d(TAG, "Output $i data type: ${outputTensor.dataType()}")
                Log.d(TAG, "Output $i name: ${outputTensor.name()}")

                if (i == 0) {
                    outputShape = shape
                }
            }

            // Determine model type based on output shape
            modelType = when {
                // YOLO: typically [1, num_detections, 85] or [1, 25200, 85]
                outputShape.size == 3 && outputShape[2] > 80 -> ModelType.YOLO

                // EfficientDet: typically multiple outputs
                interpreter.outputTensorCount > 2 -> ModelType.EFFICIENTDET

                // MobileNet SSD: typically [1, num_detections, 4] + [1, num_detections, num_classes]
                outputShape.size == 3 && outputShape[2] <= 10 -> ModelType.MOBILENET

                else -> ModelType.UNKNOWN
            }

            Log.d(TAG, "🎯 Detected model type: $modelType")

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing model: ${e.message}", e)
            modelType = ModelType.UNKNOWN
        }
    }

    private fun initializeBuffers() {
        // Get the ACTUAL tensor size in bytes directly from TensorFlow Lite
        val inputTensor = interpreter.getInputTensor(inputTensorIndex)
        val actualTensorBytes = try {
            inputTensor.numBytes()
        } catch (e: Exception) {
            Log.w(TAG, "Could not get numBytes, using calculated size")
            inputShape.fold(1) { acc, dim -> acc * dim } * 4
        }

        // Calculate the real input dimensions
        val actualInputSize = kotlin.math.sqrt((actualTensorBytes / 4 / 3).toDouble()).toInt()

        Log.d(TAG, "✅ Using actual tensor size: $actualTensorBytes bytes")
        Log.d(TAG, "✅ Calculated real input size: ${actualInputSize}x$actualInputSize")

        // Initialize input buffer with ACTUAL size
        inputBuffer = ByteBuffer.allocateDirect(actualTensorBytes)
        inputBuffer.order(ByteOrder.nativeOrder())

        Log.d(TAG, "✅ Input buffer allocated: $actualTensorBytes bytes")

        // Store the real input size for preprocessing
        realInputSize = actualInputSize

        // Initialize output buffer based on model type
        outputBuffer = when (modelType) {
            ModelType.YOLO -> {
                // For YOLO: [batch, detections, features]
                Array(outputShape[0]) {
                    Array(outputShape[1]) {
                        FloatArray(outputShape[2])
                    }
                }
            }
            ModelType.EFFICIENTDET -> {
                // For EfficientDet: Multiple outputs, use first one
                val shape = outputShape
                Array(shape[0]) { FloatArray(shape[1]) }
            }
            ModelType.MOBILENET -> {
                // For MobileNet: [batch, detections, 4 or classes]
                Array(outputShape[0]) {
                    Array(outputShape[1]) {
                        FloatArray(outputShape[2])
                    }
                }
            }
            else -> {
                // Generic fallback - try to handle any shape
                when (outputShape.size) {
                    3 -> Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
                    2 -> Array(outputShape[0]) { FloatArray(outputShape[1]) }
                    1 -> FloatArray(outputShape[0])
                    else -> Array(outputShape[0]) { FloatArray(outputShape[1]) }
                }
            }
        }

        Log.d(TAG, "✅ Buffers initialized for model type: $modelType")
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Use the REAL input size calculated from actual tensor bytes
        val modelInputSize = realInputSize

        Log.d(TAG, "Preprocessing image: ${bitmap.width}x${bitmap.height} -> ${modelInputSize}x$modelInputSize")

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, modelInputSize, modelInputSize, true)

        inputBuffer.clear()

        val pixels = IntArray(modelInputSize * modelInputSize)
        scaledBitmap.getPixels(pixels, 0, modelInputSize, 0, 0, modelInputSize, modelInputSize)

        // Check if model expects different channel order
        val isRGB = true // Most models expect RGB, but some might expect BGR

        // Normalize pixels based on common practices
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            if (isRGB) {
                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
            } else {
                // BGR order for some models
                inputBuffer.putFloat(b)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(r)
            }
        }

        scaledBitmap.recycle()
        inputBuffer.rewind()

        Log.d(TAG, "✅ Image preprocessed: buffer size = ${inputBuffer.capacity()} bytes")
        return inputBuffer
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        if (!isInitialized) {
            Log.w(TAG, "Detector not initialized")
            return emptyList()
        }

        return try {
            Log.d(TAG, "🔍 Starting detection on ${bitmap.width}x${bitmap.height} bitmap")

            val preprocessStart = System.currentTimeMillis()
            val input = preprocessImage(bitmap)
            val preprocessTime = System.currentTimeMillis() - preprocessStart

            Log.d(TAG, "✅ Preprocessing completed in ${preprocessTime}ms")
            Log.d(TAG, "📥 Input buffer size: ${input.capacity()} bytes")
            Log.d(TAG, "📏 Expected tensor size: ${inputShape.contentToString()}")

            // Debug: Check if buffer size matches tensor expectation
            // Use the REAL tensor size, not calculated from shape
            val inputTensor = interpreter.getInputTensor(inputTensorIndex)
            val expectedBytes = try {
                inputTensor.numBytes()
            } catch (e: Exception) {
                inputShape.fold(1) { acc, dim -> acc * dim } * 4
            }

            Log.d(TAG, "🔍 Expected bytes: $expectedBytes, Actual bytes: ${input.capacity()}")

            if (input.capacity() != expectedBytes) {
                Log.e(TAG, "❌ BUFFER SIZE MISMATCH!")
                Log.e(TAG, "Model expects: $expectedBytes bytes")
                Log.e(TAG, "We provided: ${input.capacity()} bytes")
                Log.e(TAG, "Input shape: ${inputShape.contentToString()}")
                return emptyList()
            }

            // Run inference
            val inferenceStart = System.currentTimeMillis()
            interpreter.run(input, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - inferenceStart

            Log.d(TAG, "✅ Inference completed in ${inferenceTime}ms")

            // Process output based on model type
            val postprocessStart = System.currentTimeMillis()
            val detections = when (modelType) {
                ModelType.YOLO -> processYOLOOutput()
                ModelType.EFFICIENTDET -> processEfficientDetOutput()
                ModelType.MOBILENET -> processMobileNetOutput()
                else -> processGenericOutput()
            }
            val postprocessTime = System.currentTimeMillis() - postprocessStart

            Log.d(TAG, "✅ Post-processing completed in ${postprocessTime}ms")
            Log.d(TAG, "🎯 Raw detections found: ${detections.size}")

            // Apply NMS and return results
            val nmsStart = System.currentTimeMillis()
            val filteredDetections = applyNMS(detections)
            val nmsTime = System.currentTimeMillis() - nmsStart

            Log.d(TAG, "✅ NMS completed in ${nmsTime}ms")
            Log.d(TAG, "🎯 Final detections: ${filteredDetections.size}")

            if (filteredDetections.isNotEmpty()) {
                Log.d(TAG, "🎯 Detected ${filteredDetections.size} objects with generic detector")
                filteredDetections.forEachIndexed { index, detection ->
                    Log.d(TAG, "Detection $index: conf=${String.format("%.3f", detection.score)}, " +
                            "bbox=[${String.format("%.3f", detection.bbox.left)}, " +
                            "${String.format("%.3f", detection.bbox.top)}, " +
                            "${String.format("%.3f", detection.bbox.right)}, " +
                            "${String.format("%.3f", detection.bbox.bottom)}]")
                }
            } else {
                Log.d(TAG, "⚠️ No valid detections found")
                // Debug: Log some raw output to understand what the model is producing
                Log.d(TAG, "🔍 Debugging raw output...")

                // Create local reference to avoid smart cast issues
                val localOutputBuffer = outputBuffer

                when (localOutputBuffer) {
                    is Array<*> -> {
                        if (localOutputBuffer.isNotEmpty() && localOutputBuffer[0] is Array<*>) {
                            val output3D = localOutputBuffer as Array<Array<FloatArray>>
                            Log.d(TAG, "Raw output shape: [${output3D.size}][${output3D[0].size}][${output3D[0][0].size}]")
                            if (output3D[0].isNotEmpty()) {
                                val firstDetection = output3D[0][0]
                                Log.d(TAG, "First detection raw: ${firstDetection.take(10).joinToString(", ")}")
                                Log.d(TAG, "Max confidence in first detection: ${firstDetection.maxOrNull()}")
                            }
                        } else if (localOutputBuffer.isNotEmpty() && localOutputBuffer[0] is FloatArray) {
                            val output2D = localOutputBuffer as Array<FloatArray>
                            Log.d(TAG, "Raw output shape: [${output2D.size}][${output2D[0].size}]")
                            if (output2D.isNotEmpty()) {
                                Log.d(TAG, "First row: ${output2D[0].take(10).joinToString(", ")}")
                                Log.d(TAG, "Max value in first row: ${output2D[0].maxOrNull()}")
                            }
                        }
                    }
                    is FloatArray -> {
                        val output1D = localOutputBuffer as FloatArray
                        Log.d(TAG, "Raw output size: ${output1D.size}")
                        Log.d(TAG, "First 10 values: ${output1D.take(10).joinToString(", ")}")
                        Log.d(TAG, "Max value: ${output1D.maxOrNull()}")
                    }
                    else -> {
                        Log.d(TAG, "Unknown output buffer type: ${localOutputBuffer.javaClass.simpleName}")
                    }
                }
            }

            // Log performance summary
            val totalTime = preprocessTime + inferenceTime + postprocessTime + nmsTime
            Log.d(TAG, "📊 Performance: Total=${totalTime}ms (prep=${preprocessTime}ms, inf=${inferenceTime}ms, post=${postprocessTime}ms, nms=${nmsTime}ms)")

            filteredDetections

        } catch (e: Exception) {
            Log.e(TAG, "❌ Detection error: ${e.message}", e)
            Log.e(TAG, "Input shape: ${inputShape.contentToString()}")
            Log.e(TAG, "Input buffer capacity: ${if (::inputBuffer.isInitialized) inputBuffer.capacity() else "not initialized"}")
            Log.e(TAG, "Model type: $modelType")
            emptyList()
        }
    }

    private fun processYOLOOutput(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            val output = outputBuffer as Array<Array<FloatArray>>
            val numDetections = output[0].size
            val numFeatures = output[0][0].size

            Log.d(TAG, "Processing YOLO output: $numDetections detections, $numFeatures features")

            for (i in 0 until numDetections) {
                val detection = output[0][i]

                // YOLO format: [x, y, w, h, confidence, class_scores...]
                if (detection.size >= 5) {
                    val x = detection[0]
                    val y = detection[1]
                    val w = detection[2]
                    val h = detection[3]
                    val confidence = detection[4]

                    if (confidence >= confThreshold) {
                        val left = (x - w / 2f).coerceIn(0f, 1f)
                        val top = (y - h / 2f).coerceIn(0f, 1f)
                        val right = (x + w / 2f).coerceIn(0f, 1f)
                        val bottom = (y + h / 2f).coerceIn(0f, 1f)

                        if (right > left && bottom > top) {
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
            Log.e(TAG, "Error processing YOLO output: ${e.message}")
        }

        return detections
    }

    private fun processEfficientDetOutput(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            val localOutputBuffer = outputBuffer

            when (localOutputBuffer) {
                is Array<*> -> {
                    if (localOutputBuffer.isNotEmpty() && localOutputBuffer[0] is FloatArray) {
                        val output2D = localOutputBuffer as Array<FloatArray>

                        Log.d(TAG, "Processing EfficientDet output: ${output2D.size} x ${output2D[0].size}")

                        // Check if this is actually a classification output [1, 25]
                        if (output2D.size == 1 && output2D[0].size == 25) {
                            Log.d(TAG, "🎯 Detected classification-style output with 25 classes")

                            // Treat as classification - find highest confidence class
                            val classScores = output2D[0]
                            val maxConfidence = classScores.maxOrNull() ?: 0f
                            val maxClassIndex = classScores.indexOfFirst { it == maxConfidence }

                            Log.d(TAG, "Max confidence: $maxConfidence at class $maxClassIndex")

                            // If confidence is above threshold, create a detection
                            // Since this is classification, we don't have real bbox coordinates
                            // We'll create a generic detection that the tracker can use
                            if (maxConfidence > confThreshold) {
                                // Create a detection covering a reasonable area in the center
                                // The tracker will handle the actual object tracking
                                detections.add(Detection(
                                    bbox = android.graphics.RectF(0.25f, 0.25f, 0.75f, 0.75f), // Center 50% area
                                    score = maxConfidence,
                                    classId = maxClassIndex
                                ))
                                Log.d(TAG, "✅ Created classification detection: conf=$maxConfidence, class=$maxClassIndex")
                            } else {
                                Log.d(TAG, "⚠️ Confidence $maxConfidence below threshold $confThreshold")
                            }
                        } else {
                            // Standard EfficientDet processing
                            Log.d(TAG, "Processing standard EfficientDet format")

                            for (i in output2D.indices) {
                                val row = output2D[i]
                                if (row.size >= 5) {
                                    val confidence = row[4]
                                    if (confidence >= confThreshold) {
                                        val left = row[0].coerceIn(0f, 1f)
                                        val top = row[1].coerceIn(0f, 1f)
                                        val right = row[2].coerceIn(0f, 1f)
                                        val bottom = row[3].coerceIn(0f, 1f)

                                        if (right > left && bottom > top) {
                                            detections.add(Detection(
                                                bbox = android.graphics.RectF(left, top, right, bottom),
                                                score = confidence,
                                                classId = 0
                                            ))
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else -> {
                    Log.d(TAG, "Unknown EfficientDet output format: ${localOutputBuffer.javaClass.simpleName}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing EfficientDet output: ${e.message}")
        }

        return detections
    }

    private fun processMobileNetOutput(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            val output = outputBuffer as Array<Array<FloatArray>>

            Log.d(TAG, "Processing MobileNet output")

            // MobileNet SSD format processing
            for (i in output[0].indices) {
                val detection = output[0][i]
                if (detection.size >= 4) {
                    // Assume confidence is in a separate tensor or last element
                    val confidence = if (detection.size > 4) detection[4] else 0.5f

                    if (confidence >= confThreshold) {
                        val left = detection[0].coerceIn(0f, 1f)
                        val top = detection[1].coerceIn(0f, 1f)
                        val right = detection[2].coerceIn(0f, 1f)
                        val bottom = detection[3].coerceIn(0f, 1f)

                        if (right > left && bottom > top) {
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
            Log.e(TAG, "Error processing MobileNet output: ${e.message}")
        }

        return detections
    }

    private fun processGenericOutput(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            Log.d(TAG, "Processing generic model output")

            // Try to interpret as a generic detection format
            when (outputBuffer) {
                is Array<*> -> {
                    val output = outputBuffer as Array<*>
                    Log.d(TAG, "Generic array output with ${output.size} elements")

                    // Try different interpretations
                    if (output.isNotEmpty() && output[0] is FloatArray) {
                        val floatOutput = output as Array<FloatArray>
                        // Interpret as [num_detections, features]
                        for (i in floatOutput.indices) {
                            val row = floatOutput[i]
                            if (row.size >= 5) {
                                val confidence = row[4]
                                if (confidence >= confThreshold) {
                                    detections.add(Detection(
                                        bbox = RectF(row[0], row[1], row[2], row[3]),
                                        score = confidence,
                                        classId = 0
                                    ))
                                }
                            }
                        }
                    }
                }
                is FloatArray -> {
                    val output = outputBuffer as FloatArray
                    Log.d(TAG, "Generic float array output with ${output.size} elements")

                    // Try to parse as flattened detection results
                    val stride = 6 // Assuming [x1, y1, x2, y2, conf, class]
                    for (i in 0 until output.size step stride) {
                        if (i + stride <= output.size) {
                            val confidence = output[i + 4]
                            if (confidence >= confThreshold) {
                                detections.add(Detection(
                                    bbox = RectF(output[i], output[i + 1], output[i + 2], output[i + 3]),
                                    score = confidence,
                                    classId = output[i + 5].toInt()
                                ))
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing generic output: ${e.message}")
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

    // Interface methods for compatibility
    fun getClassLabel(classId: Int): String = classLabels.getOrElse(classId) { "Object" }

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

    fun getPerformanceInfo(): String = "Generic TFLite ($modelType) - ${getClassLabel(0)}"

    fun cleanup() {
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
    }

    fun close() = cleanup()
}

/**
 * Detection quality data class for compatibility
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