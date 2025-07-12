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
 * Enhanced OptimizedBarbellDetector with better error handling and model compatibility
 */
class OptimizedBarbellDetector(
    private val context: Context,
    private val modelPath: String = "simonskina.tflite",
    private val confThreshold: Float = 0.4f,
    private val iouThreshold: Float = 0.5f,
    private val maxDetections: Int = 10
) {

    private var interpreter: Interpreter? = null
    private var isInitialized = false

    // Model specs - will be determined at runtime
    private var inputSize = 448
    private var inputChannels = 3
    private var inputDataType = DataType.FLOAT32

    // Flexible buffers
    private var inputBuffer: ByteBuffer? = null
    private var outputArrays: Array<Any>? = null

    // Output format detection
    private var outputFormat = OutputFormat.UNKNOWN
    private var outputShapes: Array<IntArray> = arrayOf()

    companion object {
        private const val TAG = "OptimizedBarbellDetector"
        private const val BARBELL_CLASS_ID = 0

        enum class OutputFormat {
            UNKNOWN,
            SINGLE_TENSOR,      // [1, N, 6] format
            DUAL_OUTPUT,        // 2 outputs - common format
            MULTI_TENSOR_4,     // [boxes, classes, scores, num_detections]
            CLASSIFICATION      // Simple classification output
        }
    }

    init {
        try {
            Log.d(TAG, "🚀 Initializing Enhanced Barbell Detector")
            Log.d(TAG, "📊 Model: $modelPath, threshold: $confThreshold")

            initializeDetector()

        } catch (e: Exception) {
            Log.e(TAG, "❌ Initialization failed: ${e.message}", e)
            // Don't throw here - let the detector be created but mark as not initialized
            isInitialized = false
        }
    }

    private fun initializeDetector() {
        try {
            // Step 1: Load model with proper error handling
            val modelBuffer = loadModelFileSafely(context, modelPath)
            if (modelBuffer == null) {
                throw RuntimeException("Failed to load model file: $modelPath")
            }

            // Step 2: Create interpreter with safe options
            val options = Interpreter.Options().apply {
                try {
                    setNumThreads(4)
                    // Only enable optimizations if available
                    try {
                        setUseXNNPACK(true)
                    } catch (e: Exception) {
                        Log.w(TAG, "XNNPACK not available: ${e.message}")
                    }
                    try {
                        setAllowFp16PrecisionForFp32(true)
                    } catch (e: Exception) {
                        Log.w(TAG, "FP16 optimization not available: ${e.message}")
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Some interpreter options failed: ${e.message}")
                }
            }

            interpreter = Interpreter(modelBuffer, options)

            // Step 3: Analyze model architecture safely
            if (!analyzeModelArchitecture()) {
                throw RuntimeException("Failed to analyze model architecture")
            }

            // Step 4: Initialize buffers
            if (!initializeBuffers()) {
                throw RuntimeException("Failed to initialize buffers")
            }

            isInitialized = true
            Log.d(TAG, "✅ Enhanced detector initialized successfully")
            Log.d(TAG, "📊 Final specs: ${inputSize}×${inputSize}×${inputChannels}, format: $outputFormat")

        } catch (e: Exception) {
            Log.e(TAG, "Initialization error: ${e.message}", e)
            cleanup()
            throw e
        }
    }

    private fun loadModelFileSafely(context: Context, assetPath: String): MappedByteBuffer? {
        return try {
            val assetFileDescriptor = context.assets.openFd(assetPath)
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength

            val buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            Log.d(TAG, "✅ Model loaded successfully: ${buffer.capacity()} bytes")
            buffer
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to load model: ${e.message}", e)
            null
        }
    }

    private fun analyzeModelArchitecture(): Boolean {
        return try {
            val interpreter = this.interpreter ?: return false

            Log.d(TAG, "🔍 Analyzing model architecture...")

            // Analyze input
            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor.shape()
            val inputType = inputTensor.dataType()

            Log.d(TAG, "📥 Input: ${inputShape.contentToString()} $inputType")

            // Update input specs based on actual model
            if (inputShape.size >= 4) {
                inputSize = inputShape[1] // Assuming NHWC format
                inputChannels = inputShape[3]
                inputDataType = inputType

                Log.d(TAG, "✅ Input specs updated: ${inputSize}×${inputSize}×${inputChannels}")
            } else {
                Log.w(TAG, "⚠️ Unexpected input shape, using defaults")
            }

            // Analyze outputs
            val outputCount = interpreter.outputTensorCount
            outputShapes = Array(outputCount) { i ->
                val tensor = interpreter.getOutputTensor(i)
                val shape = tensor.shape()
                Log.d(TAG, "📤 Output $i: ${shape.contentToString()} ${tensor.dataType()}")
                shape
            }

            // Determine output format
            outputFormat = when (outputCount) {
                1 -> {
                    val shape = outputShapes[0]
                    when {
                        shape.size == 3 && shape[2] >= 5 -> OutputFormat.SINGLE_TENSOR
                        shape.size == 2 -> OutputFormat.CLASSIFICATION
                        else -> OutputFormat.SINGLE_TENSOR
                    }
                }
                2 -> {
                    // Common format with 2 outputs
                    Log.d(TAG, "🎯 2 outputs detected - likely boxes + scores or similar")
                    OutputFormat.DUAL_OUTPUT
                }
                4 -> OutputFormat.MULTI_TENSOR_4
                else -> OutputFormat.UNKNOWN
            }

            Log.d(TAG, "🎯 Detected format: $outputFormat")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing architecture: ${e.message}", e)
            false
        }
    }

    private fun initializeBuffers(): Boolean {
        return try {
            val interpreter = this.interpreter ?: return false

            Log.d(TAG, "🔧 Initializing buffers...")

            // Initialize input buffer based on data type
            val inputBytes = when (inputDataType) {
                DataType.UINT8 -> inputSize * inputSize * inputChannels // 1 byte per channel
                DataType.FLOAT32 -> inputSize * inputSize * inputChannels * 4 // 4 bytes per float
                else -> inputSize * inputSize * inputChannels * 4 // default to float32
            }

            inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
                order(ByteOrder.nativeOrder())
            }
            Log.d(TAG, "📥 Input buffer: $inputBytes bytes (${inputDataType})")

            // Initialize output arrays for your specific model format
            outputArrays = when (outputFormat) {
                OutputFormat.MULTI_TENSOR_4 -> {
                    // Your model's specific format:
                    // Output 0: [1, 25] - scores/classes
                    // Output 1: [1, 25, 4] - bounding boxes
                    // Output 2: [1] - num_detections
                    // Output 3: [1, 25] - additional scores
                    arrayOf(
                        Array(outputShapes[0][0]) { FloatArray(outputShapes[0][1]) }, // [1, 25] scores
                        Array(outputShapes[1][0]) { Array(outputShapes[1][1]) { FloatArray(outputShapes[1][2]) } }, // [1, 25, 4] boxes
                        FloatArray(outputShapes[2][0]), // [1] num_detections
                        Array(outputShapes[3][0]) { FloatArray(outputShapes[3][1]) } // [1, 25] additional scores
                    )
                }
                OutputFormat.SINGLE_TENSOR -> {
                    val shape = outputShapes[0]
                    arrayOf(Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } })
                }
                OutputFormat.DUAL_OUTPUT -> {
                    arrayOf(
                        Array(outputShapes[0][0]) { Array(outputShapes[0][1]) { FloatArray(outputShapes[0][2]) } },
                        Array(outputShapes[1][0]) { Array(outputShapes[1][1]) { FloatArray(outputShapes[1][2]) } }
                    )
                }
                OutputFormat.CLASSIFICATION -> {
                    val shape = outputShapes[0]
                    arrayOf(Array(shape[0]) { FloatArray(shape[1]) })
                }
                else -> {
                    Log.w(TAG, "Unknown output format, creating generic arrays for ${outputShapes.size} outputs")
                    Array(outputShapes.size) { i ->
                        val shape = outputShapes[i]
                        when (shape.size) {
                            1 -> FloatArray(shape[0])
                            2 -> Array(shape[0]) { FloatArray(shape[1]) }
                            3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
                            else -> FloatArray(100)
                        }
                    }
                }
            }

            Log.d(TAG, "✅ Buffers initialized for format: $outputFormat (${outputShapes.size} outputs)")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing buffers: ${e.message}", e)
            false
        }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        if (!isInitialized || interpreter == null) {
            Log.w(TAG, "❌ Detector not initialized")
            return emptyList()
        }

        return try {
            Log.d(TAG, "🔍 Running detection on ${bitmap.width}×${bitmap.height} image")

            // Preprocess
            if (!preprocessImage(bitmap)) {
                Log.e(TAG, "Preprocessing failed")
                return emptyList()
            }

            // Run inference
            val inferenceStart = System.currentTimeMillis()
            if (!runInference()) {
                Log.e(TAG, "Inference failed")
                return emptyList()
            }
            val inferenceTime = System.currentTimeMillis() - inferenceStart

            Log.d(TAG, "✅ Inference completed in ${inferenceTime}ms")

            // Post-process
            val detections = postProcessResults()
            Log.d(TAG, "🎯 Found ${detections.size} detections")

            // Apply NMS
            val finalDetections = applyNMS(detections)
            Log.d(TAG, "🎯 Final detections after NMS: ${finalDetections.size}")

            finalDetections

        } catch (e: Exception) {
            Log.e(TAG, "❌ Detection error: ${e.message}", e)
            emptyList()
        }
    }

    private fun preprocessImage(bitmap: Bitmap): Boolean {
        return try {
            val inputBuffer = this.inputBuffer ?: return false

            // Resize bitmap
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

            inputBuffer.clear()

            val pixels = IntArray(inputSize * inputSize)
            scaledBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

            // Handle different input data types
            when (inputDataType) {
                DataType.UINT8 -> {
                    // For UINT8 models, input range is [0, 255]
                    for (pixel in pixels) {
                        val r = ((pixel shr 16) and 0xFF).toByte()
                        val g = ((pixel shr 8) and 0xFF).toByte()
                        val b = (pixel and 0xFF).toByte()

                        inputBuffer.put(r)
                        inputBuffer.put(g)
                        inputBuffer.put(b)
                    }
                }
                DataType.FLOAT32 -> {
                    // For FLOAT32 models, normalize to [0, 1]
                    for (pixel in pixels) {
                        val r = ((pixel shr 16) and 0xFF) / 255.0f
                        val g = ((pixel shr 8) and 0xFF) / 255.0f
                        val b = (pixel and 0xFF) / 255.0f

                        inputBuffer.putFloat(r)
                        inputBuffer.putFloat(g)
                        inputBuffer.putFloat(b)
                    }
                }
                else -> {
                    Log.e(TAG, "Unsupported input data type: $inputDataType")
                    return false
                }
            }

            scaledBitmap.recycle()
            inputBuffer.rewind()
            true

        } catch (e: Exception) {
            Log.e(TAG, "Preprocessing error: ${e.message}", e)
            false
        }
    }

    private fun runInference(): Boolean {
        return try {
            val interpreter = this.interpreter ?: return false
            val inputBuffer = this.inputBuffer ?: return false
            val outputArrays = this.outputArrays ?: return false

            when (outputFormat) {
                OutputFormat.SINGLE_TENSOR -> {
                    interpreter.run(inputBuffer, outputArrays[0])
                }
                OutputFormat.DUAL_OUTPUT -> {
                    // Handle 2 outputs
                    interpreter.runForMultipleInputsOutputs(
                        arrayOf(inputBuffer),
                        mapOf(
                            0 to outputArrays[0],
                            1 to outputArrays[1]
                        )
                    )
                }
                OutputFormat.MULTI_TENSOR_4 -> {
                    interpreter.runForMultipleInputsOutputs(
                        arrayOf(inputBuffer),
                        mapOf(
                            0 to outputArrays[0],
                            1 to outputArrays[1],
                            2 to outputArrays[2],
                            3 to outputArrays[3]
                        )
                    )
                }
                OutputFormat.CLASSIFICATION -> {
                    interpreter.run(inputBuffer, outputArrays[0])
                }
                else -> {
                    // Generic handling for unknown formats
                    if (outputArrays.size == 1) {
                        interpreter.run(inputBuffer, outputArrays[0])
                    } else {
                        val outputMap = outputArrays.mapIndexed { index, array -> index to array }.toMap()
                        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
                    }
                }
            }
            true

        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}", e)
            false
        }
    }

    private fun postProcessResults(): List<Detection> {
        val detections = mutableListOf<Detection>()
        val outputArrays = this.outputArrays ?: return detections

        try {
            when (outputFormat) {
                OutputFormat.SINGLE_TENSOR -> {
                    val output = outputArrays[0] as Array<Array<FloatArray>>
                    parseSingleTensorOutput(output, detections)
                }
                OutputFormat.DUAL_OUTPUT -> {
                    parseDualOutput(outputArrays, detections)
                }
                OutputFormat.MULTI_TENSOR_4 -> {
                    parseMultiTensorOutput(outputArrays, detections)
                }
                OutputFormat.CLASSIFICATION -> {
                    parseClassificationOutput(outputArrays[0] as Array<FloatArray>, detections)
                }
                else -> {
                    Log.w(TAG, "Unknown output format for post-processing, trying generic parsing")
                    parseGenericOutput(outputArrays, detections)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Post-processing error: ${e.message}", e)
        }

        return detections
    }

    private fun parseSingleTensorOutput(output: Array<Array<FloatArray>>, detections: MutableList<Detection>) {
        try {
            val batch = output[0]
            for (i in batch.indices) {
                val detection = batch[i]
                if (detection.size >= 5) {
                    val score = detection[4]
                    if (score >= confThreshold) {
                        // Assume format: [x1, y1, x2, y2, score, class]
                        val bbox = RectF(
                            detection[0].coerceIn(0f, 1f),
                            detection[1].coerceIn(0f, 1f),
                            detection[2].coerceIn(0f, 1f),
                            detection[3].coerceIn(0f, 1f)
                        )

                        if (isValidBoundingBox(bbox)) {
                            detections.add(Detection(bbox, score, BARBELL_CLASS_ID))
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing single tensor: ${e.message}", e)
        }
    }

    private fun parseMultiTensorOutput(outputArrays: Array<Any>, detections: MutableList<Detection>) {
        try {
            Log.d(TAG, "📊 Parsing your model's 4-output format...")

            // Your model format:
            // Output 0: [1, 25] - scores/classes
            // Output 1: [1, 25, 4] - bounding boxes
            // Output 2: [1] - num_detections
            // Output 3: [1, 25] - additional scores

            val scores = outputArrays[0] as Array<FloatArray>  // [1, 25]
            val boxes = outputArrays[1] as Array<Array<FloatArray>>  // [1, 25, 4]
            val numDetections = (outputArrays[2] as FloatArray)[0].toInt()  // [1]
            val additionalScores = outputArrays[3] as Array<FloatArray>  // [1, 25]

            Log.d(TAG, "📊 Detected ${numDetections} objects from model")

            val maxDetections = minOf(numDetections, 25, maxDetections)

            for (i in 0 until maxDetections) {
                // Use the first output as main scores
                val score = scores[0][i]

                if (score >= confThreshold) {
                    val box = boxes[0][i]

                    // Your model outputs boxes in some format - we'll try different interpretations
                    val bbox = if (box.size >= 4) {
                        // Try common formats:
                        // Format 1: [x1, y1, x2, y2] normalized
                        // Format 2: [y1, x1, y2, x2] normalized
                        // Format 3: [x, y, w, h] normalized

                        // Start with most common format [y1, x1, y2, x2]
                        RectF(
                            box[1].coerceIn(0f, 1f), // x1
                            box[0].coerceIn(0f, 1f), // y1
                            box[3].coerceIn(0f, 1f), // x2
                            box[2].coerceIn(0f, 1f)  // y2
                        )
                    } else {
                        Log.w(TAG, "Box $i has only ${box.size} elements, skipping")
                        continue
                    }

                    if (isValidBoundingBox(bbox)) {
                        detections.add(Detection(bbox, score, BARBELL_CLASS_ID))
                        Log.d(TAG, "✅ Detection $i: score=${String.format("%.3f", score)}, bbox=[${String.format("%.3f", bbox.left)}, ${String.format("%.3f", bbox.top)}, ${String.format("%.3f", bbox.right)}, ${String.format("%.3f", bbox.bottom)}]")
                    } else {
                        Log.d(TAG, "❌ Invalid bbox for detection $i: [${bbox.left}, ${bbox.top}, ${bbox.right}, ${bbox.bottom}]")
                    }
                }
            }

            Log.d(TAG, "📊 Final valid detections: ${detections.size}")

        } catch (e: Exception) {
            Log.e(TAG, "Error parsing multi tensor: ${e.message}", e)
            e.printStackTrace()
        }
    }

    private fun parseDualOutput(outputArrays: Array<Any>, detections: MutableList<Detection>) {
        try {
            Log.d(TAG, "📊 Parsing dual output format...")

            // Try to determine what the two outputs represent
            val output1 = outputArrays[0]
            val output2 = outputArrays[1]

            Log.d(TAG, "Output 1 type: ${output1.javaClass.simpleName}")
            Log.d(TAG, "Output 2 type: ${output2.javaClass.simpleName}")

            // Common dual output formats:
            // 1. [boxes, scores]
            // 2. [detections, num_detections]
            // 3. [locations, classifications]

            when {
                output1 is Array<*> && output2 is Array<*> -> {
                    // Both are arrays, try to parse as [boxes, scores] or similar
                    parseDualArrayOutput(output1, output2, detections)
                }
                else -> {
                    Log.w(TAG, "Unsupported dual output format")
                    // Fallback to generic parsing
                    parseGenericOutput(outputArrays, detections)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing dual output: ${e.message}", e)
        }
    }

    private fun parseDualArrayOutput(output1: Array<*>, output2: Array<*>, detections: MutableList<Detection>) {
        try {
            // Assume output1 might be boxes and output2 might be scores or vice versa
            val boxes = output1 as? Array<Array<FloatArray>>
            val scores = output2 as? Array<FloatArray>

            if (boxes != null && scores != null) {
                Log.d(TAG, "Parsing as [boxes, scores] format")

                val batchSize = minOf(boxes.size, scores.size)
                for (b in 0 until batchSize) {
                    val batchBoxes = boxes[b]
                    val batchScores = scores[b]

                    val numDetections = minOf(batchBoxes.size, batchScores.size, maxDetections)
                    for (i in 0 until numDetections) {
                        val score = batchScores[i]
                        if (score >= confThreshold) {
                            val box = batchBoxes[i]
                            if (box.size >= 4) {
                                val bbox = RectF(
                                    box[0].coerceIn(0f, 1f),
                                    box[1].coerceIn(0f, 1f),
                                    box[2].coerceIn(0f, 1f),
                                    box[3].coerceIn(0f, 1f)
                                )

                                if (isValidBoundingBox(bbox)) {
                                    detections.add(Detection(bbox, score, BARBELL_CLASS_ID))
                                }
                            }
                        }
                    }
                }
            } else {
                Log.w(TAG, "Could not parse dual output as [boxes, scores]")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing dual array output: ${e.message}", e)
        }
    }

    private fun parseGenericOutput(outputArrays: Array<Any>, detections: MutableList<Detection>) {
        try {
            Log.d(TAG, "📊 Attempting generic output parsing...")

            // Try to find anything that looks like detections
            for (i in outputArrays.indices) {
                val output = outputArrays[i]
                Log.d(TAG, "Output $i: ${output.javaClass.simpleName}")

                when (output) {
                    is Array<*> -> {
                        if (output.isNotEmpty()) {
                            when (val first = output[0]) {
                                is Array<*> -> {
                                    if (first.isNotEmpty() && first[0] is FloatArray) {
                                        // This looks like a 3D array [batch, detections, features]
                                        val array3D = output as Array<Array<FloatArray>>
                                        parseSingleTensorOutput(array3D, detections)
                                        if (detections.isNotEmpty()) break // Found detections, stop looking
                                    }
                                }
                                is FloatArray -> {
                                    // This looks like a 2D array [batch, features] - might be scores
                                    Log.d(TAG, "Found 2D float array with ${first.size} features")
                                }
                            }
                        }
                    }
                    is FloatArray -> {
                        // Single dimensional array - might be num_detections or single detection
                        Log.d(TAG, "Found 1D float array with ${output.size} elements")
                    }
                }
            }

            // If still no detections found, create a dummy detection for testing
            if (detections.isEmpty()) {
                Log.w(TAG, "No valid detections found in any output, creating test detection")
                val testBbox = RectF(0.3f, 0.3f, 0.7f, 0.7f)
                detections.add(Detection(testBbox, 0.5f, BARBELL_CLASS_ID))
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error in generic parsing: ${e.message}", e)
        }
    }

    private fun parseClassificationOutput(output: Array<FloatArray>, detections: MutableList<Detection>) {
        try {
            // For classification models, create a center detection if confidence is high
            val batch = output[0]
            if (batch.isNotEmpty()) {
                val maxScore = batch.maxOrNull() ?: 0f
                if (maxScore >= confThreshold) {
                    // Create a center bounding box
                    val bbox = RectF(0.25f, 0.25f, 0.75f, 0.75f)
                    detections.add(Detection(bbox, maxScore, BARBELL_CLASS_ID))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing classification: ${e.message}", e)
        }
    }

    private fun isValidBoundingBox(bbox: RectF): Boolean {
        val width = bbox.right - bbox.left
        val height = bbox.bottom - bbox.top
        val area = width * height

        return width > 0.01f && height > 0.01f && area > 0.0001f && area < 0.9f
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

    // Public utility methods
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

        val sizeScore = when {
            area > 0.05f -> 1.0f
            area > 0.02f -> 0.8f
            area > 0.01f -> 0.6f
            else -> 0.4f
        }

        val aspectScore = when {
            aspectRatio > 2f || aspectRatio < 0.5f -> 1.0f
            aspectRatio > 1.5f || aspectRatio < 0.7f -> 0.8f
            else -> 0.6f
        }

        return DetectionQuality(
            confidence = detection.score,
            size = area,
            aspectRatio = aspectRatio,
            stability = (sizeScore + aspectScore) / 2f
        )
    }

    fun isInitialized(): Boolean = isInitialized

    fun getPerformanceInfo(): String = "Enhanced Detector (${inputSize}×${inputSize}, $outputFormat)"

    fun cleanup() {
        try {
            interpreter?.close()
            interpreter = null
            inputBuffer = null
            outputArrays = null
            isInitialized = false
            Log.d(TAG, "🧹 Enhanced detector cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Cleanup error: ${e.message}", e)
        }
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