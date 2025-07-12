package com.example.atry.detector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Comprehensive diagnostic tool for simonskina.tflite model
 * Run this to identify exactly what's wrong with your model setup
 */
class ModelDiagnostic(private val context: Context) {

    companion object {
        private const val TAG = "ModelDiagnostic"
    }

    data class DiagnosticResult(
        val modelLoaded: Boolean,
        val modelSize: Long,
        val interpreterCreated: Boolean,
        val inputShape: IntArray?,
        val inputDataType: DataType?,
        val outputCount: Int,
        val outputShapes: List<IntArray>,
        val outputDataTypes: List<DataType>,
        val inferenceTest: Boolean,
        val errors: List<String>,
        val recommendations: List<String>
    )

    fun runComprehensiveDiagnostic(): DiagnosticResult {
        Log.d(TAG, "🔬 Starting comprehensive model diagnostic...")

        val errors = mutableListOf<String>()
        val recommendations = mutableListOf<String>()

        var modelLoaded = false
        var modelSize = 0L
        var interpreterCreated = false
        var inputShape: IntArray? = null
        var inputDataType: DataType? = null
        var outputCount = 0
        val outputShapes = mutableListOf<IntArray>()
        val outputDataTypes = mutableListOf<DataType>()
        var inferenceTest = false

        var interpreter: Interpreter? = null

        try {
            // Step 1: Test model loading
            Log.d(TAG, "📁 Step 1: Testing model loading...")
            val modelBuffer = try {
                loadModelFile("simonskina.tflite")
            } catch (e: Exception) {
                errors.add("Model loading failed: ${e.message}")
                Log.e(TAG, "❌ Model loading failed", e)
                null
            }

            if (modelBuffer != null) {
                modelLoaded = true
                modelSize = modelBuffer.capacity().toLong()
                Log.d(TAG, "✅ Model loaded successfully: ${modelSize} bytes")

                // Check model size reasonableness
                when {
                    modelSize < 100_000 -> {
                        errors.add("Model seems too small (${modelSize} bytes)")
                        recommendations.add("Verify model file is complete and not corrupted")
                    }
                    modelSize > 500_000_000 -> {
                        errors.add("Model seems very large (${modelSize} bytes)")
                        recommendations.add("Consider using a smaller model for mobile deployment")
                    }
                    else -> {
                        Log.d(TAG, "✅ Model size looks reasonable: ${modelSize} bytes")
                    }
                }
            } else {
                return DiagnosticResult(
                    false, 0L, false, null, null, 0, emptyList(), emptyList(),
                    false, errors, listOf("Cannot proceed without loading model")
                )
            }

            // Step 2: Test interpreter creation
            Log.d(TAG, "🏗️ Step 2: Testing interpreter creation...")
            try {
                val options = Interpreter.Options().apply {
                    setNumThreads(1) // Start with minimal configuration
                }
                interpreter = Interpreter(modelBuffer, options)
                interpreterCreated = true
                Log.d(TAG, "✅ Interpreter created successfully")
            } catch (e: Exception) {
                errors.add("Interpreter creation failed: ${e.message}")
                Log.e(TAG, "❌ Interpreter creation failed", e)

                // Try with even more basic options
                try {
                    Log.d(TAG, "🔄 Trying with minimal options...")
                    interpreter = Interpreter(modelBuffer)
                    interpreterCreated = true
                    Log.d(TAG, "✅ Interpreter created with minimal options")
                } catch (e2: Exception) {
                    errors.add("Even minimal interpreter creation failed: ${e2.message}")
                    Log.e(TAG, "❌ Minimal interpreter creation failed", e2)
                }
            }

            if (!interpreterCreated) {
                return DiagnosticResult(
                    modelLoaded, modelSize, false, null, null, 0, emptyList(), emptyList(),
                    false, errors, recommendations + "Try updating TensorFlow Lite dependencies"
                )
            }

            // Step 3: Analyze input tensor
            Log.d(TAG, "📊 Step 3: Analyzing input tensor...")
            try {
                val inputTensor = interpreter!!.getInputTensor(0)
                inputShape = inputTensor.shape()
                inputDataType = inputTensor.dataType()

                Log.d(TAG, "✅ Input analysis complete:")
                Log.d(TAG, "  Shape: ${inputShape.contentToString()}")
                Log.d(TAG, "  Data Type: $inputDataType")

                // Validate input specifications
                when {
                    inputShape.size != 4 -> {
                        errors.add("Unexpected input shape dimensions: ${inputShape.size} (expected 4)")
                        recommendations.add("Model should have 4D input [batch, height, width, channels]")
                    }
                    inputShape[0] != 1 -> {
                        errors.add("Unexpected batch size: ${inputShape[0]} (expected 1)")
                        recommendations.add("Model should have batch size of 1 for mobile inference")
                    }
                    inputShape[1] != inputShape[2] -> {
                        errors.add("Non-square input: ${inputShape[1]}×${inputShape[2]}")
                        recommendations.add("Most vision models expect square inputs")
                    }
                    inputShape[3] != 3 -> {
                        errors.add("Unexpected channels: ${inputShape[3]} (expected 3 for RGB)")
                        recommendations.add("Model should expect RGB input (3 channels)")
                    }
                    else -> {
                        Log.d(TAG, "✅ Input shape looks good: ${inputShape.contentToString()}")
                    }
                }

                // Check data type
                when (inputDataType) {
                    DataType.FLOAT32 -> Log.d(TAG, "✅ Input data type is FLOAT32 (good)")
                    DataType.UINT8 -> {
                        Log.d(TAG, "⚠️ Input data type is UINT8 (quantized)")
                        recommendations.add("UINT8 models require different preprocessing (0-255 range)")
                    }
                    else -> {
                        errors.add("Unsupported input data type: $inputDataType")
                        recommendations.add("Only FLOAT32 and UINT8 are commonly supported")
                    }
                }

            } catch (e: Exception) {
                errors.add("Input tensor analysis failed: ${e.message}")
                Log.e(TAG, "❌ Input analysis failed", e)
            }

            // Step 4: Analyze output tensors
            Log.d(TAG, "📤 Step 4: Analyzing output tensors...")
            try {
                outputCount = interpreter!!.outputTensorCount
                Log.d(TAG, "✅ Found $outputCount output tensors")

                for (i in 0 until outputCount) {
                    val outputTensor = interpreter!!.getOutputTensor(i)
                    val shape = outputTensor.shape()
                    val dataType = outputTensor.dataType()
                    val name = try { outputTensor.name() } catch (e: Exception) { "unknown" }

                    outputShapes.add(shape)
                    outputDataTypes.add(dataType)

                    Log.d(TAG, "  Output $i:")
                    Log.d(TAG, "    Name: $name")
                    Log.d(TAG, "    Shape: ${shape.contentToString()}")
                    Log.d(TAG, "    Data Type: $dataType")
                }

                // Analyze output format
                analyzeOutputFormat(outputCount, outputShapes, recommendations)

            } catch (e: Exception) {
                errors.add("Output tensor analysis failed: ${e.message}")
                Log.e(TAG, "❌ Output analysis failed", e)
            }

            // Step 5: Test inference with dummy data
            Log.d(TAG, "🧪 Step 5: Testing inference...")
            try {
                inferenceTest = testInference(interpreter!!, inputShape, inputDataType)
                if (inferenceTest) {
                    Log.d(TAG, "✅ Inference test passed")
                } else {
                    errors.add("Inference test failed")
                    Log.e(TAG, "❌ Inference test failed")
                }
            } catch (e: Exception) {
                errors.add("Inference test crashed: ${e.message}")
                Log.e(TAG, "❌ Inference test crashed", e)
            }

        } catch (e: Exception) {
            errors.add("Diagnostic crashed: ${e.message}")
            Log.e(TAG, "❌ Diagnostic crashed", e)
        } finally {
            interpreter?.close()
        }

        // Generate final recommendations
        generateFinalRecommendations(errors, recommendations, modelLoaded, interpreterCreated, inferenceTest)

        val result = DiagnosticResult(
            modelLoaded = modelLoaded,
            modelSize = modelSize,
            interpreterCreated = interpreterCreated,
            inputShape = inputShape,
            inputDataType = inputDataType,
            outputCount = outputCount,
            outputShapes = outputShapes,
            outputDataTypes = outputDataTypes,
            inferenceTest = inferenceTest,
            errors = errors,
            recommendations = recommendations
        )

        Log.d(TAG, "🏁 Diagnostic complete. Errors: ${errors.size}, Recommendations: ${recommendations.size}")
        return result
    }

    private fun loadModelFile(assetPath: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(assetPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun analyzeOutputFormat(
        outputCount: Int,
        outputShapes: List<IntArray>,
        recommendations: MutableList<String>
    ) {
        when (outputCount) {
            1 -> {
                val shape = outputShapes[0]
                Log.d(TAG, "📊 Single output detected: ${shape.contentToString()}")

                when {
                    shape.size == 3 && shape[2] >= 5 -> {
                        Log.d(TAG, "✅ Looks like object detection format [batch, detections, features]")
                        if (shape[2] == 6) {
                            recommendations.add("Detected format: [x1, y1, x2, y2, score, class]")
                        } else if (shape[2] == 5) {
                            recommendations.add("Detected format: [x, y, w, h, score] or [x1, y1, x2, y2, score]")
                        }
                    }
                    shape.size == 2 -> {
                        Log.d(TAG, "✅ Looks like classification format [batch, classes]")
                        recommendations.add("Classification model - create center bounding box for detections")
                    }
                    else -> {
                        recommendations.add("Unknown single output format - may need custom parsing")
                    }
                }
            }
            4 -> {
                Log.d(TAG, "✅ Standard object detection format (4 outputs)")
                recommendations.add("Standard format: [boxes, classes, scores, num_detections]")

                // Validate expected shapes
                if (outputShapes.size >= 4) {
                    val boxShape = outputShapes[0]
                    val scoreShape = outputShapes[2]

                    if (boxShape.size >= 3 && boxShape[2] == 4) {
                        Log.d(TAG, "✅ Box output looks correct: ${boxShape.contentToString()}")
                    } else {
                        recommendations.add("⚠️ Box output shape unexpected: ${boxShape.contentToString()}")
                    }

                    if (scoreShape.size >= 2) {
                        Log.d(TAG, "✅ Score output looks correct: ${scoreShape.contentToString()}")
                    } else {
                        recommendations.add("⚠️ Score output shape unexpected: ${scoreShape.contentToString()}")
                    }
                }
            }
            else -> {
                recommendations.add("Unusual output count: $outputCount - may need custom handling")
            }
        }
    }

    private fun testInference(
        interpreter: Interpreter,
        inputShape: IntArray?,
        inputDataType: DataType?
    ): Boolean {
        if (inputShape == null || inputDataType == null) {
            Log.e(TAG, "Cannot test inference without input specifications")
            return false
        }

        return try {
            Log.d(TAG, "🧪 Creating test input...")

            // Create input buffer
            val inputSize = inputShape.fold(1) { acc, dim -> acc * dim }
            val inputBuffer = when (inputDataType) {
                DataType.FLOAT32 -> {
                    ByteBuffer.allocateDirect(inputSize * 4).apply {
                        order(ByteOrder.nativeOrder())
                        // Fill with dummy data
                        repeat(inputSize) { putFloat(0.5f) }
                        rewind()
                    }
                }
                DataType.UINT8 -> {
                    ByteBuffer.allocateDirect(inputSize).apply {
                        order(ByteOrder.nativeOrder())
                        // Fill with dummy data
                        repeat(inputSize) { put(128.toByte()) }
                        rewind()
                    }
                }
                else -> {
                    Log.e(TAG, "Unsupported input data type for testing: $inputDataType")
                    return false
                }
            }

            Log.d(TAG, "🧪 Creating output arrays...")

            // Create output arrays based on output count
            val outputArrays = when (interpreter.outputTensorCount) {
                1 -> {
                    val outputShape = interpreter.getOutputTensor(0).shape()
                    when (outputShape.size) {
                        3 -> arrayOf(Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } })
                        2 -> arrayOf(Array(outputShape[0]) { FloatArray(outputShape[1]) })
                        else -> arrayOf(Array(1) { Array(100) { FloatArray(6) } }) // fallback
                    }
                }
                4 -> {
                    val boxShape = interpreter.getOutputTensor(0).shape()
                    val classShape = interpreter.getOutputTensor(1).shape()
                    val scoreShape = interpreter.getOutputTensor(2).shape()
                    val numShape = interpreter.getOutputTensor(3).shape()

                    arrayOf(
                        Array(boxShape[0]) { Array(boxShape[1]) { FloatArray(boxShape[2]) } },
                        Array(classShape[0]) { Array(classShape[1]) { FloatArray(classShape[2]) } },
                        Array(scoreShape[0]) { FloatArray(scoreShape[1]) },
                        FloatArray(numShape[0])
                    )
                }
                else -> {
                    // Generic fallback
                    Array(interpreter.outputTensorCount) { i ->
                        val shape = interpreter.getOutputTensor(i).shape()
                        when (shape.size) {
                            1 -> FloatArray(shape[0])
                            2 -> Array(shape[0]) { FloatArray(shape[1]) }
                            3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
                            else -> Array(1) { FloatArray(10) } // fallback
                        }
                    }
                }
            }

            Log.d(TAG, "🧪 Running inference...")

            // Run inference
            when (interpreter.outputTensorCount) {
                1 -> {
                    interpreter.run(inputBuffer, outputArrays[0])
                }
                else -> {
                    val outputMap = outputArrays.mapIndexed { index, array -> index to array }.toMap()
                    interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
                }
            }

            Log.d(TAG, "✅ Inference completed successfully")

            // Basic output validation
            when (interpreter.outputTensorCount) {
                1 -> {
                    val output = outputArrays[0]
                    Log.d(TAG, "📊 Single output validation:")
                    when (output) {
                        is Array<*> -> {
                            if (output.isNotEmpty()) {
                                Log.d(TAG, "  Output array size: ${output.size}")
                                when (val firstElement = output[0]) {
                                    is Array<*> -> Log.d(TAG, "  First element size: ${firstElement.size}")
                                    is FloatArray -> Log.d(TAG, "  First element size: ${firstElement.size}")
                                }
                            }
                        }
                    }
                }
                4 -> {
                    Log.d(TAG, "📊 Multi-output validation:")
                    Log.d(TAG, "  Boxes: ${(outputArrays[0] as Array<Array<FloatArray>>)[0].size} detections")
                    Log.d(TAG, "  Scores: ${(outputArrays[2] as Array<FloatArray>)[0].size} scores")
                    Log.d(TAG, "  Num detections: ${(outputArrays[3] as FloatArray)[0]}")
                }
            }

            true

        } catch (e: Exception) {
            Log.e(TAG, "❌ Inference test failed: ${e.message}", e)
            false
        }
    }

    private fun generateFinalRecommendations(
        errors: List<String>,
        recommendations: MutableList<String>,
        modelLoaded: Boolean,
        interpreterCreated: Boolean,
        inferenceTest: Boolean
    ) {
        when {
            !modelLoaded -> {
                recommendations.add("🔧 CRITICAL: Ensure simonskina.tflite is in src/main/assets/")
                recommendations.add("🔧 Check file permissions and asset folder structure")
            }
            !interpreterCreated -> {
                recommendations.add("🔧 CRITICAL: Update TensorFlow Lite dependencies")
                recommendations.add("🔧 Try reducing interpreter options (threads, optimizations)")
                recommendations.add("🔧 Check device compatibility with TensorFlow Lite")
            }
            !inferenceTest -> {
                recommendations.add("🔧 Model loads but inference fails")
                recommendations.add("🔧 Check input/output tensor compatibility")
                recommendations.add("🔧 Verify model format matches expected structure")
            }
            errors.isEmpty() -> {
                recommendations.add("✅ Model appears to be working correctly!")
                recommendations.add("🎯 Focus on optimizing detection thresholds and post-processing")
            }
            else -> {
                recommendations.add("🔧 Fix the identified errors above")
                recommendations.add("🔧 Test with different confidence thresholds")
            }
        }

        // Always add these general recommendations
        recommendations.add("📱 Test on physical device (not just emulator)")
        recommendations.add("🐛 Enable verbose logging to see detailed error messages")
        recommendations.add("🔄 Try different input image sizes and formats")
    }

    /**
     * Generate a human-readable diagnostic report
     */
    fun generateReport(result: DiagnosticResult): String {
        return buildString {
            appendLine("🔬 MODEL DIAGNOSTIC REPORT")
            appendLine("=".repeat(50))
            appendLine()

            appendLine("📁 MODEL LOADING:")
            appendLine("  Status: ${if (result.modelLoaded) "✅ SUCCESS" else "❌ FAILED"}")
            appendLine("  Size: ${result.modelSize} bytes")
            appendLine()

            appendLine("🏗️ INTERPRETER CREATION:")
            appendLine("  Status: ${if (result.interpreterCreated) "✅ SUCCESS" else "❌ FAILED"}")
            appendLine()

            result.inputShape?.let { shape ->
                appendLine("📊 INPUT SPECIFICATIONS:")
                appendLine("  Shape: ${shape.contentToString()}")
                appendLine("  Data Type: ${result.inputDataType}")
                appendLine("  Expected: [1, height, width, 3] FLOAT32")
                appendLine()
            }

            appendLine("📤 OUTPUT SPECIFICATIONS:")
            appendLine("  Output Count: ${result.outputCount}")
            result.outputShapes.forEachIndexed { index, shape ->
                val dataType = if (index < result.outputDataTypes.size) result.outputDataTypes[index] else "Unknown"
                appendLine("  Output $index: ${shape.contentToString()} $dataType")
            }
            appendLine()

            appendLine("🧪 INFERENCE TEST:")
            appendLine("  Status: ${if (result.inferenceTest) "✅ SUCCESS" else "❌ FAILED"}")
            appendLine()

            if (result.errors.isNotEmpty()) {
                appendLine("❌ ERRORS FOUND:")
                result.errors.forEach { error ->
                    appendLine("  • $error")
                }
                appendLine()
            }

            if (result.recommendations.isNotEmpty()) {
                appendLine("💡 RECOMMENDATIONS:")
                result.recommendations.forEach { recommendation ->
                    appendLine("  • $recommendation")
                }
                appendLine()
            }

            appendLine("🎯 NEXT STEPS:")
            when {
                !result.modelLoaded -> appendLine("  1. Fix model loading issue first")
                !result.interpreterCreated -> appendLine("  1. Fix interpreter creation issue")
                !result.inferenceTest -> appendLine("  1. Fix inference compatibility issues")
                result.errors.isEmpty() -> {
                    appendLine("  1. Model is working! Focus on:")
                    appendLine("     - Adjusting confidence thresholds")
                    appendLine("     - Optimizing preprocessing")
                    appendLine("     - Tuning post-processing")
                }
                else -> appendLine("  1. Address the errors listed above")
            }

            appendLine()
            appendLine("=".repeat(50))
            appendLine("Generated by ModelDiagnostic")
        }
    }
}