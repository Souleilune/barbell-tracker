package com.example.atry.detector

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Helper class to verify your simonskina.tflite model specifications
 * Run this BEFORE using the OptimizedBarbellDetector
 */
class ModelVerifier(private val context: Context) {

    companion object {
        private const val TAG = "ModelVerifier"
    }

    fun verifySimonskinaModel(): ModelSpecs? {
        return try {
            Log.d(TAG, "🔍 Analyzing simonskina.tflite model...")

            val modelBuffer = loadModelFile(context, "simonskina.tflite")
            val interpreter = Interpreter(modelBuffer)

            val specs = analyzeModelArchitecture(interpreter)
            interpreter.close()

            Log.d(TAG, "✅ Model analysis complete")
            specs

        } catch (e: Exception) {
            Log.e(TAG, "❌ Model analysis failed: ${e.message}", e)
            null
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

    private fun analyzeModelArchitecture(interpreter: Interpreter): ModelSpecs {
        Log.d(TAG, "📊 Analyzing model architecture...")

        // Input analysis
        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        val inputDataType = inputTensor.dataType()
        val inputBytes = try {
            inputTensor.numBytes()
        } catch (e: Exception) {
            inputShape.fold(1) { acc, dim -> acc * dim } * 4
        }

        Log.d(TAG, "📥 INPUT ANALYSIS:")
        Log.d(TAG, "  Shape: ${inputShape.contentToString()}")
        Log.d(TAG, "  Data Type: $inputDataType")
        Log.d(TAG, "  Size: $inputBytes bytes")
        Log.d(TAG, "  Expected for 448×448×3 Float32: ${448 * 448 * 3 * 4} bytes")

        // Output analysis
        Log.d(TAG, "📤 OUTPUT ANALYSIS:")
        Log.d(TAG, "  Number of outputs: ${interpreter.outputTensorCount}")

        val outputSpecs = mutableListOf<OutputTensorSpec>()
        for (i in 0 until interpreter.outputTensorCount) {
            val outputTensor = interpreter.getOutputTensor(i)
            val shape = outputTensor.shape()
            val dataType = outputTensor.dataType()
            val name = outputTensor.name()

            Log.d(TAG, "  Output $i:")
            Log.d(TAG, "    Name: $name")
            Log.d(TAG, "    Shape: ${shape.contentToString()}")
            Log.d(TAG, "    Data Type: $dataType")

            outputSpecs.add(OutputTensorSpec(i, name, shape, dataType.toString()))
        }

        // Determine model format
        val modelFormat = determineModelFormat(outputSpecs)
        Log.d(TAG, "🎯 Detected model format: $modelFormat")

        // Provide recommendations
        val recommendations = generateRecommendations(inputShape, outputSpecs, modelFormat)
        recommendations.forEach { Log.d(TAG, "💡 $it") }

        return ModelSpecs(
            inputShape = inputShape,
            inputDataType = inputDataType.toString(),
            inputBytes = inputBytes,
            outputSpecs = outputSpecs,
            modelFormat = modelFormat,
            recommendations = recommendations
        )
    }

    private fun determineModelFormat(outputSpecs: List<OutputTensorSpec>): String {
        return when (outputSpecs.size) {
            1 -> {
                val shape = outputSpecs[0].shape
                when {
                    shape.size == 3 && shape[2] >= 85 -> "YOLO-style (single tensor with classes)"
                    shape.size == 3 && shape[2] == 6 -> "Detection format [x1,y1,x2,y2,score,class]"
                    shape.size == 3 && shape[2] == 5 -> "Detection format [x,y,w,h,score]"
                    shape.size == 2 -> "Classification or simple detection"
                    else -> "Unknown single output format"
                }
            }
            2 -> "Dual output (possibly boxes + scores)"
            4 -> "Standard EfficientDet (boxes, classes, scores, num_detections)"
            else -> "Unknown multi-output format"
        }
    }

    private fun generateRecommendations(
        inputShape: IntArray,
        outputSpecs: List<OutputTensorSpec>,
        modelFormat: String
    ): List<String> {
        val recommendations = mutableListOf<String>()

        // Input recommendations
        if (inputShape.contentEquals(intArrayOf(1, 448, 448, 3))) {
            recommendations.add("✅ Input shape matches OptimizedBarbellDetector expectations")
        } else {
            recommendations.add("⚠️ Update INPUT_SIZE to ${inputShape[1]} in OptimizedBarbellDetector")
        }

        // Output format recommendations
        when (modelFormat) {
            "Standard EfficientDet (boxes, classes, scores, num_detections)" -> {
                recommendations.add("✅ Use multi-output inference path in OptimizedBarbellDetector")
            }
            "Detection format [x1,y1,x2,y2,score,class]" -> {
                recommendations.add("✅ Use single output parsing in OptimizedBarbellDetector")
                recommendations.add("⚠️ Verify coordinate format (normalized vs absolute)")
            }
            else -> {
                recommendations.add("❌ May need custom post-processing for this model format")
                recommendations.add("🔧 Consider updating OptimizedBarbellDetector for your specific format")
            }
        }

        // Threshold recommendations
        recommendations.add("🎯 Test confidence thresholds: 0.3, 0.4, 0.5 for optimal results")
        recommendations.add("📏 Verify coordinate ranges: check if outputs are [0,1] or [0,input_size]")

        return recommendations
    }

    data class ModelSpecs(
        val inputShape: IntArray,
        val inputDataType: String,
        val inputBytes: Int,
        val outputSpecs: List<OutputTensorSpec>,
        val modelFormat: String,
        val recommendations: List<String>
    )

    data class OutputTensorSpec(
        val index: Int,
        val name: String,
        val shape: IntArray,
        val dataType: String
    )
}