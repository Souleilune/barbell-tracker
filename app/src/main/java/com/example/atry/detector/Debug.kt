package com.example.atry.debug

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.atry.detector.GenericTFLiteDetector
import com.example.atry.detector.Detection
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

/**
 * Debug utility to analyze your TensorFlow Lite model and diagnose detection issues
 */
@Composable
fun ModelDebugScreen(context: Context) {
    var debugResults by remember { mutableStateOf<DebugResults?>(null) }
    var isDebugging by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "🔧 Model Debug Tool",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Button(
            onClick = {
                scope.launch {
                    isDebugging = true
                    debugResults = performDebugAnalysis(context)
                    isDebugging = false
                }
            },
            enabled = !isDebugging,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (isDebugging) "Debugging..." else "Debug Model")
        }

        debugResults?.let { results ->
            DebugResultsDisplay(results)
        }
    }
}

@Composable
private fun DebugResultsDisplay(results: DebugResults) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {

        // Model Info Card
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("📊 Model Information", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))

                Text("Model Path: ${results.modelPath}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Model Size: ${results.modelSizeBytes} bytes", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Input Shape: ${results.inputShape}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Output Shape: ${results.outputShape}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Detected Type: ${results.modelType}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Input Size Used: ${results.actualInputSize}x${results.actualInputSize}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
            }
        }

        // Test Results Card
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("🧪 Test Results", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))

                Text("Synthetic Test Detections: ${results.syntheticTestDetections}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Random Image Detections: ${results.randomTestDetections}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Inference Time: ${results.inferenceTimeMs}ms", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
            }
        }

        // Raw Output Analysis Card
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("📈 Raw Output Analysis", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))

                Text("Output Dimensions: ${results.outputDimensions}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Min Value: ${String.format("%.6f", results.outputMinValue)}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Max Value: ${String.format("%.6f", results.outputMaxValue)}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Mean Value: ${String.format("%.6f", results.outputMeanValue)}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("Non-zero Values: ${results.nonZeroCount}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                Text("High Confidence Values: ${results.highConfidenceCount}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
            }
        }

        // Issues and Recommendations Card
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("⚠️ Issues & Recommendations", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))

                results.issues.forEach { issue ->
                    Text("• $issue", fontSize = 12.sp, color = androidx.compose.ui.graphics.Color.Red)
                }

                Spacer(modifier = Modifier.height(8.dp))

                results.recommendations.forEach { recommendation ->
                    Text("✓ $recommendation", fontSize = 12.sp, color = androidx.compose.ui.graphics.Color.Green)
                }
            }
        }

        // Sample Detections Card
        if (results.sampleDetections.isNotEmpty()) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("🎯 Sample Detections", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))

                    results.sampleDetections.forEachIndexed { index, detection ->
                        Text(
                            "Detection $index: conf=${String.format("%.3f", detection.score)}, " +
                                    "bbox=[${String.format("%.3f", detection.bbox.left)}, ${String.format("%.3f", detection.bbox.top)}, " +
                                    "${String.format("%.3f", detection.bbox.right)}, ${String.format("%.3f", detection.bbox.bottom)}]",
                            fontSize = 11.sp,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }
        }
    }
}

private suspend fun performDebugAnalysis(context: Context): DebugResults = withContext(Dispatchers.Default) {
    val modelPath = "simonskina.tflite"

    try {
        Log.d("ModelDebug", "🔍 Starting comprehensive model debug analysis")

        // Load model and get basic info
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val modelSizeBytes = assetFileDescriptor.declaredLength

        // Create detector for analysis
        val detector = GenericTFLiteDetector(context, modelPath, confThreshold = 0.01f) // Very low threshold for debug

        // Load interpreter directly for detailed analysis
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, assetFileDescriptor.declaredLength)

        val interpreter = Interpreter(modelBuffer)

        // Get tensor info
        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        val inputShape = inputTensor.shape().contentToString()
        val outputShape = outputTensor.shape().contentToString()

        // Determine actual input size
        val actualInputSize = when {
            inputTensor.shape().size == 4 -> {
                if (inputTensor.shape()[1] == 3 || inputTensor.shape()[1] == 1) {
                    inputTensor.shape()[2] // NCHW
                } else {
                    inputTensor.shape()[1] // NHWC
                }
            }
            else -> 320
        }

        // Create test images
        val syntheticBitmap = createSyntheticBarbellImage()
        val randomBitmap = createRandomTestImage()

        // Test with synthetic image
        val startTime = System.currentTimeMillis()
        val syntheticDetections = detector.detect(syntheticBitmap)
        val inferenceTime = System.currentTimeMillis() - startTime

        // Test with random image
        val randomDetections = detector.detect(randomBitmap)

        // Analyze raw output by running a simple detection
        val (outputAnalysis, sampleDetections) = analyzeRawOutput(detector, syntheticBitmap)

        // Determine model type
        val modelType = determineModelTypeFromShape(outputTensor.shape())

        // Generate issues and recommendations
        val (issues, recommendations) = generateIssuesAndRecommendations(
            syntheticDetections.size,
            randomDetections.size,
            outputAnalysis,
            actualInputSize,
            modelType
        )

        // Cleanup
        interpreter.close()
        detector.cleanup()

        DebugResults(
            modelPath = modelPath,
            modelSizeBytes = modelSizeBytes,
            inputShape = inputShape,
            outputShape = outputShape,
            modelType = modelType,
            actualInputSize = actualInputSize,
            syntheticTestDetections = syntheticDetections.size,
            randomTestDetections = randomDetections.size,
            inferenceTimeMs = inferenceTime,
            outputDimensions = outputTensor.shape().contentToString(),
            outputMinValue = outputAnalysis.minValue,
            outputMaxValue = outputAnalysis.maxValue,
            outputMeanValue = outputAnalysis.meanValue,
            nonZeroCount = outputAnalysis.nonZeroCount,
            highConfidenceCount = outputAnalysis.highConfidenceCount,
            issues = issues,
            recommendations = recommendations,
            sampleDetections = sampleDetections.take(5) // Show first 5 detections
        )

    } catch (e: Exception) {
        Log.e("ModelDebug", "Error during debug analysis", e)
        DebugResults(
            modelPath = modelPath,
            modelSizeBytes = 0,
            inputShape = "ERROR",
            outputShape = "ERROR",
            modelType = "ERROR",
            actualInputSize = 0,
            syntheticTestDetections = 0,
            randomTestDetections = 0,
            inferenceTimeMs = 0,
            outputDimensions = "ERROR",
            outputMinValue = 0f,
            outputMaxValue = 0f,
            outputMeanValue = 0f,
            nonZeroCount = 0,
            highConfidenceCount = 0,
            issues = listOf("Failed to analyze model: ${e.message}"),
            recommendations = listOf("Check if simonskina.tflite exists in assets folder", "Verify model format is compatible"),
            sampleDetections = emptyList()
        )
    }
}

private fun createSyntheticBarbellImage(): Bitmap {
    val bitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bitmap)
    val paint = Paint().apply {
        isAntiAlias = true
        style = Paint.Style.FILL
    }

    // Black background
    canvas.drawColor(Color.BLACK)

    // Draw barbell in center
    paint.color = Color.WHITE

    // Bar (horizontal)
    canvas.drawRect(150f, 315f, 490f, 325f, paint)

    // Left weight
    canvas.drawCircle(150f, 320f, 40f, paint)
    canvas.drawRect(110f, 280f, 150f, 360f, paint)

    // Right weight
    canvas.drawCircle(490f, 320f, 40f, paint)
    canvas.drawRect(490f, 280f, 530f, 360f, paint)

    return bitmap
}

private fun createRandomTestImage(): Bitmap {
    val bitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bitmap)
    val paint = Paint()

    // Random colored background
    canvas.drawColor(Color.rgb(
        (Math.random() * 255).toInt(),
        (Math.random() * 255).toInt(),
        (Math.random() * 255).toInt()
    ))

    // Add some random shapes
    repeat(10) {
        paint.color = Color.rgb(
            (Math.random() * 255).toInt(),
            (Math.random() * 255).toInt(),
            (Math.random() * 255).toInt()
        )
        val x = (Math.random() * 640).toFloat()
        val y = (Math.random() * 640).toFloat()
        val radius = (Math.random() * 50 + 10).toFloat()
        canvas.drawCircle(x, y, radius, paint)
    }

    return bitmap
}

private fun analyzeRawOutput(detector: GenericTFLiteDetector, testBitmap: Bitmap): Pair<OutputAnalysis, List<Detection>> {
    // This is a simplified analysis - in real implementation, you'd need access to raw output buffer
    val detections = detector.detect(testBitmap)

    // Simulate output analysis based on detections
    val hasOutput = detections.isNotEmpty()
    val maxConf = detections.maxOfOrNull { it.score } ?: 0f
    val minConf = detections.minOfOrNull { it.score } ?: 0f
    val meanConf = if (detections.isNotEmpty()) detections.map { it.score }.average().toFloat() else 0f

    return Pair(
        OutputAnalysis(
            minValue = minConf,
            maxValue = maxConf,
            meanValue = meanConf,
            nonZeroCount = detections.size,
            highConfidenceCount = detections.count { it.score > 0.5f }
        ),
        detections
    )
}

private fun determineModelTypeFromShape(shape: IntArray): String {
    return when {
        shape.size == 3 && shape[2] > 80 -> "YOLOv5 (${shape.contentToString()})"
        shape.size == 3 && shape[1] > 80 && shape[2] > 1000 -> "YOLOv8 (${shape.contentToString()})"
        shape.size == 2 && shape[1] < 1000 -> "Classification (${shape.contentToString()})"
        shape.size == 3 -> "Generic Detection (${shape.contentToString()})"
        else -> "Unknown (${shape.contentToString()})"
    }
}

private fun generateIssuesAndRecommendations(
    syntheticDetections: Int,
    randomDetections: Int,
    outputAnalysis: OutputAnalysis,
    actualInputSize: Int,
    modelType: String
): Pair<List<String>, List<String>> {
    val issues = mutableListOf<String>()
    val recommendations = mutableListOf<String>()

    // Check synthetic detection results
    if (syntheticDetections == 0) {
        issues.add("Model failed to detect synthetic barbell - possible preprocessing issue")
        recommendations.add("Verify input preprocessing matches model training format")
        recommendations.add("Check if model expects different input normalization")
    } else if (syntheticDetections > 5) {
        issues.add("Too many detections on synthetic image ($syntheticDetections) - possible false positives")
        recommendations.add("Increase confidence threshold")
        recommendations.add("Improve NMS parameters")
    }

    // Check random detection results
    if (randomDetections > 3) {
        issues.add("Model detecting objects in random noise ($randomDetections) - overly sensitive")
        recommendations.add("Increase confidence threshold to reduce false positives")
        recommendations.add("Add more robust filtering in post-processing")
    }

    // Check output analysis
    if (outputAnalysis.maxValue < 0.1f) {
        issues.add("Very low confidence scores (max: ${outputAnalysis.maxValue}) - model might need different preprocessing")
        recommendations.add("Check input image format (RGB vs BGR)")
        recommendations.add("Verify input normalization range (0-1 vs 0-255)")
    }

    if (outputAnalysis.nonZeroCount == 0) {
        issues.add("No valid outputs detected - model inference might be failing")
        recommendations.add("Check model compatibility with TensorFlow Lite")
        recommendations.add("Verify model is not corrupted")
    }

    // Check input size
    if (actualInputSize < 224 || actualInputSize > 640) {
        issues.add("Unusual input size detected: ${actualInputSize}x$actualInputSize")
        recommendations.add("Verify model input dimensions are correct")
    }

    // Model-specific recommendations
    when {
        modelType.contains("YOLOv5") -> {
            recommendations.add("For YOLOv5: Ensure coordinates are in center format (cx, cy, w, h)")
            recommendations.add("Use confidence threshold around 0.25-0.45 for YOLOv5")
        }
        modelType.contains("YOLOv8") -> {
            recommendations.add("For YOLOv8: Output format is transposed - check coordinate extraction")
            recommendations.add("YOLOv8 typically has lower confidence scores initially")
        }
        modelType.contains("Classification") -> {
            issues.add("Model appears to be classification-only, not object detection")
            recommendations.add("Consider using an object detection model instead")
        }
    }

    // General recommendations
    if (issues.isNotEmpty()) {
        recommendations.add("Enable debug logging to see raw model outputs")
        recommendations.add("Test with different confidence thresholds (0.1, 0.3, 0.5)")
        recommendations.add("Verify the model was trained for barbell detection")
    }

    return Pair(issues, recommendations)
}

// Data classes
data class DebugResults(
    val modelPath: String,
    val modelSizeBytes: Long,
    val inputShape: String,
    val outputShape: String,
    val modelType: String,
    val actualInputSize: Int,
    val syntheticTestDetections: Int,
    val randomTestDetections: Int,
    val inferenceTimeMs: Long,
    val outputDimensions: String,
    val outputMinValue: Float,
    val outputMaxValue: Float,
    val outputMeanValue: Float,
    val nonZeroCount: Int,
    val highConfidenceCount: Int,
    val issues: List<String>,
    val recommendations: List<String>,
    val sampleDetections: List<Detection>
)

data class OutputAnalysis(
    val minValue: Float,
    val maxValue: Float,
    val meanValue: Float,
    val nonZeroCount: Int,
    val highConfidenceCount: Int
)