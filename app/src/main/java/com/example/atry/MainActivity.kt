package com.example.atry

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.atry.detector.*
import com.example.atry.ui.theme.TryTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

val Color.Companion.Orange: Color
    get() = Color(0xFFFFA500)

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "MainActivity onCreate with Optimized Barbell Detector (EfficientDet-Lite2)")

        setContent {
            TryTheme {
                Surface(color = MaterialTheme.colorScheme.background) {
                    MainContent()
                }
            }
        }
    }

    @Composable
    private fun MainContent() {
        val context = LocalContext.current

        // Track permission state
        var hasCameraPermission by remember {
            mutableStateOf(
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            )
        }

        var permissionRequested by remember { mutableStateOf(false) }
        var permissionDenied by remember { mutableStateOf(false) }

        // Permission launcher
        val permissionLauncher = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            Log.d(TAG, "Camera permission result: $isGranted")
            hasCameraPermission = isGranted
            permissionRequested = true

            if (!isGranted) {
                permissionDenied = true
                Toast.makeText(context, "Camera permission is required for optimized barbell tracking", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(context, "Camera permission granted - Optimized detector ready!", Toast.LENGTH_SHORT).show()
            }
        }

        // Request permission on first launch
        LaunchedEffect(Unit) {
            if (!hasCameraPermission && !permissionRequested) {
                Log.d(TAG, "Requesting camera permission for optimized barbell detector")
                delay(300)
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        Box(modifier = Modifier.fillMaxSize()) {
            when {
                hasCameraPermission -> {
                    Log.d(TAG, "Camera permission granted, showing optimized barbell camera preview")
                    OptimizedBarbellCameraPreview()
                }
                permissionDenied -> {
                    PermissionDeniedScreen {
                        permissionDenied = false
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }
                else -> {
                    PermissionRequestScreen {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }
            }
        }
    }

    @Composable
    private fun OptimizedBarbellCameraPreview() {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val scope = rememberCoroutineScope()

        Log.d("OptimizedCamera", "Initializing Optimized Barbell Detector (EfficientDet-Lite2)")

        // Create optimized detector specifically for your model
        val detector = remember {
            try {
                Log.d("OptimizedCamera", "Creating Optimized Barbell Detector")
                OptimizedBarbellDetector(
                    context = context,
                    modelPath = "simonskina.tflite",
                    confThreshold = 0.4f,  // Optimal for your model
                    iouThreshold = 0.5f,
                    maxDetections = 8      // Reasonable limit for barbells
                )
            } catch (e: Exception) {
                Log.e("OptimizedCamera", "Failed to create Optimized Detector: ${e.message}", e)
                null
            }
        }

        if (detector == null) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "⚠️ Optimized Barbell Detector Failed",
                        color = Color.Red,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Model: EfficientDet-Lite2 (448×448)\n• Float32 format\n• Single class (barbells)\n• Optimized preprocessing",
                        color = Color.Gray,
                        fontSize = 14.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(16.dp)
                    )

                    Button(
                        onClick = {
                            scope.launch {
                                try {
                                    // Test optimized detector
                                    val testDetector = OptimizedBarbellDetector(context)

                                    // Create high-quality test image
                                    val testBitmap = createOptimizedTestBarbell()
                                    val detections = testDetector.detect(testBitmap)

                                    val message = buildString {
                                        appendLine("✅ Optimized Test Results:")
                                        appendLine("Detections: ${detections.size}")
                                        detections.forEachIndexed { index, detection ->
                                            val quality = testDetector.getDetectionQuality(detection)
                                            appendLine("$index: ${String.format("%.1f%%", detection.score * 100)} (${quality.getQualityGrade()})")
                                        }
                                    }

                                    Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                                    testDetector.cleanup()

                                } catch (e: Exception) {
                                    Toast.makeText(context, "❌ Test failed: ${e.message}", Toast.LENGTH_LONG).show()
                                    Log.e("OptimizedCamera", "Test error", e)
                                }
                            }
                        }
                    ) {
                        Text("Test Optimized Detector")
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    Button(
                        onClick = {
                            // Run model verification
                            scope.launch {
                                try {
                                    val verifier = ModelVerifier(context)
                                    val specs = verifier.verifySimonskinaModel()

                                    val message = if (specs != null) {
                                        "✅ Model verified!\nFormat: ${specs.modelFormat}\nInput: ${specs.inputShape.contentToString()}"
                                    } else {
                                        "❌ Model verification failed"
                                    }

                                    Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                                } catch (e: Exception) {
                                    Toast.makeText(context, "Verification error: ${e.message}", Toast.LENGTH_SHORT).show()
                                }
                            }
                        }
                    ) {
                        Text("Verify Model")
                    }
                }
            }
            return
        }

        // State variables
        val previewView = remember { PreviewView(context) }
        var barbellDetections by remember { mutableStateOf<List<Detection>>(emptyList()) }
        var isProcessing by remember { mutableStateOf(false) }
        var fps by remember { mutableStateOf(0f) }
        var cameraError by remember { mutableStateOf<String?>(null) }

        // Performance tracking
        var frameCount by remember { mutableStateOf(0) }
        var lastFpsUpdate by remember { mutableStateOf(System.currentTimeMillis()) }
        var processingTime by remember { mutableStateOf(0L) }
        var averageConfidence by remember { mutableStateOf(0f) }
        var detectionHistory by remember { mutableStateOf<List<Int>>(emptyList()) }

        // Enhanced barbell tracking
        val tracker = remember {
            try {
                EnhancedBarbellTracker(
                    context = context,
                    modelPath = "simonskina.tflite",
                    confThreshold = 0.4f,
                    iouThreshold = 0.5f,
                    maxAge = 30
                )
            } catch (e: Exception) {
                Log.e("OptimizedCamera", "Failed to create Enhanced Tracker: ${e.message}", e)
                null
            }
        }

        // Session management
        var isRecording by remember { mutableStateOf(false) }
        var sessionStartTime by remember { mutableStateOf(0L) }
        var sessionEndTime by remember { mutableStateOf(0L) }
        var isGeneratingReport by remember { mutableStateOf(false) }
        var analytics by remember { mutableStateOf<BarbellAnalytics?>(null) }

        // Report generator
        val reportGenerator = remember { ReportGenerator(context) }

        // Cleanup
        DisposableEffect(detector, tracker) {
            onDispose {
                try {
                    detector.cleanup()
                    tracker?.cleanup()
                    Log.d("OptimizedCamera", "Optimized detector and tracker cleaned up")
                } catch (e: Exception) {
                    Log.e("OptimizedCamera", "Cleanup error: ${e.message}", e)
                }
            }
        }

        // Optimized camera setup
        LaunchedEffect(previewView) {
            try {
                Log.d("OptimizedCamera", "Setting up optimized camera for barbell detection")
                val cameraProvider = ProcessCameraProvider.getInstance(context).get()

                val preview = Preview.Builder()
                    .setTargetResolution(Size(1280, 720)) // High quality preview
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                val imageAnalysis = androidx.camera.core.ImageAnalysis.Builder()
                    .setBackpressureStrategy(androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(Size(640, 480)) // Good balance for 448×448 model
                    .setOutputImageFormat(androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                    .also { analyzer ->
                        analyzer.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                            if (isProcessing) {
                                imageProxy.close()
                                return@setAnalyzer
                            }

                            isProcessing = true
                            val processingStart = System.currentTimeMillis()

                            scope.launch(Dispatchers.Default) {
                                try {
                                    // Convert to bitmap with optimization
                                    val bitmap = BitmapUtils.imageProxyToBitmapOptimized(imageProxy)
                                    val timestamp = System.currentTimeMillis()

                                    // Run optimized barbell detection
                                    val newDetections = detector.detect(bitmap)

                                    // Use enhanced tracker if available
                                    val trackingResult = tracker?.track(bitmap, timestamp)

                                    withContext(Dispatchers.Main) {
                                        barbellDetections = newDetections
                                        processingTime = System.currentTimeMillis() - processingStart

                                        // Update detection history for trending
                                        detectionHistory = (detectionHistory + newDetections.size).takeLast(10)

                                        // Calculate average confidence
                                        averageConfidence = if (newDetections.isNotEmpty()) {
                                            newDetections.map { it.score }.average().toFloat()
                                        } else {
                                            0f
                                        }

                                        // Update analytics if recording
                                        if (isRecording && tracker != null) {
                                            analytics = tracker.getAnalytics()
                                        }

                                        // Enhanced logging
                                        if (newDetections.isNotEmpty()) {
                                            Log.d("OptimizedCamera", "🏋️ Found ${newDetections.size} barbells, avg conf: ${String.format("%.2f", averageConfidence)}")

                                            newDetections.forEachIndexed { index, detection ->
                                                val quality = detector.getDetectionQuality(detection)
                                                Log.d("OptimizedCamera", "  Barbell $index: ${String.format("%.1f%%", detection.score * 100)} " +
                                                        "(${quality.getQualityGrade()}, size: ${String.format("%.3f", quality.size)})")
                                            }
                                        } else if (frameCount % 30 == 0) {
                                            Log.d("OptimizedCamera", "No barbells detected (frame $frameCount)")
                                        }
                                    }

                                    // FPS calculation
                                    frameCount++
                                    val currentTime = System.currentTimeMillis()
                                    if (currentTime - lastFpsUpdate >= 1500) {
                                        val newFps = frameCount * 1000f / (currentTime - lastFpsUpdate)
                                        withContext(Dispatchers.Main) {
                                            fps = newFps
                                        }
                                        frameCount = 0
                                        lastFpsUpdate = currentTime
                                    }

                                } catch (e: Exception) {
                                    Log.e("OptimizedCamera", "Optimized detection error: ${e.message}")
                                } finally {
                                    isProcessing = false
                                    imageProxy.close()
                                }
                            }
                        }
                    }

                // Bind camera
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis
                )

                Log.d("OptimizedCamera", "Optimized camera setup complete")
                cameraError = null

            } catch (e: Exception) {
                val errorMsg = "Optimized camera setup failed: ${e.message}"
                Log.e("OptimizedCamera", errorMsg, e)
                cameraError = errorMsg
            }
        }

        // UI Layout
        Box(modifier = Modifier.fillMaxSize()) {
            if (cameraError != null) {
                // Show error state
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "📷 Optimized Camera Error",
                            color = Color.Red,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = cameraError!!,
                            color = Color.Gray,
                            fontSize = 12.sp,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.padding(16.dp)
                        )
                        Button(
                            onClick = { cameraError = null }
                        ) {
                            Text("Retry")
                        }
                    }
                }
            } else {
                // Camera Preview
                AndroidView(
                    factory = { previewView },
                    modifier = Modifier.fillMaxSize()
                )

                // Optimized Detection Overlay
                Canvas(modifier = Modifier.fillMaxSize()) {
                    drawOptimizedBarbellDetections(barbellDetections, detector)
                }

                // Enhanced Info Panel
                OptimizedBarbellInfoPanel(
                    detections = barbellDetections,
                    analytics = analytics,
                    fps = fps,
                    processingTime = processingTime,
                    averageConfidence = averageConfidence,
                    detectionHistory = detectionHistory,
                    isProcessing = isProcessing,
                    isRecording = isRecording,
                    isGeneratingReport = isGeneratingReport,
                    detector = detector,
                    onStartStopRecording = {
                        isRecording = !isRecording
                        Log.d("OptimizedCamera", "Recording toggled - isRecording: $isRecording")
                        if (isRecording) {
                            tracker?.reset()
                            sessionStartTime = System.currentTimeMillis()
                            Log.d("OptimizedCamera", "Started optimized recording")
                        } else {
                            sessionEndTime = System.currentTimeMillis()
                            analytics = tracker?.getAnalytics()
                            Log.d("OptimizedCamera", "Stopped optimized recording")
                        }
                    },
                    onClearData = {
                        tracker?.reset()
                        analytics = null
                        detectionHistory = emptyList()
                        Log.d("OptimizedCamera", "Cleared all tracking data")
                    },
                    onGenerateExcelReport = {
                        scope.launch {
                            isGeneratingReport = true
                            try {
                                val trackingData = tracker?.getTrackingData() ?: emptyList()
                                val session = ReportGenerator.WorkoutSession(
                                    startTime = sessionStartTime,
                                    endTime = if (sessionEndTime > 0) sessionEndTime else System.currentTimeMillis(),
                                    actualRepCount = analytics?.repCount ?: 0,
                                    paths = convertTrackingDataToPaths(trackingData),
                                    movements = emptyList()
                                )

                                val result = reportGenerator.generateExcelReport(session,
                                    com.example.atry.detector.BarPathAnalyzer())
                                result.fold(
                                    onSuccess = { file ->
                                        Toast.makeText(context, "📊 Excel report generated: ${file.name}", Toast.LENGTH_LONG).show()
                                        reportGenerator.shareReport(file)
                                    },
                                    onFailure = { error ->
                                        Toast.makeText(context, "❌ Excel report error: ${error.message}", Toast.LENGTH_LONG).show()
                                        Log.e("OptimizedCamera", "Excel report error", error)
                                    }
                                )
                            } finally {
                                isGeneratingReport = false
                            }
                        }
                    },
                    onGenerateCSVReport = {
                        scope.launch {
                            isGeneratingReport = true
                            try {
                                val trackingData = tracker?.getTrackingData() ?: emptyList()
                                val session = ReportGenerator.WorkoutSession(
                                    startTime = sessionStartTime,
                                    endTime = if (sessionEndTime > 0) sessionEndTime else System.currentTimeMillis(),
                                    actualRepCount = analytics?.repCount ?: 0,
                                    paths = convertTrackingDataToPaths(trackingData),
                                    movements = emptyList()
                                )

                                val result = reportGenerator.generateCSVReport(session,
                                    com.example.atry.detector.BarPathAnalyzer())
                                result.fold(
                                    onSuccess = { file ->
                                        Toast.makeText(context, "📋 CSV report generated: ${file.name}", Toast.LENGTH_LONG).show()
                                        reportGenerator.shareReport(file)
                                    },
                                    onFailure = { error ->
                                        Toast.makeText(context, "❌ CSV report error: ${error.message}", Toast.LENGTH_LONG).show()
                                        Log.e("OptimizedCamera", "CSV report error", error)
                                    }
                                )
                            } finally {
                                isGeneratingReport = false
                            }
                        }
                    },
                    modifier = Modifier.align(Alignment.TopStart)
                )
            }
        }
    }

    // Add this composable to your MainActivity.kt (inside the MainActivity class)

    @Composable
    private fun OptimizedBarbellInfoPanel(
        detections: List<Detection>,
        analytics: BarbellAnalytics?,
        fps: Float,
        processingTime: Long,
        averageConfidence: Float,
        detectionHistory: List<Int>,
        isProcessing: Boolean,
        isRecording: Boolean,
        isGeneratingReport: Boolean,
        detector: OptimizedBarbellDetector,
        onStartStopRecording: () -> Unit,
        onClearData: () -> Unit,
        onGenerateExcelReport: () -> Unit,
        onGenerateCSVReport: () -> Unit,
        modifier: Modifier = Modifier
    ) {
        Box(
            modifier = modifier
                .fillMaxWidth()
                .padding(top = 50.dp)
        ) {
            Column(
                modifier = Modifier.align(Alignment.TopCenter),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Title with model info
                Text(
                    text = "🏋️ Optimized Barbell Detector",
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .background(
                            Color.Black.copy(alpha = 0.7f),
                            shape = RoundedCornerShape(8.dp)
                        )
                        .padding(8.dp)
                )
                Text(
                    text = "EfficientDet-Lite2 • 448×448 • Float32",
                    color = Color.Cyan,
                    fontSize = 11.sp,
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .background(
                            Color.Black.copy(alpha = 0.6f),
                            shape = RoundedCornerShape(6.dp)
                        )
                        .padding(4.dp)
                )

                Spacer(modifier = Modifier.height(6.dp))

                // Performance metrics row
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier
                        .background(
                            Color.Black.copy(alpha = 0.6f),
                            shape = RoundedCornerShape(8.dp)
                        )
                        .padding(8.dp)
                ) {
                    // FPS with color coding
                    val fpsColor = when {
                        fps >= 25f -> Color.Green
                        fps >= 15f -> Color.Yellow
                        else -> Color.Red
                    }
                    Text(
                        text = "FPS: ${String.format("%.1f", fps)}",
                        color = fpsColor,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )

                    // Processing time with color coding
                    val timeColor = when {
                        processingTime <= 50L -> Color.Green
                        processingTime <= 100L -> Color.Yellow
                        else -> Color.Red
                    }
                    Text(
                        text = "Time: ${processingTime}ms",
                        color = timeColor,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )

                    // Detection count
                    Text(
                        text = "Detected: ${detections.size}",
                        color = if (detections.isNotEmpty()) Color.Green else Color.Gray,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                Spacer(modifier = Modifier.height(4.dp))

                // Detection quality summary
                if (detections.isNotEmpty()) {
                    val qualityStats = detections.map { detector.getDetectionQuality(it) }
                    val avgQuality = qualityStats.map { it.getOverallQuality() }.average().toFloat()
                    val bestQuality = qualityStats.maxByOrNull { it.getOverallQuality() }

                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier
                            .background(
                                Color.Black.copy(alpha = 0.6f),
                                shape = RoundedCornerShape(6.dp)
                            )
                            .padding(6.dp)
                    ) {
                        Text(
                            text = "Avg Conf: ${String.format("%.1f%%", averageConfidence * 100)}",
                            color = Color.Cyan,
                            fontSize = 10.sp
                        )
                        Text(
                            text = "Quality: ${String.format("%.1f%%", avgQuality * 100)}",
                            color = when {
                                avgQuality >= 0.8f -> Color.Green
                                avgQuality >= 0.6f -> Color.Yellow
                                else -> Color.Orange
                            },
                            fontSize = 10.sp
                        )
                        bestQuality?.let {
                            Text(
                                text = "Best: ${it.getQualityGrade()}",
                                color = Color.Green,
                                fontSize = 10.sp
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(6.dp))

                // Control buttons
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.padding(horizontal = 8.dp)
                ) {
                    Button(
                        onClick = onStartStopRecording,
                        modifier = Modifier.height(32.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isRecording) Color.Red else Color.Green
                        )
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text(
                                text = if (isRecording) "⏹" else "⏺",
                                fontSize = 12.sp,
                                color = Color.White
                            )
                            Text(
                                text = if (isRecording) "Stop" else "Start",
                                fontSize = 10.sp,
                                color = Color.White
                            )
                        }
                    }

                    Button(
                        onClick = onClearData,
                        modifier = Modifier.height(32.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color.Blue)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text("🗑", fontSize = 12.sp, color = Color.White)
                            Text("Clear", fontSize = 10.sp, color = Color.White)
                        }
                    }
                }

                Spacer(modifier = Modifier.height(4.dp))

                // Recording status with enhanced info
                if (isRecording) {
                    Text(
                        text = "🔴 RECORDING OPTIMIZED SESSION",
                        color = Color.Red,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center,
                        modifier = Modifier
                            .background(
                                Color.Black.copy(alpha = 0.7f),
                                shape = RoundedCornerShape(6.dp)
                            )
                            .padding(6.dp)
                    )
                    analytics?.let { stats ->
                        Text(
                            text = "Reps: ${stats.repCount} • Distance: ${String.format("%.2f", stats.totalDistance)}",
                            color = Color.Yellow,
                            fontSize = 10.sp,
                            textAlign = TextAlign.Center
                        )
                    }
                } else {
                    Text(
                        text = "⚪ Ready for optimized tracking",
                        color = Color.Gray,
                        fontSize = 11.sp,
                        textAlign = TextAlign.Center
                    )
                }

                // Report generation buttons (only show when we have analytics data)
                AnimatedVisibility(visible = analytics?.repCount ?: 0 > 0) {
                    Column(
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Spacer(modifier = Modifier.height(6.dp))

                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            modifier = Modifier.padding(horizontal = 8.dp)
                        ) {
                            Button(
                                onClick = onGenerateExcelReport,
                                enabled = !isGeneratingReport && (analytics?.repCount ?: 0) > 0,
                                modifier = Modifier.height(30.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Color(0xFF007ACC),
                                    disabledContainerColor = Color.Gray
                                )
                            ) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(
                                        text = if (isGeneratingReport) "⏳" else "📊",
                                        fontSize = 10.sp,
                                        color = Color.White
                                    )
                                    Text(
                                        text = if (isGeneratingReport) "..." else "Excel",
                                        fontSize = 9.sp,
                                        color = Color.White
                                    )
                                }
                            }

                            Button(
                                onClick = onGenerateCSVReport,
                                enabled = !isGeneratingReport && (analytics?.repCount ?: 0) > 0,
                                modifier = Modifier.height(30.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Color(0xFF228B22),
                                    disabledContainerColor = Color.Gray
                                )
                            ) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(
                                        text = if (isGeneratingReport) "⏳" else "📋",
                                        fontSize = 10.sp,
                                        color = Color.White
                                    )
                                    Text(
                                        text = if (isGeneratingReport) "..." else "CSV",
                                        fontSize = 9.sp,
                                        color = Color.White
                                    )
                                }
                            }
                        }
                    }
                }

                // Detection history trend (mini chart)
                if (detectionHistory.size >= 3) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier
                            .background(
                                Color.Black.copy(alpha = 0.6f),
                                shape = RoundedCornerShape(6.dp)
                            )
                            .padding(6.dp)
                    ) {
                        Text(
                            text = "Detection Trend:",
                            color = Color.White,
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold
                        )

                        Row(
                            horizontalArrangement = Arrangement.spacedBy(2.dp),
                            modifier = Modifier.padding(horizontal = 8.dp)
                        ) {
                            val maxDetections = detectionHistory.maxOrNull() ?: 1
                            detectionHistory.takeLast(8).forEach { count ->
                                val height = if (maxDetections > 0) {
                                    (count.toFloat() / maxDetections * 16).coerceAtLeast(2f)
                                } else 2f

                                Box(
                                    modifier = Modifier
                                        .width(6.dp)
                                        .height(height.dp)
                                        .background(
                                            color = when {
                                                count == 0 -> Color.Gray
                                                count <= 2 -> Color.Yellow
                                                else -> Color.Green
                                            },
                                            shape = RoundedCornerShape(1.dp)
                                        )
                                )
                            }
                        }

                        Text(
                            text = "Recent: ${detectionHistory.takeLast(3).joinToString("-")}",
                            color = Color.Cyan,
                            fontSize = 9.sp,
                            textAlign = TextAlign.Center
                        )
                    }
                }

                // Detailed analytics display
                analytics?.let { stats ->
                    Spacer(modifier = Modifier.height(6.dp))

                    Card(
                        modifier = Modifier
                            .padding(horizontal = 8.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = Color(0xFF1E3A8A).copy(alpha = 0.9f)
                        ),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                text = "📈 Optimized Analytics",
                                color = Color.White,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold
                            )

                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Text(
                                    text = "Reps: ${stats.repCount}",
                                    color = Color.Cyan,
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                if (stats.totalDistance > 0) {
                                    Text(
                                        text = "Dist: ${String.format("%.1f", stats.totalDistance)}",
                                        color = Color.Yellow,
                                        fontSize = 10.sp
                                    )
                                }
                                if (stats.averageVelocity > 0) {
                                    Text(
                                        text = "Vel: ${String.format("%.1f", stats.averageVelocity)}",
                                        color = Color.Green,
                                        fontSize = 10.sp
                                    )
                                }
                            }

                            if (stats.pathConsistency > 0) {
                                Text(
                                    text = "Consistency: ${String.format("%.0f%%", stats.pathConsistency * 100)}",
                                    color = when {
                                        stats.pathConsistency >= 0.8f -> Color.Green
                                        stats.pathConsistency >= 0.6f -> Color.Yellow
                                        else -> Color.Orange
                                    },
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }

                            stats.primaryTrackingId?.let { id ->
                                Text(
                                    text = "Primary ID: $id",
                                    color = Color.Magenta,
                                    fontSize = 9.sp
                                )
                            }
                        }
                    }
                }

                // Processing status indicator
                if (isProcessing) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        modifier = Modifier
                            .background(
                                Color.Black.copy(alpha = 0.6f),
                                shape = RoundedCornerShape(6.dp)
                            )
                            .padding(6.dp)
                    ) {
                        Text(
                            text = "⚡",
                            color = Color.Yellow,
                            fontSize = 12.sp
                        )
                        Text(
                            text = "Processing optimized detection...",
                            color = Color.Yellow,
                            fontSize = 9.sp,
                            fontStyle = androidx.compose.ui.text.font.FontStyle.Italic
                        )
                    }
                }

                // Report generation status
                if (isGeneratingReport) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        modifier = Modifier
                            .background(
                                Color.Black.copy(alpha = 0.7f),
                                shape = RoundedCornerShape(6.dp)
                            )
                            .padding(6.dp)
                    ) {
                        Text(
                            text = "📄",
                            color = Color.Cyan,
                            fontSize = 12.sp
                        )
                        Text(
                            text = "Generating optimized report...",
                            color = Color.Cyan,
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                // Individual detection details (for debugging)
                if (detections.isNotEmpty() && detections.size <= 3) {
                    Spacer(modifier = Modifier.height(4.dp))
                    detections.forEachIndexed { index, detection ->
                        val quality = detector.getDetectionQuality(detection)
                        val center = detector.getDetectionCenter(detection)

                        Text(
                            text = "B$index: ${String.format("%.0f%%", detection.score * 100)} " +
                                    "(${String.format("%.2f", center.first)}, ${String.format("%.2f", center.second)}) " +
                                    "${quality.getQualityGrade()}",
                            color = when {
                                quality.getOverallQuality() >= 0.8f -> Color.Green
                                quality.getOverallQuality() >= 0.6f -> Color.Yellow
                                else -> Color.Orange
                            },
                            fontSize = 8.sp,
                            textAlign = TextAlign.Center,
                            modifier = Modifier
                                .background(
                                    Color.Black.copy(alpha = 0.5f),
                                    shape = RoundedCornerShape(4.dp)
                                )
                                .padding(4.dp)
                        )
                    }
                }

                // Model performance info
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = detector.getPerformanceInfo(),
                    color = Color.Gray,
                    fontSize = 8.sp,
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .padding(horizontal = 8.dp)
                        .background(
                            Color.Black.copy(alpha = 0.5f),
                            shape = RoundedCornerShape(4.dp)
                        )
                        .padding(4.dp)
                )
            }
        }
    }

    // Helper function to create test image for optimized detector
    private fun createOptimizedTestBarbell(): Bitmap {
        val bitmap = Bitmap.createBitmap(448, 448, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = android.graphics.Paint().apply {
            isAntiAlias = true
            style = android.graphics.Paint.Style.FILL
        }

        // Black background
        canvas.drawColor(android.graphics.Color.BLACK)

        // Draw realistic barbell for 448x448 input
        paint.color = android.graphics.Color.LTGRAY

        // Bar (horizontal)
        canvas.drawRect(100f, 210f, 348f, 238f, paint)

        // Weight plates
        paint.color = android.graphics.Color.GRAY
        canvas.drawCircle(120f, 224f, 35f, paint)
        canvas.drawCircle(328f, 224f, 35f, paint)

        // Inner plates
        paint.color = android.graphics.Color.DKGRAY
        canvas.drawCircle(120f, 224f, 25f, paint)
        canvas.drawCircle(328f, 224f, 25f, paint)

        Log.d("OptimizedCamera", "Created optimized test barbell image: 448×448")
        return bitmap
    }

    // Optimized drawing function for barbell detections
    private fun DrawScope.drawOptimizedBarbellDetections(
        detections: List<Detection>,
        detector: OptimizedBarbellDetector
    ) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        detections.forEachIndexed { index, detection ->
            val bbox = detection.bbox
            val left = bbox.left * canvasWidth
            val top = bbox.top * canvasHeight
            val right = bbox.right * canvasWidth
            val bottom = bbox.bottom * canvasHeight

            // Get quality assessment for color coding
            val quality = detector.getDetectionQuality(detection)
            val qualityScore = quality.getOverallQuality()

            // Color based on quality
            val color = when {
                qualityScore >= 0.8f -> Color.Green      // Excellent detection
                qualityScore >= 0.6f -> Color.Yellow     // Good detection
                qualityScore >= 0.4f -> Color.Orange     // Fair detection
                else -> Color.Red                        // Poor detection
            }

            // Draw enhanced bounding box
            val strokeWidth = 4.dp.toPx()
            drawRect(
                color = color,
                topLeft = Offset(left, top),
                size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                style = Stroke(width = strokeWidth)
            )

            // Draw quality indicator
            val qualityColor = color.copy(alpha = 0.3f)
            drawRect(
                color = qualityColor,
                topLeft = Offset(left, top),
                size = androidx.compose.ui.geometry.Size(right - left, bottom - top)
            )

            // Draw center point
            val centerX = (left + right) / 2f
            val centerY = (top + bottom) / 2f
            drawCircle(
                color = Color.White,
                radius = 8.dp.toPx(),
                center = Offset(centerX, centerY)
            )
            drawCircle(
                color = color,
                radius = 5.dp.toPx(),
                center = Offset(centerX, centerY)
            )

            // Draw enhanced labels
            drawContext.canvas.nativeCanvas.apply {
                val paint = android.graphics.Paint().apply {
                    this.color = color.toArgb()
                    textSize = 16.sp.toPx()
                    isAntiAlias = true
                    isFakeBoldText = true
                }

                val confidenceText = "${String.format("%.0f", detection.score * 100)}%"
                val qualityText = quality.getQualityGrade()
                val sizeText = "Size: ${String.format("%.3f", quality.size)}"

                // Background for text
                val backgroundPaint = android.graphics.Paint().apply {
                    this.color = android.graphics.Color.BLACK
                    alpha = 180
                }

                val textHeight = 20.sp.toPx()
                val textY = maxOf(top - 10f, textHeight)

                drawRect(left - 5f, textY - textHeight, right + 5f, textY + textHeight * 2, backgroundPaint)

                drawText(confidenceText, left + 5f, textY, paint)
                drawText(qualityText, left + 5f, textY + textHeight, paint)
                if (top > 80f) {
                    drawText(sizeText, left + 5f, textY + textHeight * 2, paint)
                }
            }

            // Draw aspect ratio indicator
            val aspectRatio = quality.aspectRatio
            val isHorizontal = aspectRatio > 1.5f
            val isVertical = aspectRatio < 0.7f

            if (isHorizontal) {
                // Draw horizontal indicator
                drawLine(
                    color = Color.Cyan,
                    start = Offset(left + 10f, centerY),
                    end = Offset(right - 10f, centerY),
                    strokeWidth = 3.dp.toPx()
                )
            } else if (isVertical) {
                // Draw vertical indicator
                drawLine(
                    color = Color.Magenta,
                    start = Offset(centerX, top + 10f),
                    end = Offset(centerX, bottom - 10f),
                    strokeWidth = 3.dp.toPx()
                )
            }
        }
    }

    // Helper function to convert tracking data to BarPath objects
    private fun convertTrackingDataToPaths(trackingData: List<EnhancedBarbellTracker.TrackingDataPoint>): List<BarPath> {
        val groupedData = trackingData.groupBy { it.id }
        return groupedData.map { (id, points) ->
            val barPath = BarPath(id = "optimized_path_$id")
            points.forEach { dataPoint ->
                barPath.addPoint(PathPoint(dataPoint.x, dataPoint.y, dataPoint.timestamp))
            }
            barPath
        }
    }

    @Composable
    private fun PermissionRequestScreen(onRequest: () -> Unit) {
        val pulse by animateDpAsState(
            targetValue = 120.dp,
            animationSpec = tween(800, easing = FastOutSlowInEasing),
            label = "pulse_animation"
        )

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFF1E1E1E))
                .pointerInput(Unit) {
                    detectTapGestures { onRequest() }
                },
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "🏋️",
                    fontSize = 48.sp,
                    color = Color.White
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Optimized Barbell Detector",
                    style = MaterialTheme.typography.titleLarge,
                    color = Color.White,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "EfficientDet-Lite2 • 448×448 • High Precision\nCamera permission required for barbell tracking",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.Gray,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(24.dp))

                Button(
                    onClick = onRequest,
                    modifier = Modifier
                        .size(pulse)
                        .clip(RoundedCornerShape(16.dp)),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF007ACC)
                    )
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "📷",
                            fontSize = 24.sp,
                            color = Color.White
                        )
                        Text(
                            text = "Grant Permission",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp
                        )
                    }
                }
            }
        }
    }

    @Composable
    private fun PermissionDeniedScreen(onRetry: () -> Unit) {
        val context = LocalContext.current

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFF1E1E1E)),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "❌",
                    fontSize = 48.sp,
                    color = Color.Red
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Camera Permission Denied",
                    style = MaterialTheme.typography.titleLarge,
                    color = Color.White,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Optimized barbell tracking requires camera access to analyze your lifting form in real-time. Please grant permission in Settings or try again.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.Gray,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(horizontal = 32.dp)
                )
                Spacer(modifier = Modifier.height(24.dp))

                Column(
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Button(
                        onClick = onRetry,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF007ACC)
                        )
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Text("🔄", fontSize = 16.sp)
                            Text(
                                text = "Try Again",
                                color = Color.White,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    Button(
                        onClick = {
                            // Open app settings
                            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                                data = Uri.fromParts("package", context.packageName, null)
                            }
                            context.startActivity(intent)
                        },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color.Gray
                        )
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Text("⚙️", fontSize = 16.sp)
                            Text(
                                text = "Open Settings",
                                color = Color.White,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }
        }
    }
}