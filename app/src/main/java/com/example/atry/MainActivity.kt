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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Executors

val Color.Companion.Orange: Color
    get() = Color(0xFFFFA500)

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "MainActivity onCreate with Enhanced Barbell Detector")

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

        val permissionLauncher = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            Log.d(TAG, "Camera permission result: $isGranted")
            hasCameraPermission = isGranted
            permissionRequested = true

            if (!isGranted) {
                permissionDenied = true
                Toast.makeText(context, "Camera permission is required for barbell tracking", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(context, "Camera permission granted - Detector ready!", Toast.LENGTH_SHORT).show()
            }
        }

        LaunchedEffect(Unit) {
            if (!hasCameraPermission && !permissionRequested) {
                Log.d(TAG, "Requesting camera permission")
                delay(300)
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        Box(modifier = Modifier.fillMaxSize()) {
            when {
                hasCameraPermission -> {
                    Log.d(TAG, "Camera permission granted, showing camera preview")
                    EnhancedBarbellCameraPreview()
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
    private fun EnhancedBarbellCameraPreview() {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val scope = rememberCoroutineScope()

        Log.d("EnhancedCamera", "Initializing Enhanced Barbell Detector")

        // Enhanced detector state management
        var detector by remember { mutableStateOf<OptimizedBarbellDetector?>(null) }
        var detectorError by remember { mutableStateOf<String?>(null) }
        var isDetectorInitializing by remember { mutableStateOf(true) }

        // Initialize detector with proper error handling
        LaunchedEffect(Unit) {
            scope.launch(Dispatchers.IO) {
                try {
                    Log.d("EnhancedCamera", "Creating Enhanced Barbell Detector...")

                    // First verify the model exists
                    val modelVerifier = ModelVerifier(context)
                    val specs = modelVerifier.verifySimonskinaModel()

                    if (specs != null) {
                        Log.d("EnhancedCamera", "✅ Model verified: ${specs.modelFormat}")

                        // Create detector with verified specs
                        val newDetector = OptimizedBarbellDetector(
                            context = context,
                            modelPath = "simonskina.tflite",
                            confThreshold = 0.3f,  // Lower threshold for testing
                            iouThreshold = 0.5f,
                            maxDetections = 5
                        )

                        if (newDetector.isInitialized()) {
                            withContext(Dispatchers.Main) {
                                detector = newDetector
                                detectorError = null
                                isDetectorInitializing = false
                                Log.d("EnhancedCamera", "✅ Enhanced detector initialized successfully")
                            }
                        } else {
                            throw RuntimeException("Detector failed to initialize properly")
                        }
                    } else {
                        throw RuntimeException("Model verification failed")
                    }

                } catch (e: Exception) {
                    Log.e("EnhancedCamera", "❌ Failed to create detector: ${e.message}", e)
                    withContext(Dispatchers.Main) {
                        detectorError = "Detector initialization failed: ${e.message}"
                        isDetectorInitializing = false
                    }
                }
            }
        }

        // UI State
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

        // Enhanced tracking
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
                Log.e("EnhancedCamera", "Failed to create Enhanced Tracker: ${e.message}", e)
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

        // Show appropriate UI based on detector state
        when {
            isDetectorInitializing -> {
                LoadingScreen("Initializing Enhanced Barbell Detector...")
            }
            detectorError != null -> {
                EnhancedErrorScreen(
                    error = detectorError!!,
                    onRetry = {
                        isDetectorInitializing = true
                        detectorError = null
                        // Trigger re-initialization
                        scope.launch(Dispatchers.IO) {
                            try {
                                delay(500)
                                val newDetector = OptimizedBarbellDetector(context)
                                if (newDetector.isInitialized()) {
                                    withContext(Dispatchers.Main) {
                                        detector = newDetector
                                        detectorError = null
                                        isDetectorInitializing = false
                                    }
                                } else {
                                    throw RuntimeException("Retry initialization failed")
                                }
                            } catch (e: Exception) {
                                withContext(Dispatchers.Main) {
                                    detectorError = "Retry failed: ${e.message}"
                                    isDetectorInitializing = false
                                }
                            }
                        }
                    },
                    onTest = {
                        scope.launch {
                            try {
                                val testBitmap = createTestBarbell()
                                val testDetector = OptimizedBarbellDetector(context)
                                val detections = testDetector.detect(testBitmap)

                                val message = "Test Results:\n" +
                                        "Detections: ${detections.size}\n" +
                                        "Detector initialized: ${testDetector.isInitialized()}"

                                Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                                testDetector.cleanup()
                            } catch (e: Exception) {
                                Toast.makeText(context, "Test failed: ${e.message}", Toast.LENGTH_LONG).show()
                            }
                        }
                    }
                )
            }
            detector != null -> {
                CameraPreviewWithDetection(
                    detector = detector!!,
                    tracker = tracker,
                    barbellDetections = barbellDetections,
                    isProcessing = isProcessing,
                    fps = fps,
                    processingTime = processingTime,
                    averageConfidence = averageConfidence,
                    detectionHistory = detectionHistory,
                    isRecording = isRecording,
                    isGeneratingReport = isGeneratingReport,
                    analytics = analytics,
                    onDetectionsUpdated = { newDetections ->
                        barbellDetections = newDetections
                        detectionHistory = (detectionHistory + newDetections.size).takeLast(10)
                        averageConfidence = if (newDetections.isNotEmpty()) {
                            newDetections.map { it.score }.average().toFloat()
                        } else {
                            0f
                        }
                    },
                    onProcessingStateChanged = { processing ->
                        isProcessing = processing
                    },
                    onProcessingTimeUpdated = { time ->
                        processingTime = time
                    },
                    onFpsUpdated = { newFps ->
                        fps = newFps
                        frameCount++
                        val currentTime = System.currentTimeMillis()
                        if (currentTime - lastFpsUpdate >= 1500) {
                            frameCount = 0
                            lastFpsUpdate = currentTime
                        }
                    },
                    onCameraError = { error ->
                        cameraError = error
                    },
                    onStartStopRecording = {
                        isRecording = !isRecording
                        Log.d("EnhancedCamera", "Recording toggled - isRecording: $isRecording")
                        if (isRecording) {
                            tracker?.reset()
                            sessionStartTime = System.currentTimeMillis()
                            Log.d("EnhancedCamera", "Started enhanced recording")
                        } else {
                            sessionEndTime = System.currentTimeMillis()
                            analytics = tracker?.getAnalytics()
                            Log.d("EnhancedCamera", "Stopped enhanced recording")
                        }
                    },
                    onClearData = {
                        tracker?.reset()
                        analytics = null
                        detectionHistory = emptyList()
                        Log.d("EnhancedCamera", "Cleared all tracking data")
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
                                        Log.e("EnhancedCamera", "Excel report error", error)
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
                                        Log.e("EnhancedCamera", "CSV report error", error)
                                    }
                                )
                            } finally {
                                isGeneratingReport = false
                            }
                        }
                    }
                )
            }
        }

        // Cleanup
        DisposableEffect(detector, tracker) {
            onDispose {
                try {
                    detector?.cleanup()
                    tracker?.cleanup()
                    Log.d("EnhancedCamera", "Enhanced detector and tracker cleaned up")
                } catch (e: Exception) {
                    Log.e("EnhancedCamera", "Cleanup error: ${e.message}", e)
                }
            }
        }
    }

    @Composable
    private fun LoadingScreen(message: String) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator(
                    modifier = Modifier.size(48.dp),
                    color = Color.Cyan
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = message,
                    color = Color.White,
                    fontSize = 16.sp,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Please wait...",
                    color = Color.Gray,
                    fontSize = 14.sp,
                    textAlign = TextAlign.Center
                )
            }
        }
    }

    @Composable
    private fun EnhancedErrorScreen(
        error: String,
        onRetry: () -> Unit,
        onTest: () -> Unit
    ) {
        val context = LocalContext.current
        val scope = rememberCoroutineScope()

        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.padding(32.dp)
            ) {
                Text(
                    text = "⚠️",
                    fontSize = 48.sp,
                    color = Color.Red
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Detector Initialization Failed",
                    color = Color.Red,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = error,
                    color = Color.Gray,
                    fontSize = 12.sp,
                    textAlign = TextAlign.Center
                )

                Spacer(modifier = Modifier.height(24.dp))

                // Enhanced button row with debugging
                Column(
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Button(
                            onClick = onRetry,
                            colors = ButtonDefaults.buttonColors(containerColor = Color.Blue)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text("🔄", fontSize = 14.sp)
                                Text("Retry", fontSize = 12.sp)
                            }
                        }

                        Button(
                            onClick = onTest,
                            colors = ButtonDefaults.buttonColors(containerColor = Color.Green)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text("🧪", fontSize = 14.sp)
                                Text("Test", fontSize = 12.sp)
                            }
                        }
                    }

                    // Add comprehensive debug button
                    DebugButton(context, scope)

                    // Model verification button
                    Button(
                        onClick = {
                            scope.launch {
                                try {
                                    val verifier = ModelVerifier(context)
                                    val specs = verifier.verifySimonskinaModel()

                                    val message = if (specs != null) {
                                        "✅ Model verified!\n" +
                                                "Format: ${specs.modelFormat}\n" +
                                                "Input: ${specs.inputShape.contentToString()}\n" +
                                                "Outputs: ${specs.outputSpecs.size}"
                                    } else {
                                        "❌ Model verification failed"
                                    }

                                    Toast.makeText(context, message, Toast.LENGTH_LONG).show()

                                    // Log detailed specs
                                    specs?.let { s ->
                                        Log.d("VERIFY", "📊 DETAILED MODEL SPECS:")
                                        Log.d("VERIFY", "Input: ${s.inputShape.contentToString()} ${s.inputDataType}")
                                        Log.d("VERIFY", "Input bytes: ${s.inputBytes}")
                                        Log.d("VERIFY", "Format: ${s.modelFormat}")
                                        s.outputSpecs.forEachIndexed { index, spec ->
                                            Log.d("VERIFY", "Output $index: ${spec.name} ${spec.shape.contentToString()} ${spec.dataType}")
                                        }
                                        Log.d("VERIFY", "📋 RECOMMENDATIONS:")
                                        s.recommendations.forEach { rec ->
                                            Log.d("VERIFY", "  • $rec")
                                        }
                                    }
                                } catch (e: Exception) {
                                    Toast.makeText(context, "Verification error: ${e.message}", Toast.LENGTH_LONG).show()
                                    Log.e("VERIFY", "Verification failed", e)
                                }
                            }
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = Color.Cyan)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text("🔍", fontSize = 14.sp)
                            Text("Verify", fontSize = 12.sp)
                        }
                    }

                    // Dependencies check button
                    Button(
                        onClick = {
                            scope.launch {
                                checkDependencies(context)
                            }
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = Color.Orange)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text("📦", fontSize = 14.sp)
                            Text("Deps", fontSize = 12.sp)
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Quick troubleshooting tips
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = Color.Black.copy(alpha = 0.8f)
                    ),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Text(
                            text = "🛠️ Quick Troubleshooting",
                            color = Color.White,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "1. Check LogCat for detailed errors\n" +
                                    "2. Ensure simonskina.tflite is in assets/\n" +
                                    "3. Try on physical device (not emulator)\n" +
                                    "4. Update TensorFlow Lite dependencies",
                            color = Color.Gray,
                            fontSize = 10.sp
                        )
                    }
                }
            }
        }
    }

    @Composable
    private fun DebugButton(context: android.content.Context, scope: CoroutineScope) {
        Button(
            onClick = {
                scope.launch {
                    runComprehensiveDebug(context)
                }
            },
            colors = ButtonDefaults.buttonColors(containerColor = Color.Magenta)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text("🔬", fontSize = 14.sp)
                Text("Full Debug", fontSize = 12.sp)
            }
        }
    }

    @Composable
    private fun CameraPreviewWithDetection(
        detector: OptimizedBarbellDetector,
        tracker: EnhancedBarbellTracker?,
        barbellDetections: List<Detection>,
        isProcessing: Boolean,
        fps: Float,
        processingTime: Long,
        averageConfidence: Float,
        detectionHistory: List<Int>,
        isRecording: Boolean,
        isGeneratingReport: Boolean,
        analytics: BarbellAnalytics?,
        onDetectionsUpdated: (List<Detection>) -> Unit,
        onProcessingStateChanged: (Boolean) -> Unit,
        onProcessingTimeUpdated: (Long) -> Unit,
        onFpsUpdated: (Float) -> Unit,
        onCameraError: (String) -> Unit,
        onStartStopRecording: () -> Unit,
        onClearData: () -> Unit,
        onGenerateExcelReport: () -> Unit,
        onGenerateCSVReport: () -> Unit
    ) {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val scope = rememberCoroutineScope()
        val previewView = remember { PreviewView(context) }

        // Camera setup
        LaunchedEffect(previewView) {
            try {
                Log.d("CameraPreview", "Setting up camera with enhanced detection")
                val cameraProvider = ProcessCameraProvider.getInstance(context).get()

                val preview = Preview.Builder()
                    .setTargetResolution(Size(1280, 720))
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                val imageAnalysis = androidx.camera.core.ImageAnalysis.Builder()
                    .setBackpressureStrategy(androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(Size(640, 480))
                    .setOutputImageFormat(androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                    .also { analyzer ->
                        analyzer.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                            if (isProcessing) {
                                imageProxy.close()
                                return@setAnalyzer
                            }

                            onProcessingStateChanged(true)
                            val processingStart = System.currentTimeMillis()

                            scope.launch(Dispatchers.Default) {
                                try {
                                    // Convert to bitmap
                                    val bitmap = BitmapUtils.imageProxyToBitmapOptimized(imageProxy)
                                    val timestamp = System.currentTimeMillis()

                                    // Run detection
                                    val newDetections = detector.detect(bitmap)

                                    // Use enhanced tracker if available
                                    val trackingResult = tracker?.track(bitmap, timestamp)

                                    withContext(Dispatchers.Main) {
                                        onDetectionsUpdated(newDetections)
                                        onProcessingTimeUpdated(System.currentTimeMillis() - processingStart)
                                        onFpsUpdated(0f) // Will be calculated by parent

                                        // Enhanced logging
                                        if (newDetections.isNotEmpty()) {
                                            Log.d("CameraPreview", "🏋️ Found ${newDetections.size} barbells")
                                            newDetections.forEachIndexed { index, detection ->
                                                val quality = detector.getDetectionQuality(detection)
                                                Log.d("CameraPreview", "  Barbell $index: ${String.format("%.1f%%", detection.score * 100)} (${quality.getQualityGrade()})")
                                            }
                                        }
                                    }

                                } catch (e: Exception) {
                                    Log.e("CameraPreview", "Detection error: ${e.message}", e)
                                } finally {
                                    onProcessingStateChanged(false)
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

                Log.d("CameraPreview", "Camera setup complete")

            } catch (e: Exception) {
                val errorMsg = "Camera setup failed: ${e.message}"
                Log.e("CameraPreview", errorMsg, e)
                onCameraError(errorMsg)
            }
        }

        // UI Layout
        Box(modifier = Modifier.fillMaxSize()) {
            // Camera Preview
            AndroidView(
                factory = { previewView },
                modifier = Modifier.fillMaxSize()
            )

            // Detection Overlay
            Canvas(modifier = Modifier.fillMaxSize()) {
                drawBarbellDetections(barbellDetections, detector)
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
                onStartStopRecording = onStartStopRecording,
                onClearData = onClearData,
                onGenerateExcelReport = onGenerateExcelReport,
                onGenerateCSVReport = onGenerateCSVReport,
                modifier = Modifier.align(Alignment.TopStart)
            )
        }
    }

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
                            text = "Processing...",
                            color = Color.Yellow,
                            fontSize = 9.sp,
                            fontStyle = androidx.compose.ui.text.font.FontStyle.Italic
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
            }
        }
    }

    // Drawing function for barbell detections
    private fun DrawScope.drawBarbellDetections(
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

                // Background for text
                val backgroundPaint = android.graphics.Paint().apply {
                    this.color = android.graphics.Color.BLACK
                    alpha = 180
                }

                val textHeight = 20.sp.toPx()
                val textY = maxOf(top - 10f, textHeight)

                drawRect(left - 5f, textY - textHeight, right + 5f, textY + textHeight, backgroundPaint)

                drawText(confidenceText, left + 5f, textY, paint)
                drawText(qualityText, left + 5f, textY + textHeight * 0.7f, paint)
            }
        }
    }

    // Helper function to create test image for optimized detector
    private fun createTestBarbell(): Bitmap {
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

    // Debug functions
    private suspend fun runComprehensiveDebug(context: android.content.Context) {
        withContext(Dispatchers.IO) {
            try {
                Log.d("DEBUG", "🔬 Starting comprehensive debug...")

                // Step 1: Check if model file exists
                Log.d("DEBUG", "📁 Step 1: Checking model file...")
                val modelExists = checkModelExists(context)
                Log.d("DEBUG", "Model exists: $modelExists")

                // Step 2: Run model diagnostic
                Log.d("DEBUG", "🧪 Step 2: Running model diagnostic...")
                val diagnostic = ModelDiagnostic(context)
                val result = diagnostic.runComprehensiveDiagnostic()

                val report = diagnostic.generateReport(result)
                Log.d("DEBUG", "📋 DIAGNOSTIC REPORT:\n$report")

                // Step 3: Test basic TensorFlow Lite functionality
                Log.d("DEBUG", "⚡ Step 3: Testing basic TFLite...")
                testBasicTensorFlowLite(context)

                // Step 4: Test with minimal detector
                Log.d("DEBUG", "🧪 Step 4: Testing minimal detector...")
                testMinimalDetector(context)

                withContext(Dispatchers.Main) {
                    Toast.makeText(context, "Debug complete! Check logs for details.", Toast.LENGTH_LONG).show()
                }

            } catch (e: Exception) {
                Log.e("DEBUG", "❌ Debug failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(context, "Debug failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun checkModelExists(context: android.content.Context): Boolean {
        return try {
            val assetManager = context.assets
            val modelFiles = assetManager.list("") ?: emptyArray()

            Log.d("DEBUG", "📁 Files in assets:")
            modelFiles.forEach { file ->
                Log.d("DEBUG", "  - $file")
            }

            val modelExists = modelFiles.contains("simonskina.tflite")
            if (modelExists) {
                // Check file size
                val fd = assetManager.openFd("simonskina.tflite")
                val fileSize = fd.length
                Log.d("DEBUG", "✅ simonskina.tflite found, size: $fileSize bytes")
                fd.close()
            } else {
                Log.e("DEBUG", "❌ simonskina.tflite not found in assets")
            }

            modelExists
        } catch (e: Exception) {
            Log.e("DEBUG", "❌ Error checking model file: ${e.message}", e)
            false
        }
    }

    private fun testBasicTensorFlowLite(context: android.content.Context) {
        try {
            Log.d("DEBUG", "⚡ Testing basic TensorFlow Lite...")

            // Test 1: Create a simple interpreter with minimal options
            val assetFileDescriptor = context.assets.openFd("simonskina.tflite")
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            Log.d("DEBUG", "✅ Model buffer created: ${modelBuffer.capacity()} bytes")

            // Test 2: Try different interpreter options
            val basicOptions = org.tensorflow.lite.Interpreter.Options()
            Log.d("DEBUG", "🧪 Testing with basic options...")

            val interpreter1 = try {
                org.tensorflow.lite.Interpreter(modelBuffer, basicOptions)
            } catch (e: Exception) {
                Log.e("DEBUG", "❌ Basic interpreter failed: ${e.message}")
                null
            }

            if (interpreter1 != null) {
                Log.d("DEBUG", "✅ Basic interpreter created successfully")
                Log.d("DEBUG", "  Input tensors: ${interpreter1.inputTensorCount}")
                Log.d("DEBUG", "  Output tensors: ${interpreter1.outputTensorCount}")

                // Test input tensor access
                try {
                    val inputTensor = interpreter1.getInputTensor(0)
                    Log.d("DEBUG", "  Input shape: ${inputTensor.shape().contentToString()}")
                    Log.d("DEBUG", "  Input type: ${inputTensor.dataType()}")
                } catch (e: Exception) {
                    Log.e("DEBUG", "❌ Input tensor access failed: ${e.message}")
                }

                interpreter1.close()
            }

            // Test 3: Try with single thread
            val singleThreadOptions = org.tensorflow.lite.Interpreter.Options().apply {
                setNumThreads(1)
            }

            val interpreter2 = try {
                org.tensorflow.lite.Interpreter(modelBuffer, singleThreadOptions)
            } catch (e: Exception) {
                Log.e("DEBUG", "❌ Single thread interpreter failed: ${e.message}")
                null
            }

            if (interpreter2 != null) {
                Log.d("DEBUG", "✅ Single thread interpreter created successfully")
                interpreter2.close()
            }

        } catch (e: Exception) {
            Log.e("DEBUG", "❌ Basic TensorFlow Lite test failed: ${e.message}", e)
        }
    }

    private fun testMinimalDetector(context: android.content.Context) {
        try {
            Log.d("DEBUG", "🧪 Testing minimal detector...")

            val minimalDetector = object {
                private var interpreter: org.tensorflow.lite.Interpreter? = null

                fun initialize(): Boolean {
                    return try {
                        val assetFileDescriptor = context.assets.openFd("simonskina.tflite")
                        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
                        val fileChannel = inputStream.channel
                        val startOffset = assetFileDescriptor.startOffset
                        val declaredLength = assetFileDescriptor.declaredLength
                        val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

                        interpreter = org.tensorflow.lite.Interpreter(modelBuffer)
                        Log.d("DEBUG", "✅ Minimal detector initialized")
                        true
                    } catch (e: Exception) {
                        Log.e("DEBUG", "❌ Minimal detector initialization failed: ${e.message}", e)
                        false
                    }
                }

                fun testInference(): Boolean {
                    return try {
                        val interp = interpreter ?: return false

                        // Get input specs
                        val inputTensor = interp.getInputTensor(0)
                        val inputShape = inputTensor.shape()
                        val inputSize = inputShape.fold(1) { acc, dim -> acc * dim }

                        // Create dummy input
                        val inputBuffer = ByteBuffer.allocateDirect(inputSize * 4).apply {
                            order(ByteOrder.nativeOrder())
                            repeat(inputSize) { putFloat(0.5f) }
                            rewind()
                        }

                        // Create dummy output
                        val outputArray = Array(1) { Array(100) { FloatArray(6) } }

                        // Run inference
                        interp.run(inputBuffer, outputArray)

                        Log.d("DEBUG", "✅ Minimal inference test passed")
                        true
                    } catch (e: Exception) {
                        Log.e("DEBUG", "❌ Minimal inference test failed: ${e.message}", e)
                        false
                    }
                }

                fun cleanup() {
                    interpreter?.close()
                }
            }

            val initSuccess = minimalDetector.initialize()
            if (initSuccess) {
                val inferenceSuccess = minimalDetector.testInference()
                Log.d("DEBUG", "Minimal detector inference: ${if (inferenceSuccess) "✅ SUCCESS" else "❌ FAILED"}")
            }

            minimalDetector.cleanup()

        } catch (e: Exception) {
            Log.e("DEBUG", "❌ Minimal detector test failed: ${e.message}", e)
        }
    }

    private suspend fun checkDependencies(context: android.content.Context) {
        withContext(Dispatchers.IO) {
            try {
                Log.d("DEPS", "📦 Checking TensorFlow Lite dependencies...")

                // Check TensorFlow Lite version
                try {
                    val version = org.tensorflow.lite.TensorFlowLite.schemaVersion()
                    Log.d("DEPS", "✅ TensorFlow Lite schema version: $version")
                } catch (e: Exception) {
                    Log.e("DEPS", "❌ Could not get TF Lite version: ${e.message}")
                }

                // Check if basic TF Lite classes are available
                val classes = listOf(
                    "org.tensorflow.lite.Interpreter",
                    "org.tensorflow.lite.DataType",
                    "org.tensorflow.lite.Tensor"
                )

                classes.forEach { className ->
                    try {
                        Class.forName(className)
                        Log.d("DEPS", "✅ Found: $className")
                    } catch (e: ClassNotFoundException) {
                        Log.e("DEPS", "❌ Missing: $className")
                    } catch (e: Exception) {
                        Log.e("DEPS", "❌ Error loading $className: ${e.message}")
                    }
                }

                // Check Android API level
                val apiLevel = android.os.Build.VERSION.SDK_INT
                Log.d("DEPS", "📱 Android API Level: $apiLevel")

                if (apiLevel < 26) {
                    Log.w("DEPS", "⚠️ API level $apiLevel may have limited TF Lite support")
                } else {
                    Log.d("DEPS", "✅ API level $apiLevel is good for TF Lite")
                }

                // Check device architecture
                val abis = android.os.Build.SUPPORTED_ABIS
                Log.d("DEPS", "🏗️ Supported ABIs: ${abis.joinToString(", ")}")

                // Check available memory
                val runtime = Runtime.getRuntime()
                val maxMemory = runtime.maxMemory() / 1024 / 1024 // MB
                val totalMemory = runtime.totalMemory() / 1024 / 1024 // MB
                val freeMemory = runtime.freeMemory() / 1024 / 1024 // MB

                Log.d("DEPS", "💾 Memory - Max: ${maxMemory}MB, Total: ${totalMemory}MB, Free: ${freeMemory}MB")

                withContext(Dispatchers.Main) {
                    Toast.makeText(context, "Dependencies checked! See logs for details.", Toast.LENGTH_SHORT).show()
                }

            } catch (e: Exception) {
                Log.e("DEPS", "❌ Dependency check failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(context, "Dependency check failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
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
                    text = "Enhanced Barbell Detector",
                    style = MaterialTheme.typography.titleLarge,
                    color = Color.White,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "AI-Powered Barbell Tracking\nCamera permission required",
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
                    text = "Barbell tracking requires camera access. Please grant permission in Settings or try again.",
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