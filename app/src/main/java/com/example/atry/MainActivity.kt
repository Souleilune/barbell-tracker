package com.example.atry

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Bundle
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

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "MainActivity onCreate with Hybrid Classification Support")

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
                Toast.makeText(context, "Camera permission is required for this app", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(context, "Camera permission granted", Toast.LENGTH_SHORT).show()
            }
        }

        // Request permission on first launch
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
                    Log.d(TAG, "Camera permission granted, showing hybrid camera preview")
                    HybridCameraPreview()
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
    private fun HybridCameraPreview() {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val scope = rememberCoroutineScope()

        Log.d("HybridCamera", "Initializing Hybrid Classification Barbell Tracker")

        // Create enhanced tracker with hybrid classification support
        val tracker = remember {
            try {
                Log.d("HybridCamera", "Creating Enhanced Tracker with Hybrid Classification")
                EnhancedBarbellTracker(
                    context = context,
                    modelPath = "simonskina.tflite",
                    confThreshold = 0.05f,  // Very low threshold for classification
                    iouThreshold = 0.3f,    // Lower IoU for motion areas
                    maxAge = 40             // Longer tracking for stability
                )
            } catch (e: Exception) {
                Log.e("HybridCamera", "Failed to create Hybrid Tracker: ${e.message}", e)
                null
            }
        }

        // Create report generator
        val reportGenerator = remember { ReportGenerator(context) }

        if (tracker == null) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "⚠️ Hybrid Tracker Loading Failed",
                        color = Color.Red,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Classification Model Debug:\n• Check simonskina.tflite in assets\n• Verify model outputs [1, 25] shape\n• Model should be classification format",
                        color = Color.Gray,
                        fontSize = 14.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(16.dp)
                    )

                    Button(
                        onClick = {
                            scope.launch {
                                try {
                                    // Test hybrid detector directly
                                    val hybridDetector = HybridClassificationDetector(context)

                                    // Create test bitmap
                                    val testBitmap = createTestClassificationImage()
                                    val detections = hybridDetector.detect(testBitmap)

                                    Toast.makeText(context,
                                        "✅ Hybrid test: ${detections.size} detections",
                                        Toast.LENGTH_LONG).show()

                                    Log.d("HybridCamera", "Hybrid test detections: ${detections.size}")
                                    detections.forEach { detection ->
                                        Log.d("HybridCamera", "Detection: conf=${detection.score}, bbox=${detection.bbox}")
                                    }

                                    hybridDetector.cleanup()
                                } catch (e: Exception) {
                                    Toast.makeText(context, "❌ Hybrid test failed: ${e.message}", Toast.LENGTH_LONG).show()
                                    Log.e("HybridCamera", "Hybrid test error", e)
                                }
                            }
                        }
                    ) {
                        Text("Test Hybrid Model")
                    }
                }
            }
            return
        }

        // PreviewView instance
        val previewView = remember { PreviewView(context) }

        // Enhanced state variables
        var trackingResult by remember { mutableStateOf<TrackingResult?>(null) }
        var isProcessing by remember { mutableStateOf(false) }
        var fps by remember { mutableStateOf(0f) }
        var cameraError by remember { mutableStateOf<String?>(null) }
        var analytics by remember { mutableStateOf<BarbellAnalytics?>(null) }

        // Session management
        var isRecording by remember { mutableStateOf(false) }
        var sessionStartTime by remember { mutableStateOf(0L) }
        var sessionEndTime by remember { mutableStateOf(0L) }
        var isGeneratingReport by remember { mutableStateOf(false) }

        // Performance tracking
        var frameCount by remember { mutableStateOf(0) }
        var lastFpsUpdate by remember { mutableStateOf(System.currentTimeMillis()) }
        var processingTime by remember { mutableStateOf(0L) }

        // Dispose tracker when composable is removed
        DisposableEffect(tracker) {
            onDispose {
                try {
                    tracker.cleanup()
                    Log.d("HybridCamera", "Hybrid tracker cleaned up successfully")
                } catch (e: Exception) {
                    Log.e("HybridCamera", "Error cleaning up hybrid tracker: ${e.message}", e)
                }
            }
        }

        // Hybrid Camera setup optimized for classification + motion
        LaunchedEffect(previewView) {
            try {
                Log.d("HybridCamera", "Hybrid camera setup with classification + motion detection")
                val cameraProvider = ProcessCameraProvider.getInstance(context).get()

                val preview = Preview.Builder()
                    .setTargetResolution(Size(1280, 720)) // Good quality for motion detection
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                val imageAnalysis = androidx.camera.core.ImageAnalysis.Builder()
                    .setBackpressureStrategy(androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(Size(640, 480)) // Balanced for hybrid processing
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
                                    // Use optimized bitmap conversion
                                    val bitmap = BitmapUtils.imageProxyToBitmapOptimized(imageProxy)
                                    val timestamp = System.currentTimeMillis()

                                    // Hybrid tracking (classification + motion)
                                    val newTrackingResult = tracker.track(bitmap, timestamp)

                                    withContext(Dispatchers.Main) {
                                        trackingResult = newTrackingResult
                                        processingTime = System.currentTimeMillis() - processingStart

                                        // Enhanced logging for hybrid tracking
                                        if (newTrackingResult.trackedObjects.isNotEmpty()) {
                                            Log.d("HybridCamera", "✅ Hybrid tracker found ${newTrackingResult.trackedObjects.size} objects")

                                            newTrackingResult.trackedObjects.forEach { obj ->
                                                Log.d("HybridCamera", "  Hybrid Object ${obj.id}: " +
                                                        "conf=${String.format("%.3f", obj.confidence)}, " +
                                                        "center=[${String.format("%.3f", obj.center.first)}, ${String.format("%.3f", obj.center.second)}], " +
                                                        "path_length=${obj.path.size}")
                                            }
                                        } else if (frameCount % 30 == 0) { // Log every 30 frames when no detection
                                            Log.d("HybridCamera", "No hybrid objects detected (frame $frameCount)")
                                        }

                                        // Update analytics if recording
                                        if (isRecording) {
                                            analytics = tracker.getAnalytics()
                                            analytics?.let { stats ->
                                                if (stats.repCount > 0) {
                                                    Log.d("HybridCamera", "📊 Hybrid Analytics: reps=${stats.repCount}, " +
                                                            "distance=${String.format("%.2f", stats.totalDistance)}, " +
                                                            "consistency=${String.format("%.1f%%", stats.pathConsistency * 100)}")
                                                }
                                            }
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
                                    Log.e("HybridCamera", "Hybrid tracking error: ${e.message}", e)
                                    withContext(Dispatchers.Main) {
                                        processingTime = System.currentTimeMillis() - processingStart
                                    }
                                } finally {
                                    isProcessing = false
                                    imageProxy.close()
                                }
                            }
                        }
                    }

                // Bind camera
                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageAnalysis
                    )

                    Log.d("HybridCamera", "✅ Hybrid camera setup complete")
                    cameraError = null

                } catch (bindException: Exception) {
                    val errorMsg = "Hybrid camera binding failed: ${bindException.message}"
                    Log.e("HybridCamera", errorMsg, bindException)
                    cameraError = errorMsg
                }

            } catch (e: Exception) {
                val errorMsg = "Hybrid camera setup failed: ${e.message}"
                Log.e("HybridCamera", errorMsg, e)
                cameraError = errorMsg
            }
        }

        // UI Layout
        Box(modifier = Modifier.fillMaxSize()) {
            if (cameraError != null) {
                // Error state
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "📷 Hybrid Camera Error",
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

                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { cameraError = null }) {
                                Text("Retry")
                            }
                            Button(onClick = {
                                cameraError = null
                                tracker.reset()
                                analytics = null
                            }) {
                                Text("Reset")
                            }
                        }
                    }
                }
            } else {
                // Camera Preview
                AndroidView(
                    factory = { previewView },
                    modifier = Modifier.fillMaxSize()
                )

                // Hybrid Detection Overlay
                Canvas(modifier = Modifier.fillMaxSize()) {
                    trackingResult?.let { result ->
                        drawHybridTracking(result)
                    }
                }

                // Hybrid Info Panel
                HybridTrackingInfoPanel(
                    trackingResult = trackingResult,
                    analytics = analytics,
                    fps = fps,
                    processingTime = processingTime,
                    isProcessing = isProcessing,
                    isRecording = isRecording,
                    isGeneratingReport = isGeneratingReport,
                    onStartStopRecording = {
                        isRecording = !isRecording
                        Log.d("HybridCamera", "Hybrid recording toggled - isRecording: $isRecording")
                        if (isRecording) {
                            tracker.reset()
                            sessionStartTime = System.currentTimeMillis()
                            Log.d("HybridCamera", "Started hybrid recording - reset tracker")
                        } else {
                            sessionEndTime = System.currentTimeMillis()
                            analytics = tracker.getAnalytics()
                            Log.d("HybridCamera", "Stopped hybrid recording - final analytics: ${analytics?.repCount} reps")
                        }
                    },
                    onClearTracking = {
                        tracker.reset()
                        analytics = null
                        Log.d("HybridCamera", "Cleared all hybrid tracking data")
                    },
                    onGenerateExcelReport = {
                        scope.launch {
                            isGeneratingReport = true
                            try {
                                val trackingData = tracker.getTrackingData()
                                val session = ReportGenerator.WorkoutSession(
                                    startTime = sessionStartTime,
                                    endTime = if (sessionEndTime > 0) sessionEndTime else System.currentTimeMillis(),
                                    actualRepCount = analytics?.repCount ?: 0,
                                    paths = convertTrackingDataToPaths(trackingData),
                                    movements = emptyList()
                                )

                                val result = reportGenerator.generateExcelReport(session, BarPathAnalyzer())
                                result.fold(
                                    onSuccess = { file ->
                                        Toast.makeText(context, "✅ Hybrid Excel report: ${file.name}", Toast.LENGTH_LONG).show()
                                        reportGenerator.shareReport(file)
                                    },
                                    onFailure = { error ->
                                        Toast.makeText(context, "❌ Excel error: ${error.message}", Toast.LENGTH_LONG).show()
                                        Log.e("HybridCamera", "Excel report error", error)
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
                                val trackingData = tracker.getTrackingData()
                                val session = ReportGenerator.WorkoutSession(
                                    startTime = sessionStartTime,
                                    endTime = if (sessionEndTime > 0) sessionEndTime else System.currentTimeMillis(),
                                    actualRepCount = analytics?.repCount ?: 0,
                                    paths = convertTrackingDataToPaths(trackingData),
                                    movements = emptyList()
                                )

                                val result = reportGenerator.generateCSVReport(session, BarPathAnalyzer())
                                result.fold(
                                    onSuccess = { file ->
                                        Toast.makeText(context, "✅ Hybrid CSV report: ${file.name}", Toast.LENGTH_LONG).show()
                                        reportGenerator.shareReport(file)
                                    },
                                    onFailure = { error ->
                                        Toast.makeText(context, "❌ CSV error: ${error.message}", Toast.LENGTH_LONG).show()
                                        Log.e("HybridCamera", "CSV report error", error)
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

    @Composable
    private fun HybridTrackingInfoPanel(
        trackingResult: TrackingResult?,
        analytics: BarbellAnalytics?,
        fps: Float,
        processingTime: Long,
        isProcessing: Boolean,
        isRecording: Boolean,
        isGeneratingReport: Boolean,
        onStartStopRecording: () -> Unit,
        onClearTracking: () -> Unit,
        onGenerateExcelReport: () -> Unit,
        onGenerateCSVReport: () -> Unit,
        modifier: Modifier = Modifier
    ) {
        Box(
            modifier = modifier
                .fillMaxWidth()
                .padding(top = 60.dp)
        ) {
            Column(
                modifier = Modifier.align(Alignment.TopCenter),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "🧠 Hybrid Classification + Motion Tracker",
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
                    text = "Model: simonskina.tflite (Classification + Motion)",
                    color = Color.Cyan,
                    fontSize = 12.sp,
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .background(
                            Color.Black.copy(alpha = 0.6f),
                            shape = RoundedCornerShape(6.dp)
                        )
                        .padding(4.dp)
                )

                Spacer(modifier = Modifier.height(6.dp))

                // Control buttons
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.padding(horizontal = 8.dp)
                ) {
                    Button(
                        onClick = onStartStopRecording,
                        modifier = Modifier.height(32.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isRecording) Color.Red else Color.Green
                        ),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(
                            text = if (isRecording) "⏹ Stop" else "▶ Start",
                            fontSize = 11.sp,
                            color = Color.White,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Button(
                        onClick = onClearTracking,
                        modifier = Modifier.height(32.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color.Blue),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("🧹 Clear", fontSize = 11.sp, color = Color.White, fontWeight = FontWeight.Bold)
                    }
                }

                Spacer(modifier = Modifier.height(6.dp))

                // Report generation buttons
                AnimatedVisibility(visible = analytics?.repCount ?: 0 > 0) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.padding(horizontal = 8.dp)
                    ) {
                        Button(
                            onClick = onGenerateExcelReport,
                            enabled = !isGeneratingReport && (analytics?.repCount ?: 0) > 0,
                            modifier = Modifier.height(32.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF007ACC),
                                disabledContainerColor = Color.Gray
                            ),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text(
                                text = if (isGeneratingReport) "⏳ ..." else "📊 Excel",
                                fontSize = 11.sp,
                                color = Color.White,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        Button(
                            onClick = onGenerateCSVReport,
                            enabled = !isGeneratingReport && (analytics?.repCount ?: 0) > 0,
                            modifier = Modifier.height(32.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF228B22),
                                disabledContainerColor = Color.Gray
                            ),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text(
                                text = if (isGeneratingReport) "⏳ ..." else "📋 CSV",
                                fontSize = 11.sp,
                                color = Color.White,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Performance and status info with enhanced styling
                Card(
                    modifier = Modifier
                        .padding(horizontal = 8.dp)
                        .background(
                            Color.Black.copy(alpha = 0.8f),
                            shape = RoundedCornerShape(12.dp)
                        ),
                    colors = CardDefaults.cardColors(
                        containerColor = Color.Transparent
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        // Performance metrics
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            Text(
                                text = "FPS: ${String.format("%.1f", fps)}",
                                color = if (fps > 20) Color.Green else if (fps > 10) Color.Yellow else Color.Red,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = "Process: ${processingTime}ms",
                                color = if (processingTime < 100) Color.Green else if (processingTime < 200) Color.Yellow else Color.Red,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Spacer(modifier = Modifier.height(4.dp))

                        // Tracking status
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            Text(
                                text = "Objects: ${trackingResult?.trackedObjects?.size ?: 0}",
                                color = Color.Cyan,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = "Paths: ${trackingResult?.barPaths?.size ?: 0}",
                                color = Color.Magenta,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Spacer(modifier = Modifier.height(4.dp))

                        // Recording status
                        Text(
                            text = "🔴 Recording: ${if (isRecording) "ACTIVE" else "STOPPED"}",
                            color = if (isRecording) Color.Red else Color.Gray,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = "🤖 Processing: ${if (isProcessing) "ACTIVE" else "IDLE"}",
                            color = if (isProcessing) Color.Yellow else Color.Green,
                            fontSize = 11.sp
                        )
                    }
                }

                // Analytics display with enhanced visualization
                analytics?.let { stats ->
                    Spacer(modifier = Modifier.height(8.dp))

                    Card(
                        modifier = Modifier
                            .padding(horizontal = 8.dp)
                            .background(
                                Color(0xFF1E3A8A).copy(alpha = 0.9f),
                                shape = RoundedCornerShape(12.dp)
                            ),
                        colors = CardDefaults.cardColors(
                            containerColor = Color.Transparent
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            // Main rep counter
                            Text(
                                text = "🏋️ REPS: ${stats.repCount}",
                                color = Color.White,
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold
                            )

                            if (stats.totalDistance > 0) {
                                Spacer(modifier = Modifier.height(6.dp))

                                // Detailed metrics
                                Row(
                                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                        Text(
                                            text = "Distance",
                                            color = Color.Yellow,
                                            fontSize = 10.sp
                                        )
                                        Text(
                                            text = String.format("%.2f", stats.totalDistance),
                                            color = Color.White,
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }

                                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                        Text(
                                            text = "Avg Speed",
                                            color = Color.Yellow,
                                            fontSize = 10.sp
                                        )
                                        Text(
                                            text = String.format("%.2f", stats.averageVelocity),
                                            color = Color.White,
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }

                                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                        Text(
                                            text = "Consistency",
                                            color = Color.Yellow,
                                            fontSize = 10.sp
                                        )
                                        Text(
                                            text = "${String.format("%.0f", stats.pathConsistency * 100)}%",
                                            color = when {
                                                stats.pathConsistency > 0.8f -> Color.Green
                                                stats.pathConsistency > 0.6f -> Color.Yellow
                                                else -> Color.Red
                                            },
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            }

                            stats.primaryTrackingId?.let { id ->
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "Primary Tracker ID: $id",
                                    color = Color.Cyan,
                                    fontSize = 10.sp
                                )
                            }
                        }
                    }
                }

                // Report generation status
                if (isGeneratingReport) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Card(
                        modifier = Modifier
                            .padding(horizontal = 8.dp)
                            .background(
                                Color(0xFF8B4513).copy(alpha = 0.9f),
                                shape = RoundedCornerShape(12.dp)
                            ),
                        colors = CardDefaults.cardColors(
                            containerColor = Color.Transparent
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                text = "📄 Generating Hybrid Report...",
                                color = Color.White,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = "Classification + Motion Data",
                                color = Color.Yellow,
                                fontSize = 10.sp
                            )
                        }
                    }
                }

                // Individual tracked objects info (for debugging)
                trackingResult?.trackedObjects?.take(2)?.forEachIndexed { index, obj ->
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "🎯 ID ${obj.id}: conf=${String.format("%.2f", obj.confidence)}, path=${obj.path.size}",
                        color = Color.White,
                        fontSize = 9.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier
                            .background(
                                Color.Black.copy(alpha = 0.6f),
                                shape = RoundedCornerShape(4.dp)
                            )
                            .padding(4.dp)
                    )
                }

                // Model-specific info
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "🔬 Hybrid: Classification detects presence + Motion tracks movement",
                    color = Color.Gray,
                    fontSize = 9.sp,
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

    // Helper function to create test image for classification
    private fun createTestClassificationImage(): Bitmap {
        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        // Dark background
        canvas.drawColor(android.graphics.Color.DKGRAY)

        // Simple barbell shape for classification test
        val paint = android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            style = android.graphics.Paint.Style.FILL
            isAntiAlias = true
        }

        // Horizontal bar
        canvas.drawRect(50f, 110f, 174f, 114f, paint)

        // Left weight plate
        canvas.drawCircle(50f, 112f, 15f, paint)
        canvas.drawRect(35f, 97f, 50f, 127f, paint)

        // Right weight plate
        canvas.drawCircle(174f, 112f, 15f, paint)
        canvas.drawRect(174f, 97f, 189f, 127f, paint)

        return bitmap
    }

    // Drawing function for hybrid tracking visualization
    private fun DrawScope.drawHybridTracking(result: TrackingResult) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        // Draw tracked objects with hybrid visualization
        result.trackedObjects.forEach { obj ->
            val bbox = obj.bbox
            val left = bbox.left * canvasWidth
            val top = bbox.top * canvasHeight
            val right = bbox.right * canvasWidth
            val bottom = bbox.bottom * canvasHeight

            // Color based on tracking ID
            val colors = listOf(
                Color.Cyan, Color.Yellow, Color.Magenta,
                Color.Green, Color.Blue, Color.Red
            )
            val color = colors[obj.id % colors.size]

            // Draw bounding box with thicker stroke for better visibility
            val strokeWidth = 4.dp.toPx()
            drawRect(
                color = color,
                topLeft = Offset(left, top),
                size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                style = Stroke(width = strokeWidth)
            )

            // Draw tracking ID and confidence with better visibility
            drawContext.canvas.nativeCanvas.apply {
                val paint = android.graphics.Paint().apply {
                    this.color = color.toArgb()
                    textSize = 16.sp.toPx()
                    isAntiAlias = true
                    isFakeBoldText = true
                    setShadowLayer(2f, 1f, 1f, android.graphics.Color.BLACK)
                }

                val label = "ID: ${obj.id}"
                val confText = "Hybrid: ${String.format("%.2f", obj.confidence)}"
                val pathText = "Path: ${obj.path.size}"

                drawText(label, left + 8f, top - 8f, paint)
                drawText(confText, left + 8f, top - 28f, paint)
                drawText(pathText, left + 8f, top - 48f, paint)
            }

            // Draw center point
            val centerX = obj.center.first * canvasWidth
            val centerY = obj.center.second * canvasHeight
            drawCircle(
                color = color,
                radius = 8.dp.toPx(),
                center = Offset(centerX, centerY)
            )
        }

        // Draw motion paths with enhanced visualization
        result.barPaths.forEach { (trackingId, pathPoints) ->
            if (pathPoints.size < 2) return@forEach

            val color = when (trackingId % 6) {
                0 -> Color.Cyan
                1 -> Color.Yellow
                2 -> Color.Magenta
                3 -> Color.Green
                4 -> Color.Blue
                else -> Color.Red
            }

            // Draw path lines with gradient effect
            pathPoints.zipWithNext { a, b ->
                drawLine(
                    color = color.copy(alpha = 0.8f),
                    start = Offset(a.x * canvasWidth, a.y * canvasHeight),
                    end = Offset(b.x * canvasWidth, b.y * canvasHeight),
                    strokeWidth = 3.dp.toPx()
                )
            }

            // Draw recent points with decreasing size
            pathPoints.takeLast(10).forEachIndexed { index, point ->
                val alpha = (index + 1) / 10f
                val radius = (3 + index * 0.5f).dp.toPx()
                drawCircle(
                    color = color.copy(alpha = alpha),
                    radius = radius,
                    center = Offset(point.x * canvasWidth, point.y * canvasHeight)
                )
            }

            // Highlight the most recent point
            if (pathPoints.isNotEmpty()) {
                val lastPoint = pathPoints.last()
                drawCircle(
                    color = Color.White,
                    radius = 6.dp.toPx(),
                    center = Offset(lastPoint.x * canvasWidth, lastPoint.y * canvasHeight),
                    style = Stroke(width = 2.dp.toPx())
                )
            }
        }
    }

    // Helper function to convert tracking data to BarPath objects
    private fun convertTrackingDataToPaths(trackingData: List<EnhancedBarbellTracker.TrackingDataPoint>): List<BarPath> {
        val groupedData = trackingData.groupBy { it.id }
        return groupedData.map { (id, points) ->
            val barPath = BarPath(id = "hybrid_path_$id")
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
                    text = "📱",
                    fontSize = 48.sp,
                    color = Color.White
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Camera Permission Required",
                    style = MaterialTheme.typography.titleLarge,
                    color = Color.White,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Hybrid TFLite tracking needs camera access",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.Gray,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(24.dp))

                Button(
                    onClick = onRequest,
                    modifier = Modifier
                        .size(pulse)
                        .clip(RoundedCornerShape(16.dp))
                ) {
                    Text(
                        text = "Grant Permission",
                        color = Color.White,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }

    @Composable
    private fun PermissionDeniedScreen(onRetry: () -> Unit) {
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
                    text = "Hybrid tracking requires camera permission. Please grant permission in Settings or try again.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.Gray,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(horizontal = 32.dp)
                )
                Spacer(modifier = Modifier.height(24.dp))

                Button(
                    onClick = onRetry,
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Try Again",
                        color = Color.White,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}