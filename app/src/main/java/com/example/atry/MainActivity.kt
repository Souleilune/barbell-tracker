package com.example.atry

import android.Manifest
import android.content.pm.PackageManager
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
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
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
import com.example.atry.detector.BitmapUtils
import com.example.atry.detector.EnhancedBarbellTracker
import com.example.atry.detector.BarbellAnalytics
import com.example.atry.detector.TrackingResult
import com.example.atry.detector.ReportGenerator
import com.example.atry.detector.PathPoint
import com.example.atry.ui.theme.TryTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "MainActivity onCreate with Generic TFLite Support")

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
                    Log.d(TAG, "Camera permission granted, showing generic camera preview")
                    GenericCameraPreview()
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
    private fun GenericCameraPreview() {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val scope = rememberCoroutineScope()

        Log.d("GenericCamera", "Initializing Generic TFLite barbell tracker")

        // Create enhanced tracker with generic TFLite support
        val tracker = remember {
            try {
                Log.d("GenericCamera", "Creating Enhanced Tracker with Generic TFLite support")
                EnhancedBarbellTracker(
                    context = context,
                    modelPath = "simonskina.tflite", // Your model
                    confThreshold = 0.3f,  // Lower threshold for generic models
                    iouThreshold = 0.5f,
                    maxAge = 30
                )
            } catch (e: Exception) {
                Log.e("GenericCamera", "Failed to create Enhanced Tracker: ${e.message}", e)
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
                        text = "⚠️ Generic Tracker Loading Failed",
                        color = Color.Red,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Please check if simonskina.tflite is in assets folder and is a valid TensorFlow Lite model",
                        color = Color.Gray,
                        fontSize = 14.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(16.dp)
                    )

                    Button(
                        onClick = {
                            // Try to provide debug info
                            try {
                                val inputStream = context.assets.open("simonskina.tflite")
                                val size = inputStream.available()
                                inputStream.close()
                                Toast.makeText(context, "Model found: ${size} bytes", Toast.LENGTH_LONG).show()
                            } catch (e: Exception) {
                                Toast.makeText(context, "Model not found: ${e.message}", Toast.LENGTH_LONG).show()
                            }
                        }
                    ) {
                        Text("Check Model")
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

        // FPS calculation
        var frameCount by remember { mutableStateOf(0) }
        var lastFpsUpdate by remember { mutableStateOf(System.currentTimeMillis()) }

        // Dispose tracker when composable is removed
        DisposableEffect(tracker) {
            onDispose {
                try {
                    tracker.cleanup()
                    Log.d("GenericCamera", "Generic tracker cleaned up successfully")
                } catch (e: Exception) {
                    Log.e("GenericCamera", "Error cleaning up tracker: ${e.message}", e)
                }
            }
        }

        // Generic Camera setup for any TFLite model
        LaunchedEffect(previewView) {
            try {
                Log.d("GenericCamera", "Generic camera setup with TFLite support")
                val cameraProvider = ProcessCameraProvider.getInstance(context).get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val imageAnalysis = androidx.camera.core.ImageAnalysis.Builder()
                    .setBackpressureStrategy(androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(Size(640, 480)) // Good resolution for most models
                    .setOutputImageFormat(androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                    .also { analyzer ->
                        analyzer.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                            if (isProcessing) {
                                imageProxy.close()
                                return@setAnalyzer
                            }

                            isProcessing = true

                            scope.launch(Dispatchers.Default) {
                                try {
                                    val bitmap = BitmapUtils.imageProxyToBitmap(imageProxy)
                                    val timestamp = System.currentTimeMillis()

                                    // Use generic tracking
                                    val newTrackingResult = tracker.track(bitmap, timestamp)

                                    withContext(Dispatchers.Main) {
                                        trackingResult = newTrackingResult

                                        // Generic logging for tracking results
                                        if (newTrackingResult.trackedObjects.isNotEmpty()) {
                                            Log.d("GenericCamera", "Generic tracker found ${newTrackingResult.trackedObjects.size} tracked objects")
                                            newTrackingResult.trackedObjects.forEachIndexed { index, obj ->
                                                Log.d("GenericCamera", "Tracked Object $index: " +
                                                        "ID=${obj.id}, " +
                                                        "conf=${String.format("%.3f", obj.confidence)}, " +
                                                        "center=[${String.format("%.3f", obj.center.first)}, " +
                                                        "${String.format("%.3f", obj.center.second)}]")
                                            }
                                        }

                                        // Update analytics if recording
                                        if (isRecording) {
                                            analytics = tracker.getAnalytics()
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
                                    Log.e("GenericCamera", "Generic tracking error: ${e.message}")
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

                Log.d("GenericCamera", "Generic camera setup complete")
                cameraError = null

            } catch (e: Exception) {
                val errorMsg = "Generic camera setup failed: ${e.message}"
                Log.e("GenericCamera", errorMsg, e)
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
                            text = "📷 Generic Camera Error",
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

                // Generic Detection and Path Overlay
                Canvas(modifier = Modifier.fillMaxSize()) {
                    trackingResult?.let { result ->
                        drawGenericTracking(result)
                    }
                }

                // Generic Info Panel
                GenericTrackingInfoPanel(
                    trackingResult = trackingResult,
                    analytics = analytics,
                    fps = fps,
                    isProcessing = isProcessing,
                    isRecording = isRecording,
                    isGeneratingReport = isGeneratingReport,
                    onStartStopRecording = {
                        isRecording = !isRecording
                        Log.d("GenericCamera", "Generic recording toggled - isRecording: $isRecording")
                        if (isRecording) {
                            tracker.reset() // Clear previous tracking data
                            sessionStartTime = System.currentTimeMillis()
                            Log.d("GenericCamera", "Started generic recording - reset tracker")
                        } else {
                            sessionEndTime = System.currentTimeMillis()
                            analytics = tracker.getAnalytics()
                            Log.d("GenericCamera", "Stopped generic recording")
                        }
                    },
                    onClearTracking = {
                        tracker.reset()
                        analytics = null
                        Log.d("GenericCamera", "Cleared all generic tracking data")
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

                                val result = reportGenerator.generateExcelReport(session,
                                    com.example.atry.detector.BarPathAnalyzer())
                                result.fold(
                                    onSuccess = { file ->
                                        Toast.makeText(context, "Generic Excel report generated: ${file.name}", Toast.LENGTH_LONG).show()
                                        reportGenerator.shareReport(file)
                                    },
                                    onFailure = { error ->
                                        Toast.makeText(context, "Error generating generic Excel report: ${error.message}", Toast.LENGTH_LONG).show()
                                        Log.e("GenericCamera", "Generic Excel report error", error)
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

                                val result = reportGenerator.generateCSVReport(session,
                                    com.example.atry.detector.BarPathAnalyzer())
                                result.fold(
                                    onSuccess = { file ->
                                        Toast.makeText(context, "Generic CSV report generated: ${file.name}", Toast.LENGTH_LONG).show()
                                        reportGenerator.shareReport(file)
                                    },
                                    onFailure = { error ->
                                        Toast.makeText(context, "Error generating generic CSV report: ${error.message}", Toast.LENGTH_LONG).show()
                                        Log.e("GenericCamera", "Generic CSV report error", error)
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

    // Helper function to convert tracking data to BarPath objects
    private fun convertTrackingDataToPaths(trackingData: List<EnhancedBarbellTracker.TrackingDataPoint>): List<com.example.atry.detector.BarPath> {
        val groupedData = trackingData.groupBy { it.id }
        return groupedData.map { (id, points) ->
            val barPath = com.example.atry.detector.BarPath(id = "generic_path_$id")
            points.forEach { dataPoint ->
                barPath.addPoint(PathPoint(dataPoint.x, dataPoint.y, dataPoint.timestamp))
            }
            barPath
        }
    }
}

// Generic drawing function for any TFLite model tracking
private fun DrawScope.drawGenericTracking(result: TrackingResult) {
    val canvasWidth = size.width
    val canvasHeight = size.height

    // Draw tracked objects with generic visualization
    result.trackedObjects.forEach { obj ->
        val bbox = obj.bbox
        val left = bbox.left * canvasWidth
        val top = bbox.top * canvasHeight
        val right = bbox.right * canvasWidth
        val bottom = bbox.bottom * canvasHeight

        // Color based on tracking ID for consistency
        val colors = listOf(
            Color.Cyan, Color.Yellow, Color.Magenta,
            Color.Green, Color.Blue, Color.Red
        )
        val color = colors[obj.id % colors.size]

        // Draw bounding box
        val strokeWidth = 3.dp.toPx()
        drawRect(
            color = color,
            topLeft = Offset(left, top),
            size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
            style = Stroke(width = strokeWidth)
        )

        // Draw tracking ID and confidence
        drawContext.canvas.nativeCanvas.apply {
            val paint = android.graphics.Paint().apply {
                this.color = color.toArgb()
                textSize = 14.sp.toPx()
                isAntiAlias = true
                isFakeBoldText = true
            }

            val label = "ID: ${obj.id}"
            val confText = "Conf: ${String.format("%.2f", obj.confidence)}"

            drawText(label, left + 5f, top - 5f, paint)
            drawText(confText, left + 5f, top - 25f, paint)
        }

        // Draw center point
        val centerX = obj.center.first * canvasWidth
        val centerY = obj.center.second * canvasHeight
        drawCircle(
            color = color,
            radius = 6.dp.toPx(),
            center = Offset(centerX, centerY)
        )
    }

    // Draw bar paths with generic visualization
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

        // Draw path lines
        pathPoints.zipWithNext { a, b ->
            drawLine(
                color = color,
                start = Offset(a.x * canvasWidth, a.y * canvasHeight),
                end = Offset(b.x * canvasWidth, b.y * canvasHeight),
                strokeWidth = 2.dp.toPx()
            )
        }

        // Draw recent points
        pathPoints.takeLast(5).forEach { point ->
            drawCircle(
                color = color,
                radius = 3.dp.toPx(),
                center = Offset(point.x * canvasWidth, point.y * canvasHeight)
            )
        }

        // Highlight the most recent point
        if (pathPoints.isNotEmpty()) {
            val lastPoint = pathPoints.last()
            drawCircle(
                color = Color.White,
                radius = 5.dp.toPx(),
                center = Offset(lastPoint.x * canvasWidth, lastPoint.y * canvasHeight),
                style = Stroke(width = 2.dp.toPx())
            )
        }
    }
}

@Composable
private fun GenericTrackingInfoPanel(
    trackingResult: TrackingResult?,
    analytics: BarbellAnalytics?,
    fps: Float,
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
                text = "Generic TFLite Barbell Tracker",
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
            Text(
                text = "Model: simonskina.tflite",
                color = Color.Cyan,
                fontSize = 12.sp,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(4.dp))

            // Control buttons
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.padding(horizontal = 8.dp)
            ) {
                Button(
                    onClick = onStartStopRecording,
                    modifier = Modifier.height(28.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (isRecording) Color.Red else Color.Green
                    )
                ) {
                    Text(
                        text = if (isRecording) "Stop" else "Start",
                        fontSize = 10.sp,
                        color = Color.White
                    )
                }
                Button(
                    onClick = onClearTracking,
                    modifier = Modifier.height(28.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color.Blue)
                ) {
                    Text("Clear", fontSize = 10.sp, color = Color.White)
                }
            }

            Spacer(modifier = Modifier.height(4.dp))

            // Report generation buttons
            AnimatedVisibility(visible = analytics?.repCount ?: 0 > 0) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.padding(horizontal = 8.dp)
                ) {
                    Button(
                        onClick = onGenerateExcelReport,
                        enabled = !isGeneratingReport && (analytics?.repCount ?: 0) > 0,
                        modifier = Modifier.height(28.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF007ACC),
                            disabledContainerColor = Color.Gray
                        )
                    ) {
                        Text(
                            text = if (isGeneratingReport) "..." else "📊 Excel",
                            fontSize = 10.sp,
                            color = Color.White
                        )
                    }
                    Button(
                        onClick = onGenerateCSVReport,
                        enabled = !isGeneratingReport && (analytics?.repCount ?: 0) > 0,
                        modifier = Modifier.height(28.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF228B22),
                            disabledContainerColor = Color.Gray
                        )
                    ) {
                        Text(
                            text = if (isGeneratingReport) "..." else "📋 CSV",
                            fontSize = 10.sp,
                            color = Color.White
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            // Status info
            Text(
                text = "FPS: ${String.format("%.1f", fps)}",
                color = Color.White,
                fontSize = 11.sp,
                textAlign = TextAlign.Center
            )
            Text(
                text = "Tracked Objects: ${trackingResult?.trackedObjects?.size ?: 0}",
                color = Color.White,
                fontSize = 11.sp,
                textAlign = TextAlign.Center
            )
            Text(
                text = "Active Paths: ${trackingResult?.barPaths?.size ?: 0}",
                color = Color.Cyan,
                fontSize = 11.sp,
                textAlign = TextAlign.Center
            )
            Text(
                text = "Recording: ${if (isRecording) "ON" else "OFF"}",
                color = if (isRecording) Color.Green else Color.Gray,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )

            // Analytics display
            analytics?.let { stats ->
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Reps: ${stats.repCount}",
                    color = Color.Cyan,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
                if (stats.totalDistance > 0) {
                    Text(
                        text = "Distance: ${String.format("%.2f", stats.totalDistance)}",
                        color = Color.Yellow,
                        fontSize = 11.sp,
                        textAlign = TextAlign.Center
                    )
                    Text(
                        text = "Avg Speed: ${String.format("%.2f", stats.averageVelocity)}",
                        color = Color.Yellow,
                        fontSize = 11.sp,
                        textAlign = TextAlign.Center
                    )
                    Text(
                        text = "Consistency: ${String.format("%.1f%%", stats.pathConsistency * 100)}",
                        color = Color.Green,
                        fontSize = 11.sp,
                        textAlign = TextAlign.Center
                    )
                }
                stats.primaryTrackingId?.let { id ->
                    Text(
                        text = "Primary ID: $id",
                        color = Color.Magenta,
                        fontSize = 10.sp,
                        textAlign = TextAlign.Center
                    )
                }
            }

            if (isGeneratingReport) {
                Text(
                    text = "📄 Generating Generic Report...",
                    color = Color.Yellow,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
            }

            // Display tracked objects info
            trackingResult?.trackedObjects?.take(2)?.forEachIndexed { index, obj ->
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "ID ${obj.id}: ${String.format("%.2f", obj.confidence)}",
                    color = Color.Cyan,
                    fontSize = 9.sp,
                    textAlign = TextAlign.Center
                )
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
                text = "Generic TFLite tracking needs camera access",
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
                text = "Generic tracking requires camera permission. Please grant permission in Settings or try again.",
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