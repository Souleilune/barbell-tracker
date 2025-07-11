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
import androidx.compose.ui.geometry.Size as ComposeSize
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
import com.example.atry.detector.Detection
import com.example.atry.detector.YOLOv8ObjectDetector
import com.example.atry.detector.BarPathAnalyzer
import com.example.atry.detector.ReportGenerator
import com.example.atry.detector.PathPoint
import com.example.atry.detector.BarPath
import com.example.atry.detector.MovementDirection
import com.example.atry.detector.MovementAnalysis
import com.example.atry.ui.theme.TryTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "MainActivity onCreate started")

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

        // Track permission state more robustly
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

        // Enhanced permission launcher with better error handling
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
                delay(300) // Small delay to ensure UI is ready
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        Box(modifier = Modifier.fillMaxSize()) {
            when {
                hasCameraPermission -> {
                    Log.d(TAG, "Camera permission granted, showing camera preview")
                    CameraPreviewWithYOLOv11()
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
                text = "This app needs camera access to detect barbells",
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
                text = "Please grant camera permission in Settings or tap below to try again",
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

@Composable
private fun CameraPreviewWithYOLOv11() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    Log.d("CameraPreview", "Initializing camera preview")

    // Create detector with FIXED parameters (removed iouThreshold)
    val detector = remember {
        try {
            Log.d("CameraPreview", "Creating YOLOv8 detector based on working YOLOv11 logic")
            YOLOv8ObjectDetector(
                context = context,
                modelPath = "yolofinal.tflite", // Your trained YOLOv8 model
                confThreshold = 0.1f,  // Start with low threshold for debugging
                iouThreshold = 0.3f,   // Same as working YOLOv11
                inputSize = 320        // Match your training size
            )
        } catch (e: Exception) {
            Log.e("CameraPreview", "Failed to create YOLOv8 detector: ${e.message}", e)
            Toast.makeText(context, "Failed to load YOLOv8 model: ${e.message}", Toast.LENGTH_LONG).show()
            null
        }
    }
    // Create analyzer and report generator
    val analyzer = remember { BarPathAnalyzer() }
    val reportGenerator = remember { ReportGenerator(context) }

    // Early return if detector creation failed
    if (detector == null) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "⚠️ Model Loading Failed",
                    color = Color.Red,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Check if new model.tflite is in assets folder",
                    color = Color.Gray,
                    fontSize = 14.sp,
                    textAlign = TextAlign.Center
                )
            }
        }
        return
    }

    // PreviewView instance
    val previewView = remember { PreviewView(context) }

    // State variables
    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) }
    var isProcessing by remember { mutableStateOf(false) }
    var fps by remember { mutableStateOf(0f) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    // Enhanced bar path tracking state with session management
    var barPaths by remember { mutableStateOf<List<BarPath>>(listOf(BarPath())) }
    var currentMovement by remember { mutableStateOf<MovementAnalysis?>(null) }
    var repCount by remember { mutableStateOf(0) }
    var isRecording by remember { mutableStateOf(false) }

    // Session management state
    var sessionStartTime by remember { mutableStateOf(0L) }
    var sessionEndTime by remember { mutableStateOf(0L) }
    var allMovements by remember { mutableStateOf<List<MovementAnalysis>>(emptyList()) }
    var isGeneratingReport by remember { mutableStateOf(false) }

    // FPS calculation
    var frameCount by remember { mutableStateOf(0) }
    var lastFpsUpdate by remember { mutableStateOf(System.currentTimeMillis()) }

    // Performance optimization variables
    var frameSkipCounter by remember { mutableStateOf(0) }

    // Dispose detector when composable is removed
    DisposableEffect(detector) {
        onDispose {
            try {
                detector.cleanup()
                detector.close()
                Log.d("CameraPreview", "Detector closed successfully")
            } catch (e: Exception) {
                Log.e("CameraPreview", "Error closing detector: ${e.message}", e)
            }
        }
    }

    // EXTREME Camera setup for Redmi 13C
    LaunchedEffect(previewView) {
        try {
            Log.d("CameraPreview", "YOLOv8 camera setup for Redmi 13C")
            val cameraProvider = ProcessCameraProvider.getInstance(context).get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalysis = androidx.camera.core.ImageAnalysis.Builder()
                .setBackpressureStrategy(androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                // YOLOv8 optimized resolution
                .setTargetResolution(Size(480, 320)) // 3:2 aspect ratio works well with YOLOv8
                .setOutputImageFormat(androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also { analyzer ->
                    analyzer.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                        // YOLOv8 frame skipping - every 2nd frame (YOLOv8 is more efficient)
                        frameSkipCounter++
                        if (frameSkipCounter % 2 != 0) {
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        if (isProcessing) {
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        isProcessing = true

                        scope.launch(Dispatchers.Default) {
                            try {
                                val bitmap = BitmapUtils.imageProxyToBitmap(imageProxy)
                                val newDetections = detector.detect(bitmap)

                                withContext(Dispatchers.Main) {
                                    detections = newDetections

                                    // Enhanced logging for YOLOv8
                                    if (newDetections.isNotEmpty()) {
                                        Log.d("CameraPreview", "YOLOv8 found ${newDetections.size} detections")
                                        newDetections.forEachIndexed { index, detection ->
                                            Log.d("CameraPreview", "Detection $index: " +
                                                    "conf=${String.format("%.3f", detection.score)}, " +
                                                    "class=${detector.getClassLabel(detection.classId)}, " +
                                                    "bbox=[${String.format("%.3f", detection.bbox.left)}, " +
                                                    "${String.format("%.3f", detection.bbox.top)}, " +
                                                    "${String.format("%.3f", detection.bbox.right)}, " +
                                                    "${String.format("%.3f", detection.bbox.bottom)}]")
                                        }
                                    }
                                }

                                // YOLOv8 optimized bar path processing
                                if (isRecording && newDetections.isNotEmpty()) {
                                    val updatedData = processBarPathYOLOv8(
                                        detections = newDetections,
                                        currentPaths = barPaths
                                    )

                                    withContext(Dispatchers.Main) {
                                        barPaths = updatedData.paths
                                        repCount = updatedData.repCount
                                    }
                                }

                                // FPS calculation
                                frameCount++
                                val currentTime = System.currentTimeMillis()
                                if (currentTime - lastFpsUpdate >= 1500) { // Update every 1.5 seconds
                                    val newFps = frameCount * 1000f / (currentTime - lastFpsUpdate)
                                    withContext(Dispatchers.Main) {
                                        fps = newFps
                                    }
                                    frameCount = 0
                                    lastFpsUpdate = currentTime
                                }

                            } catch (e: Exception) {
                                Log.e("CameraPreview", "YOLOv8 processing error: ${e.message}")
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

            Log.d("CameraPreview", "YOLOv8 camera setup complete")
            cameraError = null

        } catch (e: Exception) {
            val errorMsg = "YOLOv8 camera setup failed: ${e.message}"
            Log.e("CameraPreview", errorMsg, e)
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
                        text = "📷 Camera Error",
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
                        onClick = {
                            cameraError = null
                        }
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

            // Detection and Path Overlay
            Canvas(modifier = Modifier.fillMaxSize()) {
                // Use YOLOv8 optimized drawing
                drawDetectionsYOLOv8(detections, detector)
                drawBarPathsMinimal(barPaths)
            }

            // Enhanced Info Panel
            EnhancedInfoPanel(
                detections = detections,
                fps = fps,
                isProcessing = isProcessing,
                currentMovement = currentMovement,
                repCount = repCount,
                isRecording = isRecording,
                isGeneratingReport = isGeneratingReport,
                isUsingGPU = false,
                performanceInfo = detector.getPerformanceInfo(),
                onStartStopRecording = {
                    isRecording = !isRecording
                    Log.d("CameraPreview", "Recording toggled - isRecording: $isRecording")
                    if (isRecording) {
                        // Start fresh when recording starts
                        barPaths = listOf(BarPath())
                        repCount = 0
                        allMovements = emptyList()
                        sessionStartTime = System.currentTimeMillis()
                        Log.d("CameraPreview", "Started recording - cleared paths")
                    } else {
                        sessionEndTime = System.currentTimeMillis()
                        Log.d("CameraPreview", "Stopped recording")
                    }
                },
                onClearPath = {
                    barPaths = listOf(BarPath())
                    repCount = 0
                    currentMovement = null
                    allMovements = emptyList()
                    Log.d("CameraPreview", "Cleared all paths")
                },
                onGenerateExcelReport = {
                    scope.launch {
                        isGeneratingReport = true
                        try {
                            val session = ReportGenerator.WorkoutSession(
                                startTime = sessionStartTime,
                                endTime = if (sessionEndTime > 0) sessionEndTime else System.currentTimeMillis(),
                                actualRepCount = repCount,
                                paths = barPaths,
                                movements = allMovements
                            )

                            val result = reportGenerator.generateExcelReport(session, analyzer)
                            result.fold(
                                onSuccess = { file ->
                                    Toast.makeText(context, "Excel report generated: ${file.name}", Toast.LENGTH_LONG).show()
                                    reportGenerator.shareReport(file)
                                },
                                onFailure = { error ->
                                    Toast.makeText(context, "Error generating Excel report: ${error.message}", Toast.LENGTH_LONG).show()
                                    Log.e("CameraPreview", "Excel report error", error)
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
                            val session = ReportGenerator.WorkoutSession(
                                startTime = sessionStartTime,
                                endTime = if (sessionEndTime > 0) sessionEndTime else System.currentTimeMillis(),
                                actualRepCount = repCount,
                                paths = barPaths,
                                movements = allMovements
                            )

                            val result = reportGenerator.generateCSVReport(session, analyzer)
                            result.fold(
                                onSuccess = { file ->
                                    Toast.makeText(context, "CSV report generated: ${file.name}", Toast.LENGTH_LONG).show()
                                    reportGenerator.shareReport(file)
                                },
                                onFailure = { error ->
                                    Toast.makeText(context, "Error generating CSV report: ${error.message}", Toast.LENGTH_LONG).show()
                                    Log.e("CameraPreview", "CSV report error", error)
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

// Data class for bar path processing results
data class BarPathResult(
    val paths: List<BarPath>,
    val movement: MovementAnalysis?,
    val repCount: Int
)

// EXTREME bar path processing function
private fun processBarPathYOLOv8(
    detections: List<Detection>,
    currentPaths: List<BarPath>
): BarPathResult {
    if (detections.isEmpty()) {
        return BarPathResult(currentPaths, null, 0)
    }

    // YOLOv8 usually gives better detections, so use the highest confidence one
    val bestDetection = detections.maxByOrNull { it.score } ?: detections.first()
    val centerX = (bestDetection.bbox.left + bestDetection.bbox.right) / 2f
    val centerY = (bestDetection.bbox.top + bestDetection.bbox.bottom) / 2f
    val currentTime = System.currentTimeMillis()

    val newPoint = PathPoint(centerX, centerY, currentTime)

    // Get or create active path
    val activePath = currentPaths.lastOrNull() ?: BarPath()

    // YOLOv8 movement validation (more sensitive due to better accuracy)
    val shouldAddPoint = if (activePath.points.isEmpty()) {
        true
    } else {
        val lastPoint = activePath.points.last()
        val dx = newPoint.x - lastPoint.x
        val dy = newPoint.y - lastPoint.y
        val distance = kotlin.math.sqrt(dx * dx + dy * dy)
        distance > 0.008f // More sensitive threshold for YOLOv8
    }

    if (shouldAddPoint) {
        activePath.addPoint(newPoint, 150) // Larger buffer for YOLOv8 accuracy
    }

    // Update paths
    val updatedPaths = if (currentPaths.isEmpty()) {
        listOf(activePath)
    } else {
        currentPaths.dropLast(1) + activePath
    }

    // YOLOv8 rep counting (more frequent due to better detection)
    val repCount = if (activePath.points.size % 10 == 0) { // Every 10th point
        countRepsYOLOv8(activePath.points)
    } else {
        0
    }

    return BarPathResult(updatedPaths, null, repCount)
}

// MINIMAL rep counting function
private fun countRepsYOLOv8(points: List<PathPoint>): Int {
    if (points.size < 20) return 0

    var repCount = 0
    var lastDirection: MovementDirection? = null
    var inUpPhase = false
    var lastPeakY = 0f
    var lastValleyY = 0f

    // Use a smoothing window for YOLOv8 data
    val windowSize = 5

    for (i in windowSize until points.size - windowSize) {
        // Calculate smoothed Y position
        val beforeY = points.subList(i - windowSize, i).map { it.y }.average().toFloat()
        val afterY = points.subList(i + 1, i + windowSize + 1).map { it.y }.average().toFloat()
        val currentY = points[i].y

        val currentDirection = when {
            afterY - beforeY > 0.02f -> MovementDirection.DOWN
            afterY - beforeY < -0.02f -> MovementDirection.UP
            else -> MovementDirection.STABLE
        }

        // Enhanced rep detection for YOLOv8
        when {
            // Starting upward movement (concentric phase)
            lastDirection == MovementDirection.DOWN && currentDirection == MovementDirection.UP -> {
                if (!inUpPhase) {
                    inUpPhase = true
                    lastValleyY = currentY
                }
            }

            // Completing upward movement, starting downward (eccentric phase)
            lastDirection == MovementDirection.UP && currentDirection == MovementDirection.DOWN -> {
                if (inUpPhase) {
                    lastPeakY = currentY
                    val rangeOfMotion = kotlin.math.abs(lastPeakY - lastValleyY)

                    // Only count as rep if sufficient range of motion
                    if (rangeOfMotion > 0.05f) { // 5% of screen height
                        repCount++
                        Log.d("YOLOv8RepCount", "Rep detected: ROM=$rangeOfMotion, peak=$lastPeakY, valley=$lastValleyY")
                    }
                    inUpPhase = false
                }
            }
        }

        if (currentDirection != MovementDirection.STABLE) {
            lastDirection = currentDirection
        }
    }

    return repCount
}


// MINIMAL drawing functions for maximum performance
// Replace the drawDetectionsYOLOv8 function in MainActivity.kt with this:

private fun DrawScope.drawDetectionsYOLOv8(detections: List<Detection>, detector: YOLOv8ObjectDetector) {
    if (detections.isEmpty()) return

    val canvasWidth = size.width
    val canvasHeight = size.height

    detections.forEach { detection ->
        val bbox = detection.bbox

        // Convert normalized coordinates to pixel coordinates
        val left = bbox.left * canvasWidth
        val top = bbox.top * canvasHeight
        val right = bbox.right * canvasWidth
        val bottom = bbox.bottom * canvasHeight

        // Color based on confidence
        val color = when {
            detection.score > 0.7f -> androidx.compose.ui.graphics.Color.Green
            detection.score > 0.4f -> androidx.compose.ui.graphics.Color.Yellow
            else -> androidx.compose.ui.graphics.Color.Red
        }

        // Draw bounding box with confidence-based thickness
        val strokeWidth = (2 + detection.score * 4).dp.toPx()

        drawRect(
            color = color,
            topLeft = androidx.compose.ui.geometry.Offset(left, top),
            size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = strokeWidth)
        )

        // Draw center point
        val centerX = (left + right) / 2f
        val centerY = (top + bottom) / 2f

        drawCircle(
            color = color,
            radius = (6 + detection.score * 6).dp.toPx(),
            center = androidx.compose.ui.geometry.Offset(centerX, centerY)
        )

        // Draw text info - FIXED VERSION
        drawContext.canvas.nativeCanvas.apply {
            val paint = android.graphics.Paint().apply {
                this.color = color.toArgb()
                textSize = 14.sp.toPx()
                isAntiAlias = true
                isFakeBoldText = true
            }

            val label = "${detector.getClassLabel(detection.classId)}: ${String.format("%.2f", detection.score)}"
            // Get quality from detector and show overall quality
            val quality = detector.getDetectionQuality(detection)
            val qualityText = "Q: ${String.format("%.2f", quality.getOverallQuality())}"

            drawText(label, left + 5f, top - 5f, paint)
            drawText(qualityText, left + 5f, top - 25f, paint)
        }
    }
}

private fun DrawScope.drawBarPathsMinimal(paths: List<BarPath>) {
    if (paths.isEmpty()) return

    val path = paths.first()
    val canvasWidth = size.width
    val canvasHeight = size.height

    // Draw only last 5 points for performance
    val recentPoints = path.points.takeLast(5)

    recentPoints.forEach { point ->
        val pointX = point.x * canvasWidth
        val pointY = point.y * canvasHeight

        drawCircle(
            color = Color.Cyan,
            radius = 6.dp.toPx(),
            center = Offset(pointX, pointY)
        )
    }
}

@Composable
private fun EnhancedInfoPanel(
    detections: List<Detection>,
    fps: Float,
    isProcessing: Boolean,
    currentMovement: MovementAnalysis?,
    repCount: Int,
    isRecording: Boolean,
    isGeneratingReport: Boolean,
    isUsingGPU: Boolean = false,
    performanceInfo: String = "",
    onStartStopRecording: () -> Unit,
    onClearPath: () -> Unit,
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
                text = "Bar Path Detector - YOLOv8",
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(4.dp))

            // Control buttons row 1
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
                    onClick = onClearPath,
                    modifier = Modifier.height(28.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color.Blue)
                ) {
                    Text("Clear", fontSize = 10.sp, color = Color.White)
                }
            }

            Spacer(modifier = Modifier.height(4.dp))

            // Report generation buttons row 2
            AnimatedVisibility(visible = repCount > 0) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.padding(horizontal = 8.dp)
                ) {
                    Button(
                        onClick = onGenerateExcelReport,
                        enabled = !isGeneratingReport && repCount > 0,
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
                        enabled = !isGeneratingReport && repCount > 0,
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
                text = performanceInfo,
                color = Color.Yellow,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
            Text(
                text = "Detections: ${detections.size}",
                color = Color.White,
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
            Text(
                text = "Reps: $repCount",
                color = Color.Cyan,
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )

            if (isGeneratingReport) {
                Text(
                    text = "📄 Generating Report...",
                    color = Color.Yellow,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
            }

            if (detections.isNotEmpty()) {
                Spacer(modifier = Modifier.height(6.dp))
                detections.take(1).forEachIndexed { index, detection ->
                    Text(
                        text = "Barbell: ${String.format("%.2f", detection.score)}",
                        color = Color.Cyan,
                        fontSize = 9.sp,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}