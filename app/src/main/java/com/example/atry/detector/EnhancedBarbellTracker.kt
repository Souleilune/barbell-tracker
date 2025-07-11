package com.example.atry.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import androidx.compose.ui.graphics.Color
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.*

/**
 * Updated Enhanced Barbell Tracker that works with Hybrid Classification Detector
 * Perfect for simonskina.tflite classification model
 */
class EnhancedBarbellTracker(
    private val context: Context,
    private val modelPath: String = "simonskina.tflite",
    private val confThreshold: Float = 0.1f, // Lower threshold for classification
    private val iouThreshold: Float = 0.3f,  // Lower IoU for less strict matching
    private val maxAge: Int = 30
) {

    companion object {
        private const val TAG = "EnhancedBarbellTracker"
        private const val MIN_DETECTION_SIZE = 0.01f // Very small minimum for motion areas
        private const val MAX_DETECTION_SIZE = 0.9f  // Allow larger detections
        private const val CENTER_REGION_THRESHOLD = 0.05f // Allow detections closer to edges
    }

    // Use Hybrid Classification Detector instead of Generic TFLite Detector
    private val detector: HybridClassificationDetector
    private val trackingScope = CoroutineScope(Dispatchers.Default)

    // Tracking state
    private val trackedObjects = ConcurrentHashMap<Int, TrackedObject>()
    private val trackingHistory = mutableListOf<TrackingDataPoint>()
    private var nextTrackingId = 1
    private var lastProcessedTimestamp = 0L

    // Analytics
    private var analytics: BarbellAnalytics? = null

    init {
        Log.d(TAG, "🏋️ Initializing Enhanced Barbell Tracker with Hybrid Detector")
        detector = HybridClassificationDetector(
            context = context,
            modelPath = modelPath,
            confThreshold = confThreshold,
            maxDetections = 3 // Limit detections for stability
        )
        Log.d(TAG, "✅ Enhanced Barbell Tracker with Hybrid Detector initialized")
    }

    /**
     * Enhanced tracking optimized for classification + motion detection
     */
    fun track(bitmap: Bitmap, timestamp: Long): TrackingResult {
        try {
            Log.d(TAG, "🔍 Enhanced hybrid tracking frame at $timestamp")

            // Run hybrid detection (classification + motion)
            val rawDetections = detector.detect(bitmap)
            Log.d(TAG, "Hybrid detections: ${rawDetections.size}")

            // Apply relaxed filtering for hybrid approach
            val filteredDetections = applyHybridFiltering(rawDetections)
            Log.d(TAG, "Filtered hybrid detections: ${filteredDetections.size}")

            // Update tracking with hybrid approach
            updateHybridTracking(filteredDetections, timestamp)

            // Create result
            val result = createTrackingResult()

            // Update analytics
            updateAnalytics()

            lastProcessedTimestamp = timestamp
            return result

        } catch (e: Exception) {
            Log.e(TAG, "Error in hybrid tracking: ${e.message}", e)
            return TrackingResult(emptyList(), emptyMap())
        }
    }

    /**
     * Relaxed filtering for hybrid classification + motion detection
     */
    private fun applyHybridFiltering(detections: List<Detection>): List<Detection> {
        return detections.filter { detection ->
            val bbox = detection.bbox

            // Filter 1: Very relaxed size validation (motion areas can be small)
            val width = bbox.right - bbox.left
            val height = bbox.bottom - bbox.top
            val area = width * height

            if (area < MIN_DETECTION_SIZE || area > MAX_DETECTION_SIZE) {
                Log.d(TAG, "❌ Filtered detection: invalid size (area: $area)")
                return@filter false
            }

            // Filter 2: More permissive position validation
            val centerX = (bbox.left + bbox.right) / 2f
            val centerY = (bbox.top + bbox.bottom) / 2f

            val tooCloseToEdge = centerX < CENTER_REGION_THRESHOLD ||
                    centerX > (1f - CENTER_REGION_THRESHOLD) ||
                    centerY < CENTER_REGION_THRESHOLD ||
                    centerY > (1f - CENTER_REGION_THRESHOLD)

            if (tooCloseToEdge) {
                Log.d(TAG, "❌ Filtered detection: too close to edge (center: $centerX, $centerY)")
                return@filter false
            }

            // Filter 3: Very relaxed aspect ratio (motion areas have varied shapes)
            val aspectRatio = width / height
            if (aspectRatio < 0.1f || aspectRatio > 20f) {
                Log.d(TAG, "❌ Filtered detection: extreme aspect ratio ($aspectRatio)")
                return@filter false
            }

            // Filter 4: Lower confidence threshold for hybrid approach
            val adjustedThreshold = confThreshold * 0.5f // Even lower for hybrid
            if (detection.score < adjustedThreshold) {
                Log.d(TAG, "❌ Filtered detection: low confidence (${detection.score} < $adjustedThreshold)")
                return@filter false
            }

            // Filter 5: Validate bounding box coordinates
            if (bbox.left >= bbox.right || bbox.top >= bbox.bottom ||
                bbox.left < 0f || bbox.top < 0f || bbox.right > 1f || bbox.bottom > 1f) {
                Log.d(TAG, "❌ Filtered detection: invalid coordinates")
                return@filter false
            }

            Log.d(TAG, "✅ Valid hybrid detection: conf=${detection.score}, area=$area, pos=($centerX,$centerY)")
            true
        }
    }

    /**
     * Hybrid tracking update with more permissive matching
     */
    private fun updateHybridTracking(detections: List<Detection>, timestamp: Long) {
        val currentIds = mutableSetOf<Int>()

        // Age existing tracks
        trackedObjects.values.forEach { it.age++ }

        // More permissive association for hybrid approach
        val unassignedDetections = detections.toMutableList()

        for (detection in detections) {
            val detectionCenter = Pair(
                (detection.bbox.left + detection.bbox.right) / 2f,
                (detection.bbox.top + detection.bbox.bottom) / 2f
            )

            // Find best matching track with relaxed criteria
            val bestMatch = findBestHybridTrackMatch(detection, detectionCenter)

            if (bestMatch != null) {
                // Update existing track
                updateExistingTrack(bestMatch, detection, detectionCenter, timestamp)
                currentIds.add(bestMatch.id)
                unassignedDetections.remove(detection)

                Log.d(TAG, "📍 Updated hybrid track ${bestMatch.id} with detection")
            }
        }

        // Create new tracks for unassigned detections
        for (detection in unassignedDetections) {
            val detectionCenter = Pair(
                (detection.bbox.left + detection.bbox.right) / 2f,
                (detection.bbox.top + detection.bbox.bottom) / 2f
            )

            val newTrack = createNewTrack(detection, detectionCenter, timestamp)
            trackedObjects[newTrack.id] = newTrack
            currentIds.add(newTrack.id)

            Log.d(TAG, "🆕 Created new hybrid track ${newTrack.id}")
        }

        // More permissive track removal (longer lifetime for hybrid approach)
        val extendedMaxAge = maxAge * 2 // Double the lifetime for hybrid
        val toRemove = trackedObjects.filter { (id, track) ->
            track.age > extendedMaxAge || (!currentIds.contains(id) && track.age > maxAge)
        }

        toRemove.forEach { (id, track) ->
            trackedObjects.remove(id)
            Log.d(TAG, "🗑️ Removed hybrid track $id (age: ${track.age})")
        }

        Log.d(TAG, "📊 Active hybrid tracks: ${trackedObjects.size}")
    }

    /**
     * More permissive track matching for hybrid approach
     */
    private fun findBestHybridTrackMatch(detection: Detection, detectionCenter: Pair<Float, Float>): TrackedObject? {
        var bestTrack: TrackedObject? = null
        var bestDistance = Float.MAX_VALUE
        val maxMatchDistance = 0.25f // Increased from 0.15f for hybrid approach

        for (track in trackedObjects.values) {
            val trackCenter = track.center
            val distance = sqrt(
                (detectionCenter.first - trackCenter.first).pow(2) +
                        (detectionCenter.second - trackCenter.second).pow(2)
            )

            // More permissive IoU calculation for hybrid
            val iou = calculateIoU(detection.bbox, track.bbox)
            val combinedScore = distance * 0.8f + (1f - iou) * 0.2f // Emphasize distance over IoU

            if (combinedScore < bestDistance && distance < maxMatchDistance) {
                bestDistance = combinedScore
                bestTrack = track
            }
        }

        return bestTrack
    }

    /**
     * Smoother update for hybrid tracks
     */
    private fun updateExistingTrack(
        track: TrackedObject,
        detection: Detection,
        detectionCenter: Pair<Float, Float>,
        timestamp: Long
    ) {
        // More aggressive smoothing for hybrid approach
        val alpha = 0.6f // Reduced from 0.7f for smoother motion

        track.center = Pair(
            track.center.first * (1f - alpha) + detectionCenter.first * alpha,
            track.center.second * (1f - alpha) + detectionCenter.second * alpha
        )

        // Update bounding box with more smoothing
        track.bbox = RectF(
            track.bbox.left * (1f - alpha) + detection.bbox.left * alpha,
            track.bbox.top * (1f - alpha) + detection.bbox.top * alpha,
            track.bbox.right * (1f - alpha) + detection.bbox.right * alpha,
            track.bbox.bottom * (1f - alpha) + detection.bbox.bottom * alpha
        )

        // Smoother confidence update
        track.confidence = track.confidence * 0.9f + detection.score * 0.1f

        // Reset age
        track.age = 0
        track.lastSeen = timestamp

        // Add to path
        track.path.add(PathPoint(track.center.first, track.center.second, timestamp))

        // Longer path for better analytics
        if (track.path.size > 200) { // Increased from 100
            track.path.removeAt(0)
        }

        // Add to tracking history
        trackingHistory.add(TrackingDataPoint(
            id = track.id,
            x = track.center.first,
            y = track.center.second,
            timestamp = timestamp,
            confidence = track.confidence
        ))
    }

    /**
     * Create new track for hybrid detection
     */
    private fun createNewTrack(
        detection: Detection,
        detectionCenter: Pair<Float, Float>,
        timestamp: Long
    ): TrackedObject {
        val newId = nextTrackingId++
        val colors = listOf(Color.Cyan, Color.Yellow, Color.Magenta, Color.Green, Color.Blue, Color.Red)

        val track = TrackedObject(
            id = newId,
            center = detectionCenter,
            bbox = detection.bbox,
            confidence = detection.score,
            age = 0,
            lastSeen = timestamp,
            path = mutableListOf(PathPoint(detectionCenter.first, detectionCenter.second, timestamp)),
            color = colors[newId % colors.size]
        )

        // Add to tracking history
        trackingHistory.add(TrackingDataPoint(
            id = newId,
            x = detectionCenter.first,
            y = detectionCenter.second,
            timestamp = timestamp,
            confidence = detection.score
        ))

        return track
    }

    /**
     * Create tracking result from current state
     */
    private fun createTrackingResult(): TrackingResult {
        val activeObjects = trackedObjects.values.toList()
        val barPaths = mutableMapOf<Int, List<PathPoint>>()

        activeObjects.forEach { track ->
            if (track.path.size >= 2) {
                barPaths[track.id] = track.path.toList()
            }
        }

        return TrackingResult(activeObjects, barPaths)
    }

    /**
     * Enhanced analytics for hybrid approach
     */
    private fun updateAnalytics() {
        if (trackedObjects.isEmpty()) {
            analytics = null
            return
        }

        val primaryTrack = trackedObjects.values.maxByOrNull { it.path.size }
        if (primaryTrack == null) {
            analytics = null
            return
        }

        val path = primaryTrack.path
        if (path.size < 5) { // Reduced minimum path length for hybrid
            analytics = BarbellAnalytics(
                repCount = 0,
                totalDistance = 0f,
                averageVelocity = 0f,
                pathConsistency = 0f,
                primaryTrackingId = primaryTrack.id
            )
            return
        }

        // Calculate enhanced analytics for hybrid approach
        val repCount = calculateHybridRepCount(path)
        val totalDistance = calculateTotalDistance(path)
        val averageVelocity = calculateAverageVelocity(path)
        val pathConsistency = calculatePathConsistency(path)

        analytics = BarbellAnalytics(
            repCount = repCount,
            totalDistance = totalDistance,
            averageVelocity = averageVelocity,
            pathConsistency = pathConsistency,
            primaryTrackingId = primaryTrack.id
        )

        Log.d(TAG, "📈 Hybrid analytics updated: reps=$repCount, distance=$totalDistance, consistency=$pathConsistency")
    }

    /**
     * Enhanced rep counting optimized for hybrid detection
     */
    private fun calculateHybridRepCount(path: List<PathPoint>): Int {
        if (path.size < 10) return 0 // Reduced from 20 for hybrid

        var repCount = 0
        var inRepPhase = false
        var repStartY: Float? = null
        var currentDirection: MovementDirection? = null
        val minRepDisplacement = 0.03f // Reduced from 0.05f for more sensitive detection
        val smoothingWindow = 2 // Reduced from 3 for faster response

        for (i in smoothingWindow until path.size - smoothingWindow) {
            val prevY = path[i - smoothingWindow].y
            val nextY = path[i + smoothingWindow].y
            val displacement = nextY - prevY

            val direction = when {
                displacement > 0.008f -> MovementDirection.DOWN // More sensitive
                displacement < -0.008f -> MovementDirection.UP
                else -> MovementDirection.STABLE
            }

            // Detect rep phases with hybrid approach
            if (currentDirection != direction && direction != MovementDirection.STABLE) {
                when {
                    // Starting upward phase (concentric)
                    currentDirection == MovementDirection.DOWN && direction == MovementDirection.UP -> {
                        if (!inRepPhase) {
                            inRepPhase = true
                            repStartY = path[i].y
                        }
                    }
                    // Completing downward phase (eccentric) - rep completed
                    currentDirection == MovementDirection.UP && direction == MovementDirection.DOWN -> {
                        if (inRepPhase && repStartY != null) {
                            val totalDisplacement = abs(path[i].y - repStartY!!)
                            if (totalDisplacement > minRepDisplacement) {
                                repCount++
                                Log.d(TAG, "🏋️ Hybrid rep detected! Total: $repCount, displacement: $totalDisplacement")
                            }
                            inRepPhase = false
                            repStartY = null
                        }
                    }
                }
                currentDirection = direction
            }
        }

        return repCount
    }

    /**
     * Calculate total distance traveled
     */
    private fun calculateTotalDistance(path: List<PathPoint>): Float {
        if (path.size < 2) return 0f

        return path.zipWithNext { a, b ->
            val dx = b.x - a.x
            val dy = b.y - a.y
            sqrt(dx * dx + dy * dy)
        }.sum()
    }

    /**
     * Calculate average velocity
     */
    private fun calculateAverageVelocity(path: List<PathPoint>): Float {
        if (path.size < 2) return 0f

        val totalDistance = calculateTotalDistance(path)
        val totalTime = (path.last().timestamp - path.first().timestamp) / 1000f

        return if (totalTime > 0) totalDistance / totalTime else 0f
    }

    /**
     * Calculate path consistency (how straight the movement is)
     */
    private fun calculatePathConsistency(path: List<PathPoint>): Float {
        if (path.size < 3) return 1f

        val centerX = path.map { it.x }.average().toFloat()
        val deviations = path.map { abs(it.x - centerX) }
        val averageDeviation = deviations.average().toFloat()

        // Convert to consistency score (0-1, where 1 is perfectly consistent)
        return max(0f, 1f - averageDeviation * 8f) // Adjusted for hybrid approach
    }

    /**
     * Calculate IoU between two bounding boxes
     */
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

    // Public interface methods
    fun getAnalytics(): BarbellAnalytics? = analytics

    fun getTrackingData(): List<TrackingDataPoint> = trackingHistory.toList()

    fun reset() {
        trackedObjects.clear()
        trackingHistory.clear()
        analytics = null
        nextTrackingId = 1
        Log.d(TAG, "🔄 Enhanced hybrid tracker reset")
    }

    fun cleanup() {
        reset()
        detector.cleanup()
        Log.d(TAG, "🧹 Enhanced hybrid tracker cleaned up")
    }

    // Data classes
    data class TrackedObject(
        val id: Int,
        var center: Pair<Float, Float>,
        var bbox: RectF,
        var confidence: Float,
        var age: Int,
        var lastSeen: Long,
        val path: MutableList<PathPoint>,
        val color: Color
    )

    data class TrackingDataPoint(
        val id: Int,
        val x: Float,
        val y: Float,
        val timestamp: Long,
        val confidence: Float
    )
}

// Supporting data classes
data class TrackingResult(
    val trackedObjects: List<EnhancedBarbellTracker.TrackedObject>,
    val barPaths: Map<Int, List<PathPoint>>
)

data class BarbellAnalytics(
    val repCount: Int,
    val totalDistance: Float,
    val averageVelocity: Float,
    val pathConsistency: Float,
    val primaryTrackingId: Int?
)

