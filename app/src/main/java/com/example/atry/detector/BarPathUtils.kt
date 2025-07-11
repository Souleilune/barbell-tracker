package com.example.atry.detector

import androidx.compose.ui.graphics.Color
import kotlin.math.*

/**
 * Optimized utility classes and functions for bar path tracking and analysis
 */

data class PathPoint(
    val x: Float,
    val y: Float,
    val timestamp: Long
) {
    fun distanceTo(other: PathPoint): Float {
        return sqrt((x - other.x).pow(2) + (y - other.y).pow(2))
    }

    fun verticalDistanceTo(other: PathPoint): Float {
        return abs(y - other.y)
    }

    fun horizontalDistanceTo(other: PathPoint): Float {
        return abs(x - other.x)
    }
}

data class BarPath(
    val id: String = generatePathId(),
    val points: MutableList<PathPoint> = mutableListOf(),
    val isActive: Boolean = true,
    val color: Color = Color.Cyan,
    val startTime: Long = System.currentTimeMillis()
) {
    companion object {
        private var pathCounter = 0
        fun generatePathId(): String = "path_${++pathCounter}"
    }

    fun addPoint(point: PathPoint, maxPoints: Int = 300) { // Reduced from 500
        points.add(point)
        if (points.size > maxPoints) {
            points.removeAt(0)
        }
    }

    fun getTotalDistance(): Float {
        if (points.size < 2) return 0f
        return points.zipWithNext { a, b -> a.distanceTo(b) }.sum()
    }

    fun getVerticalRange(): Float {
        if (points.isEmpty()) return 0f
        val minY = points.minOf { it.y }
        val maxY = points.maxOf { it.y }
        return maxY - minY
    }

    fun getDuration(): Long {
        if (points.isEmpty()) return 0L
        return points.last().timestamp - points.first().timestamp
    }
}

enum class MovementDirection {
    UP, DOWN, STABLE
}

data class MovementPhase(
    val direction: MovementDirection,
    val startPoint: PathPoint,
    val endPoint: PathPoint?,
    val maxDisplacement: Float = 0f
)

data class MovementAnalysis(
    val direction: MovementDirection,
    val velocity: Float, // pixels per second
    val acceleration: Float = 0f,
    val totalDistance: Float,
    val repCount: Int,
    val currentPhase: MovementPhase? = null,
    val averageBarSpeed: Float = 0f,
    val peakVelocity: Float = 0f
)

data class LiftingMetrics(
    val totalReps: Int,
    val averageRepTime: Float,
    val averageRangeOfMotion: Float,
    val barPathDeviation: Float, // How much the bar deviates from vertical
    val consistencyScore: Float, // How consistent the movement pattern is
    val phases: List<MovementPhase> = emptyList()
)

/**
 * Optimized bar path analyzer with reduced computational complexity
 */
class BarPathAnalyzer(
    private val smoothingWindow: Int = 3,     // Reduced from 5
    private val minRepDisplacement: Float = 0.06f, // Slightly reduced threshold
    private val velocityThreshold: Float = 0.01f,
    private val stableThreshold: Float = 0.005f
) {

    private var lastDirection: MovementDirection? = null
    private var repPhases = mutableListOf<MovementPhase>()
    private var currentPhase: MovementPhase? = null

    fun analyzeMovement(points: List<PathPoint>): MovementAnalysis {
        if (points.size < 3) {
            return MovementAnalysis(
                direction = MovementDirection.STABLE,
                velocity = 0f,
                totalDistance = 0f,
                repCount = 0
            )
        }

        // Use simplified analysis for better performance
        val recentPoints = points.takeLast(10) // Reduced from all points
        val direction = calculateDirectionOptimized(recentPoints)
        val velocity = calculateVelocityOptimized(recentPoints)
        val totalDistance = calculateTotalDistanceOptimized(recentPoints)
        val repCount = countRepsOptimized(points)
        val averageSpeed = calculateAverageSpeedOptimized(recentPoints)

        return MovementAnalysis(
            direction = direction,
            velocity = velocity,
            acceleration = 0f, // Skip acceleration calculation for performance
            totalDistance = totalDistance,
            repCount = repCount,
            currentPhase = currentPhase,
            averageBarSpeed = averageSpeed,
            peakVelocity = velocity // Simplified
        )
    }

    fun calculateLiftingMetrics(paths: List<BarPath>): LiftingMetrics {
        val allPoints = paths.flatMap { it.points }
        if (allPoints.isEmpty()) {
            return LiftingMetrics(0, 0f, 0f, 0f, 0f)
        }

        val totalReps = countRepsOptimized(allPoints)
        val averageRepTime = calculateAverageRepTimeOptimized(allPoints, totalReps)
        val averageROM = calculateAverageRangeOfMotionOptimized(paths)
        val pathDeviation = calculateBarPathDeviationOptimized(allPoints)
        val consistency = calculateConsistencyScoreOptimized(paths)

        return LiftingMetrics(
            totalReps = totalReps,
            averageRepTime = averageRepTime,
            averageRangeOfMotion = averageROM,
            barPathDeviation = pathDeviation,
            consistencyScore = consistency,
            phases = repPhases.toList()
        )
    }

    // Optimized calculation methods
    private fun calculateDirectionOptimized(points: List<PathPoint>): MovementDirection {
        if (points.size < 2) return MovementDirection.STABLE

        val verticalChange = points.last().y - points.first().y

        return when {
            verticalChange > stableThreshold -> MovementDirection.DOWN
            verticalChange < -stableThreshold -> MovementDirection.UP
            else -> MovementDirection.STABLE
        }
    }

    private fun calculateVelocityOptimized(points: List<PathPoint>): Float {
        if (points.size < 2) return 0f

        val totalDisplacement = points.zipWithNext { a, b ->
            val dx = b.x - a.x
            val dy = b.y - a.y
            sqrt(dx * dx + dy * dy)
        }.sum()

        val timeSpan = (points.last().timestamp - points.first().timestamp) / 1000f

        return if (timeSpan > 0) totalDisplacement / timeSpan else 0f
    }

    private fun calculateTotalDistanceOptimized(points: List<PathPoint>): Float {
        return points.zipWithNext { a, b ->
            val dx = b.x - a.x
            val dy = b.y - a.y
            sqrt(dx * dx + dy * dy)
        }.sum()
    }

    private fun countRepsOptimized(points: List<PathPoint>): Int {
        if (points.size < 15) return 0 // Reduced minimum points

        var repCount = 0
        var inRepPhase = false
        var repStartY: Float? = null
        var currentDirection: MovementDirection? = null

        // Process every 2nd point for better performance
        for (i in smoothingWindow until points.size - smoothingWindow step 2) {
            val prevY = points[i - smoothingWindow].y
            val nextY = points[i + smoothingWindow].y
            val displacement = nextY - prevY

            val direction = when {
                displacement > stableThreshold -> MovementDirection.DOWN
                displacement < -stableThreshold -> MovementDirection.UP
                else -> MovementDirection.STABLE
            }

            // Simplified phase detection
            if (currentDirection != direction && direction != MovementDirection.STABLE) {
                when {
                    // Starting upward phase (concentric)
                    currentDirection == MovementDirection.DOWN && direction == MovementDirection.UP -> {
                        if (!inRepPhase) {
                            inRepPhase = true
                            repStartY = points[i].y
                        }
                    }
                    // Completing downward phase (eccentric) - rep completed
                    currentDirection == MovementDirection.UP && direction == MovementDirection.DOWN -> {
                        if (inRepPhase && repStartY != null) {
                            val totalDisplacement = abs(points[i].y - repStartY!!)
                            if (totalDisplacement > minRepDisplacement) {
                                repCount++

                                // Record the rep phase (simplified)
                                val repPhase = MovementPhase(
                                    direction = MovementDirection.UP,
                                    startPoint = PathPoint(0f, repStartY!!, 0L),
                                    endPoint = points[i],
                                    maxDisplacement = totalDisplacement
                                )
                                repPhases.add(repPhase)
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

    private fun calculateAverageSpeedOptimized(points: List<PathPoint>): Float {
        if (points.size < 2) return 0f

        val totalDistance = calculateTotalDistanceOptimized(points)
        val totalTime = (points.last().timestamp - points.first().timestamp) / 1000f

        return if (totalTime > 0) totalDistance / totalTime else 0f
    }

    private fun calculateAverageRepTimeOptimized(points: List<PathPoint>, repCount: Int): Float {
        if (repCount == 0 || points.isEmpty()) return 0f

        val totalTime = (points.last().timestamp - points.first().timestamp) / 1000f
        return totalTime / repCount
    }

    private fun calculateAverageRangeOfMotionOptimized(paths: List<BarPath>): Float {
        if (paths.isEmpty()) return 0f

        val ranges = paths.map { it.getVerticalRange() }.filter { it > 0f }
        return if (ranges.isNotEmpty()) ranges.average().toFloat() else 0f
    }

    private fun calculateBarPathDeviationOptimized(points: List<PathPoint>): Float {
        if (points.isEmpty()) return 0f

        // Simplified deviation calculation
        val centerX = points.map { it.x }.average().toFloat()
        return points.map { abs(it.x - centerX) }.average().toFloat()
    }

    private fun calculateConsistencyScoreOptimized(paths: List<BarPath>): Float {
        if (paths.size < 2) return 1f

        // Simplified consistency calculation
        val pathCharacteristics = paths.map { path ->
            listOf(
                path.getVerticalRange(),
                path.getTotalDistance()
            )
        }

        if (pathCharacteristics.isEmpty()) return 1f

        // Calculate coefficient of variation for each characteristic
        val consistencyScores = (0..1).map { index ->
            val values = pathCharacteristics.map { it[index] }.filter { it > 0f }
            if (values.isEmpty()) return@map 1f

            val mean = values.average()
            val variance = values.map { (it - mean).pow(2) }.average()
            val stdDev = sqrt(variance)

            // Lower coefficient of variation = higher consistency
            val coefficientOfVariation = if (mean > 0) stdDev / mean else 0.0
            maxOf(0f, 1f - coefficientOfVariation.toFloat())
        }

        return consistencyScores.average().toFloat()
    }
}