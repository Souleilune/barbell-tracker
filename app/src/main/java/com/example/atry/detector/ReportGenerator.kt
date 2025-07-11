package com.example.atry.detector

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Environment
import android.util.Log
import androidx.core.content.FileProvider
import org.apache.poi.ss.usermodel.*
import org.apache.poi.xssf.usermodel.XSSFWorkbook
import org.apache.poi.xssf.usermodel.XSSFCellStyle
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Optimized report generator for bar path analysis with improved performance
 */
class ReportGenerator(private val context: Context) {

    companion object {
        private const val TAG = "ReportGenerator"
        private const val PROVIDER_AUTHORITY = "com.example.atry.fileprovider"
    }

    data class WorkoutSession(
        val startTime: Long,
        val endTime: Long,
        val actualRepCount: Int,
        val paths: List<BarPath>,
        val movements: List<MovementAnalysis>,
        val sessionNotes: String = ""
    )

    // Optimized rep detection data structure
    data class RepData(
        val repNumber: Int,
        val startIndex: Int,
        val endIndex: Int,
        val startPoint: PathPoint,
        val endPoint: PathPoint,
        val duration: Float,
        val rangeOfMotion: Float,
        val totalDistance: Float,
        val pathDeviation: Float
    )

    /**
     * Optimized rep extraction with reduced complexity
     */
    private fun extractRepData(path: BarPath): List<RepData> {
        val points = path.points
        if (points.size < 15) return emptyList() // Reduced from 20

        val reps = mutableListOf<RepData>()
        var inRepPhase = false
        var repStartIndex: Int? = null
        var repStartY: Float? = null
        var currentDirection: MovementDirection? = null
        var repNumber = 1

        val smoothingWindow = 3 // Reduced from 5
        val stableThreshold = 0.02f
        val minRepDisplacement = 0.05f

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

            // Detect phase transitions
            if (currentDirection != direction && direction != MovementDirection.STABLE) {
                when {
                    // Starting upward phase (concentric)
                    currentDirection == MovementDirection.DOWN && direction == MovementDirection.UP -> {
                        if (!inRepPhase) {
                            inRepPhase = true
                            repStartIndex = i
                            repStartY = points[i].y
                        }
                    }
                    // Completing downward phase (eccentric) - rep completed
                    currentDirection == MovementDirection.UP && direction == MovementDirection.DOWN -> {
                        if (inRepPhase && repStartIndex != null && repStartY != null) {
                            val totalDisplacement = abs(points[i].y - repStartY!!)
                            if (totalDisplacement > minRepDisplacement) {
                                // Calculate rep metrics (optimized)
                                val repPoints = points.subList(repStartIndex!!, i + 1)
                                val duration = (points[i].timestamp - points[repStartIndex!!].timestamp) / 1000f
                                val rangeOfMotion = totalDisplacement
                                val totalDistance = calculateDistanceOptimized(repPoints)
                                val pathDeviation = calculatePathDeviationOptimized(repPoints)

                                val repData = RepData(
                                    repNumber = repNumber,
                                    startIndex = repStartIndex!!,
                                    endIndex = i,
                                    startPoint = points[repStartIndex!!],
                                    endPoint = points[i],
                                    duration = duration,
                                    rangeOfMotion = rangeOfMotion,
                                    totalDistance = totalDistance,
                                    pathDeviation = pathDeviation
                                )

                                reps.add(repData)
                                repNumber++
                                Log.d(TAG, "Rep $repNumber detected: ROM=$rangeOfMotion, Duration=$duration")
                            }
                            inRepPhase = false
                            repStartIndex = null
                            repStartY = null
                        }
                    }
                }
                currentDirection = direction
            }
        }

        Log.d(TAG, "Extracted ${reps.size} reps from path data")
        return reps
    }

    // Optimized distance calculation
    private fun calculateDistanceOptimized(points: List<PathPoint>): Float {
        return points.zipWithNext { a, b ->
            val dx = b.x - a.x
            val dy = b.y - a.y
            kotlin.math.sqrt(dx * dx + dy * dy)
        }.sum()
    }

    // Optimized path deviation calculation
    private fun calculatePathDeviationOptimized(points: List<PathPoint>): Float {
        if (points.isEmpty()) return 0f
        val centerX = points.map { it.x }.average().toFloat()
        return points.map { abs(it.x - centerX) }.average().toFloat()
    }

    /**
     * Generate optimized Excel report
     */
    fun generateExcelReport(
        session: WorkoutSession,
        analyzer: BarPathAnalyzer
    ): Result<File> {
        return try {
            val workbook = XSSFWorkbook()

            // Extract rep data from the continuous path
            val allRepData = session.paths.flatMap { extractRepData(it) }

            // Create sheets (optimized)
            createSummarySheetOptimized(workbook, session, analyzer, allRepData)
            createDetailedPathSheetOptimized(workbook, session)
            createMovementAnalysisSheetOptimized(workbook, session, allRepData)
            createRepAnalysisSheetOptimized(workbook, session, allRepData)

            // Save to file
            val file = saveWorkbookToFile(workbook, "barpath_report")
            workbook.close()

            Log.d(TAG, "Excel report generated successfully: ${file.absolutePath}")
            Result.success(file)
        } catch (e: Exception) {
            Log.e(TAG, "Error generating Excel report: ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Generate optimized CSV report
     */
    fun generateCSVReport(
        session: WorkoutSession,
        analyzer: BarPathAnalyzer
    ): Result<File> {
        return try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "barpath_report_$timestamp.csv"
            val file = File(context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), fileName)

            val csv = StringBuilder()

            // Extract rep data for proper analysis
            val allRepData = session.paths.flatMap { extractRepData(it) }
            val actualRepCount = max(session.actualRepCount, allRepData.size)

            // Header information
            csv.appendLine("Bar Path Analysis Report")
            csv.appendLine("Generated: ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())}")
            csv.appendLine("Session Duration: ${formatDuration(session.endTime - session.startTime)}")
            csv.appendLine("Total Reps: $actualRepCount")
            csv.appendLine("Detected Reps: ${allRepData.size}")
            csv.appendLine("")

            // Lifting metrics using corrected rep count
            val metrics = analyzer.calculateLiftingMetrics(session.paths)
            csv.appendLine("LIFTING METRICS")
            csv.appendLine("Metric,Value")
            csv.appendLine("Total Reps,$actualRepCount")
            csv.appendLine("Detected Reps,${allRepData.size}")
            csv.appendLine("Average Rep Time (s),${String.format("%.2f", calculateAverageRepTimeOptimized(allRepData))}")
            csv.appendLine("Average Range of Motion,${String.format("%.3f", calculateAverageROMOptimized(allRepData))}")
            csv.appendLine("Bar Path Deviation,${String.format("%.3f", metrics.barPathDeviation)}")
            csv.appendLine("Consistency Score,${String.format("%.3f", calculateRepConsistencyOptimized(allRepData))}")
            csv.appendLine("")

            // Individual rep analysis (limited for performance)
            if (allRepData.isNotEmpty()) {
                csv.appendLine("INDIVIDUAL REP ANALYSIS")
                csv.appendLine("Rep,Duration(s),Range_of_Motion,Total_Distance,Path_Deviation,Quality_Score")
                allRepData.take(20).forEach { rep -> // Limit to 20 reps for performance
                    val qualityScore = calculateRepQualityScoreOptimized(rep.rangeOfMotion, rep.duration, rep.totalDistance, rep.pathDeviation)
                    csv.appendLine("${rep.repNumber},${String.format("%.2f", rep.duration)},${String.format("%.3f", rep.rangeOfMotion)},${String.format("%.3f", rep.totalDistance)},${String.format("%.3f", rep.pathDeviation)},${String.format("%.3f", qualityScore)}")
                }
                csv.appendLine("")
            }

            // Detailed path data (sampled for performance)
            csv.appendLine("DETAILED PATH DATA (SAMPLED)")
            csv.appendLine("Point_Index,X_Position,Y_Position,Timestamp,Distance_From_Previous")

            session.paths.forEach { path ->
                val sampledPoints = if (path.points.size > 100) {
                    path.points.filterIndexed { index, _ -> index % 2 == 0 } // Take every 2nd point
                } else {
                    path.points
                }

                sampledPoints.forEachIndexed { pointIndex, point ->
                    val distanceFromPrevious = if (pointIndex > 0) {
                        calculateDistanceOptimized(listOf(sampledPoints[pointIndex - 1], point))
                    } else 0f

                    csv.appendLine("$pointIndex,${point.x},${point.y},${point.timestamp},$distanceFromPrevious")
                }
            }

            file.writeText(csv.toString())
            Log.d(TAG, "CSV report generated successfully: ${file.absolutePath}")
            Result.success(file)
        } catch (e: Exception) {
            Log.e(TAG, "Error generating CSV report: ${e.message}", e)
            Result.failure(e)
        }
    }

    // Optimized sheet creation methods
    private fun createSummarySheetOptimized(
        workbook: XSSFWorkbook,
        session: WorkoutSession,
        analyzer: BarPathAnalyzer,
        repData: List<RepData>
    ) {
        val sheet = workbook.createSheet("Summary")
        val headerStyle = createHeaderStyle(workbook)
        val dataStyle = createDataStyle(workbook)

        var rowNum = 0

        // Title
        val titleRow = sheet.createRow(rowNum++)
        val titleCell = titleRow.createCell(0)
        titleCell.setCellValue("Bar Path Analysis Report")
        titleCell.cellStyle = createTitleStyle(workbook)
        sheet.addMergedRegion(org.apache.poi.ss.util.CellRangeAddress(0, 0, 0, 3))

        rowNum++ // Empty row

        val actualRepCount = max(session.actualRepCount, repData.size)
        val detectedReps = repData.size

        // Session info (simplified)
        val sessionInfoData = arrayOf(
            arrayOf("Generated", SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())),
            arrayOf("Duration", formatDuration(session.endTime - session.startTime)),
            arrayOf("Total Reps (App Count)", actualRepCount.toString()),
            arrayOf("Detected Reps (Analysis)", detectedReps.toString())
        )

        // Session info header
        val sessionHeaderRow = sheet.createRow(rowNum++)
        val sessionHeaderCell = sessionHeaderRow.createCell(0)
        sessionHeaderCell.setCellValue("SESSION INFORMATION")
        sessionHeaderCell.cellStyle = headerStyle

        sessionInfoData.forEach { data ->
            val row = sheet.createRow(rowNum++)
            row.createCell(0).apply { setCellValue(data[0]); cellStyle = dataStyle }
            row.createCell(1).apply { setCellValue(data[1]); cellStyle = dataStyle }
        }

        rowNum++ // Empty row

        // Lifting metrics using corrected calculations
        val metricsHeaderRow = sheet.createRow(rowNum++)
        val metricsHeaderCell = metricsHeaderRow.createCell(0)
        metricsHeaderCell.setCellValue("LIFTING METRICS")
        metricsHeaderCell.cellStyle = headerStyle

        val metricsData = arrayOf(
            arrayOf("Total Reps", actualRepCount.toString()),
            arrayOf("Detected Reps", detectedReps.toString()),
            arrayOf("Average Rep Time (s)", String.format("%.2f", calculateAverageRepTimeOptimized(repData))),
            arrayOf("Average Range of Motion", String.format("%.3f", calculateAverageROMOptimized(repData))),
            arrayOf("Bar Path Deviation", String.format("%.3f", analyzer.calculateLiftingMetrics(session.paths).barPathDeviation)),
            arrayOf("Rep Consistency Score", String.format("%.3f", calculateRepConsistencyOptimized(repData)))
        )

        metricsData.forEach { data ->
            val row = sheet.createRow(rowNum++)
            row.createCell(0).apply { setCellValue(data[0]); cellStyle = dataStyle }
            row.createCell(1).apply { setCellValue(data[1]); cellStyle = dataStyle }
        }

        // Auto-size columns
        sheet.setColumnWidth(0, 6000)
        sheet.setColumnWidth(1, 4000)
    }

    private fun createDetailedPathSheetOptimized(workbook: XSSFWorkbook, session: WorkoutSession) {
        val sheet = workbook.createSheet("Path Data")
        val headerStyle = createHeaderStyle(workbook)
        val dataStyle = createDataStyle(workbook)

        var rowNum = 0

        // Headers
        val headerRow = sheet.createRow(rowNum++)
        val headers = arrayOf("Point_Index", "X_Position", "Y_Position", "Timestamp", "Distance_From_Previous")
        headers.forEachIndexed { index, header ->
            val cell = headerRow.createCell(index)
            cell.setCellValue(header)
            cell.cellStyle = headerStyle
        }

        // Data - optimized with sampling for large datasets
        var globalPointIndex = 0
        session.paths.forEach { path ->
            val sampledPoints = if (path.points.size > 200) {
                path.points.filterIndexed { index, _ -> index % 2 == 0 } // Take every 2nd point
            } else {
                path.points
            }

            sampledPoints.forEachIndexed { pointIndex, point ->
                val row = sheet.createRow(rowNum++)

                val distanceFromPrevious = if (pointIndex > 0) {
                    calculateDistanceOptimized(listOf(sampledPoints[pointIndex - 1], point))
                } else 0f

                row.createCell(0).apply { setCellValue(globalPointIndex.toDouble()); cellStyle = dataStyle }
                row.createCell(1).apply { setCellValue(point.x.toDouble()); cellStyle = dataStyle }
                row.createCell(2).apply { setCellValue(point.y.toDouble()); cellStyle = dataStyle }
                row.createCell(3).apply { setCellValue(point.timestamp.toDouble()); cellStyle = dataStyle }
                row.createCell(4).apply { setCellValue(distanceFromPrevious.toDouble()); cellStyle = dataStyle }

                globalPointIndex++
            }
        }

        // Auto-size columns
        headers.forEachIndexed { index, _ ->
            sheet.setColumnWidth(index, 3000)
        }
    }

    private fun createMovementAnalysisSheetOptimized(workbook: XSSFWorkbook, session: WorkoutSession, repData: List<RepData>) {
        val sheet = workbook.createSheet("Movement Analysis")
        val headerStyle = createHeaderStyle(workbook)
        val dataStyle = createDataStyle(workbook)

        var rowNum = 0

        // Headers
        val headerRow = sheet.createRow(rowNum++)
        val headers = arrayOf("Rep", "Duration(s)", "Range_of_Motion", "Total_Distance", "Path_Deviation", "Quality_Score")
        headers.forEachIndexed { index, header ->
            val cell = headerRow.createCell(index)
            cell.setCellValue(header)
            cell.cellStyle = headerStyle
        }

        // Use detected rep data (limited for performance)
        repData.take(50).forEach { rep -> // Limit to 50 reps
            val row = sheet.createRow(rowNum++)
            val qualityScore = calculateRepQualityScoreOptimized(rep.rangeOfMotion, rep.duration, rep.totalDistance, rep.pathDeviation)

            row.createCell(0).apply { setCellValue(rep.repNumber.toDouble()); cellStyle = dataStyle }
            row.createCell(1).apply { setCellValue(rep.duration.toDouble()); cellStyle = dataStyle }
            row.createCell(2).apply { setCellValue(rep.rangeOfMotion.toDouble()); cellStyle = dataStyle }
            row.createCell(3).apply { setCellValue(rep.totalDistance.toDouble()); cellStyle = dataStyle }
            row.createCell(4).apply { setCellValue(rep.pathDeviation.toDouble()); cellStyle = dataStyle }
            row.createCell(5).apply { setCellValue(qualityScore.toDouble()); cellStyle = dataStyle }
        }

        // Auto-size columns
        headers.forEachIndexed { index, _ ->
            sheet.setColumnWidth(index, 3500)
        }
    }

    private fun createRepAnalysisSheetOptimized(workbook: XSSFWorkbook, session: WorkoutSession, repData: List<RepData>) {
        val sheet = workbook.createSheet("Rep Analysis")
        val headerStyle = createHeaderStyle(workbook)
        val dataStyle = createDataStyle(workbook)

        var rowNum = 0

        // Headers
        val headerRow = sheet.createRow(rowNum++)
        val headers = arrayOf("Rep", "Duration(s)", "Range_of_Motion", "Quality_Score", "Grade")
        headers.forEachIndexed { index, header ->
            val cell = headerRow.createCell(index)
            cell.setCellValue(header)
            cell.cellStyle = headerStyle
        }

        // Analyze each detected rep (limited)
        repData.take(30).forEach { rep -> // Limit to 30 reps
            val row = sheet.createRow(rowNum++)
            val qualityScore = calculateRepQualityScoreOptimized(rep.rangeOfMotion, rep.duration, rep.totalDistance, rep.pathDeviation)
            val grade = getQualityGrade(qualityScore)

            row.createCell(0).apply { setCellValue(rep.repNumber.toDouble()); cellStyle = dataStyle }
            row.createCell(1).apply { setCellValue(rep.duration.toDouble()); cellStyle = dataStyle }
            row.createCell(2).apply { setCellValue(rep.rangeOfMotion.toDouble()); cellStyle = dataStyle }
            row.createCell(3).apply { setCellValue(qualityScore.toDouble()); cellStyle = dataStyle }
            row.createCell(4).apply { setCellValue(grade); cellStyle = dataStyle }
        }

        // Auto-size columns
        headers.forEachIndexed { index, _ ->
            sheet.setColumnWidth(index, 3500)
        }
    }

    // Optimized helper functions
    private fun calculateAverageRepTimeOptimized(repData: List<RepData>): Float {
        return if (repData.isNotEmpty()) repData.map { it.duration }.average().toFloat() else 0f
    }

    private fun calculateAverageROMOptimized(repData: List<RepData>): Float {
        return if (repData.isNotEmpty()) repData.map { it.rangeOfMotion }.average().toFloat() else 0f
    }

    private fun calculateRepConsistencyOptimized(repData: List<RepData>): Float {
        if (repData.size < 2) return 1f

        val durations = repData.map { it.duration }
        val roms = repData.map { it.rangeOfMotion }

        val durationVariability = calculateCoefficientOfVariationOptimized(durations)
        val romVariability = calculateCoefficientOfVariationOptimized(roms)

        return max(0f, 1f - (durationVariability + romVariability) / 2f)
    }

    private fun calculateCoefficientOfVariationOptimized(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val mean = values.average()
        if (mean <= 0) return 0f

        val variance = values.map { (it - mean) * (it - mean) }.average()
        val stdDev = kotlin.math.sqrt(variance)
        return (stdDev / mean).toFloat()
    }

    private fun calculateRepQualityScoreOptimized(rom: Float, duration: Float, distance: Float, deviation: Float): Float {
        // Simplified quality score calculation
        val romScore = min(rom * 8f, 1f) // Assumes good ROM is ~0.125
        val durationScore = if (duration > 0) min(1.5f / duration, 1f) else 0f // Optimal around 1.5s
        val consistencyScore = max(0f, 1f - deviation * 15f)

        return (romScore * 0.4f + durationScore * 0.3f + consistencyScore * 0.3f)
    }

    private fun getQualityGrade(score: Float): String {
        return when {
            score >= 0.9f -> "A+"
            score >= 0.8f -> "A"
            score >= 0.7f -> "B+"
            score >= 0.6f -> "B"
            score >= 0.5f -> "C+"
            score >= 0.4f -> "C"
            score >= 0.3f -> "D"
            else -> "F"
        }
    }

    private fun saveWorkbookToFile(workbook: XSSFWorkbook, baseFileName: String): File {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val fileName = "${baseFileName}_$timestamp.xlsx"
        val file = File(context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), fileName)

        FileOutputStream(file).use { outputStream ->
            workbook.write(outputStream)
        }

        return file
    }

    // Style creation methods
    private fun createTitleStyle(workbook: XSSFWorkbook): XSSFCellStyle {
        val style = workbook.createCellStyle()
        val font = workbook.createFont()
        font.bold = true
        font.fontHeightInPoints = 16
        style.setFont(font)
        style.alignment = HorizontalAlignment.CENTER
        return style
    }

    private fun createHeaderStyle(workbook: XSSFWorkbook): XSSFCellStyle {
        val style = workbook.createCellStyle()
        val font = workbook.createFont()
        font.bold = true
        font.fontHeightInPoints = 12
        style.setFont(font)
        style.fillForegroundColor = IndexedColors.GREY_25_PERCENT.getIndex()
        style.fillPattern = FillPatternType.SOLID_FOREGROUND
        style.borderBottom = BorderStyle.THIN
        style.borderTop = BorderStyle.THIN
        style.borderRight = BorderStyle.THIN
        style.borderLeft = BorderStyle.THIN
        return style
    }

    private fun createDataStyle(workbook: XSSFWorkbook): XSSFCellStyle {
        val style = workbook.createCellStyle()
        style.borderBottom = BorderStyle.THIN
        style.borderTop = BorderStyle.THIN
        style.borderRight = BorderStyle.THIN
        style.borderLeft = BorderStyle.THIN
        return style
    }

    private fun formatDuration(durationMs: Long): String {
        val minutes = durationMs / 60000
        val seconds = (durationMs % 60000) / 1000
        return "${minutes}m ${seconds}s"
    }

    /**
     * Share the generated report file
     */
    fun shareReport(file: File) {
        try {
            val uri = FileProvider.getUriForFile(context, PROVIDER_AUTHORITY, file)
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = if (file.extension == "xlsx") {
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                } else {
                    "text/csv"
                }
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "Bar Path Analysis Report")
                putExtra(Intent.EXTRA_TEXT, "Bar path analysis report generated by Bar Path Detector app.")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }

            val chooser = Intent.createChooser(intent, "Share Report")
            chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(chooser)
        } catch (e: Exception) {
            Log.e(TAG, "Error sharing report: ${e.message}", e)
        }
    }
}