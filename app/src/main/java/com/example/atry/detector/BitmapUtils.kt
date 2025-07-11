package com.example.atry.detector

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

/**
 * Optimized BitmapUtils for better performance
 * Includes object reuse and efficient YUV conversion
 */
object BitmapUtils {

    // Reuse these objects to avoid allocation overhead
    private val reusableMatrix = Matrix()
    private val reusableByteArrayOutputStream = ByteArrayOutputStream()

    fun imageProxyToBitmapOptimized(imageProxy: ImageProxy): Bitmap {
        return when (imageProxy.format) {
            ImageFormat.YUV_420_888 -> convertYuv420888ToBitmap(imageProxy)
            else -> imageProxyToBitmap(imageProxy) // fallback to original method
        }
    }

    private fun convertYuv420888ToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        // Reuse byte array if possible
        val nv21 = ByteArray(ySize + uSize + vSize)

        // Copy Y plane
        yBuffer.get(nv21, 0, ySize)

        // Copy UV planes (interleaved for NV21)
        val uvPixelStride = imageProxy.planes[1].pixelStride
        if (uvPixelStride == 1) {
            // Planes are already interleaved
            uBuffer.get(nv21, ySize, uSize)
            vBuffer.get(nv21, ySize + uSize, vSize)
        } else {
            // Need to interleave U and V
            val uvBuffer = ByteArray(uSize + vSize)
            uBuffer.get(uvBuffer, 0, uSize)
            vBuffer.get(uvBuffer, uSize, vSize)

            // Interleave
            var uvIndex = 0
            for (i in 0 until uSize step uvPixelStride) {
                nv21[ySize + uvIndex] = uvBuffer[uSize + i]
                nv21[ySize + uvIndex + 1] = uvBuffer[i]
                uvIndex += 2
            }
        }

        // Create YuvImage and convert to bitmap
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)

        // Reuse ByteArrayOutputStream
        reusableByteArrayOutputStream.reset()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 80, reusableByteArrayOutputStream)
        val jpegArray = reusableByteArrayOutputStream.toByteArray()

        // Decode bitmap
        val bitmap = android.graphics.BitmapFactory.decodeByteArray(jpegArray, 0, jpegArray.size)

        // Apply rotation if needed
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        return if (rotationDegrees != 0) {
            reusableMatrix.reset()
            reusableMatrix.postRotate(rotationDegrees.toFloat())
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, reusableMatrix, false)
        } else {
            bitmap
        }
    }

    // Keep original method as fallback
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val bytes = out.toByteArray()

        val bitmap = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}