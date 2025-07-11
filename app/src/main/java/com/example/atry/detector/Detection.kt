package com.example.atry.detector

import android.graphics.RectF

/** Holds one detected box (normalized [0..1]), its confidence score and class id. */
data class Detection(
    val bbox: RectF,
    val score: Float,
    val classId: Int
)