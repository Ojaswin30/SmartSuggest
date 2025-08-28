package com.example.appusagelogger

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import java.text.SimpleDateFormat
import java.util.*

class DailyUsageWorker(appContext: Context, params: WorkerParameters) :
    CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        val now = System.currentTimeMillis()
        val yesterday = now - 1000L * 60 * 60 * 24
        val events = getAppEvents(applicationContext, yesterday, now)
        val sessions = buildAppUsageSessions(applicationContext, events)
        val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

        sessions.forEach { entry ->
            val duration = entry.endTime - entry.startTime
            val logLine =
                "${formatter.format(Date(entry.startTime))},${formatter.format(Date(entry.endTime))},${entry.appName},${entry.eventType},$duration"
            writeAppUsageLog(applicationContext, logLine)
        }

        return Result.success()
    }
}
