package com.example.appusagelogger

import android.Manifest
import android.app.AppOpsManager
import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Process
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
//import androidx.compose.foundation.lazy.LazyColumn
//import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.edit
import androidx.work.*
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit

class MainActivity : ComponentActivity() {

    private val _logStatus = mutableStateOf("Initializing...")
    val logStatus: State<String> get() = _logStatus

    private val logsPerPage = 30
    private var allLogs = listOf<String>()
    private val _displayLogs = mutableStateListOf<String>()
    private var currentIndex = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        requestPermissions()
        if (!hasUsageStatsPermission(this)) {
            startActivity(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS))
        }

        val prefs = getSharedPreferences("app_usage_prefs", MODE_PRIVATE)
        val firstRunDone = prefs.getBoolean("first_run_done", false)
        if (!firstRunDone) {
            dumpMonthlyUsage()
            prefs.edit {
                putBoolean("first_run_done", true)
            }
        }

        scheduleDailyWork()

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    AppUi()
                }
            }
        }
    }

    @Composable
    private fun AppUi() {
        var showLogsDialog by remember { mutableStateOf(false) }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text(
                text = "📱 Smart App Recommender",
                style = MaterialTheme.typography.headlineMedium
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = logStatus.value)
            Spacer(modifier = Modifier.height(16.dp))

            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = {
                    dumpDailyUsage()
                }) {
                    Text("Refresh Logs")
                }
                Button(onClick = {
                    CoroutineScope(Dispatchers.IO).launch {
                        val logsText = readAppUsageLog(this@MainActivity)
                        allLogs = logsText.lines().filter { it.isNotBlank() }
                        currentIndex = 0
                        _displayLogs.clear()
                        if (allLogs.isNotEmpty()) {
                            val start = (allLogs.size - logsPerPage).coerceAtLeast(0)
                            _displayLogs.addAll(allLogs.subList(start, allLogs.size).asReversed())
                            currentIndex = _displayLogs.size
                        }
                        withContext(Dispatchers.Main) {
                            showLogsDialog = true
                        }
                    }
                }) {
                    Text("Show Logs")
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Start) {
                Button(onClick = {
                    val pred = predictOutcome()
                    _logStatus.value = "Predicted: $pred"
                }) {
                    Text("Predict Outcome")
                }
            }
        }

        if (showLogsDialog) {
            AlertDialog(
                onDismissRequest = { showLogsDialog = false },
                confirmButton = {
                    TextButton(onClick = { showLogsDialog = false }) {
                        Text("Close")
                    }
                },
                title = { Text("App Usage Log") },
                text = {
                    Column(modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState())) {
                        if (_displayLogs.isEmpty()) {
                            Text("No log file present or file is empty.")
                        } else {
                            _displayLogs.forEach { logLine ->
                                Text(logLine)
                            }
                            if (currentIndex < allLogs.size) {
                                Button(
                                    onClick = {
                                        val nextIndex = (currentIndex + logsPerPage).coerceAtMost(allLogs.size)
                                        _displayLogs.addAll(allLogs.takeLast(nextIndex).dropLast(currentIndex).asReversed())
                                        currentIndex = nextIndex
                                    },
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Text("Load More")
                                }
                            }
                        }
                    }
                }
            )
        }
    }

    private fun predictOutcome(): String {
        return "Feature unchanged" // Keeping same signature, but no longer using sessions list
    }

    fun dumpMonthlyUsage() {
        val prefs = getSharedPreferences("app_usage_prefs", MODE_PRIVATE)
        val lastSavedTime = prefs.getLong("last_saved_time", 0L)

        val now = System.currentTimeMillis()
        val allEvents = getAppEvents(this, lastSavedTime, now)

        if (allEvents.isEmpty()) {
            _logStatus.value = "No new usage events."
            return
        }

        val sessionsBuilt = buildAppUsageSessions(this, allEvents)
        val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

        val camGranted = ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        val micGranted = ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

        val logs = sessionsBuilt.joinToString("\n") { entry ->
            val duration = entry.endTime - entry.startTime
            "${formatter.format(Date(entry.startTime))}," +
                    "${formatter.format(Date(entry.endTime))}," +
                    "${entry.appName},${entry.eventType},$duration," +
                    "camera=$camGranted,audio=$micGranted"
        }

        writeAppUsageLog(this, logs)

        val latest = sessionsBuilt.maxOfOrNull { it.endTime } ?: now
        prefs.edit{
            putLong("last_saved_time", latest)
        }

        _logStatus.value = "Usage logs updated: ${sessionsBuilt.size} new sessions"
    }

    fun dumpDailyUsage() {
        val prefs = getSharedPreferences("app_usage_prefs", MODE_PRIVATE)
        val lastSavedTime = prefs.getLong("last_saved_time", 0L)

        val now = System.currentTimeMillis()
        val allEvents = getAppEvents(this, lastSavedTime, now)

        if (allEvents.isEmpty()) {
            _logStatus.value = "No new usage events."
            return
        }

        val sessionsBuilt = buildAppUsageSessions(this, allEvents)
        val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

        val camGranted = ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        val micGranted = ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

        val logs = sessionsBuilt.joinToString("\n") { entry ->
            val duration = entry.endTime - entry.startTime
            "${formatter.format(Date(entry.startTime))}," +
                    "${formatter.format(Date(entry.endTime))}," +
                    "${entry.appName},${entry.eventType},$duration," +
                    "camera=$camGranted,audio=$micGranted"
        }

        writeAppUsageLog(this, logs)

        val latest = sessionsBuilt.maxOfOrNull { it.endTime } ?: now
        prefs.edit{
            putLong("last_saved_time", latest)
        }

        _logStatus.value = "Daily logs updated: ${sessionsBuilt.size} new sessions"
    }

    private fun scheduleDailyWork() {
        val dailyWorkRequest = PeriodicWorkRequestBuilder<DailyUsageWorker>(1, TimeUnit.DAYS)
            .setInitialDelay(getDelayUntil1AM(), TimeUnit.MILLISECONDS)
            .build()

        WorkManager.getInstance(this).enqueueUniquePeriodicWork(
            "DailyUsageLog",
            ExistingPeriodicWorkPolicy.UPDATE,
            dailyWorkRequest
        )
    }

    private fun requestPermissions() {
        val requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
                val camGranted = permissions[Manifest.permission.CAMERA] == true
                val micGranted = permissions[Manifest.permission.RECORD_AUDIO] == true

                _logStatus.value = when {
                    camGranted && micGranted -> "Camera & Audio permissions granted"
                    camGranted -> "Only Camera granted"
                    micGranted -> "Only Audio granted"
                    else -> "Camera & Audio permissions denied"
                }
            }

        val neededPermissions = mutableListOf<String>()
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            neededPermissions.add(Manifest.permission.CAMERA)
        }
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            neededPermissions.add(Manifest.permission.RECORD_AUDIO)
        }

        if (neededPermissions.isNotEmpty()) {
            requestPermissionLauncher.launch(neededPermissions.toTypedArray())
        }
    }
}

@Suppress("DEPRECATION")
fun hasUsageStatsPermission(context: Context): Boolean {
    val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
    val mode = appOps.checkOpNoThrow(
        AppOpsManager.OPSTR_GET_USAGE_STATS,
        Process.myUid(),
        context.packageName
    )
    return mode == AppOpsManager.MODE_ALLOWED
}

// --- Data classes & helpers ---

data class AppUsageEvent(
    val timeStamp: Long,
    val eventType: Int,
    val packageName: String?
)

data class AppUsageLogEntry(
    val appName: String,
    val startTime: Long,
    var endTime: Long,
    val eventType: String
)

fun getAppEvents(context: Context, start: Long, end: Long): List<AppUsageEvent> {
    val usageStatsManager = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
    val usageEvents = usageStatsManager.queryEvents(start, end)
    val events = mutableListOf<AppUsageEvent>()
    val event = UsageEvents.Event()
    while (usageEvents.hasNextEvent()) {
        usageEvents.getNextEvent(event)
        events.add(AppUsageEvent(event.timeStamp, event.eventType, event.packageName))
    }
    return events
}

fun buildAppUsageSessions(context: Context, events: List<AppUsageEvent>): List<AppUsageLogEntry> {
    val sessions = mutableListOf<AppUsageLogEntry>()
    val activeApps = mutableMapOf<String, Long>()

    val systemPrefixes = listOf(
        "android", "com.android.", "com.google.android.", "com.google.android.gms",
        "com.sec.android.", "com.samsung.android.", "com.qualcomm.", "com.huawei.",
        "com.mi.", "com.coloros.", "com.vivo.", "com.oppo.", "launcher", "setupwizard"
    )

    for (ev in events) {
        val appName = getAppName(context, ev.packageName)
        if (systemPrefixes.any { appName.startsWith(it) }) continue
        val evLabel = eventTypeLabel(ev.eventType)
        when (ev.eventType) {
            UsageEvents.Event.ACTIVITY_RESUMED -> activeApps[appName] = ev.timeStamp
            UsageEvents.Event.ACTIVITY_PAUSED -> {
                val start = activeApps.remove(appName)
                if (start != null) {
                    sessions.add(AppUsageLogEntry(appName, start, ev.timeStamp, evLabel))
                }
            }
        }
    }
    return sessions
}

fun getAppName(context: Context, packageName: String?): String {
    if (packageName.isNullOrBlank()) return "UNKNOWN"
    return try {
        val pm = context.packageManager
        val appInfo = pm.getApplicationInfo(packageName, 0)
        pm.getApplicationLabel(appInfo).toString()
    } catch (e: Exception) {
        e.printStackTrace()
        packageName.substringAfterLast('.')
    }
}

fun eventTypeLabel(type: Int): String {
    return when (type) {
        UsageEvents.Event.ACTIVITY_RESUMED -> "RESUMED"
        UsageEvents.Event.ACTIVITY_PAUSED -> "PAUSED"
        else -> "UNKNOWN"
    }
}

fun writeAppUsageLog(context: Context, logContent: String) {
    try {
        val file = File(context.filesDir, "app_usage_log.txt")
        FileOutputStream(file, true).use { fos ->
            fos.write((logContent + "\n").toByteArray())
        }
        android.util.Log.d("AppUsageLogger", "Wrote ${logContent.length} chars to usage log")
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

fun readAppUsageLog(context: Context): String {
    return try {
        val file = File(context.filesDir, "app_usage_log.txt")
        if (file.exists()) file.readText() else ""
    } catch (e: Exception) {
        e.printStackTrace()
        return ""
    }
}

fun getDelayUntil1AM(): Long {
    val now = Calendar.getInstance()
    val next1AM = Calendar.getInstance().apply {
        set(Calendar.HOUR_OF_DAY, 1)
        set(Calendar.MINUTE, 0)
        set(Calendar.SECOND, 0)
        set(Calendar.MILLISECOND, 0)
        if (before(now)) add(Calendar.DAY_OF_MONTH, 1)
    }
    return next1AM.timeInMillis - now.timeInMillis
}
