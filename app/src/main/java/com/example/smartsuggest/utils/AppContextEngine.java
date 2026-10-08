package com.example.smartsuggest.utils;

import android.app.usage.UsageEvents;
import android.app.usage.UsageStatsManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.media.AudioManager;
import android.os.BatteryManager;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AppContextEngine {

    public static final int MAX_CANDIDATE_APPS = 10;
    public static final int FEATURE_COUNT = 10;

    public static class AppItem {
        public final String packageName;
        public final String label;

        public AppItem(String packageName, String label) {
            this.packageName = packageName;
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    public static List<AppItem> getInstalledCandidateApps(Context context) {
        PackageManager pm = context.getPackageManager();
        Intent mainIntent = new Intent(Intent.ACTION_MAIN, null);
        mainIntent.addCategory(Intent.CATEGORY_LAUNCHER);

        List<ResolveInfo> resolvedList = pm.queryIntentActivities(mainIntent, 0);
        List<AppItem> appList = new ArrayList<>();
        String myPkg = context.getPackageName();

        if (resolvedList != null) {
            // Sort resolved list by package name deterministically
            Collections.sort(resolvedList, (a, b) -> {
                String pkgA = (a.activityInfo != null && a.activityInfo.packageName != null) ? a.activityInfo.packageName : "";
                String pkgB = (b.activityInfo != null && b.activityInfo.packageName != null) ? b.activityInfo.packageName : "";
                return pkgA.compareTo(pkgB);
            });

            for (ResolveInfo info : resolvedList) {
                if (info.activityInfo == null) continue;
                String pkg = info.activityInfo.packageName;
                if (pkg.equals(myPkg)) continue; // ignore SmartSuggest itself

                CharSequence labelCs = info.loadLabel(pm);
                String label = labelCs != null ? labelCs.toString().trim() : pkg;

                // Deduplicate by package name
                boolean exists = false;
                for (AppItem existing : appList) {
                    if (existing.packageName.equals(pkg)) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    appList.add(new AppItem(pkg, label));
                }

                if (appList.size() >= MAX_CANDIDATE_APPS) break;
            }
        }

        // Fallback filler if device has fewer than 10 launcher apps
        while (appList.size() < MAX_CANDIDATE_APPS) {
            int idx = appList.size() + 1;
            appList.add(new AppItem("com.app.slot" + idx, "App Slot #" + idx));
        }

        return appList;
    }

    public static List<String> getPackageNames(List<AppItem> apps) {
        List<String> names = new ArrayList<>(apps.size());
        for (AppItem app : apps) {
            names.add(app.packageName);
        }
        return names;
    }

    /**
     * Extracts real-time device context features into a 3D INDArray for LSTM input [1, 10, 1].
     */
    public static INDArray extractCurrentContextFeatures(Context context, List<AppItem> candidateApps) {
        float[] features = new float[FEATURE_COUNT];

        // 1. Time Context
        Calendar now = Calendar.getInstance();
        int hour = now.get(Calendar.HOUR_OF_DAY);
        int minute = now.get(Calendar.MINUTE);
        int dayOfWeek = now.get(Calendar.DAY_OF_WEEK); // 1 = Sunday, 7 = Saturday

        features[0] = (hour + (minute / 60.0f)) / 24.0f;           // Feature 0: Hour of Day (0.0 to 1.0)
        features[1] = (dayOfWeek - 1.0f) / 6.0f;                    // Feature 1: Day of Week (0.0 to 1.0)
        features[2] = (dayOfWeek == Calendar.SATURDAY || dayOfWeek == Calendar.SUNDAY) ? 1.0f : 0.0f; // Feature 2: Is Weekend

        // 2. Battery & Charging State
        IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        Intent batteryStatus = context.registerReceiver(null, ifilter);
        if (batteryStatus != null) {
            int level = batteryStatus.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
            int scale = batteryStatus.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
            int status = batteryStatus.getIntExtra(BatteryManager.EXTRA_STATUS, -1);
            features[3] = (scale > 0) ? (level / (float) scale) : 0.5f; // Feature 3: Battery Level (0.0 to 1.0)
            boolean isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                    status == BatteryManager.BATTERY_STATUS_FULL;
            features[4] = isCharging ? 1.0f : 0.0f;                     // Feature 4: Is Charging
        } else {
            features[3] = 0.5f;
            features[4] = 0.0f;
        }

        // 3. Audio / Media Context
        AudioManager audioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
        if (audioManager != null) {
            features[5] = audioManager.isMusicActive() ? 1.0f : 0.0f;  // Feature 5: Music/Audio Playing
            int ringer = audioManager.getRingerMode();
            features[6] = (ringer == AudioManager.RINGER_MODE_NORMAL) ? 1.0f : 0.0f; // Feature 6: Normal Ringer
        } else {
            features[5] = 0.0f;
            features[6] = 1.0f;
        }

        // 4. Usage Context: Last Used App from UsageEvents
        int lastAppIndex = getLastUsedAppIndex(context, candidateApps);
        features[7] = (lastAppIndex >= 0) ? (lastAppIndex / (float) MAX_CANDIDATE_APPS) : 0.0f; // Feature 7: Last App Index
        features[8] = (lastAppIndex >= 0) ? 1.0f : 0.0f;                                         // Feature 8: Has Recent App Transition
        features[9] = (hour >= 22 || hour < 6) ? 1.0f : 0.0f;                                  // Feature 9: Night Time Mode

        // Format into 3D tensor: [batch=1, input_size=10, time_steps=1]
        INDArray tensor = Nd4j.create(1, FEATURE_COUNT, 1);
        for (int i = 0; i < FEATURE_COUNT; i++) {
            tensor.putScalar(new int[]{0, i, 0}, features[i]);
        }
        return tensor;
    }

    /**
     * Reads recent UsageEvents to identify the index of the last active candidate app.
     */
    private static int getLastUsedAppIndex(Context context, List<AppItem> candidateApps) {
        UsageStatsManager usm = (UsageStatsManager) context.getSystemService(Context.USAGE_STATS_SERVICE);
        if (usm == null) return -1;

        long endTime = System.currentTimeMillis();
        long startTime = endTime - (1000 * 60 * 60 * 4); // last 4 hours

        UsageEvents events = usm.queryEvents(startTime, endTime);
        if (events == null) return -1;

        UsageEvents.Event event = new UsageEvents.Event();
        String lastPkg = null;

        while (events.hasNextEvent()) {
            events.getNextEvent(event);
            if (event.getEventType() == UsageEvents.Event.ACTIVITY_RESUMED) {
                lastPkg = event.getPackageName();
            }
        }

        if (lastPkg == null) return -1;

        for (int i = 0; i < candidateApps.size(); i++) {
            if (candidateApps.get(i).packageName.equals(lastPkg)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Extracts actual app launch transitions from UsageStatsManager over the last 24-48h
     * and constructs a valid training DataSet for on-device LSTM fine-tuning.
     */
    public static DataSet buildTrainingDataSet(Context context, List<AppItem> candidateApps) {
        UsageStatsManager usm = (UsageStatsManager) context.getSystemService(Context.USAGE_STATS_SERVICE);
        if (usm == null) return null;

        long endTime = System.currentTimeMillis();
        long startTime = endTime - (1000L * 60 * 60 * 24); // past 24 hours

        Map<String, Integer> pkgToIndex = new HashMap<>();
        for (int i = 0; i < candidateApps.size(); i++) {
            pkgToIndex.put(candidateApps.get(i).packageName, i);
        }

        List<Integer> targetIndices = new ArrayList<>();
        List<Long> eventTimestamps = new ArrayList<>();

        UsageEvents events = usm.queryEvents(startTime, endTime);
        if (events != null) {
            UsageEvents.Event event = new UsageEvents.Event();
            while (events.hasNextEvent()) {
                events.getNextEvent(event);
                if (event.getEventType() == UsageEvents.Event.ACTIVITY_RESUMED) {
                    String pkg = event.getPackageName();
                    if (pkgToIndex.containsKey(pkg)) {
                        targetIndices.add(pkgToIndex.get(pkg));
                        eventTimestamps.add(event.getTimeStamp());
                    }
                }
            }
        }

        if (targetIndices.isEmpty()) {
            return null;
        }

        int timeSteps = targetIndices.size();
        INDArray input = Nd4j.zeros(1, FEATURE_COUNT, timeSteps);
        INDArray labels = Nd4j.zeros(1, MAX_CANDIDATE_APPS, timeSteps);

        for (int t = 0; t < timeSteps; t++) {
            int targetAppIdx = targetIndices.get(t);
            long timestamp = eventTimestamps.get(t);
            int prevAppIdx = (t > 0) ? targetIndices.get(t - 1) : -1;

            Calendar c = Calendar.getInstance();
            c.setTimeInMillis(timestamp);
            int hour = c.get(Calendar.HOUR_OF_DAY);
            int minute = c.get(Calendar.MINUTE);
            int dayOfWeek = c.get(Calendar.DAY_OF_WEEK);

            float hourFeature = (hour + (minute / 60.0f)) / 24.0f;
            float dayFeature = (dayOfWeek - 1.0f) / 6.0f;
            float isWeekend = (dayOfWeek == Calendar.SATURDAY || dayOfWeek == Calendar.SUNDAY) ? 1.0f : 0.0f;
            float prevAppFeature = (prevAppIdx >= 0) ? (prevAppIdx / (float) MAX_CANDIDATE_APPS) : 0.0f;
            float hasTransition = (prevAppIdx >= 0) ? 1.0f : 0.0f;
            float nightMode = (hour >= 22 || hour < 6) ? 1.0f : 0.0f;

            // Populate features matching real-time context
            input.putScalar(new int[]{0, 0, t}, hourFeature);
            input.putScalar(new int[]{0, 1, t}, dayFeature);
            input.putScalar(new int[]{0, 2, t}, isWeekend);
            input.putScalar(new int[]{0, 3, t}, 0.5f);
            input.putScalar(new int[]{0, 4, t}, 0.0f);
            input.putScalar(new int[]{0, 5, t}, 0.0f);
            input.putScalar(new int[]{0, 6, t}, 1.0f);
            input.putScalar(new int[]{0, 7, t}, prevAppFeature);
            input.putScalar(new int[]{0, 8, t}, hasTransition);
            input.putScalar(new int[]{0, 9, t}, nightMode);

            // Valid one-hot ground truth label for softmax loss
            labels.putScalar(new int[]{0, targetAppIdx, t}, 1.0f);
        }

        return new DataSet(input, labels);
    }
}
