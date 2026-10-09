package com.example.smartsuggest.utils;

import android.app.usage.UsageEvents;
import android.app.usage.UsageStats;
import android.app.usage.UsageStatsManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Engine responsible for extracting real-time device context, ranking and discovering
 * installed candidate apps using foreground usage stats, and preparing sequence datasets
 * for LSTM model training and inference.
 */
public class AppContextEngine {

    public static final int MAX_CANDIDATE_APPS = 10;
    public static final int FEATURE_COUNT = 10;

    /**
     * Data representation of an installed application candidate.
     */
    public static class AppItem {
        public final String packageName;
        public final String label;

        /**
         * Constructs an AppItem with package name and human-readable label.
         *
         * @param packageName Application unique package name.
         * @param label Human-readable application display name.
         */
        public AppItem(String packageName, String label) {
            this.packageName = packageName;
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    /**
     * Discovers installed candidate launcher apps on the device, ranking them by actual
     * foreground usage time and recency over the past 7 days, prioritizing user-installed
     * interactive applications over unused system packages.
     *
     * @param context Application context used to query PackageManager and UsageStatsManager.
     * @return Deterministically ordered list of top candidate {@link AppItem} instances.
     */
    public static List<AppItem> getInstalledCandidateApps(Context context) {
        PackageManager pm = context.getPackageManager();
        Intent mainIntent = new Intent(Intent.ACTION_MAIN, null);
        mainIntent.addCategory(Intent.CATEGORY_LAUNCHER);

        List<ResolveInfo> resolvedList = pm.queryIntentActivities(mainIntent, 0);
        List<AppItem> launchableApps = new ArrayList<>();
        Map<String, Boolean> isSystemAppMap = new HashMap<>();
        String myPkg = context.getPackageName();

        if (resolvedList != null) {
            for (ResolveInfo info : resolvedList) {
                if (info.activityInfo == null) continue;
                String pkg = info.activityInfo.packageName;
                if (pkg.equals(myPkg)) continue; // ignore SmartSuggest itself

                CharSequence labelCs = info.loadLabel(pm);
                String label = labelCs != null ? labelCs.toString().trim() : pkg;

                // Deduplicate by package name
                boolean exists = false;
                for (AppItem existing : launchableApps) {
                    if (existing.packageName.equals(pkg)) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    launchableApps.add(new AppItem(pkg, label));
                    boolean isSys = (info.activityInfo.applicationInfo != null) &&
                            ((info.activityInfo.applicationInfo.flags & android.content.pm.ApplicationInfo.FLAG_SYSTEM) != 0);
                    isSystemAppMap.put(pkg, isSys);
                }
            }
        }

        // Query usage stats to rank launchable apps by foreground usage time
        Map<String, Long> usageTimeMap = new HashMap<>();
        Map<String, Long> lastUsedMap = new HashMap<>();
        UsageStatsManager usm = (UsageStatsManager) context.getSystemService(Context.USAGE_STATS_SERVICE);
        if (usm != null) {
            long now = System.currentTimeMillis();
            long startTime = now - (1000L * 60 * 60 * 24 * 7); // last 7 days
            Map<String, UsageStats> aggregatedStats = usm.queryAndAggregateUsageStats(startTime, now);
            if (aggregatedStats != null && !aggregatedStats.isEmpty()) {
                for (Map.Entry<String, UsageStats> entry : aggregatedStats.entrySet()) {
                    UsageStats stats = entry.getValue();
                    if (stats != null) {
                        usageTimeMap.put(entry.getKey(), stats.getTotalTimeInForeground());
                        lastUsedMap.put(entry.getKey(), stats.getLastTimeUsed());
                    }
                }
            } else {
                // Fallback to queryUsageStats if queryAndAggregateUsageStats is empty
                List<UsageStats> statsList = usm.queryUsageStats(UsageStatsManager.INTERVAL_DAILY, startTime, now);
                if (statsList != null) {
                    for (UsageStats stats : statsList) {
                        long existing = usageTimeMap.containsKey(stats.getPackageName()) ? usageTimeMap.get(stats.getPackageName()) : 0L;
                        usageTimeMap.put(stats.getPackageName(), existing + stats.getTotalTimeInForeground());
                        long last = lastUsedMap.containsKey(stats.getPackageName()) ? lastUsedMap.get(stats.getPackageName()) : 0L;
                        if (stats.getLastTimeUsed() > last) {
                            lastUsedMap.put(stats.getPackageName(), stats.getLastTimeUsed());
                        }
                    }
                }
            }
        }

        // Separate apps with real foreground activity (> 0 ms) from unused/background apps
        List<AppItem> activeForegroundApps = new ArrayList<>();
        List<AppItem> otherApps = new ArrayList<>();

        for (AppItem app : launchableApps) {
            long fgTime = usageTimeMap.containsKey(app.packageName) ? usageTimeMap.get(app.packageName) : 0L;
            if (fgTime > 0) {
                activeForegroundApps.add(app);
            } else {
                otherApps.add(app);
            }
        }

        // Sort active foreground apps descending by total foreground time, then last time used
        Collections.sort(activeForegroundApps, (a, b) -> {
            long timeA = usageTimeMap.containsKey(a.packageName) ? usageTimeMap.get(a.packageName) : 0L;
            long timeB = usageTimeMap.containsKey(b.packageName) ? usageTimeMap.get(b.packageName) : 0L;
            if (timeA != timeB) {
                return Long.compare(timeB, timeA);
            }
            long lastA = lastUsedMap.containsKey(a.packageName) ? lastUsedMap.get(a.packageName) : 0L;
            long lastB = lastUsedMap.containsKey(b.packageName) ? lastUsedMap.get(b.packageName) : 0L;
            return Long.compare(lastB, lastA);
        });

        // Sort other launchable apps (non-system user apps prioritized over system apps)
        Collections.sort(otherApps, (a, b) -> {
            boolean sysA = isSystemAppMap.containsKey(a.packageName) && Boolean.TRUE.equals(isSystemAppMap.get(a.packageName));
            boolean sysB = isSystemAppMap.containsKey(b.packageName) && Boolean.TRUE.equals(isSystemAppMap.get(b.packageName));
            if (sysA != sysB) {
                return sysA ? 1 : -1; // user-installed apps first
            }
            return a.label.compareToIgnoreCase(b.label);
        });

        // Pick top candidate apps prioritizing active foreground apps first
        List<AppItem> candidates = new ArrayList<>();
        for (AppItem app : activeForegroundApps) {
            if (candidates.size() >= MAX_CANDIDATE_APPS) break;
            candidates.add(app);
        }
        for (AppItem app : otherApps) {
            if (candidates.size() >= MAX_CANDIDATE_APPS) break;
            candidates.add(app);
        }

        // Sort the chosen candidates by package name for stable, deterministic indexing in model
        Collections.sort(candidates, (a, b) -> a.packageName.compareTo(b.packageName));

        // Fallback filler if device has fewer than 10 launcher apps
        while (candidates.size() < MAX_CANDIDATE_APPS) {
            int idx = candidates.size() + 1;
            candidates.add(new AppItem("com.app.slot" + idx, "App Slot #" + idx));
        }

        return candidates;
    }

    /**
     * Extracts a list of package name strings from a list of {@link AppItem} instances.
     *
     * @param apps List of AppItem objects.
     * @return List containing package names as strings.
     */
    public static List<String> getPackageNames(List<AppItem> apps) {
        List<String> names = new ArrayList<>(apps.size());
        for (AppItem app : apps) {
            names.add(app.packageName);
        }
        return names;
    }

    /**
     * Extracts real-time device context features into a 3D INDArray for LSTM input [1, 10, 1].
     *
     * @param context Application context used to query system services.
     * @param candidateApps List of candidate apps for target indexing.
     * @return 3D INDArray representing context tensor [batch=1, features=10, timesteps=1].
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

        // 2. Battery & Charging State (consistent with training constants)
        features[3] = 0.5f; // Feature 3: Nominal Battery Level
        features[4] = 0.0f; // Feature 4: Is Charging

        // 3. Audio / Media Context (consistent with training constants)
        features[5] = 0.0f; // Feature 5: Music/Audio Playing
        features[6] = 1.0f; // Feature 6: Normal Ringer

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
     * Ignores our own app so the previous foreground app is correctly identified.
     *
     * @param context Application context used to query UsageStatsManager.
     * @param candidateApps List of candidate apps to search against.
     * @return Index of the last active candidate app in the candidate list, or -1 if none found.
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
        String myPkg = context.getPackageName();

        while (events.hasNextEvent()) {
            events.getNextEvent(event);
            if (event.getEventType() == UsageEvents.Event.ACTIVITY_RESUMED) {
                String pkg = event.getPackageName();
                // Exclude our own app from being recognized as the previous app
                if (pkg != null && !pkg.equals(myPkg)) {
                    lastPkg = pkg;
                }
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
     *
     * @param context Application context used to query UsageStatsManager.
     * @param candidateApps List of candidate apps mapped to output classes.
     * @return {@link DataSet} containing sequences of input context features and one-hot labels, or null if insufficient data.
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
        String myPkg = context.getPackageName();

        UsageEvents events = usm.queryEvents(startTime, endTime);
        if (events != null) {
            UsageEvents.Event event = new UsageEvents.Event();
            String prevPkg = null;
            while (events.hasNextEvent()) {
                events.getNextEvent(event);
                if (event.getEventType() == UsageEvents.Event.ACTIVITY_RESUMED) {
                    String pkg = event.getPackageName();
                    if (pkg != null && !pkg.equals(myPkg) && pkgToIndex.containsKey(pkg)) {
                        // Avoid consecutive duplicate activity resumes of the exact same app
                        if (!pkg.equals(prevPkg)) {
                            targetIndices.add(pkgToIndex.get(pkg));
                            eventTimestamps.add(event.getTimeStamp());
                            prevPkg = pkg;
                        }
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
