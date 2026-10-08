package com.example.smartsuggest.workers;

import android.app.usage.UsageEvents;
import android.app.usage.UsageStatsManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;

import androidx.annotation.NonNull;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import com.example.smartsuggest.data.AppDatabase;
import com.example.smartsuggest.data.InferenceResult;
import com.example.smartsuggest.model.LSTMModel;
import com.example.smartsuggest.utils.AppContextEngine;
import com.example.smartsuggest.utils.ModelFileUtils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Locale;

public class InferenceWorker extends Worker {

    public InferenceWorker(@NonNull Context context, @NonNull WorkerParameters params) {
        super(context, params);
    }

    @NonNull
    @Override
    public Result doWork() {

        // step 1 - check if foreground app is in rest list / launcher
        if (!isForegroundAppInRestList()) {
            return Result.success(); // skip silently when user is active in an app
        }

        MultiLayerNetwork network;
        List<AppContextEngine.AppItem> installedApps;
        INDArray input;
        INDArray output;

        synchronized (ModelFileUtils.MODEL_LOCK) {
            // step 2 - discover installed candidate apps
            installedApps = AppContextEngine.getInstalledCandidateApps(getApplicationContext());
            List<String> currentPkgs = AppContextEngine.getPackageNames(installedApps);

            // step 3 - load model (from file if valid, else fresh)
            if (ModelFileUtils.isModelValid(getApplicationContext(), currentPkgs)) {
                network = ModelFileUtils.loadModel(getApplicationContext());
            } else {
                if (ModelFileUtils.modelExists(getApplicationContext())) {
                    ModelFileUtils.invalidateModel(getApplicationContext());
                }
                LSTMModel freshModel = new LSTMModel();
                network = freshModel.getNetwork();
            }

            if (network == null) return Result.failure();

            // step 4 - extract real live device context
            input = AppContextEngine.extractCurrentContextFeatures(getApplicationContext(), installedApps);
            output = network.output(input);

            // step 5 - release model reference
            network = null;
        }

        // step 6 - find top predicted app
        int bestIndex = 0;
        double maxProb = -1.0;
        for (int i = 0; i < installedApps.size(); i++) {
            double prob = output.getDouble(0, i, 0);
            if (prob > maxProb) {
                maxProb = prob;
                bestIndex = i;
            }
        }
        String topAppName = installedApps.get(bestIndex).label;

        // step 7 - save output and input snapshot to Room database
        AppDatabase db = AppDatabase.getInstance(getApplicationContext());
        InferenceResult result = new InferenceResult(
                System.currentTimeMillis(),
                String.format(Locale.getDefault(), "Top: %s (%.1f%%)", topAppName, maxProb * 100),
                input.toString()
        );
        db.inferenceResultDao().insert(result);
        db.inferenceResultDao().pruneOld();

        return Result.success();
    }

    private boolean isForegroundAppInRestList() {
        UsageStatsManager usageStatsManager = (UsageStatsManager)
                getApplicationContext().getSystemService(Context.USAGE_STATS_SERVICE);

        if (usageStatsManager == null) return false;

        long currentTime = System.currentTimeMillis();
        long startTime = currentTime - (1000L * 60 * 60 * 4); // search last 4 hours

        UsageEvents events = usageStatsManager.queryEvents(startTime, currentTime);
        if (events == null) return false;

        UsageEvents.Event event = new UsageEvents.Event();
        String foregroundApp = null;

        while (events.hasNextEvent()) {
            events.getNextEvent(event);
            if (event.getEventType() == UsageEvents.Event.ACTIVITY_RESUMED) {
                foregroundApp = event.getPackageName();
            }
        }

        if (foregroundApp == null) return false;

        // define rest list (common launchers and system home apps)
        return isInRestList(foregroundApp);
    }

    private boolean isInRestList(String packageName) {
        if (packageName == null) return false;
        try {
            Intent homeIntent = new Intent(Intent.ACTION_MAIN);
            homeIntent.addCategory(Intent.CATEGORY_HOME);
            ResolveInfo defaultLauncher = getApplicationContext().getPackageManager()
                    .resolveActivity(homeIntent, PackageManager.MATCH_DEFAULT_ONLY);
            if (defaultLauncher != null && defaultLauncher.activityInfo != null) {
                if (packageName.equalsIgnoreCase(defaultLauncher.activityInfo.packageName)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
        }

        String[] restList = {
                "com.android.launcher3",
                "com.google.android.apps.nexuslauncher",
                "com.sec.android.app.launcher",       // Samsung OneUI Launcher
                "com.miui.home",                       // Xiaomi MIUI / HyperOS Launcher
                "com.oppo.launcher",                   // Oppo / Realme ColorOS Launcher
                "com.oneplus.launcher",                // OnePlus Launcher
                "com.huawei.android.launcher",         // Huawei EMUI Launcher
                "com.vivo.launcher"                    // Vivo Funtouch Launcher
        };

        for (String app : restList) {
            if (app.equalsIgnoreCase(packageName)) return true;
        }
        return false;
    }
}