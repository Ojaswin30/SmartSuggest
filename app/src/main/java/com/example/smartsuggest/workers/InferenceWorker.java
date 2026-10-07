package com.example.smartsuggest.workers;

import android.app.usage.UsageEvents;
import android.app.usage.UsageStatsManager;
import android.content.Context;

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

        // step 2 - discover installed candidate apps
        List<AppContextEngine.AppItem> installedApps = AppContextEngine.getInstalledCandidateApps(getApplicationContext());

        // step 3 - load model (from file if exists, else fresh)
        MultiLayerNetwork network;
        if (ModelFileUtils.modelExists(getApplicationContext())) {
            network = ModelFileUtils.loadModel(getApplicationContext());
        } else {
            LSTMModel freshModel = new LSTMModel();
            network = freshModel.getNetwork();
        }

        if (network == null) return Result.failure();

        // step 4 - extract real live device context
        INDArray input = AppContextEngine.extractCurrentContextFeatures(getApplicationContext(), installedApps);
        INDArray output = network.output(input);

        // step 5 - find top predicted app
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

        // step 6 - save output and input snapshot to Room database
        AppDatabase db = AppDatabase.getInstance(getApplicationContext());
        InferenceResult result = new InferenceResult(
                System.currentTimeMillis(),
                String.format(Locale.getDefault(), "Top: %s (%.1f%%)", topAppName, maxProb * 100),
                input.toString()
        );
        db.inferenceResultDao().insert(result);
        db.inferenceResultDao().pruneOld();

        // step 7 - let go of model
        network = null;

        return Result.success();
    }

    private boolean isForegroundAppInRestList() {
        UsageStatsManager usageStatsManager = (UsageStatsManager)
                getApplicationContext().getSystemService(Context.USAGE_STATS_SERVICE);

        if (usageStatsManager == null) return false;

        long currentTime = System.currentTimeMillis();
        long startTime = currentTime - 1000 * 60; // last 60 seconds

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