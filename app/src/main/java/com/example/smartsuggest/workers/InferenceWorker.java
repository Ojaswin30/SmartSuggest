package com.example.smartsuggest.workers;

import android.app.usage.UsageStats;
import android.app.usage.UsageStatsManager;
import android.content.Context;

import androidx.annotation.NonNull;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import com.example.smartsuggest.model.LSTMModel;
import com.example.smartsuggest.utils.ModelFileUtils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

public class InferenceWorker extends Worker {

    public InferenceWorker(@NonNull Context context, @NonNull WorkerParameters params) {
        super(context, params);
    }

    @NonNull
    @Override
    public Result doWork() {

        // step 1 - check if foreground app is in rest list
        if (!isForegroundAppInRestList()) {
            return Result.success(); // skip silently
        }

        // step 2 - load model (from file if exists, else fresh)
        MultiLayerNetwork network;
        if (ModelFileUtils.modelExists(getApplicationContext())) {
            network = ModelFileUtils.loadModel(getApplicationContext());
        } else {
            LSTMModel freshModel = new LSTMModel();
            network = freshModel.getNetwork();
        }

        if (network == null) return Result.failure();

        // step 3 - run inference
        // replace this dummy input with your real input data later
        INDArray input = Nd4j.zeros(1, 10, 1); // shape: [batch, input_size, time_steps]
        INDArray output = network.output(input);

        // step 4 - do something with output (log it for now)
        System.out.println("Inference output: " + output);

        // step 5 - let go of model, GC will reclaim memory
        network = null;

        return Result.success();
    }

    private boolean isForegroundAppInRestList() {
        UsageStatsManager usageStatsManager = (UsageStatsManager)
                getApplicationContext().getSystemService(Context.USAGE_STATS_SERVICE);

        long currentTime = System.currentTimeMillis();
        List<UsageStats> stats = usageStatsManager.queryUsageStats(
                UsageStatsManager.INTERVAL_DAILY,
                currentTime - 1000 * 10, // last 10 seconds
                currentTime
        );

        if (stats == null || stats.isEmpty()) return false;

        // find the most recently used app
        SortedMap<Long, UsageStats> sortedMap = new TreeMap<>();
        for (UsageStats usageStats : stats) {
            sortedMap.put(usageStats.getLastTimeUsed(), usageStats);
        }

        String foregroundApp = sortedMap.get(sortedMap.lastKey()).getPackageName();

        // define your rest list here
        return isInRestList(foregroundApp);
    }

    private boolean isInRestList(String packageName) {
        // add apps here where it's safe to run inference
        String[] restList = {
                "com.android.launcher3",   // home screen
                "com.google.android.apps.nexuslauncher" // pixel launcher
        };

        for (String app : restList) {
            if (app.equals(packageName)) return true;
        }
        return false;
    }
}