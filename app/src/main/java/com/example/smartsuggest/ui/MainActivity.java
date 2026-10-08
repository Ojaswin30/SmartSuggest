package com.example.smartsuggest.ui;

import android.app.AppOpsManager;
import android.app.usage.UsageStats;
import android.app.usage.UsageStatsManager;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.smartsuggest.R;
import com.example.smartsuggest.data.AppDatabase;
import com.example.smartsuggest.data.InferenceResult;
import com.example.smartsuggest.model.LSTMModel;
import com.example.smartsuggest.utils.AppContextEngine;
import com.example.smartsuggest.utils.BootReceiver;
import com.example.smartsuggest.utils.ModelFileUtils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private TextView statusText;
    private TextView resultText;
    private TextView topAppNameText;
    private TextView topAppConfidenceText;
    private Button permissionButton;
    private Button runInferenceButton;
    private Button runRetrainButton;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private static class AppPrediction implements Comparable<AppPrediction> {
        String name;
        String packageName;
        double probability;

        AppPrediction(String name, String packageName, double probability) {
            this.name = name;
            this.packageName = packageName;
            this.probability = probability;
        }

        @Override
        public int compareTo(AppPrediction other) {
            return Double.compare(other.probability, this.probability); // descending
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusText = findViewById(R.id.statusText);
        resultText = findViewById(R.id.resultText);
        topAppNameText = findViewById(R.id.topAppNameText);
        topAppConfidenceText = findViewById(R.id.topAppConfidenceText);
        permissionButton = findViewById(R.id.permissionButton);
        runInferenceButton = findViewById(R.id.runInferenceButton);
        runRetrainButton = findViewById(R.id.runRetrainButton);

        updateStatusText();

        permissionButton.setOnClickListener(v -> {
            try {
                Intent intent = new Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS);
                intent.setData(Uri.parse("package:" + getPackageName()));
                startActivity(intent);
            } catch (Exception e) {
                Intent fallbackIntent = new Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS);
                startActivity(fallbackIntent);
            }
        });

        runInferenceButton.setOnClickListener(v -> executeInferenceNow());
        runRetrainButton.setOnClickListener(v -> executeRetrainingNow());

        if (isUsageStatsPermissionGranted()) {
            BootReceiver.scheduleWorkers(this);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateStatusText();
        if (isUsageStatsPermissionGranted()) {
            BootReceiver.scheduleWorkers(this);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdown();
    }

    private void updateStatusText() {
        boolean granted = isUsageStatsPermissionGranted();
        if (granted) {
            statusText.setText("Status: Active & Running ✓\nBackground workers are scheduled.");
            if (permissionButton != null) {
                permissionButton.setText("Permission Granted ✓");
                permissionButton.setEnabled(false);
            }
        } else {
            statusText.setText("Status: Permission needed\nTap button below to grant Usage Access.");
            if (permissionButton != null) {
                permissionButton.setText("Grant Usage Access");
                permissionButton.setEnabled(true);
            }
        }
    }

    private void executeInferenceNow() {
        setButtonsEnabled(false);
        resultText.setText("⚡ Extracting live device context & running LSTM inference...");

        executor.execute(() -> {
            long startTime = System.currentTimeMillis();
            try {
                MultiLayerNetwork network;
                List<AppContextEngine.AppItem> installedApps;
                INDArray inputContext;
                INDArray output;

                synchronized (ModelFileUtils.MODEL_LOCK) {
                    // 1. Discover user's actual installed candidate apps
                    installedApps = AppContextEngine.getInstalledCandidateApps(getApplicationContext());
                    List<String> currentPkgs = AppContextEngine.getPackageNames(installedApps);

                    // 2. Load or initialize neural network
                    if (ModelFileUtils.isModelValid(getApplicationContext(), currentPkgs)) {
                        network = ModelFileUtils.loadModel(getApplicationContext());
                    } else {
                        if (ModelFileUtils.modelExists(getApplicationContext())) {
                            ModelFileUtils.invalidateModel(getApplicationContext());
                        }
                        LSTMModel freshModel = new LSTMModel();
                        network = freshModel.getNetwork();
                    }

                    if (network == null) {
                        runOnUiThread(() -> {
                            resultText.setText("❌ Error: Failed to load or initialize LSTM model.");
                            setButtonsEnabled(true);
                        });
                        return;
                    }

                    // 3. Extract real-time context features (time, day, battery, last app, media state)
                    inputContext = AppContextEngine.extractCurrentContextFeatures(getApplicationContext(), installedApps);

                    // 4. Run LSTM forward pass
                    output = network.output(inputContext);
                }

                long latency = System.currentTimeMillis() - startTime;

                // 5. Map output probabilities to actual installed apps
                List<AppPrediction> predictions = new ArrayList<>();
                for (int i = 0; i < installedApps.size(); i++) {
                    double prob = output.getDouble(0, i, 0);
                    predictions.add(new AppPrediction(installedApps.get(i).label, installedApps.get(i).packageName, prob));
                }
                Collections.sort(predictions);
                AppPrediction topApp = predictions.get(0);

                // 6. Save prediction to Room DB
                AppDatabase db = AppDatabase.getInstance(getApplicationContext());
                InferenceResult record = new InferenceResult(
                        System.currentTimeMillis(),
                        String.format(Locale.getDefault(), "Top: %s (%.1f%%)", topApp.name, topApp.probability * 100),
                        inputContext.toString()
                );
                db.inferenceResultDao().insert(record);
                db.inferenceResultDao().pruneOld();
                InferenceResult latest = db.inferenceResultDao().getLatest();

                String timeStr = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());

                // Build detailed log
                StringBuilder sb = new StringBuilder();
                sb.append(String.format(Locale.getDefault(), "[%s] REAL CONTEXT INFERENCE COMPLETED\n", timeStr));
                sb.append("------------------------------------\n");
                sb.append(String.format(Locale.getDefault(), "• Latency: %d ms\n", latency));
                sb.append(String.format(Locale.getDefault(), "• Top Suggestion: %s (%.1f%%)\n", topApp.name, topApp.probability * 100));
                sb.append(String.format(Locale.getDefault(), "• Room DB Record: #%d (saved)\n", latest != null ? latest.id : record.id));
                sb.append("------------------------------------\n");
                sb.append("📱 REAL INSTALLED APP PROBABILITIES:\n");

                for (int i = 0; i < predictions.size(); i++) {
                    AppPrediction p = predictions.get(i);
                    sb.append(String.format(Locale.getDefault(), "%2d. %-22s : %5.1f%%\n", (i + 1), p.name, p.probability * 100));
                }

                String log = sb.toString();

                runOnUiThread(() -> {
                    topAppNameText.setText(topApp.name);
                    topAppConfidenceText.setText(String.format(Locale.getDefault(), "Confidence: %.1f%% • Latency: %d ms", topApp.probability * 100, latency));
                    resultText.setText(log);
                    setButtonsEnabled(true);
                });

            } catch (Exception e) {
                String errorMsg = e.getMessage();
                runOnUiThread(() -> {
                    resultText.setText("❌ Exception during inference: " + errorMsg);
                    setButtonsEnabled(true);
                });
            }
        });
    }

    private void executeRetrainingNow() {
        setButtonsEnabled(false);
        resultText.setText("🔄 Learning from your real 24h app usage history on-device...");

        executor.execute(() -> {
            long startTime = System.currentTimeMillis();
            try {
                int epochs = 10;
                int sampleCount = 0;

                synchronized (ModelFileUtils.MODEL_LOCK) {
                    List<AppContextEngine.AppItem> installedApps = AppContextEngine.getInstalledCandidateApps(getApplicationContext());
                    List<String> currentPkgs = AppContextEngine.getPackageNames(installedApps);

                    MultiLayerNetwork network;
                    if (ModelFileUtils.isModelValid(getApplicationContext(), currentPkgs)) {
                        network = ModelFileUtils.loadModel(getApplicationContext());
                    } else {
                        if (ModelFileUtils.modelExists(getApplicationContext())) {
                            ModelFileUtils.invalidateModel(getApplicationContext());
                        }
                        LSTMModel freshModel = new LSTMModel();
                        network = freshModel.getNetwork();
                    }

                    if (network == null) {
                        runOnUiThread(() -> {
                            resultText.setText("❌ Error: Failed to load network for retraining.");
                            setButtonsEnabled(true);
                        });
                        return;
                    }

                    // Extract actual app usage transitions and context features from the device
                    DataSet realDataSet = AppContextEngine.buildTrainingDataSet(getApplicationContext(), installedApps);

                    if (realDataSet != null && realDataSet.getFeatures() != null) {
                        sampleCount = (int) realDataSet.getFeatures().size(2);
                        for (int i = 0; i < epochs; i++) {
                            network.fit(realDataSet);
                        }
                        ModelFileUtils.saveModel(getApplicationContext(), network, currentPkgs);
                    } else {
                        runOnUiThread(() -> {
                            resultText.setText("⚠️ Not enough usage data in past 24h to retrain model.");
                            setButtonsEnabled(true);
                        });
                        return;
                    }
                }

                long duration = System.currentTimeMillis() - startTime;
                String timeStr = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
                final int finalSampleCount = sampleCount;

                runOnUiThread(() -> {
                    String log = String.format(
                            Locale.getDefault(),
                            "[%s] ON-DEVICE TRAINING COMPLETED\n" +
                            "------------------------------------\n" +
                            "• Mode: Trained on your real device usage\n" +
                            "• Usage Steps Trained: %d\n" +
                            "• Epochs Trained: %d\n" +
                            "• Duration: %d ms\n" +
                            "• Model Status: Weights saved locally\n" +
                            "• Tip: Tap 'Run Inference' to see personalized rankings!\n" +
                            "------------------------------------",
                            timeStr,
                            finalSampleCount,
                            epochs,
                            duration
                    );
                    resultText.setText(log);
                    setButtonsEnabled(true);
                });

            } catch (Exception e) {
                String errorMsg = e.getMessage();
                runOnUiThread(() -> {
                    resultText.setText("❌ Exception during retraining: " + errorMsg);
                    setButtonsEnabled(true);
                });
            }
        });
    }

    private void setButtonsEnabled(boolean enabled) {
        if (runInferenceButton != null) runInferenceButton.setEnabled(enabled);
        if (runRetrainButton != null) runRetrainButton.setEnabled(enabled);
    }

    private boolean isUsageStatsPermissionGranted() {
        try {
            AppOpsManager appOps = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
            if (appOps != null) {
                int mode;
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    mode = appOps.unsafeCheckOpNoThrow(
                            AppOpsManager.OPSTR_GET_USAGE_STATS,
                            android.os.Process.myUid(),
                            getPackageName()
                    );
                } else {
                    mode = appOps.checkOpNoThrow(
                            AppOpsManager.OPSTR_GET_USAGE_STATS,
                            android.os.Process.myUid(),
                            getPackageName()
                    );
                }
                if (mode == AppOpsManager.MODE_ALLOWED) {
                    return true;
                }
            }

            UsageStatsManager usageStatsManager = (UsageStatsManager) getSystemService(Context.USAGE_STATS_SERVICE);
            if (usageStatsManager != null) {
                long now = System.currentTimeMillis();
                List<UsageStats> stats = usageStatsManager.queryUsageStats(
                        UsageStatsManager.INTERVAL_DAILY,
                        now - 1000 * 60,
                        now
                );
                return stats != null && !stats.isEmpty();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }
}