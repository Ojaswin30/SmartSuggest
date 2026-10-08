package com.example.smartsuggest.workers;

import android.content.Context;

import androidx.annotation.NonNull;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import com.example.smartsuggest.model.LSTMModel;
import com.example.smartsuggest.utils.AppContextEngine;
import com.example.smartsuggest.utils.ModelFileUtils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;

public class RetrainingWorker extends Worker {

    public RetrainingWorker(@NonNull Context context, @NonNull WorkerParameters params) {
        super(context, params);
    }

    @NonNull
    @Override
    public Result doWork() {
        synchronized (ModelFileUtils.MODEL_LOCK) {
            // step 1 - discover installed apps
            List<AppContextEngine.AppItem> installedApps = AppContextEngine.getInstalledCandidateApps(getApplicationContext());
            List<String> currentPkgs = AppContextEngine.getPackageNames(installedApps);

            // step 2 - load existing model or create fresh
            MultiLayerNetwork network;
            if (!ModelFileUtils.modelExists(getApplicationContext())) {
                network = new LSTMModel().getNetwork();
            } else if (ModelFileUtils.isModelValid(getApplicationContext(), currentPkgs)) {
                network = ModelFileUtils.loadModel(getApplicationContext());
            } else {
                return Result.failure();
            }

            if (network == null) return Result.failure();

            // step 3 - construct training dataset from real 24h usage events
            DataSet realDataSet = AppContextEngine.buildTrainingDataSet(getApplicationContext(), installedApps);

            if (realDataSet != null && realDataSet.getFeatures() != null) {
                // step 4 - retrain on device
                int epochs = 10;
                for (int i = 0; i < epochs; i++) {
                    network.fit(realDataSet);
                }

                // step 5 - save updated weights and mapping
                try {
                    ModelFileUtils.saveModel(getApplicationContext(), network, currentPkgs);
                } catch (Exception e) {
                    return Result.failure();
                }
            }

            // step 6 - release memory
            network = null;

            return Result.success();
        }
    }
}