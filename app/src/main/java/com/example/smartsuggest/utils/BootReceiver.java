package com.example.smartsuggest.utils;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

import androidx.work.Constraints;
import androidx.work.ExistingPeriodicWorkPolicy;
import androidx.work.PeriodicWorkRequest;
import androidx.work.WorkManager;

import com.example.smartsuggest.workers.InferenceWorker;
import com.example.smartsuggest.workers.RetrainingWorker;

import java.util.concurrent.TimeUnit;

public class BootReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {
        if (Intent.ACTION_BOOT_COMPLETED.equals(intent.getAction())) {
            scheduleWorkers(context);
        }
    }

    public static void scheduleWorkers(Context context) {

        Constraints constraints = new Constraints.Builder()
                .setRequiresBatteryNotLow(true)
                .build();

        // inference every 15 minutes
        PeriodicWorkRequest inferenceRequest =
                new PeriodicWorkRequest.Builder(InferenceWorker.class, 15, TimeUnit.MINUTES)
                        .setConstraints(constraints)
                        .build();

        // retraining once a day
        PeriodicWorkRequest retrainingRequest =
                new PeriodicWorkRequest.Builder(RetrainingWorker.class, 1, TimeUnit.DAYS)
                        .setConstraints(constraints)
                        .build();

        WorkManager workManager = WorkManager.getInstance(context);

        // KEEP_EXISTING means if already scheduled, don't reschedule
        workManager.enqueueUniquePeriodicWork(
                "inference_worker",
                ExistingPeriodicWorkPolicy.KEEP,
                inferenceRequest
        );

        workManager.enqueueUniquePeriodicWork(
                "retraining_worker",
                ExistingPeriodicWorkPolicy.KEEP,
                retrainingRequest
        );
    }
}