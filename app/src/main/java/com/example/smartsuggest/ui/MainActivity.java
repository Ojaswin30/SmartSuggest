package com.example.smartsuggest.ui;

import android.app.AppOpsManager;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.provider.Settings;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.smartsuggest.R;
import com.example.smartsuggest.utils.BootReceiver;

public class MainActivity extends AppCompatActivity {

    private TextView statusText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusText = findViewById(R.id.statusText);
        Button permissionButton = findViewById(R.id.permissionButton);

        // check if usage stats permission is granted
        updateStatusText();

        // button to take user to settings to grant permission
        permissionButton.setOnClickListener(v -> {
            Intent intent = new Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS);
            startActivity(intent);
        });

        // schedule workers if permission is already granted
        if (isUsageStatsPermissionGranted()) {
            BootReceiver.scheduleWorkers(this);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        // recheck permission when user comes back from settings
        updateStatusText();
        if (isUsageStatsPermissionGranted()) {
            BootReceiver.scheduleWorkers(this);
        }
    }

    private void updateStatusText() {
        if (isUsageStatsPermissionGranted()) {
            statusText.setText("Status: Running — workers scheduled");
        } else {
            statusText.setText("Status: Permission needed — tap button below");
        }
    }

    private boolean isUsageStatsPermissionGranted() {
        AppOpsManager appOps = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
        int mode = appOps.checkOpNoThrow(
                AppOpsManager.OPSTR_GET_USAGE_STATS,
                android.os.Process.myUid(),
                getPackageName()
        );
        return mode == AppOpsManager.MODE_ALLOWED;
    }
}