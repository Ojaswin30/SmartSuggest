# SmartSuggest 🧠📱

**SmartSuggest** is an on-device intelligent prediction and suggestion engine for Android. Powered by **Deeplearning4j (DL4J)** and **ND4J**, SmartSuggest runs recurrent neural network (LSTM) inference and local model retraining directly on the device while leveraging **AndroidX WorkManager** for periodic background execution and **Room** for local persistence.

---

## 🚀 Key Features

- **On-Device Machine Learning (LSTM)**: Runs Long Short-Term Memory (LSTM) recurrent neural network inference directly on the client device without sending private data to cloud servers.
- **Dynamic Installed App Discovery**: Automatically scans and maps predictions to actual launcher applications installed on the user's phone via Android's `PackageManager`.
- **10-Dimensional Live Context Engine**: Extracts real-time device signals instead of dummy vectors:
  - Time of day & Day of week
  - Weekend detection
  - Battery level & Charging state
  - Active media/audio playback & Ringer mode
  - Previous app transitions & Night-mode flag
- **Real-History On-Device Retraining**: Extracts 24–48 hour app launch sequences via `UsageEvents` to fine-tune model weights locally on the user's real habits.
- **Context-Aware Triggering**: Inspects foreground app state via `UsageEvents` to execute inference during launcher/idle moments (OEM rest lists) without interrupting active tasks.
- **Interactive UI & Live Diagnostic Console**: Includes manual **Run Inference** and **Retrain Model** buttons with a top suggested app card and ranked probability breakdown.
- **Background Task Scheduling**: Utilizes AndroidX **WorkManager** to schedule lightweight periodic inference (every 15 minutes) and daily retraining under battery-friendly constraints.
- **Local SQLite / Room Database**: Securely logs predictions, input snapshots, and latency metrics with automated retention pruning.
- **Boot Resilience**: Listens for `BOOT_COMPLETED` via `BootReceiver` to automatically re-enqueue and maintain background tasks after device restarts.
- **Multi-Architecture Support**: Packaged with native math libraries for `arm64-v8a`, `armeabi-v7a`, `x86_64`, and `x86`.

---

## 🛠 Tech Stack & Dependencies

- **Platform**: Android (minSdk: 26 / targetSdk: 36)
- **Language**: Java 17
- **Deep Learning Framework**: [Deeplearning4j (DL4J)](https://deeplearning4j.konduit.ai/) `1.0.0-M2.1` & [ND4J Native](https://nd4j.org/)
- **Background Processing**: AndroidX WorkManager `2.9.0`
- **Database / ORM**: AndroidX Room `2.6.1`
- **UI & Support**: AndroidX AppCompat, Material Design Components, ConstraintLayout

---

## 📂 Project Architecture

```
SmartSuggest/
├── app/
│   ├── src/main/
│   │   ├── AndroidManifest.xml          # Permissions, package queries & receivers
│   │   ├── java/com/example/smartsuggest/
│   │   │   ├── data/
│   │   │   │   ├── AppDatabase.java          # Room database definition
│   │   │   │   ├── InferenceResult.java      # Room entity for model predictions
│   │   │   │   └── InferenceResultDao.java   # DAO operations & auto-pruning
│   │   │   ├── model/
│   │   │   │   └── LSTMModel.java            # DL4J LSTM configuration & Adam optimizer
│   │   │   ├── ui/
│   │   │   │   └── MainActivity.java         # Permission checks, UI dashboard & test runner
│   │   │   ├── utils/
│   │   │   │   ├── AppContextEngine.java     # App discovery, feature extraction & dataset builder
│   │   │   │   ├── BootReceiver.java         # BroadcastReceiver for BOOT_COMPLETED
│   │   │   │   └── ModelFileUtils.java       # Model serialization & deserialization helpers
│   │   │   └── workers/
│   │   │       ├── InferenceWorker.java      # Background periodic inference worker
│   │   │       └── RetrainingWorker.java     # Scheduled background on-device retraining worker
│   │   └── res/                              # Layouts, themes, drawables, and XML rules
│   └── build.gradle                          # App dependencies & NDK ABI filters
├── build.gradle                              # Root build script
├── settings.gradle                           # Settings configuration
└── .gitignore                                # Comprehensive Android & IDE ignore rules
```

---

## ⚙️ Permissions Required

- `android.permission.PACKAGE_USAGE_STATS`: Monitors foreground app transitions and extracts historical usage sequences for local training.
- `android.permission.RECEIVE_BOOT_COMPLETED`: Restores periodic background workers on device reboot.
- `android.permission.FOREGROUND_SERVICE` & `android.permission.FOREGROUND_SERVICE_DATA_SYNC`: Keeps long-running retraining tasks reliable.
- `android.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`: Prevents aggressive OS battery managers from killing on-device training.

---

## 🏃 Getting Started

### Prerequisites
- Android Studio Ladybug (or newer)
- Android SDK 36 with Build Tools
- JDK 17 configured in Android Studio
- Physical device or Emulator running Android 8.0 (API level 26) or higher

### Installation & Run

1. Clone or open the project in Android Studio:
   ```bash
   git clone https://github.com/Ojaswin30/SmartSuggest.git
   ```
2. Build the project:
   ```bash
   ./gradlew assembleDebug
   ```
3. Run the app on your connected device or emulator.
4. Grant **Usage Access Permission** on first launch.
5. Tap **"Run Inference"** to test live on-device prediction or **"Retrain Model"** to learn from your recent app usage!

---

## 📄 License

This project is licensed under the Apache License 2.0 / MIT License.
