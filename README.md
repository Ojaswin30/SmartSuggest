# SmartSuggest 🧠📱

**SmartSuggest** is an on-device intelligent prediction and suggestion engine for Android. Powered by **Deeplearning4j (DL4J)** and **ND4J**, SmartSuggest runs recurrent neural network (LSTM) inference and local model retraining directly on the device while leveraging **AndroidX WorkManager** for periodic execution and **Room** for local persistence.

---

## 🚀 Key Features

- **On-Device Machine Learning (LSTM)**: Runs Long Short-Term Memory (LSTM) recurrent neural network inference directly on the client device without sending private data to cloud servers.
- **Context-Aware Triggering**: Inspects foreground app state via `UsageStatsManager` to execute inference intelligently during launcher/idle moments (rest list) rather than interrupting active tasks.
- **On-Device Retraining & Model Persistence**: Supports periodic background model fine-tuning with local user interaction data, serializing updated network weights to local storage.
- **Background Task Scheduling**: Utilizes AndroidX **WorkManager** to schedule lightweight inference (every 15 minutes) and scheduled retraining (daily) under battery-friendly constraints.
- **Local SQLite / Room Database**: Employs Room persistence library to store inference history, suggestions, and prediction logs securely.
- **Boot Resilience**: Listens for `BOOT_COMPLETED` via `BootReceiver` to automatically re-enqueue and maintain background tasks after device restarts.

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
│   │   ├── AndroidManifest.xml
│   │   ├── java/com/example/smartsuggest/
│   │   │   ├── data/
│   │   │   │   ├── AppDatabase.java          # Room database definition
│   │   │   │   ├── InferenceResult.java      # Room entity for model predictions
│   │   │   │   └── InferenceResultDao.java   # DAO operations for stored inferences
│   │   │   ├── model/
│   │   │   │   └── LSTMModel.java            # Deeplearning4j LSTM configuration & initialization
│   │   │   ├── ui/
│   │   │   │   └── MainActivity.java         # Permission prompt & worker scheduling UI
│   │   │   ├── utils/
│   │   │   │   ├── BootReceiver.java         # BroadcastReceiver for BOOT_COMPLETED
│   │   │   │   └── ModelFileUtils.java       # Model serialization & deserialization helpers
│   │   │   └── workers/
│   │   │       ├── InferenceWorker.java      # Background periodic inference worker
│   │   │       └── RetrainingWorker.java     # Scheduled background on-device retraining worker
│   │   └── res/                              # Layouts, themes, drawables, and XML rules
├── build.gradle                              # Root build script
├── settings.gradle                           # Settings configuration
└── gradle.properties                         # JVM arguments and Gradle properties
```

---

## ⚙️ Permissions Required

To function properly in the background, SmartSuggest requires the following system permissions configured in `AndroidManifest.xml`:

- `android.permission.PACKAGE_USAGE_STATS`: Required to monitor foreground app transitions and verify launcher/idle state before running predictions.
- `android.permission.RECEIVE_BOOT_COMPLETED`: Restores periodic background workers on device reboot.
- `android.permission.FOREGROUND_SERVICE` & `android.permission.FOREGROUND_SERVICE_DATA_SYNC`: Ensures long-running inference/retraining jobs complete cleanly without termination.
- `android.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`: Prevents aggressive OS battery management from halting on-device training.

---

## 🏃 Getting Started

### Prerequisites
- Android Studio Ladybug (or newer)
- Android SDK 36 with Build Tools
- JDK 17 configured in Android Studio
- Device or Emulator running Android 8.0 (API level 26) or higher

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
4. On first launch, grant **Usage Access Permission** when prompted to enable background context monitoring and worker scheduling.

---

## 📄 License

This project is licensed under the Apache License 2.0 / MIT License.
