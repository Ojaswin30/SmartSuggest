# SmartSuggest  

SmartSuggest is an intelligent suggestion and logging framework designed to analyze **app usage behavior** and provide **context-aware recommendations**. Think of it as a smart layer that learns from how users interact with apps and surfaces smarter insights, powered by logging, modeling, and Python-backed intelligence.  

---

## 🚀 Features  
- **App Usage Logger** – Android module for tracking and logging app usage patterns.  
- **Behavior Modeling** – A `model/` directory to house ML/DL models for predictive suggestions.  
- **Python Integrations** – Dedicated scripts in `python files/` to process data, train models, or perform advanced analytics.  
- **Usage Records** – A `record/` directory to store and manage usage history and experiment outputs.  

---

## 📂 Project Structure  

```plaintext
SmartSuggest/
│
├── AppUsageLogger/       # Android app module (Kotlin/Java + Gradle)
│   ├── app/              # Core Android app source (UI, Activities, Services)
│   ├── .gradle/          # Gradle build cache
│   ├── .idea/            # IntelliJ/Android Studio configs
│   ├── build/            # Auto-generated build files
│   ├── gradle/           # Gradle wrapper files
│   └── python/           # Python integration hooks
│
├── model/                # ML/DL models for predictions
├── python files/         # Python scripts for data processing, training, etc.
└── record/               # Logs, experiment results, and app usage records


| Screenshot 1 | Screenshot 2 |
|--------------|--------------|
| ![Home Page](assets\Home Page.jpg) | ![Logs](assets\Show log.jpg) |