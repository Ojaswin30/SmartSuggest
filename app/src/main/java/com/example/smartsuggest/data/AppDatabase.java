package com.example.smartsuggest.data;

import android.content.Context;

import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;

@Database(entities = {InferenceResult.class}, version = 1, exportSchema = false)
public abstract class AppDatabase extends RoomDatabase {

    private static AppDatabase instance;

    public abstract InferenceResultDao inferenceResultDao();

    // singleton so we don't open multiple db connections
    public static synchronized AppDatabase getInstance(Context context) {
        if (instance == null) {
            instance = Room.databaseBuilder(
                            context.getApplicationContext(),
                            AppDatabase.class,
                            "smartsuggest_db"
                    )
                    .fallbackToDestructiveMigration()
                    .build();
        }
        return instance;
    }
}