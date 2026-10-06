package com.example.smartsuggest.data;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;

import java.util.List;

@Dao
public interface InferenceResultDao {

    @Insert
    void insert(InferenceResult result);

    @Query("SELECT * FROM inference_results ORDER BY timestamp DESC")
    List<InferenceResult> getAll();

    @Query("SELECT * FROM inference_results ORDER BY timestamp DESC LIMIT 1")
    InferenceResult getLatest();

    // keep only last 100 results to avoid db growing forever
    @Query("DELETE FROM inference_results WHERE id NOT IN (SELECT id FROM inference_results ORDER BY timestamp DESC LIMIT 100)")
    void pruneOld();
}