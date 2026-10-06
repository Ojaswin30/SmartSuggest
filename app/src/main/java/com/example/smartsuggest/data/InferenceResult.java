package com.example.smartsuggest.data;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "inference_results")
public class InferenceResult {

    @PrimaryKey(autoGenerate = true)
    public int id;

    public long timestamp;      // when inference ran
    public String output;       // store output as string for now
    public String inputSnapshot; // what input was used

    public InferenceResult(long timestamp, String output, String inputSnapshot) {
        this.timestamp = timestamp;
        this.output = output;
        this.inputSnapshot = inputSnapshot;
    }
}