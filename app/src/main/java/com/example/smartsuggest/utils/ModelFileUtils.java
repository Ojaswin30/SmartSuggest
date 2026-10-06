package com.example.smartsuggest.utils;

import android.content.Context;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;

public class ModelFileUtils {

    private static final String MODEL_FILENAME = "lstm_model.zip";

    // get the file path where model will be saved
    private static File getModelFile(Context context) {
        return new File(context.getFilesDir(), MODEL_FILENAME);
    }

    // save model weights to internal storage
    public static void saveModel(Context context, MultiLayerNetwork network) {
        try {
            File modelFile = getModelFile(context);
            ModelSerializer.writeModel(network, modelFile, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // load model weights from internal storage
    public static MultiLayerNetwork loadModel(Context context) {
        try {
            File modelFile = getModelFile(context);
            if (modelFile.exists()) {
                return ModelSerializer.restoreMultiLayerNetwork(modelFile);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null; // null means no saved model found, use fresh one
    }

    // check if a saved model exists
    public static boolean modelExists(Context context) {
        return getModelFile(context).exists();
    }
}