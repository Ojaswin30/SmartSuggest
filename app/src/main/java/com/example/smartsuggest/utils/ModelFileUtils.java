package com.example.smartsuggest.utils;

import android.content.Context;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.json.JSONArray;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class ModelFileUtils {

    public static final Object MODEL_LOCK = new Object();

    private static final String MODEL_FILENAME = "lstm_model.zip";
    private static final String MAPPING_FILENAME = "model_packages.json";

    // get the file path where model will be saved
    private static File getModelFile(Context context) {
        return new File(context.getFilesDir(), MODEL_FILENAME);
    }

    private static File getMappingFile(Context context) {
        return new File(context.getFilesDir(), MAPPING_FILENAME);
    }

    // save model weights and package mapping to internal storage
    public static void saveModel(Context context, MultiLayerNetwork network, List<String> packageNames) {
        synchronized (MODEL_LOCK) {
            try {
                File modelFile = getModelFile(context);
                ModelSerializer.writeModel(network, modelFile, true);

                if (packageNames != null) {
                    savePackageMapping(context, packageNames);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // save package mapping list as JSON array
    private static void savePackageMapping(Context context, List<String> packageNames) {
        try {
            JSONArray array = new JSONArray();
            for (String pkg : packageNames) {
                array.put(pkg);
            }
            File mappingFile = getMappingFile(context);
            try (FileOutputStream fos = new FileOutputStream(mappingFile)) {
                fos.write(array.toString().getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // load package mapping list from internal storage
    public static List<String> getSavedPackageMapping(Context context) {
        synchronized (MODEL_LOCK) {
            File mappingFile = getMappingFile(context);
            if (!mappingFile.exists()) return null;

            try (FileInputStream fis = new FileInputStream(mappingFile)) {
                byte[] data = new byte[(int) mappingFile.length()];
                int read = fis.read(data);
                if (read <= 0) return null;
                String json = new String(data, 0, read, StandardCharsets.UTF_8);
                JSONArray array = new JSONArray(json);
                List<String> list = new ArrayList<>();
                for (int i = 0; i < array.length(); i++) {
                    list.add(array.getString(i));
                }
                return list;
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }
    }

    // verify if saved model exists and its package mapping matches the current candidates
    public static boolean isModelValid(Context context, List<String> currentCandidatePackages) {
        synchronized (MODEL_LOCK) {
            if (!modelExists(context)) return false;
            List<String> savedMapping = getSavedPackageMapping(context);
            if (savedMapping == null || currentCandidatePackages == null) return false;
            if (savedMapping.size() != currentCandidatePackages.size()) return false;
            for (int i = 0; i < savedMapping.size(); i++) {
                if (!savedMapping.get(i).equals(currentCandidatePackages.get(i))) {
                    return false;
                }
            }
            return true;
        }
    }

    // load model weights from internal storage
    public static MultiLayerNetwork loadModel(Context context) {
        synchronized (MODEL_LOCK) {
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
    }

    // check if a saved model exists
    public static boolean modelExists(Context context) {
        synchronized (MODEL_LOCK) {
            return getModelFile(context).exists();
        }
    }

    // delete existing model and mapping
    public static void invalidateModel(Context context) {
        synchronized (MODEL_LOCK) {
            File modelFile = getModelFile(context);
            if (modelFile.exists()) {
                modelFile.delete();
            }
            File mappingFile = getMappingFile(context);
            if (mappingFile.exists()) {
                mappingFile.delete();
            }
        }
    }
}