package com.example.smartsuggest.model;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LSTMModel {

    private final MultiLayerNetwork network;

    // adjust these based on your use case later
    private static final int INPUT_SIZE = 10;
    private static final int HIDDEN_SIZE = 50;
    private static final int OUTPUT_SIZE = 10;
    private static final double LEARNING_RATE = 0.01;

    public LSTMModel() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(LEARNING_RATE))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new LSTM.Builder()
                        .nIn(INPUT_SIZE)
                        .nOut(HIDDEN_SIZE)
                        .activation(Activation.TANH)
                        .build())
                .layer(new RnnOutputLayer.Builder()
                        .nIn(HIDDEN_SIZE)
                        .nOut(OUTPUT_SIZE)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        network = new MultiLayerNetwork(config);
        network.init();
    }

    // pass in existing network (when loading saved weights)
    public LSTMModel(MultiLayerNetwork network) {
        this.network = network;
    }

    public MultiLayerNetwork getNetwork() {
        return network;
    }
}