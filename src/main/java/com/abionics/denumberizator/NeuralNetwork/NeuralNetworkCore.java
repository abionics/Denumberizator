package com.abionics.denumberizator.NeuralNetwork;

import org.jetbrains.annotations.NotNull;

import java.util.Random;

class NeuralNetworkCore {
    private final static double SPEED = 0.7;
    private final static double MOMENTUM = 0.3;
    private final static double KHEIGHT = 1.2;

    private final int inputCount;
    private final int outputCount;
    private final int hideCount;
    private final int bias;

    private int[] layersSizes;
    private Neuron[][] layers;
    private double[][][] weights;

    private double[][] deltaLayers;
    private double[][][] deltaWeights;

    private double speed = SPEED;
    private double momentum = MOMENTUM;

    public Log log = new Log();


    NeuralNetworkCore(int inputs, int outputs, int hides) {
        this(inputs, outputs, hides, true, KHEIGHT);
    }
    NeuralNetworkCore(int inputs, int outputs, int hides, boolean isBias) {
        this(inputs, outputs, hides, isBias, KHEIGHT);
    }
    NeuralNetworkCore(int inputCount, int outputCount, int hideCount, boolean isBias, double kHeight) {
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.hideCount = hideCount;
        this.bias = isBias ? 1 : 0;
        init(kHeight);
    }
    private void init(double kHeight) {
        calculateLayersSize(kHeight);
        initLayers();
        initWeights();
    }

    public void setSpeed(double speed) {
        this.speed = speed;
    }
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    private void calculateLayersSize(double k) {
        double x = hideCount + 1;
        double y = outputCount;
        double c = inputCount;

        double p = (c * (k - 1) + Math.sqrt(c * (k - 1) * (c * k - y))) / x;
        double a = p * p / (c * (1 - k));
        double b = 2 * p;

        layersSizes = new int[hideCount + 2];
        layersSizes[0] = inputCount + bias;
        layersSizes[hideCount + 2 - 1] = outputCount + bias;
        for (int i = 1; i < hideCount + 2 - 1; i++) {
            layersSizes[i] = (int) (a * i * i + b * i + c + bias);
        }
    }
    private void initLayers() {
        int count = hideCount + 2;
        layers = new Neuron[count][];
        deltaLayers = new double[count][];
        for (int q = 0; q < count; q++) {
            int size = layersSizes[q];
            layers[q] = new Neuron[size];
            deltaLayers[q] = new double[size];
            for (int i = 0; i < size; i++) {
                layers[q][i] = new Neuron();
            }
        }
    }
    private void initWeights() {
        int count = hideCount + 1;
        weights = new double[count][][];
        deltaWeights = new double[count][][];
        Random random = new Random();
        for (int q = 0; q < count; q++) {
            int nextSize = layersSizes[q + 1];
            int currentSize = layersSizes[q];
            weights[q] = new double[nextSize - bias][];
            deltaWeights[q] = new double[nextSize - bias][];
            for (int i = 0; i < nextSize - bias; i++) {
                weights[q][i] = new double[currentSize];
                deltaWeights[q][i] = new double[currentSize];
                for (int j = 0; j < currentSize; j++) {
                    weights[q][i][j] = 4 * (random.nextDouble() - 0.5);
                }
            }
        }
    }

    private void iteration(double[] data) {
        for (int i = 0; i < inputCount; i++) {
            layers[0][i].set(data[i]);
        }

        for (int layer = 0; layer < layers.length - 1; layer++) {
            for (int neuron = 0; neuron < layersSizes[layer + 1] - bias; neuron++) {
                double input = 0;
                for (int j = 0; j < layersSizes[layer]; j++) {
                    input += layers[layer][j].get() * weights[layer][neuron][j];
                }
                layers[layer + 1][neuron].active(input);
//                System.out.println("result #[" + (layer + 1) + ";" + neuron + "]: " + layers[layer + 1][neuron].get() + " *" + input);
            }
        }
    }
    private double deviation(double[] solution) {
        double result = 0;
        int last = hideCount + 1;
        for (int i = 0; i < outputCount; i++) {
            result += Math.pow(solution[i] - layers[last][i].get(), 2);
        }
        return result / outputCount;
    }
    private void backpropagation(double[] solution) {
        //output
        int last = hideCount + 1;
        for (int i = 0; i < outputCount; i++) {
            double output = layers[last][i].get();
            deltaLayers[last][i] = (solution[i] - output) * (1 - output) * output;
//            System.out.println("output #[" + i + "]: " + deltaLayers[last][i]);
        }
        //hidden
        for (int q = hideCount; q > 0; q--) {
            for (int i = 0; i < layersSizes[q]; i++) {
                double sumd = 0;
                for (int j = 0; j < layersSizes[q + 1] - bias; j++) {
                    sumd += weights[q][j][i] * deltaLayers[q + 1][j];
                }
                double output = layers[q][i].get();
                deltaLayers[q][i] = (1 - output) * output * sumd;
                for (int j = 0; j < layersSizes[q + 1] - bias; j++) {
                    deltaWeights[q][j][i] = speed * output * deltaLayers[q + 1][j] + momentum * deltaWeights[q][j][i];
                    weights[q][j][i] += deltaWeights[q][j][i];
                }
//                System.out.println("hidden #[" + q + ";" + i + "]: " + deltaLayers[q][i]);
            }
        }
        //input
        for (int i = 0; i < inputCount; i++) {
            double output = layers[0][i].get();
            for (int j = 0; j < layersSizes[1] - bias; j++) {
                deltaWeights[0][j][i] = speed * output * deltaLayers[1][j] + momentum * deltaWeights[0][j][i];
                weights[0][j][i] += deltaWeights[0][j][i];
            }
        }
    }
    private int result() {
        double max = 0;
        int result = 0;
        int last = hideCount + 1;
        for (int i = 0; i < outputCount; i++) {
            double temp = layers[last][i].get();
            if (temp > max) {
                max = temp;
                result = i;
            }
        }
        return result;
    }

    public double deviation(double[] data, double[] solution) throws NeuralNetworkException {
        check(data, solution);
        iteration(data);
        return deviation(solution);
    }
    public double deviation(double[][] datas, double[][] solutions) throws NeuralNetworkException {
        check(datas, solutions, false);
        double dev = 0;
        for (int i = 0; i < datas.length; i++) {
            dev += deviation(datas[i], solutions[i]);
        }
        return dev / datas.length;
    }
    public int corrects(double[][] datas, int[] answers) throws NeuralNetworkException {
        check(datas, answers);
        int correct = 0;
        for (int i = 0; i < datas.length; i++) {
            iteration(datas[i]);
            if (result() == answers[i]) correct++;
        }
        return correct;
    }

    protected void learn(double[] data, double[] solution) {
        iteration(data);
        backpropagation(solution);
    }
    protected void learn(@NotNull double[][] datas, double[][] solutions) {
        for (int i = 0; i < datas.length; i++) {
            learn(datas[i], solutions[i]);
        }
    }

    public int analyze(double[] data) {
        iteration(data);
        return result();
    }

    protected void check(@NotNull double[] data, double[] solution) throws NeuralNetworkException {
        if (data.length != inputCount && solution.length != outputCount)
            throw new NeuralNetworkException("Size of data[] and solution[] are different");
    }
    protected void check(@NotNull double[][] datas, @NotNull int[] answers) throws NeuralNetworkException {
        if (datas.length != answers.length)
            throw new NeuralNetworkException("Size of datas[][] and answers[] are different");
    }
    protected void check(@NotNull double[][] datas, @NotNull double[][] solutions, boolean inner) throws NeuralNetworkException {
        if (datas.length != solutions.length)
            throw new NeuralNetworkException("Size of datas[][] and solutions[][] are different");
        if (inner)
            for (int i = 0; i < datas.length; i++)
                check(datas[i], solutions[i]);
    }

    public class Log {
        public void layers() {
            for (int q = 0; q < hideCount + 2; q++) {
                System.out.print("Layer " + q + " : ");
                for (int i = 0; i < layersSizes[q]; i++) {
                    System.out.print(layers[q][i].get() + " ");
                }
                System.out.println();
            }
        }

        public void weights() {
            for (int q = 0; q < hideCount + 1; q++) {
                System.out.println("Matrix " + q + ":");
                for (int i = 0; i < weights[q].length; i++) {
                    for (int j = 0; j < weights[q][i].length; j++) {
                        System.out.print(weights[q][i][j] + " ");
                    }
                    System.out.println();
                }
            }
        }

        public void weights(double partOfMax) {
            double maxWeight = 0;
            for (int q = 0; q < hideCount + 1; q++) {
                for (int i = 0; i < weights[q].length; i++) {
                    for (int j = 0; j < weights[q][i].length; j++) {
                        if (Math.abs(weights[q][i][j]) > maxWeight) maxWeight = Math.abs(weights[q][i][j]);
                    }
                }
            }
            for (int q = 0; q < hideCount + 1; q++) {
                System.out.println("Matrix " + q + ":");
                for (int i = 0; i < weights[q].length; i++) {
                    for (int j = 0; j < weights[q][i].length; j++) {
                        boolean isImportant = Math.abs(weights[q][i][j]) > maxWeight * partOfMax;
                        if (isImportant) System.out.print(Math.round(weights[q][i][j]) + " ");
                        else System.out.print("0 ");
                    }
                    System.out.println();
                }
            }
        }

        public void results() {
            double sum = 0;
            int last = hideCount + 1;
            for (int i = 0; i < outputCount; i++) {
                sum += layers[last][i].get();
            }
            for (int i = 0; i < outputCount; i++) {
                double temp = layers[last][i].get();
                System.out.println(i + " : " + (int) (temp / sum * 100));
            }
        }
    }
}
