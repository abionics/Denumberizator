package com.abionics.denumberizator.NeuralNetwork;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.util.Random;

public class NeuralNetwork extends NeuralNetworkCore {
    private final static double TRAINING_PERCENT = 0.8;

    private double trainingPercent = TRAINING_PERCENT;

    private DataSet training;
    private DataSet validation;


    public NeuralNetwork(int inputs, int outputs, int hides) {
        super(inputs, outputs, hides);
    }
    public NeuralNetwork(int inputs, int outputs, int hides, boolean isBias) {
        super(inputs, outputs, hides, isBias);
    }
    public NeuralNetwork(int inputCount, int outputCount, int hideCount, boolean isBias, double kHeight) {
        super(inputCount, outputCount, hideCount, isBias, kHeight);
    }

    public void setTrainingPercent(double trainingPercent) throws NeuralNetworkException {
        if (trainingPercent < 0 || trainingPercent > 1)
            throw new NeuralNetworkException("Training percent must be from 0 to 1, but it is " + trainingPercent);
        this.trainingPercent = trainingPercent;
    }

    public void setup(double[][] datas, double[][] solutions) throws NeuralNetworkException {
        check(datas, solutions, true);
        int dataSize = datas.length;
        int trainingSize = (int) (dataSize * trainingPercent);

        shuffle(datas, solutions);
        training = createDataSet(datas, solutions, 0, trainingSize);
        validation = createDataSet(datas, solutions, trainingSize, dataSize);
    }

    public void learn(int cycles) {
        for (int i = 0; i < cycles; i++) {
            learn(training.datas, training.solutions);
        }
    }
    public void learnWithLog(int cycles) throws NeuralNetworkException {
        var answers = createAnswers(training.solutions);
        for (int i = 0; i < cycles; i++) {
            learn(training.datas, training.solutions);
            double dev = deviation(training.datas, training.solutions);
            double rdev = deviation(validation.datas, validation.solutions);
            System.out.println("cycle " + i + " : rdev = " + rdev + " (dev = " + dev + "), corrects = " + corrects(training.datas, answers));
        }
    }

    @NotNull
    @Contract(pure = true)
    public static int[] createAnswers(@NotNull double[][] solutions) {
        final int size = solutions.length;
        int[] answers = new int[size];
        for (int q = 0; q < size; q++) {
            int maxi = 0;
            for (int i = 1; i < solutions[q].length; i++)
                if (solutions[q][i] > solutions[q][maxi])
                    maxi = i;
            answers[q] = maxi;
        }
        return answers;
    }

    private static void shuffle(@NotNull double[][] datas, double[][] solutions) {
        final int size = datas.length;
        var random = new Random();
        for (int i = 0; i < size; i++) {
            int index = random.nextInt(size);
            swap(datas, i, index);
            swap(solutions, i, index);
        }
    }
    private static void swap(double[][] array, int pos1, int pos2) {
        if (pos1 == pos2) return;
        var temp = array[pos1];
        array[pos1] = array[pos2];
        array[pos2] = temp;
    }

    @NotNull
    private static DataSet createDataSet(double[][] datas, double[][] solutions, int from, int to) {
        var _datas = cut(datas, from, to);
        var _solutions = cut(solutions, from, to);
        return new DataSet(_datas, _solutions);
    }
    @NotNull
    private static double[][] cut(double[][] array, int from, int to) {
        int size = to - from;
        double[][] values = new double[size][];
        System.arraycopy(array, from, values, 0, size);
        return values;
    }
}
