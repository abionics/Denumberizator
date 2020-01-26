package com.abionics.denumberizator;

import com.abionics.denumberizator.NeuralNetwork.DataSet;
import com.abionics.denumberizator.NeuralNetwork.NeuralNetwork;
import com.google.gson.Gson;
import javafx.scene.control.Alert;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

class Model {
    private static final int MATRIX_WIDTH = 6;
    private static final int MATRIX_HEIGHT = 8;
    private static final int INPUT_COUNT = MATRIX_WIDTH * MATRIX_HEIGHT;
    private static final int OUTPUT_COUNT = 10;
    private static final int ONE_LEARN_CYCLES = 500;

    private boolean[][] picture;
    private boolean[][] matrix = new boolean[MATRIX_WIDTH][MATRIX_HEIGHT];
    private double[] linear = new double[INPUT_COUNT];

    private ArrayList<double[]> datas = new ArrayList<>();
    private ArrayList<double[]> solutions = new ArrayList<>();
    private NeuralNetwork neural;
    private int result;


    void initNeural(int hiddenCount, double speed, double momentum) {
        neural = new NeuralNetwork(INPUT_COUNT, OUTPUT_COUNT, hiddenCount);
        neural.setSpeed(speed);
        neural.setMomentum(momentum);
        neural.setTrainingPercent(1.0);
        alert("Neural init", "hidden count = " + hiddenCount +
                "\nspeed = " + speed +
                "\nmomentum = " + momentum + "");
    }

    boolean isInitNeural() {
        return neural != null;
    }

    int analyze(boolean[][] picture) {
        this.picture = picture;
        compress();
        analyze();
        return result;
    }

    private void compress() {
        for (boolean[] row : matrix)
            Arrays.fill(row, false);

        int width = picture.length;
        int height = picture[0].length;

        int minX = width - 1;
        int maxX = 0;
        int minY = height - 1;
        int maxY = 0;
        for (int i = 0; i < width; i++)
            for (int j = 0; j < height; j++)
                if (picture[i][j]) {
                    if (i < minX) minX = i;
                    if (i > maxX) maxX = i;
                    if (j < minY) minY = j;
                    if (j > maxY) maxY = j;
                }
        int lenX = maxX - minX + 1;
        int lenY = maxY - minY + 1;
        for (int i = 0; i < width; i++)
            for (int j = 0; j < height; j++) {
                if (picture[i][j]) {
                    int x = (i - minX) * MATRIX_WIDTH / lenX;
                    int y = (j - minY) * MATRIX_HEIGHT / lenY;
                    matrix[x][y] = true;
                }
            }

        for (int j = 0; j < MATRIX_HEIGHT; j++) {
            for (int i = 0; i < MATRIX_WIDTH; i++) {
                linear[j * MATRIX_WIDTH + i] = (matrix[i][j] ? 1 : 0);
            }
        }
    }

    private void analyze() {
        result = neural.analyze(linear);
    }

    void rememberNumber(boolean[][] picture, int number) {
        this.picture = picture;
        compress();
        double[] data = Arrays.copyOf(linear, linear.length);
        double[] solution = createSolution(number);
        datas.add(data);
        solutions.add(solution);
        System.out.println("[Remembered number " + number + "]");
    }

    @NotNull
    @Contract(pure = true)
    private double[] createSolution(int number) {
        double[] solution = new double[OUTPUT_COUNT];
        solution[number] = 1;
        return solution;
    }

    void learn() {
        var _datas = datas.toArray(double[][]::new);
        var _solutions = solutions.toArray(double[][]::new);
        neural.setup(_datas, _solutions);
        neural.learn(ONE_LEARN_CYCLES);
        statistic();
    }

    private void statistic() {
        var _datas = datas.toArray(double[][]::new);
        var _solutions = solutions.toArray(double[][]::new);
        double deviation = neural.deviation(_datas, _solutions);
        int corrects = neural.corrects(_datas, NeuralNetwork.createAnswers(_solutions));
        int count = datas.size();
        alert("Statistic", "deviation = " + deviation + ", corrects = " + corrects + " of " + count + " (" + 100. * corrects / count + ")");
    }

    public double[][] heatmap(int output) {
        double[] linear = neural.heatmap(output);
        double[][] matrix = new double[MATRIX_WIDTH][MATRIX_HEIGHT];
        int k = 0;
        for (int i = 0; i < MATRIX_WIDTH; i++)
            for (int j = 0; j < MATRIX_HEIGHT; j++)
                matrix[i][j] = linear[k++];
        return matrix;
    }

    void datasetLoad(File file) {
        DataSet dataset;
        try {
            var json = new Gson();
            BufferedReader reader = new BufferedReader(new FileReader(file));
            dataset = json.fromJson(reader.readLine(), DataSet.class);
            reader.close();
            datas = new ArrayList<>(Arrays.asList(dataset.datas));
            solutions = new ArrayList<>(Arrays.asList(dataset.solutions));
            System.out.println("[Dataset loaded]");
            alert("Loaded dataset", "size = " + datas.size());
        } catch (IOException ex) {
            System.out.println("[ERROR] Cannot load dataset from file " + file.getName());
            ex.printStackTrace();
        }
    }

    void datasetSave(File file) {
        var _datas = datas.toArray(double[][]::new);
        var _solutions = solutions.toArray(double[][]::new);
        var dataset = new DataSet(_datas, _solutions);
        try {
            var json = new Gson();
            FileWriter writer = new FileWriter(file);
            writer.write(json.toJson(dataset));
            writer.close();
            System.out.println("[Dataset saved]");
        } catch (IOException ex) {
            System.out.println("[ERROR] Cannot save dataset to file " + file.getName());
            ex.printStackTrace();
        }
    }

    private static void alert(String header, String text) {
        System.out.println(header + " : " + text);
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Denumberizator Alert");
        alert.setHeaderText(header);
        alert.setContentText(text);
        alert.showAndWait();
    }
}
