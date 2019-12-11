package com.abionics.denumberizator.NeuralNetwork;

public class DataSet {
    public double[][] datas;
    public double[][] solutions;

    public DataSet(double[][] datas, double[][] solutions) {
        this.datas = datas;
        this.solutions = solutions;
    }
}
