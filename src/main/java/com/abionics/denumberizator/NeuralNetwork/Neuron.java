package com.abionics.denumberizator.NeuralNetwork;

class Neuron {
    private double output;

    Neuron() {
        output = 1;
    }
    void active(double input) {
        output = 1.0 / (1 + Math.exp(-input));
    }
    double get() {
        return output;
    }
    void set(double value) {
        output = value;
    }
}
