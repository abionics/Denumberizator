package com.abionics.denumberizator;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.util.Arrays;
import java.util.stream.IntStream;

public class Controller {
    private static final File DATA_DIRECTORY = new File("data");

    @FXML
    Canvas canvas;
    @FXML
    Slider speedSlider;
    @FXML
    Slider momentumSlider;
    @FXML
    TextField hiddenCountTextField;
    @FXML
    ComboBox<Integer> numberComboBox;
    @FXML
    Label resultLabel;

    private Model model = new Model();
    private GraphicsContext graphics;
    private boolean needInit = false;

    private int screenWidth;
    private int screenHeight;
    private boolean[][] picture;


    @FXML
    private void initialize() {
        IntStream.range(0, 10).forEach(number -> numberComboBox.getItems().add(number));
        numberComboBox.setValue(0);

        speedSlider.valueProperty().addListener(observable -> needInit = true);
        momentumSlider.valueProperty().addListener(observable -> needInit = true);
        hiddenCountTextField.setOnAction(actionEvent -> needInit = true);

        graphics = canvas.getGraphicsContext2D();
        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED,
                event -> {
                    graphics.beginPath();
                    graphics.moveTo(event.getX(), event.getY());
                    graphics.stroke();
                });
        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED,
                event -> {
                    if (event.getX() < 0 || event.getX() >= screenWidth || event.getY() < 0 || event.getY() >= screenHeight)
                        return;
                    graphics.lineTo(event.getX(), event.getY());
                    graphics.stroke();
                    picture[(int) event.getX()][(int) event.getY()] = true;
                });

        screenWidth = (int) canvas.getWidth();
        screenHeight = (int) canvas.getHeight();
        picture = new boolean[screenWidth][screenHeight];

        DATA_DIRECTORY.mkdirs();
        initNeural();
        clearCanvas();
    }

    private void initNeural() {
        var hiddenCountText = hiddenCountTextField.getText();
        if (!isInteger(hiddenCountText)) return;
        var hiddenCount = Integer.parseInt(hiddenCountText);
        var speed = speedSlider.getValue();
        var momentum = momentumSlider.getValue();
        model.initNeural(hiddenCount, speed, momentum);
        needInit = false;
    }

    @FXML
    private void rememberNumber() {
        int number = numberComboBox.getValue();
        model.rememberNumber(picture, number);
    }

    @FXML
    private void learn() {
        if (!model.isInitNeural() || needInit) initNeural();
        model.learn();
    }

    @FXML
    private void clean() {
        for (boolean[] row : picture)
            Arrays.fill(row, false);
        clearCanvas();
    }

    @FXML
    private void analyze() {
        if (!model.isInitNeural() || needInit) initNeural();
        int result = model.analyze(picture);
        if (result != -1) {
            resultLabel.setText(Integer.toString(result));
        } else {
            resultLabel.setText("NaN");
        }
    }

    @FXML
    private void heatmap() {
        clean();
        int number = numberComboBox.getValue();
        double[][] matrix = model.heatmap(number);
        int width = matrix.length;
        int height = matrix[0].length;
        double sizeX = (double) screenWidth / width;
        double sizeY = (double) screenHeight / height;
        GraphicsContext graphics = canvas.getGraphicsContext2D();
        for (int i = 0; i < width; i++)
            for (int j = 0; j < height; j++) {
                double value = matrix[i][j];
                Color color = value > 0 ? Color.color(0, 1, 0, value) : Color.color(1, 0, 0, -value);
                graphics.setFill(color);
                graphics.fillRect(i * sizeX, j * sizeY, sizeX, sizeY);
            }
    }

    @FXML
    private void datasetLoad() {
        var chooser = getFileChooser();
        var file = chooser.showOpenDialog(canvas.getScene().getWindow());
        if (file == null) return;
        model.datasetLoad(file);
    }

    @FXML
    private void datasetSave() {
        var chooser = getFileChooser();
        var file = chooser.showSaveDialog(canvas.getScene().getWindow());
        if (file == null) return;
        model.datasetSave(file);
    }

    @Contract(pure = true)
    static void exit() {
        Platform.exit();
        System.exit(0);
    }

    @NotNull
    private FileChooser getFileChooser() {
        var chooser = new FileChooser();
        chooser.setInitialDirectory(DATA_DIRECTORY);
        FileChooser.ExtensionFilter extensions = new FileChooser.ExtensionFilter("JSON files", "*.json");
        chooser.getExtensionFilters().add(extensions);
        return chooser;
    }

    private void clearCanvas() {
        graphics.beginPath();

        graphics.setFill(Color.LIGHTGRAY);
        graphics.setStroke(Color.BLACK);
        graphics.setLineWidth(3);

        graphics.fillRect(0, 0, screenWidth, screenHeight);
        graphics.strokeRect(0, 0, screenWidth, screenHeight);

        graphics.setFill(Color.RED);
        graphics.setStroke(Color.BLACK);
        graphics.setLineWidth(2);
    }

    private static boolean isInteger(String value) {
        try {
            Integer.parseInt(value);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}
