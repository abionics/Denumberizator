package com.abionics.denumberizator;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import org.jetbrains.annotations.NotNull;

public class Main extends Application {
    @Override
    public void start(@NotNull Stage primaryStage) throws Exception {
        Parent root = FXMLLoader.load(getClass().getResource("/denumberizator.fxml"));
        primaryStage.setTitle("Denumberizator");
        primaryStage.setScene(new Scene(root, 1000, 685));
        primaryStage.setOnCloseRequest(t -> Controller.exit());
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
