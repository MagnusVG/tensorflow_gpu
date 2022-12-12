#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight
from functools import partial
import time
import argparse

def area_of_dataframe(dataframe, x_start, x_last, y_start, y_last):
    area_dataframe = dataframe[
        (dataframe["x"] >= x_start) &
        (dataframe["x"] <= x_last) & 
        (dataframe["y"] >= y_start) & 
        (dataframe["y"] <= y_last)
    ]
    return area_dataframe

def get_x_and_y(dataframe, z_index):
    x = np.array(dataframe.iloc[:, z_index:])
    y = np.array(dataframe.accepted)

    return x, y

# Look at the training, save plots
def plot_result(model, item, save_location):
    plt.plot(model.history[item], label=item)
    plt.plot(model.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and validation {} over epochs".format(item), fontsize=14)
    plt.legend()
    plt.savefig(save_location)

def main():
    parser = argparse.ArgumentParser(description='Neural Network')
    parser.add_argument('--folder', type=str, default="", metavar='N', help='Folder to look for data')
    parser.add_argument('--batchSize', type=int, default=32, metavar='N', help='Batch size for tuning')
    parser.add_argument('--model', type=str, default="", metavar='N', help='Model name')
    args = parser.parse_args()
    arg_folder = args.folder
    arg_batch_size = args.batchSize
    arg_model = args.model

    # Start timer
    startTime = time.time()

    # Standard variables
    Z_INDEX = 3
    VAL_AREA = 0.15
    EPOCHS = 50

    # Read data
    part1 = pd.read_parquet("../project_data/"+ arg_folder +"part1.parquet.gzip")
    part2 = pd.read_parquet("../project_data/"+ arg_folder +"part2.parquet.gzip")
    part3 = pd.read_parquet("../project_data/"+ arg_folder +"part3.parquet.gzip")
    part4 = pd.read_parquet("../project_data/"+ arg_folder +"part4.parquet.gzip")
    part5 = pd.read_parquet("../project_data/"+ arg_folder +"part5.parquet.gzip")

    # Combine
    combined_data = pd.concat([part1, part2, part3, part4, part5], ignore_index=True)

    # Found an area beforehand to use as validation, about 10%
    train_area = area_of_dataframe(combined_data, VAL_AREA, 1, 0, 1)
    validation_area = area_of_dataframe(combined_data, 0, VAL_AREA, 0, 1)

    # Turn into x and y
    x_train, y_train = get_x_and_y(train_area, Z_INDEX)
    x_val, y_val = get_x_and_y(validation_area, Z_INDEX)

    # Do undersampling
    sampling = RandomUnderSampler(sampling_strategy=0.1)
    x_train_us, y_train_us = sampling.fit_resample(x_train, y_train)

    # Get class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_us), y=y_train_us)
    class_weight_dict = dict(enumerate(class_weights))

    # Build a model
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=x_train[0].shape, name="Input"))
    model.add(keras.layers.Dense(100, activation="relu", name="Hidden1"))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(50, activation="relu", name="Hidden2"))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(10, activation="relu", name="Hidden3"))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(1, activation="sigmoid", name="Output"))
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.01, patience=10)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    # Train the model
    history = model.fit(
        x_train_us, 
        y_train_us, 
        validation_data=(x_val, y_val),
        class_weight=class_weight_dict,
        epochs=EPOCHS,
        callbacks=[es], 
        shuffle=True, 
        batch_size=arg_batch_size)

    # Plots
    acc_location = "../project_data/"+ arg_folder +"accuray_plot.jpg"
    plot_result(history, "binary_accuracy", acc_location)
    loss_location = "../project_data/"+ arg_folder +"loss_plot.jpg"
    plot_result(history, "loss", loss_location)

    # Save model
    model_location = "../project_data/"+ arg_folder + arg_model
    model.save(model_location)

    # Get executiontime
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

if __name__ == '__main__':
    main()