#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time

def area_of_dataframe(dataframe, x_start, x_last, y_start, y_last):
    area_dataframe = dataframe[
        (dataframe["x"] >= x_start) &
        (dataframe["x"] <= x_last) & 
        (dataframe["y"] >= y_start) & 
        (dataframe["y"] <= y_last)
    ]
    return area_dataframe

def dataframe_to_cells(dataframe, cells_in_x, cells_in_y):
    cells = []

    for ix, x in enumerate(cells_in_x):
        for iy, y in enumerate(cells_in_y):
            if ix != 0 and iy != 0:
                cell = area_of_dataframe(dataframe, cells_in_x[ix-1], x, cells_in_y[iy-1], y)
                cells.append(cell) # Use np.array here(?)
    
    return cells

def normalize_cells_area(cells, MinMaxArea):
    normalized_cells = []

    for cell in cells:
        norm_cell = (cell - MinMaxArea.min()) / (MinMaxArea.max() - MinMaxArea.min())
        normalized_cells.append(norm_cell)

    return normalized_cells

def get_cell_neighbors(cells, cells_in_x, cells_in_y, num_neighbors):
    num_cells_in_y = len(cells_in_y)-1
    neighbors = {}

    for cell_idx in range(len(cells)):
        if cell_idx < num_cells_in_y*num_neighbors or cell_idx >= len(cells)-num_cells_in_y*num_neighbors:
            continue # Continue on horizontal edges
        if (cell_idx % num_cells_in_y) < num_neighbors or (cell_idx % num_cells_in_y) >= num_cells_in_y-num_neighbors:
            continue # Continue on vertical edges

        neighbors[cell_idx] = []
        for nx in range(1, num_neighbors+1):
            neighbors[cell_idx].append(cell_idx-(num_cells_in_y*nx)) # Neighbors to the left
            neighbors[cell_idx].append(cell_idx-nx) # Neighbors above
            neighbors[cell_idx].append(cell_idx+nx) # Neighbors below
            neighbors[cell_idx].append(cell_idx+(num_cells_in_y*nx)) # Neighbors to the right
            for ny in range(1, num_neighbors+1):       
                neighbors[cell_idx].append(cell_idx-((num_cells_in_y*nx)+ny)) # Left corners
                neighbors[cell_idx].append(cell_idx-((num_cells_in_y*nx)-ny)) # Left corners
                neighbors[cell_idx].append(cell_idx+((num_cells_in_y*nx)-ny)) # Right corners
                neighbors[cell_idx].append(cell_idx+((num_cells_in_y*nx)+ny)) # Right corners

    return neighbors

def cell_statistics(cell):
    median = np.median(cell.z)
    std = np.std(cell.z)
    mx = np.max(cell.z)
    mn = np.min(cell.z)

    return median, std, mx, mn

def calculate_statistics(neighbors, cells):
    # Get all statistics
    for key in neighbors:
        # Skip if there is no points in the cell
        if len(cells[key]) == 0: 
            continue

        # Cell indexes to check for statistics
        cell_indexes = [key] + neighbors[key]

        # Find statistics if each cell
        statistics = []
        for idx in cell_indexes:
            if len(cells[idx]) == 0:
                # Empty statistics if neighbor has nothing to show for
                for i in range(len(STATISTICS)):
                    statistics.append(-1)
            else:
                median, std, mx, mn = cell_statistics(cells[idx]) 
                statistics.append(median)
                statistics.append(std)
                statistics.append(mx)
                statistics.append(mn)

        # Add it to the current cell
        for idx, stats in enumerate(statistics):
            col_name = STATISTICS[(idx % len(STATISTICS))] + str(idx)
            cells[key][col_name] = stats

    return cells

def main():
    # Start timer
    startTime = time.time()

    # Define statistics and size
    STATISTICS = ["median", "std", "max", "min"]
    CELL_SIZE = 0.1 # 0.1 meters

    # Read data
    data = pd.read_csv("data/felt1_points.csv")

    # Get an area from the data
    xMax = data.x.max()/2
    yMax = data.y.max()/2
    data_area = area_of_dataframe(data, 0, 0+xMax, 0, 0+yMax)

    # Turn area into cells
    cells_in_x = np.arange(0, xMax+CELL_SIZE, CELL_SIZE)
    cells_in_y = np.arange(0, yMax+CELL_SIZE, CELL_SIZE)
    cells = dataframe_to_cells(data_area, cells_in_x, cells_in_y)
        
    # Min-Max normalization on cells
    norm_cells = normalize_cells_area(cells, data_area)

    # Get neighbor cells for each cell
    neighbors = 5
    neighbors = get_cell_neighbors(norm_cells, cells_in_x, cells_in_y)

    # Calculate statistics and add to each cell
    stat_cells = calculate_statistics(neighbors, norm_cells)

    # Print and store plot
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    plt.plot(t, s)
    plt.xlabel("time (s)")
    plt.ylabel("voltage (mV)")
    plt.title("Simple Plot")
    plt.grid()
    plt.savefig("test.jpg")
    plt.show()

    # Get executiontime
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

if __name__ == '__main__':
    main()