#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def cell_statistics(cell, z_index):
    cell_z = cell[:, z_index]
    median = np.median(cell_z)
    std = np.std(cell_z)
    mx = np.max(cell_z)
    mn = np.min(cell_z)

    return median, std, mx, mn

def dataframe_to_normalized_stats_cells(dataframe, cells_in_x, cells_in_y, MinMaxArea, z_index):
    cells = []

    sample_min = np.min(MinMaxArea, axis=0)
    sample_max = np.max(MinMaxArea, axis=0)

    for ix, x in enumerate(cells_in_x):
        for iy, y in enumerate(cells_in_y):
            if ix != 0 and iy != 0:
                cell = np.array(area_of_dataframe(dataframe, cells_in_x[ix-1], x, cells_in_y[iy-1], y))
                
                if len(cell) <= 0:
                    cells.append(cell)
                else:
                    # Normalize
                    norm_cell = (cell - sample_min) / (sample_max - sample_min)

                    # Add cell statistics
                    median, std, mx, mn = cell_statistics(norm_cell, z_index)
                    stats = np.array([[median, std, mx, mn] for i in range(len(norm_cell))])
                    norm_cell_stats = np.append(norm_cell, stats, axis=1)

                    cells.append(norm_cell_stats)
    
    return cells

def check_neighbor(all_stats, current_cell, len_stats):
    if len(current_cell) <= 0:
        return np.concatenate((all_stats, np.array([-1 for i in range(len_stats)])), axis=0)
    else:
        return np.concatenate((all_stats, current_cell[0, -len_stats:]), axis=0)

def get_cell_neighbors(cells, cells_in_y, num_neighbors, len_stats):
    num_cells_in_y = len(cells_in_y)-1
    neighbor_cells = []

    for cell_idx in range(len(cells)):
        if len(cells[cell_idx]) <= 0:
            continue

        if cell_idx < num_cells_in_y*num_neighbors or cell_idx >= len(cells)-num_cells_in_y*num_neighbors:
            continue # Continue on horizontal edges
        if (cell_idx % num_cells_in_y) < num_neighbors or (cell_idx % num_cells_in_y) >= num_cells_in_y-num_neighbors:
            continue # Continue on vertical edges

        all_neighbor_stats = np.array([])
        for nx in range(1, num_neighbors+1):
            all_neighbor_stats = check_neighbor(all_neighbor_stats, cells[(cell_idx-(num_cells_in_y*nx))], len_stats) # Neighbors to the left
            all_neighbor_stats = check_neighbor(all_neighbor_stats, cells[(cell_idx-nx)], len_stats) # Neighbors above
            all_neighbor_stats = check_neighbor(all_neighbor_stats, cells[(cell_idx+nx)], len_stats) # Neighbors below
            all_neighbor_stats = check_neighbor(all_neighbor_stats, cells[(cell_idx+(num_cells_in_y*nx))], len_stats) # Neighbors to the right
            for ny in range(1, num_neighbors+1):
                all_neighbor_stats = check_neighbor(all_neighbor_stats, cells[(cell_idx-((num_cells_in_y*nx)+ny))], len_stats) # Left corners
                all_neighbor_stats = check_neighbor(all_neighbor_stats, cells[(cell_idx-((num_cells_in_y*nx)-ny))], len_stats) # Left corners
                all_neighbor_stats = check_neighbor(all_neighbor_stats, cells[(cell_idx+((num_cells_in_y*nx)-ny))], len_stats) # Right corners
                all_neighbor_stats = check_neighbor(all_neighbor_stats, cells[(cell_idx+((num_cells_in_y*nx)+ny))], len_stats) # Right corners

        # Add stats to current cell
        stats = np.array([all_neighbor_stats for i in range(len(cells[cell_idx]))])
        neighbor_cell_stats = np.append(cells[cell_idx], stats, axis=1)
        neighbor_cells.append(neighbor_cell_stats)

    return neighbor_cells

def main():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--part', type=int, default=1, metavar='N', help='Read area in parts')
    parser.add_argument('--neighbors', type=int, default=2, metavar='N', help='Number of neighbors')
    args = parser.parse_args()
    arg_part = args.part
    arg_neighbors = args.neighbors

    # Start timer
    startTime = time.time()

    # Define statistics and size
    STATISTICS = ["median", "std", "max", "min"]
    CELL_SIZE = 0.5 # 0.5 meters
    Z_INDEX = 3 # Where to find the z value

    # Read data
    data = pd.read_csv("../project_data/felt1_points.csv")

    # Get an area from the data
    xMin = data.x.max()*(0.1*(arg_part-1))
    xMax = data.x.max()*(0.1*arg_part)
    yMin = data.y.max()*0
    yMax = data.y.max()*1
    data_area = area_of_dataframe(data, xMin, xMax, yMin, yMax)

    # Turn area into cells
    cells_in_x = np.arange(xMin, xMax+CELL_SIZE, CELL_SIZE)
    cells_in_y = np.arange(yMin, yMax+CELL_SIZE, CELL_SIZE)
    cells = dataframe_to_normalized_stats_cells(data_area, cells_in_x, cells_in_y, np.array(data), Z_INDEX)

    # Get neighbor cells for each cell
    num_neighbors = arg_neighbors
    neighbors = get_cell_neighbors(cells, cells_in_y, num_neighbors, len(STATISTICS))

    # Create dataframe
    df_columns = ["accepted", "x", "y", "z", "thu" "tvu"]
    len_stats = neighbors[0].shape[1]-len(df_columns)
    for i in range(len_stats):
        df_columns.append(str(i))

    celled_dataframe = pd.DataFrame(np.concatenate(neighbors), columns=df_columns)
    celled_dataframe.to_csv("../project_data/cell_05/part"+ str(arg_part) +".csv", index=None)

    # Get executiontime
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

if __name__ == '__main__':
    main()