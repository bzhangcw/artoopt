# coding=utf-8
# @author:
# - Meiling Chen, brookechen@sjtu.edu.cn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import *


def main(input_file_path, output_file_path, model_type, solver_name, x_name, y_name):

    # input data
    input_data = pd.read_csv(input_file_path)
    x = input_data['x'].values.tolist()
    y = input_data['y'].values.tolist()

    # normalize
    x = np.array(x)
    x_raw = x
    y = np.array(y)
    y_raw = y
    x_min, x_max = min(x), max(x)
    x = (x - x_min) / (x_max - x_min)
    x = np.mat(x)
    y_min, y_max = min(y), max(y)
    y0 = (y - y_min) / (y_max - y_min)
    n, m = x.shape[1], x.shape[0]

    # generate the results
    # todo: errors will come up if x is bigger than one dimension
    for N in range(2, n+1, 2):
        # deal with the results from qp/mip model
        if model_type == 'mip_gurobipy':
            r_sol, q_sol, b_sol = mip_gurobipy(x, y0, n, m, N)
        elif model_type == 'qp_gurobipy':
            r_sol, q_sol, b_sol = qp_gurobipy(x, y0, n, m, N)
        elif model_type == 'mip_mip':
            r_sol, q_sol, b_sol = mip_mip(x, y0, n, m, N, solver_name)

        # get optimization results
        selected_index = [i for i in range(n) if q_sol[i] == 1]
        dropped_index = [i for i in range(n) if q_sol[i] != 1]
        s_x = x_raw[selected_index]
        d_x = x_raw[dropped_index]
        s_y0 = y_raw[selected_index]
        d_y0 = y_raw[dropped_index]

        # redo linear programming based on raw data
        y_bar = sum(s_y0) / len(s_y0)
        x_bar = sum(s_x) / len(s_x)
        beta1 = sum((s_x[i]-x_bar)*(s_y0[i]-y_bar) for i in range(len(s_x))) / sum((s_x[i]-x_bar)**2 for i in range(len(s_x)))
        beta0 = y_bar - beta1 * x_bar
        y_pred = [beta1 * s_x[i] + beta0 for i in range(len(s_x))]
        res = sum((s_y0[i]-y_pred[i])**2 for i in range(len(s_x)))
        tot = sum((s_y0[i]-y_bar)**2 for i in range(len(s_x)))
        r_sqr = (1 - res / tot)

        # plot the results
        fig = plt.figure()
        s_x = s_x.tolist()
        d_x = d_x.tolist()
        plt.plot(s_x, y_pred, label="selected_regression")
        plt.scatter(s_x, s_y0, color='green', label="selected_samples", marker='o')
        plt.scatter(d_x, d_y0, color='red', label="outliers", marker='o')
        for i in range(len(y0)):
            plt.annotate((i+1), (x_raw[i], y_raw[i]))
        plt.title("drop {} => R^2: {}\n{} = {}, {} = {}".format((n-N), r_sqr, chr(946)+'1', beta1, chr(946)+'0', beta0))
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.grid()
        plt.legend()
        plt.savefig('{}_{}_{}.png'.format(x_name, y_name, N), dpi=500)
        # plt.show()

        # write txt
        filename = output_file_path
        with open(filename, 'a') as f:
            f.write('Select {} samples from {} samples:\n'.format(N, n))
            f.write('Select: ' + str([i+1 for i in selected_index]) + '\n')
            f.write('Drop: ' + str([i+1 for i in dropped_index]) + '\n')
            f.write('Linear regression result: y = beta1 * x + beta0 \n')
            f.write('beta1 = {}, beta0 = {}\n\n'.format(beta1, beta0))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", default='input_data.csv')
    parser.add_argument("--output_file_path", default='results.txt')
    parser.add_argument("--model_type", choices=['qp_gurobipy', 'mip_gurobipy', 'mip_mip'], default='mip_mip')
    parser.add_argument('--solver_name', choices=['CBC', 'GUROBI'], default='CBC')
    parser.add_argument("--x_name", choices=['ash', 'ban2'], default='ash')
    parser.add_argument("--y_name", choices=['ban2', 'ban3'], default='ban2')
    args = parser.parse_args()

    main(input_file_path=args.input_file_path,
         output_file_path=args.output_file_path,
         model_type=args.model_type,
         solver_name=args.solver_name,
         x_name=args.x_name,
         y_name=args.y_name)
