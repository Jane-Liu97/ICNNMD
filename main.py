import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import functools
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

parser = argparse.ArgumentParser(description='ICNNMD')
# The data file to read
parser.add_argument('--nc1_file')
# The location the generated 'bin' file to save
parser.add_argument('--nc2_file')
parser.add_argument('--pdb1_file')
parser.add_argument('--pdb2_file')
parser.add_argument('--print_acc')
parser.add_argument('--save_models')
parser.add_argument('--print_detail')
args = parser.parse_args()

if __name__ == '__main__':
    file0_1 = args.nc1_file
    file0_2 = args.pdb1_file
    file1_1 = args.nc2_file
    file1_2 = args.pdb2_file
    if_prt = args.print_acc
    if_save = args.save_models
    if_dtl = args.print_detail
    try:
        from . import traj_utils
    except Exception:
        import traj_utils
    _, y_all, X_all = traj_utils.traj_pre(file0_1, file1_1, file1_2, file1_2)
    print("Preprocess Done.\n")
    try:
        from . import split_data
    except Exception:
        import split_data
    group = 20
    cap = int(len(X_all) / 20)
    k = 5
    X_train, y_train, X_test, y_test = split_data.split_by_group(X_all, y_all, group, cap, k)


    # =============================================================================
    try:
        from . import data_utils
    except Exception:
        import data_utils
    num_classes = 2
    X_train, y_train, X_test, y_test = data_utils.data_encode(X_train, y_train, X_test, y_test, k, num_classes)
    # =============================================================================

    # =============================================================================
    try:
        from . import cnn_model
    except Exception:
        import cnn_model

    batch_size = 128
    epochs = 5
    model = cnn_model.cnn_build(X_train, k, num_classes)
    model, history = cnn_model.cnn_train(model, X_train, y_train, X_test, y_test, k, batch_size, epochs, if_dtl)

    if (if_prt):
        # =============================================================================
        try:
            from . import model_evaluate
        except Exception:
            import model_evaluate

        model_evaluate.eva_cross_acc_all(model, X_train, X_test, y_train, y_test)

    if (if_save):
        for t in range(k):
            model[t].save('model' + str(t) + '.h5')

    # =============================================================================
    try:
        from . import lime_utils
    except Exception:
        import lime_utils

    importance_pic, explainer, segmenter = lime_utils.cnn_lime(model, X_all, y_all, k)
    # =============================================================================

    # =============================================================================

    import xlsxwriter

    workbook = xlsxwriter.Workbook('atom.xlsx')
    worksheet = workbook.add_worksheet()
    size = len(importance_pic)
    for i in range(size):
        for j in range(size):
            worksheet.write(i * size + j, 0, i * size + j + 1)
            worksheet.write(i * size + j, 1, importance_pic[i][j])
    workbook.close()

    # =============================================================================

    f = open(file0_2, "r")

    data = f.readlines()
    f.close()
    p = len(data) - 1
    data.pop(p)
    data.pop(p - 1)
    data.pop(0)
    #################################################################
    bond = []
    for line in data:
        bond_line = line.split(' ')
        while '' in bond_line:
            bond_line.remove('')
        bond.append([int(bond_line[1]), int(bond_line[4])])

    #################################################################

    #################################################################
    fre = len(y_all)
    import xlrd
    rbook = xlrd.open_workbook('atom.xlsx')
    rbook.sheets()
    rsheet = rbook.sheet_by_index(0)

    for row in rsheet.get_rows():
        if (int(row[0].value) - 1) < len(bond):
            bond[int(row[0].value) - 1].append(row[1].value / fre)
        else:
            break
    #################################################################

    #################################################################
    res = []
    for i in range(len(bond)):
        res.append([bond[i][1], bond[i][2]])

    #################################################################
    freq = []
    s = 0
    score = 0
    n = 0
    f = 0
    for i in range(len(res) - 1):
        if res[i][0] == res[i + 1][0]:
            score += res[i + 1][1]
            n += 1
        else:
            score += res[i + 1][1]
            n += 1
            freq.append([res[i][0], score / n])
            score = 0
            s = 0
            n = 0
    # freq.append([res[len(res)-1][0], s+res[len(res)-1][1]])
    freq_sort = sorted(freq, key=lambda x: (x[1]))
    freq_sort.reverse()

    workbook = xlsxwriter.Workbook('res_score.xlsx')
    worksheet = workbook.add_worksheet()
    for i in range(len(freq_sort)):
        worksheet.write(i, 0, freq_sort[i][0])
        worksheet.write(i, 1, freq_sort[i][1])
    workbook.close()