import xlsxwriter
import glob
import pandas as pd
import os
import numpy as np
from scipy.io import savemat

#Write corrcoefs into Excel file 'trial.xlsx', renamed later
workbook = xlsxwriter.Workbook('trial.xlsx')
worksheet = workbook.add_worksheet()

#load all csv files into folder

data_files = os.listdir("/Users/sunnyjin45/Desktop/conf database copy/ran")
#omit '.DS_Store' default hidden file
data_files = data_files[1:]
r_dict = {}


def excel():

    # Start from the first cell.
    # Rows and columns are zero indexed.
    row = 0
    column = 0
    size = 0

    #check csv file one by one, generic --
    global data_files
    global r_values_dict
    while len(data_files) != 0:
        filename = data_files[0]
        myfile = open("/Users/sunnyjin45/Desktop/conf database copy/ran/" + filename, 'r')
        writefile = open(filename + ".txt", 'w')
        header = myfile.readline()
        linelist = myfile.readlines()
        df = pd.read_csv("/Users/sunnyjin45/Desktop/conf database copy/ran/" + filename)
        subjects = df["Subj_idx"].max()
        mAcc_values = np.empty(0)
        mConf_values = np.empty(0)
        mAccA_values = np.empty(0)
        mConfA_values = np.empty(0)
        mAccB_values = np.empty(0)
        mConfB_values = np.empty(0)
        mAccC_values = np.empty(0)
        mConfC_values = np.empty(0)
        mAccD_values = np.empty(0)
        mConfD_values = np.empty(0)
        mAcc5_values = np.empty(0)
        mConf5_values = np.empty(0)
        mAcc6_values = np.empty(0)
        mConf6_values = np.empty(0)
        mAcc7_values = np.empty(0)
        mConf7_values = np.empty(0)
        mAcc8_values = np.empty(0)
        mConf8_values = np.empty(0)
        mAcc9_values = np.empty(0)
        mConf9_values = np.empty(0)
        mAcc10_values = np.empty(0)
        mConf10_values = np.empty(0)
        num_tasks = 1 #default only one condition/subset

    #---------------------------------
    #specific cases begin:

        if filename == 'data_AguilarLleyda_2018.csv' or filename == 'data_AguilarLleyda_unpub.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       trials += 2
                       conf += float(col[15])
                       conf += float(col[14])
                       acc += float(col[13])
                       acc += float(col[12])
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = np.corrcoef(mAcc_values, mConf_values)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        elif filename == 'data_Akdogan_2017_Exp1.csv' or filename == 'data_Akdogan_2017_Exp2.csv' or filename == 'data_Akdogan_2017_Exp3.csv' or filename == 'data_Akdogan_2017_Exp4.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       trials += 1
                       if filename == 'data_Akdogan_2017_Exp4.csv':
                           conf += float(col[6])
                           acc += abs(float(col[3])-float(col[5]))
                       else:
                           conf += float(col[7])
                           acc += abs(float(col[4])-float(col[6]))
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = 0 - np.corrcoef(mAcc_values, mConf_values) #flip the sign
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        elif filename == 'data_Arbuzova_unpub_1.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):

                       if col[0] != '25':
                           if col[0] != '33':
                               if not "NaN" in col[3]:
                                   trials += 1
                                   conf += float(col[3])
                                   if col[1] == col[2]:                         #check this
                                       acc += 1
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = np.corrcoef(mAcc_values, mConf_values)
            for row in range(subjects):
                if row < len(arr):
                    if float(arr[row, 0]) < (0.000000000000000001):
                        arr = np.delete(arr, row, axis=0)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        elif filename == 'data_Clark_unpub.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if col[0] != '1':
                           if col[0] != '11':
                               if not "NaN" in col[3]:
                                   trials += 1
                                   conf += float(col[3])
                                   if col[1] == col[2]:                         #check this
                                       acc += 1
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = np.corrcoef(mAcc_values, mConf_values)
            for row in range(subjects):
                if row < len(arr):
                    if float(arr[row, 0]) < (0.000000000000000001):
                        arr = np.delete(arr, row, axis=0)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        elif filename == 'data_Desender_2014.csv' or filename == 'data_Desender_2016.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if not "NA" in col[3]:
                           if not "NA" in col[2]:
                               if not "training" in col[8]:
                                   trials += 1
                                   if filename == 'data_Desender_2016.csv':
                                       if "easy" in col[3]:
                                           conf += 1
                                   else:
                                       conf += float(col[3])

                                   if col[1] == col[2]:                         #check this
                                       acc += 1
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = np.corrcoef(mAcc_values, mConf_values)
            for row in range(subjects):
                if row < len(arr):
                    if float(arr[row, 0]) < (0.000000000000000001):
                        arr = np.delete(arr, row, axis=0)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        elif filename == 'data_Duyan_2018_Exp1.csv' or filename == 'data_Duyan_2018_Exp2.csv' or filename == 'data_Duyan_2019.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if not "NaN" in col[3]:
                           trials += 1
                           if filename == 'data_Duyan_2019.csv':
                               conf += float(col[6])
                               acc += abs(float(col[3])-float(col[4]))
                           else:
                               conf += float(col[7])
                               acc += abs(float(col[6]))
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = 0 - np.corrcoef(mAcc_values, mConf_values)
            for row in range(subjects):
                if row < len(arr):
                    if float(arr[row, 0]) < (0.000000000000000001):
                        arr = np.delete(arr, row, axis=0)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)


        elif filename == 'data_Matthews_2018_exp1.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if col[14] != '1':
                           if "central_task" in col[15]:
                               trials += 1
                               conf += float(col[3])
                               if col[1] == col[2]:
                                   acc += 1
                           elif "peripheral_task" in col[15]:
                               trials += 1
                               conf += float(col[9])
                               if col[7] == col[8]:
                                   acc += 1
                           elif "dual_tasks" in col[15]:
                               trials += 2
                               conf += float(col[3])
                               conf += float(col[9])
                               if col[1] == col[2]:
                                   acc += 1
                               if col[7] == col[8]:
                                   acc += 1
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = np.corrcoef(mAcc_values, mConf_values)
            for row in range(subjects):
                if row < len(arr):
                    if float(arr[row, 0]) < (0.00000000000000000001):
                        arr = np.delete(arr, row, axis=0)
            arr = np.delete(arr, 7, axis=0)
            arr = np.delete(arr, 7, axis=0)
            arr = np.delete(arr, 7, axis=0)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        elif filename == 'data_Matthews_2018_exp2.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if col[15] != '1':
                           if "central_task" in col[16]:
                               trials += 1
                               conf += float(col[3])
                               if col[1] == col[2]:
                                   acc += 1
                           elif "peripheral_task" in col[16]:
                               trials += 1
                               conf += float(col[9])
                               if col[7] == col[8]:
                                   acc += 1
                           elif "dual_tasks" in col[16]:
                               if not "NaN" in col[2]:
                                   if not "NaN" in col[8]:
                                       trials += 2
                                       conf += float(col[3])
                                       conf += float(col[9])
                                       if col[1] == col[2]:
                                           acc += 1
                                       if col[7] == col[8]:
                                           acc += 1
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = np.corrcoef(mAcc_values, mConf_values)
            for row in range(subjects):
                if row < len(arr):
                    if float(arr[row, 0]) < (0.000000000000000001):
                        arr = np.delete(arr, row, axis=0)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        elif filename == 'data_Rausch_2014.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if not "1" in col[8]:
                           trials += 1
                           conf += float(col[3])
                           acc += abs(float(col[7]))
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = 0 - np.corrcoef(mAcc_values, mConf_values) #flip the sign
            for row in range(subjects):
                if row < len(arr):
                    if float(arr[row, 0]) < (0.000000000000000001):
                        arr = np.delete(arr, row, axis=0)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        elif filename == 'data_Samaha_2017_exp1.csv' or filename == 'data_Samaha_2017_exp2.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       trials += 1
                       conf += float(col[3])
                       acc += abs(float(col[5]))
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = 0 - np.corrcoef(mAcc_values, mConf_values) #flip the sign
            for row in range(subjects):
                if row < len(arr):
                    if float(arr[row, 0]) < (0.000000000000000001):
                        arr = np.delete(arr, row, axis=0)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        elif filename == 'data_Weidemann_2016.csv':
            person = ""
            num = 1
            mdic1 = {}
            arr1 = np.empty((0, 4), float)
            for line in linelist:
                acc = 0
                trials = 0
                conf = 0.0
                col = line.split(',')
                if col[0] != person:
                    person = col[0]
                    for line in linelist:
                        col1 = line.split(',')
                        if col1[0] == person:
                            if not "NaN" in col1[2]:
                                if not "NaN" in col1[3]:
                                   if col1[6] != "":
                                       trials += 1
                                       conf += float(col1[6])
                                       if col1[4] == col1[5]:
                                           acc += 1


                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr1 = np.append(arr1, np.array([[num, float(acc / trials), float(conf / trials), trials]]), axis=0)
                    num += 1
            correlationA = np.corrcoef(mAcc_values, mConf_values)
            mdic1 = {'data': arr1}
            savemat(filename + ".mat", mdic1)


        elif filename == 'data_Duyan_unpub_Exp1.csv' or filename == 'data_Duyan_unpub_Exp2.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       trials += 1
                       conf += float(col[6])
                       acc += abs(float(col[5]))
                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = 0 - np.corrcoef(mAcc_values, mConf_values) #flip the sign
            for row in range(subjects):
                if row < len(arr):
                    if float(arr[row, 0]) < (0.000000000000000001):
                        arr = np.delete(arr, row, axis=0)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        elif filename == 'data_Koculak_unpub.csv':
            mdic = {}
            arr = np.empty(shape=(subjects, 4))
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                        if not "NaN" in col[3]:
                            trials += 1
                            acc += abs(float(col[7]))
                            conf += float(col[3])

                if trials != 0:
                    mean_acc = round(acc / trials, 5)
                    mean_conf = round(conf / trials, 5)
                    writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                    writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                    mAcc_values = np.append(mAcc_values, mean_acc)
                    mConf_values = np.append(mConf_values, mean_conf)
                    arr[num, 0] = num + 1
                    arr[num, 1] = float(acc / trials)
                    arr[num, 2] = float(conf / trials)
                    arr[num, 3] = trials
            correlationA = np.corrcoef(mAcc_values, mConf_values)
            for row in range(subjects):
                if row < len(arr):
                    if float(arr[row, 0]) < (0.000000000000000001):
                        arr = np.delete(arr, row, axis=0)
            mdic = {'a': arr}
            print(mdic)
            savemat(filename + ".mat", mdic)

        #2 conditions
        elif filename == 'data_Yeon_2019.csv' or filename == 'data_Xu_2019_Exp1.csv' or filename == 'data_Wang_2017_Neuropsychologia.csv' or filename == 'data_Wang_2017_NatComm.csv' or filename == 'data_Rausch_2016.csv' or filename == 'data_Siedlecka_2019_Exp2.csv' or filename == 'data_Siedlecka_2019_Exp1.csv' or filename == 'data_Schmidt_2019_perception.csv' or filename == 'data_Schmidt_2019_memory.csv' or filename == 'data_Sadeghi_2017_memory.csv' or filename == 'data_Sadeghi_2017_perception.csv' or filename == 'data_Reyes_unpub.csv' or filename == 'data_Chetverikov_2014_exp3.csv' or filename == 'data_Chetverikov_2014_exp4.csv' or filename == 'data_Fallow_unpub_2.csv' or filename == 'data_Massoni_2017.csv' or filename == 'data_Massoni_unpub.csv' or filename == 'data_Paulewicz_unpub2.csv' or filename == 'data_Paulewicz_unpub3.csv':
            num_tasks = 2
            mdic1 = {}
            mdic2 = {}
            arr1 = np.empty((0, 4), float)
            arr2 = np.empty((0, 4), float)
            for num in range(subjects):
                A_acc = 0
                A_conf = 0
                B_acc = 0
                B_conf = 0
                trials = 0
                A_trials = 0
                B_trials = 0
                mean_A_acc = 0
                mean_B_acc = 0
                mean_A_conf = 0
                mean_B_conf = 0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       trials += 1
                       if filename == 'data_Chetverikov_2014_exp3.csv':
                           if "Liking before confidence" in col[7]:
                               A_trials += 1
                               A_conf += float(col[3])
                               if col[1] == col[2]:
                                   A_acc += 1
                           elif "Confidence before liking" in col[7]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1
                       elif filename == 'data_Chetverikov_2014_exp4.csv':
                           if "Liking before confidence" in col[7]:
                               A_trials += 1
                               A_conf += float(col[3])
                               if col[1] == col[2]:
                                   A_acc += 1
                           elif "Confidence before liking" in col[7]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1

                       elif filename == 'data_Massoni_2017.csv':
                           if "1" in col[8]:
                               A_trials += 1
                               A_conf += float(col[3])
                               if col[1] == col[2]:
                                   A_acc += 1
                           elif "0" in col[8]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1
                       elif filename == 'data_Massoni_unpub.csv':
                           if "1" in col[8]:
                               A_trials += 1
                               A_conf += float(col[3])
                               if col[1] == col[2]:
                                   A_acc += 1
                           elif "0" in col[8]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1
                       elif filename == 'data_Paulewicz_unpub2.csv':
                           if not "NA" in col[2]:
                               if "DS" in col[7]:
                                   A_trials += 1
                                   A_conf += float(col[3])
                                   if col[1] == col[2]:
                                       A_acc += 1
                               elif "SD" in col[7]:
                                   B_trials += 1
                                   B_conf += float(col[3])
                                   if col[1] == col[2]:
                                       B_acc += 1
                       elif filename == 'data_Paulewicz_unpub3.csv':
                           if not "NA" in col[2]:
                               if not "NA" in col[3]:
                                   if "DS" in col[7]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A_acc += 1
                                   elif "SD" in col[7]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == col[2]:
                                           B_acc += 1
                       elif filename == 'data_Reyes_unpub.csv':
                           if "P" in col[7]:
                               A_trials += 1
                               A_conf += float(col[3])
                               if col[1] == col[2]:
                                   A_acc += 1
                           elif "H" in col[7]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1
                       elif filename == 'data_Sadeghi_2017_memory.csv':
                           if "patient" in col[6]:
                               A_trials += 1
                               A_conf += float(col[3])
                               if col[1] == col[2]:
                                   A_acc += 1
                           elif "control" in col[6]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1
                       elif filename == 'data_Sadeghi_2017_perception.csv':
                           if "patient" in col[7]:
                               A_trials += 1
                               A_conf += float(col[3])
                               if col[1] == col[2]:
                                   A_acc += 1
                           elif "control" in col[7]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1
                       elif filename == 'data_Schmidt_2019_memory.csv':
                           if "MM" in col[7]:
                               A_trials += 1
                               A_conf += float(col[3])
                               if col[1] == col[2]:
                                   A_acc += 1
                           elif "SoB" in col[7]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1
                       elif filename == 'data_Schmidt_2019_perception.csv':
                           if "MM" in col[8]:
                               A_trials += 1
                               A_conf += float(col[3])
                               if col[1] == col[2]:
                                   A_acc += 1
                           elif "SoB" in col[8]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1
                       elif filename == 'data_Siedlecka_2019_Exp1.csv':
                           if not "NaN" in col[2]:
                               if not "NaN" in col[3]:
                                   if "0" in col[7]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A_acc += 1
                                   elif "1" in col[7]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == col[2]:
                                           B_acc += 1
                       elif filename == 'data_Siedlecka_2019_Exp2.csv':
                           if not "NaN" in col[2]:
                               if not "NaN" in col[3]:
                                   if "0" in col[7]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A_acc += 1
                                   elif "1" in col[7]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == col[2]:
                                           B_acc += 1
                       elif filename == 'data_Rausch_2016.csv':
                           if not "1" in col[8]:
                               if "Visibility" in col[9]:
                                   A_trials += 1
                                   A_conf += float(col[3])
                                   if col[1] == col[2]:
                                       A_acc += 1
                               elif "Confidence" in col[9]:
                                   B_trials += 1
                                   B_conf += float(col[3])
                                   if col[1] == col[2]:
                                       B_acc += 1
                       elif filename == 'data_Xu_2019_Exp1.csv':
                           if not "NA" in col[2]:
                               if not "NA" in col[3]:
                                   if "N" in col[8]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if float(col[1]) == float(col[2]):
                                           A_acc += 1
                                   elif "C" in col[8]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if float(col[1]) == float(col[2]):
                                           B_acc += 1
                       elif filename == 'data_Yeon_2019.csv':
                           if "1" in col[6]:
                               A_trials += 1
                               A_conf += float(col[3])
                               if col[1] == col[2]:
                                   A_acc += 1
                           elif "2" in col[6]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1
                       elif filename == 'data_Wang_2017_NatComm.csv':
                           if not "NaN" in col[2]:
                               if not "NaN" in col[3]:
                                   if "Epilepsy" in col[6]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == '1' or col[1] == '2' or col[1] == '3':
                                           if col[2] == '1':
                                               A_acc += 1
                                       elif col[1] == '5' or col[1] == '6' or col[1] == '7':
                                           if col[2] == '2':
                                               A_acc += 1
                                   elif "Lesion" in col[6]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == '1' or col[1] == '2' or col[1] == '3':
                                           if col[2] == '1':
                                               B_acc += 1
                                       elif col[1] == '5' or col[1] == '6' or col[1] == '7':
                                           if col[2] == '2':
                                               B_acc += 1
                       elif filename == 'data_Wang_2017_Neuropsychologia.csv':
                           if not "NaN" in col[2]:
                               if not "NaN" in col[3]:
                                   if "ASD_Ctrl" in col[6]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == '1' or col[1] == '2' or col[1] == '3':
                                           if col[2] == '1':
                                               A_acc += 1
                                       elif col[1] == '5' or col[1] == '6' or col[1] == '7':
                                           if col[2] == '2':
                                               A_acc += 1
                                   elif "ASD" in col[6]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == '1' or col[1] == '2' or col[1] == '3':
                                           if col[2] == '1':
                                               B_acc += 1
                                       elif col[1] == '5' or col[1] == '6' or col[1] == '7':
                                           if col[2] == '2':
                                               B_acc += 1
                       elif filename == 'data_Fallow_unpub_2.csv':
                            if not "NaN" in col[2]:
                                if not "NaN" in col[3]:
                                    if not "NaN" in col[4]:
                                        if not "NaN" in col[5]:
                                            if "face" in col[1]:
                                                A_trials += 1
                                                A_conf += int(float(col[5]))
                                                if col[3] == col[4]:
                                                    A_acc += 1
                                            elif "paint" in col[1]:
                                                B_trials += 1
                                                B_conf += int(float(col[5]))
                                                if col[3] == col[4]:
                                                    B_acc += 1

                if trials != 0:
                    if A_trials != 0:
                        mean_A_acc = round(A_acc / A_trials, 5)
                        mean_A_conf = round(A_conf / A_trials, 5)
                        mAccA_values = np.append(mAccA_values, mean_A_acc)
                        mConfA_values = np.append(mConfA_values, mean_A_conf)
                        arr1 = np.append(arr1, np.array([[num + 1, float(A_acc / A_trials), float(A_conf / A_trials), A_trials]]), axis=0)
                    if B_trials != 0:
                        mean_B_acc = round(B_acc / B_trials, 5)
                        mean_B_conf = round(B_conf / B_trials, 5)
                        mAccB_values = np.append(mAccB_values, mean_B_acc)
                        mConfB_values = np.append(mConfB_values, mean_B_conf)
                        arr2 = np.append(arr2, np.array([[num + 1, float(B_acc / B_trials), float(B_conf / B_trials), B_trials]]), axis=0)
                if trials != 0:
                    writefile.write("Subj: {}, Total trials: {}, Task 1 trials: {}, Task 2 trials: {},\n" .format(num + 1, trials, A_trials, B_trials))
                    writefile.write("Mean task 1 acc: {}, Mean task 1 conf: {} \n" .format(mean_A_acc, mean_A_conf))
                    writefile.write("Mean task 2 acc: {}, Mean task 2 conf: {} \n" .format(mean_B_acc, mean_B_conf))
                    writefile.write("Total 1 acc: {}, Total 1 conf: {} \n" .format(A_acc, A_conf))
                    writefile.write("Total 2 acc: {}, Total 2 conf: {} \n" .format(B_acc, B_conf))
            correlationA = np.corrcoef(mAccA_values, mConfA_values)
            correlationB = np.corrcoef(mAccB_values, mConfB_values)
            mdic1 = {'data': arr1}
            mdic2 = {'data': arr2}
            print(mdic1)
            print(mdic2)
            savemat(filename + "_C1" + ".mat", mdic1)
            savemat(filename + "_C2" + ".mat", mdic2)


        #3 conditions
        elif filename == 'data_Haddara_unpub.csv' or filename == 'data_Wierzchon_2019.csv' or filename == 'data_Fallow_unpub_1.csv' or filename == 'data_Lindsay_2014.csv' or filename == 'data_Siedlecka_2016.csv' or filename == 'data_Siedlecka_2018_bioRxiv.csv' or filename == 'data_Matthews_unpub.csv':
            mdic1 = {}
            mdic2 = {}
            mdic3 = {}
            arr1 = np.empty((0, 4), float)
            arr2 = np.empty((0, 4), float)
            arr3 = np.empty((0, 4), float)
            num_tasks = 3
            for num in range(subjects):
                A_acc = 0
                A_conf = 0
                B_acc = 0
                B_conf = 0
                C_acc = 0
                C_conf = 0
                A_trials = 0
                B_trials = 0
                C_trials = 0
                mean_A_acc = 0
                mean_B_acc = 0
                mean_C_acc = 0
                mean_A_conf = 0
                mean_B_conf = 0
                mean_C_conf = 0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if not "NaN" in col[2]:
                           if not "NaN" in col[3]:
                               if filename == 'data_Lindsay_2014.csv':
                                   if "Words" in col[5]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A_acc += 1
                                   elif "Paintings" in col[5]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == col[2]:
                                           B_acc += 1
                                   elif "Mixed" in col[5]:
                                       C_trials += 1
                                       C_conf += float(col[3])
                                       if col[1] == col[2]:
                                           C_acc += 1
                               elif filename == 'data_Siedlecka_2016.csv':
                                   if "1" in col[6]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A_acc += 1
                                   elif "2" in col[6]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == col[2]:
                                           B_acc += 1
                                   elif "3" in col[6]:
                                       C_trials += 1
                                       C_conf += float(col[3])
                                       if col[1] == col[2]:
                                           C_acc += 1
                               elif filename == 'data_Siedlecka_2018_bioRxiv.csv':
                                   if "1" in col[7]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A_acc += 1
                                   elif "2" in col[7]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == col[2]:
                                           B_acc += 1
                                   elif "3" in col[7]:
                                       C_trials += 1
                                       C_conf += float(col[3])
                                       if col[1] == col[2]:
                                           C_acc += 1
                               elif filename == 'data_Fallow_unpub_1.csv':
                                   if "face" in col[1]:
                                       A_trials += 1
                                       A_conf += float(col[5])
                                       if col[3] == col[4]:
                                           A_acc += 1
                                   elif "paint" in col[1]:
                                       B_trials += 1
                                       B_conf += float(col[5])
                                       if col[3] == col[4]:
                                           B_acc += 1
                                   elif "word" in col[1]:
                                       C_trials += 1
                                       C_conf += float(col[5])
                                       if col[3] == col[4]:
                                           C_acc += 1
                               elif filename == 'data_Haddara_unpub.csv':
                                   if "1" in col[6]:
                                       if "0" in col[7]:
                                           A_trials += 1
                                           A_conf += float(col[3])
                                           if col[1] == col[2]:
                                               A_acc += 1
                                       elif "1" in col[7]:
                                           B_trials += 1
                                           B_conf += float(col[3])
                                           if col[1] == col[2]:
                                               B_acc += 1
                                   elif "2" in col[6]:
                                       C_trials += 1
                                       C_conf += float(col[3])
                                       if col[1] == col[2]:
                                           C_acc += 1
                               elif filename == 'data_Wierzchon_2019.csv':
                                   if "cpas" in col[7]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A_acc += 1
                                   elif "cs" in col[7]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == col[2]:
                                           B_acc += 1
                                   elif "pas" in col[7]:
                                       C_trials += 1
                                       C_conf += float(col[3])
                                       if col[1] == col[2]:
                                           C_acc += 1
                               elif filename == 'data_Matthews_unpub.csv':
                                   if "control" in col[15]:
                                       if "dual_tasks" in col[16]:
                                           A_trials += 2
                                           A_conf += float(col[3])
                                           A_conf += float(col[9])
                                           A_acc += float(col[6])
                                           A_acc += float(col[11])
                                       elif "peripheral_tasks" in col[16]:
                                           A_trials += 1
                                           A_conf += float(col[3])
                                           A_acc += float(col[6])
                                       elif "central_tasks" in col[16]:
                                           A_trials += 1
                                           A_conf += float(col[9])
                                           A_acc += float(col[11])
                                   elif "functional_movement_disorder" in col[15]:
                                       if "dual_tasks" in col[16]:
                                           B_trials += 2
                                           B_conf += float(col[3])
                                           B_conf += float(col[9])
                                           B_acc += float(col[6])
                                           B_acc += float(col[11])
                                       elif "peripheral_tasks" in col[16]:
                                           B_trials += 1
                                           B_conf += float(col[3])
                                           B_acc += float(col[6])
                                       elif "central_tasks" in col[16]:
                                           B_trials += 1
                                           B_conf += float(col[9])
                                           B_acc += float(col[11])
                                   elif "organic_movement_disorder" in col[15]:
                                       if "dual_tasks" in col[16]:
                                           C_trials += 2
                                           C_conf += float(col[3])
                                           C_conf += float(col[9])
                                           C_acc += float(col[6])
                                           C_acc += float(col[11])
                                       elif "peripheral_tasks" in col[16]:
                                           C_trials += 1
                                           C_conf += float(col[3])
                                           C_acc += float(col[6])
                                       elif "central_tasks" in col[16]:
                                           C_trials += 1
                                           C_conf += float(col[9])
                                           C_acc += float(col[11])
                if A_trials != 0:
                    mean_A_acc = round(A_acc / A_trials, 5)
                    mean_A_conf = round(A_conf / A_trials, 5)
                    mAccA_values = np.append(mAccA_values, mean_A_acc)
                    mConfA_values = np.append(mConfA_values, mean_A_conf)
                    arr1 = np.append(arr1, np.array([[num + 1, float(A_acc / A_trials), float(A_conf / A_trials), A_trials]]), axis=0)
                if B_trials != 0:
                    mean_B_acc = round(B_acc / B_trials, 5)
                    mean_B_conf = round(B_conf / B_trials, 5)
                    mAccB_values = np.append(mAccB_values, mean_B_acc)
                    mConfB_values = np.append(mConfB_values, mean_B_conf)
                    arr2 = np.append(arr2, np.array([[num + 1, float(B_acc / B_trials), float(B_conf / B_trials), B_trials]]), axis=0)
                if C_trials != 0:
                    mean_C_acc = round(C_acc / C_trials, 5)
                    mean_C_conf = round(C_conf / C_trials, 5)
                    mAccC_values = np.append(mAccC_values, mean_C_acc)
                    mConfC_values = np.append(mConfC_values, mean_C_conf)
                    arr3 = np.append(arr3, np.array([[num + 1, float(C_acc / C_trials), float(C_conf / C_trials), C_trials]]), axis=0)
                writefile.write("Subj: {}, Task 1 trials: {}, Task 2 trials: {}, Task 3 trials: {},\n" .format(num + 1, A_trials, B_trials, C_trials))
                writefile.write("Mean task 1 acc: {}, Mean task 1 conf: {} \n" .format(mean_A_acc, mean_A_conf))
                writefile.write("Mean task 2 acc: {}, Mean task 2 conf: {} \n" .format(mean_B_acc, mean_B_conf))
                writefile.write("Mean task 3 acc: {}, Mean task 3 conf: {} \n" .format(mean_C_acc, mean_C_conf))
                writefile.write("Total 1 acc: {}, Total 1 conf: {} \n" .format(A_acc, A_conf))
                writefile.write("Total 2 acc: {}, Total 2 conf: {} \n" .format(B_acc, B_conf))
                writefile.write("Total 3 acc: {}, Total 3 conf: {} \n" .format(C_acc, C_conf))
            correlationA = np.corrcoef(mAccA_values, mConfA_values)
            correlationB = np.corrcoef(mAccB_values, mConfB_values)
            correlationC = np.corrcoef(mAccC_values, mConfC_values)
            mdic1 = {'data': arr1}
            mdic2 = {'data': arr2}
            mdic3 = {'data': arr3}
            print(mdic1)
            print(mdic2)
            print(mdic3)
            savemat(filename + "_C1" + ".mat", mdic1)
            savemat(filename + "_C2" + ".mat", mdic2)
            savemat(filename + "_C3" + ".mat", mdic3)

                #4 conditions
        elif filename == 'data_Kantner_2012_E3.csv' or filename == 'data_Konishi_2019.csv' or filename == 'data_OHora_unpub_1.csv':
            mdic1 = {}
            mdic2 = {}
            mdic3 = {}
            mdic4 = {}
            arr1 = np.empty((0, 4), float)
            arr2 = np.empty((0, 4), float)
            arr3 = np.empty((0, 4), float)
            arr4 = np.empty((0, 4), float)
            num_tasks = 4
            for num in range(subjects):
                A_acc = 0
                A_conf = 0
                B_acc = 0
                B_conf = 0
                C_acc = 0
                C_conf = 0
                D_acc = 0
                D_conf = 0
                A_trials = 0
                B_trials = 0
                C_trials = 0
                D_trials = 0
                mean_A_acc = 0
                mean_B_acc = 0
                mean_C_acc = 0
                mean_D_acc = 0
                mean_A_conf = 0
                mean_B_conf = 0
                mean_C_conf = 0
                mean_D_conf = 0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if not "NaN" in col[2]:
                           if not "NaN" in col[3]:
                               if filename == 'data_Kantner_2012_E3.csv':
                                   if "Paintings-Paintings" in col[5]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A_acc += 1
                                   elif "Paintings-Words" in col[5]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == col[2]:
                                           B_acc += 1
                                   elif "Words-Paintings" in col[5]:
                                       C_trials += 1
                                       C_conf += float(col[3])
                                       if col[1] == col[2]:
                                           C_acc += 1
                                   elif "Words-Words" in col[5]:
                                       D_trials += 1
                                       D_conf += float(col[3])
                                       if col[1] == col[2]:
                                           D_acc += 1
                               elif filename == 'data_OHora_unpub_1.csv':
                                   if not "TRUE" in col[12]:
                                       if not "TIMEOUT" in col[2]:
                                           if not "Practice" in col[10]:
                                               if "Expt" in col[9]:
                                                   if "TRUE" in col[11]:
                                                       A_trials += 1
                                                       A_conf += float(col[3])
                                                       A_acc += float(col[7])
                                                   else:
                                                       B_trials += 1
                                                       B_conf += float(col[3])
                                                       B_acc += float(col[7])
                                               elif "Control" in col[9]:
                                                   if "TRUE" in col[11]:
                                                       C_trials += 1
                                                       C_conf += float(col[3])
                                                       C_acc += float(col[7])
                                                   else:
                                                       D_trials += 1
                                                       D_conf += float(col[3])
                                                       D_acc += float(col[7])

                               elif filename == 'data_Konishi_2019.csv':
                                   if "long/staircase" in col[17]:
                                       if "dual" in col[1]:
                                           A_trials += 2
                                           A_conf += float(col[14])
                                           A_conf += float(col[15])
                                           A_acc += float(col[8])
                                           A_acc += float(col[9])
                                       elif "motion" in col[1]:
                                           A_trials += 1
                                           A_conf += float(col[14])
                                           A_acc += float(col[8])
                                       elif "color" in col[1]:
                                           A_trials += 1
                                           A_conf += float(col[15])
                                           A_acc += float(col[9])
                                   elif "short/staircase 2" in col[17]:
                                       if "dual" in col[1]:
                                           B_trials += 2
                                           B_conf += float(col[14])
                                           B_conf += float(col[15])
                                           B_acc += float(col[8])
                                           B_acc += float(col[9])
                                       elif "motion" in col[1]:
                                           B_trials += 1
                                           B_conf += float(col[14])
                                           B_acc += float(col[8])
                                       elif "color" in col[1]:
                                           B_trials += 1
                                           B_conf += float(col[15])
                                           B_acc += float(col[9])
                                   elif "short/staircase" in col[17]:
                                       if "dual" in col[1]:
                                           C_trials += 2
                                           C_conf += float(col[14])
                                           C_conf += float(col[15])
                                           C_acc += float(col[8])
                                           C_acc += float(col[9])
                                       elif "motion" in col[1]:
                                           C_trials += 1
                                           C_conf += float(col[14])
                                           C_acc += float(col[8])
                                       elif "color" in col[1]:
                                           C_trials += 1
                                           C_conf += float(col[15])
                                           C_acc += float(col[9])
                                   elif "short/constant" in col[17]:
                                       if "dual" in col[1]:
                                           D_trials += 2
                                           D_conf += float(col[14])
                                           D_conf += float(col[15])
                                           D_acc += float(col[8])
                                           D_acc += float(col[9])
                                       elif "motion" in col[1]:
                                           D_trials += 1
                                           D_conf += float(col[14])
                                           D_acc += float(col[8])
                                       elif "color" in col[1]:
                                           D_trials += 1
                                           D_conf += float(col[15])
                                           D_acc += float(col[9])
                if A_trials != 0:
                    mean_A_acc = round(A_acc / A_trials, 5)
                    mean_A_conf = round(A_conf / A_trials, 5)
                    mAccA_values = np.append(mAccA_values, mean_A_acc)
                    mConfA_values = np.append(mConfA_values, mean_A_conf)
                    arr1 = np.append(arr1, np.array([[num + 1, float(A_acc / A_trials), float(A_conf / A_trials), A_trials]]), axis=0)
                if B_trials != 0:
                    mean_B_acc = round(B_acc / B_trials, 5)
                    mean_B_conf = round(B_conf / B_trials, 5)
                    mAccB_values = np.append(mAccB_values, mean_B_acc)
                    mConfB_values = np.append(mConfB_values, mean_B_conf)
                    arr2 = np.append(arr2, np.array([[num + 1, float(B_acc / B_trials), float(B_conf / B_trials), B_trials]]), axis=0)
                if C_trials != 0:
                    mean_C_acc = round(C_acc / C_trials, 5)
                    mean_C_conf = round(C_conf / C_trials, 5)
                    mAccC_values = np.append(mAccC_values, mean_C_acc)
                    mConfC_values = np.append(mConfC_values, mean_C_conf)
                    arr3 = np.append(arr3, np.array([[num + 1, float(C_acc / C_trials), float(C_conf / C_trials), C_trials]]), axis=0)
                if D_trials != 0:
                    mean_D_acc = round(D_acc / D_trials, 5)
                    mean_D_conf = round(D_conf / D_trials, 5)
                    mAccD_values = np.append(mAccD_values, mean_D_acc)
                    mConfD_values = np.append(mConfD_values, mean_D_conf)
                    arr4 = np.append(arr4, np.array([[num + 1, float(D_acc / D_trials), float(D_conf / D_trials), D_trials]]), axis=0)
                writefile.write("Subj: {}, Task 1 trials: {}, Task 2 trials: {}, Task 3 trials: {}, Task 4 trials: {} \n" .format(num + 1, A_trials, B_trials, C_trials, D_trials))
                writefile.write("Mean task 1 acc: {}, Mean task 1 conf: {} \n" .format(mean_A_acc, mean_A_conf))
                writefile.write("Mean task 2 acc: {}, Mean task 2 conf: {} \n" .format(mean_B_acc, mean_B_conf))
                writefile.write("Mean task 3 acc: {}, Mean task 3 conf: {} \n" .format(mean_C_acc, mean_C_conf))
                writefile.write("Mean task 4 acc: {}, Mean task 4 conf: {} \n" .format(mean_D_acc, mean_D_conf))
                writefile.write("Total 1 acc: {}, Total 1 conf: {} \n" .format(A_acc, A_conf))
                writefile.write("Total 2 acc: {}, Total 2 conf: {} \n" .format(B_acc, B_conf))
                writefile.write("Total 3 acc: {}, Total 3 conf: {} \n" .format(C_acc, C_conf))
                writefile.write("Total 4 acc: {}, Total 4 conf: {} \n" .format(D_acc, D_conf))
            correlationA = np.corrcoef(mAccA_values, mConfA_values)
            correlationB = np.corrcoef(mAccB_values, mConfB_values)
            correlationC = np.corrcoef(mAccC_values, mConfC_values)
            correlationD = np.corrcoef(mAccD_values, mConfD_values)
            mdic1 = {'data': arr1}
            mdic2 = {'data': arr2}
            mdic3 = {'data': arr3}
            mdic4 = {'data': arr4}
            print(mdic1)
            print(mdic2)
            print(mdic3)
            print(mdic4)
            savemat(filename + "_C1" + ".mat", mdic1)
            savemat(filename + "_C2" + ".mat", mdic2)
            savemat(filename + "_C3" + ".mat", mdic3)
            savemat(filename + "_C4" + ".mat", mdic4)

        #6 conditions
        elif filename == 'data_Kantner_unpub2.csv' or filename == 'data_Matthews_2019.csv' or filename == 'data_Kantner_2010.csv':
            mdic1 = {}
            mdic2 = {}
            mdic3 = {}
            mdic4 = {}
            mdic5 = {}
            mdic6 = {}
            arr1 = np.empty((0, 4), float)
            arr2 = np.empty((0, 4), float)
            arr3 = np.empty((0, 4), float)
            arr4 = np.empty((0, 4), float)
            arr5 = np.empty((0, 4), float)
            arr6 = np.empty((0, 4), float)
            num_tasks = 6
            for num in range(subjects):
                A_acc = 0
                A_conf = 0
                B_acc = 0
                B_conf = 0
                C_acc = 0
                C_conf = 0
                D_acc = 0
                D_conf = 0
                A5_acc = 0
                A5_conf = 0
                A6_acc = 0
                A6_conf = 0
                A_trials = 0
                B_trials = 0
                C_trials = 0
                D_trials = 0
                A5_trials = 0
                A6_trials = 0
                mean_A_acc = 0
                mean_B_acc = 0
                mean_C_acc = 0
                mean_D_acc = 0
                mean_5_acc = 0
                mean_6_acc = 0
                mean_A_conf = 0
                mean_B_conf = 0
                mean_C_conf = 0
                mean_D_conf = 0
                mean_5_conf = 0
                mean_6_conf = 0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                        if filename == 'data_Kantner_2010.csv' or filename == 'data_Kantner_unpub2.csv':
                           if "1" in col[5]:
                               A_trials += 1
                               A_conf += float(col[3])
                               A_acc += float(col[6])
                           elif "2" in col[5]:
                               B_trials += 1
                               B_conf += float(col[3])
                               if col[1] == col[2]:
                                   B_acc += 1
                           elif "3" in col[5]:
                               C_trials += 1
                               C_conf += float(col[3])
                               if col[1] == col[2]:
                                   C_acc += 1
                           elif "4" in col[5]:
                               D_trials += 1
                               D_conf += float(col[3])
                               if col[1] == col[2]:
                                   D_acc += 1
                           elif "5" in col[5]:
                               A5_trials += 1
                               A5_conf += float(col[3])
                               if col[1] == col[2]:
                                   A5_acc += 1
                           elif "6" in col[5]:
                               A6_trials += 1
                               A6_conf += float(col[3])
                               if col[1] == col[2]:
                                   A6_acc += 1
                        elif filename == 'data_Matthews_2019.csv':
                           if not "NaN" in col[2]:
                               if not "NaN" in col[3]:
                                   if "across_scene" in col[8]:
                                       if "inverted" in col[9]:
                                           if "no_target_search" in col[10]:
                                               A_trials += 1
                                               A_conf += float(col[3])
                                               if col[1] == col[2]:
                                                   A_acc += 1
                                           elif "target_search" in col[10]:
                                               B_trials += 1
                                               B_conf += float(col[3])
                                               if col[1] == col[2]:
                                                   B_acc += 1
                                       elif "upright" in col[9]:
                                           if "no_target_search" in col[10]:
                                               C_trials += 1
                                               C_conf += float(col[3])
                                               if col[1] == col[2]:
                                                   C_acc += 1
                                           elif "target_search" in col[10]:
                                               D_trials += 1
                                               D_conf += float(col[3])
                                               if col[1] == col[2]:
                                                   D_acc += 1
                                   elif "within_scene" in col[8]:
                                       if "inverted" in col[9]:
                                           if "target_search" in col[10]:
                                               A5_trials += 1
                                               A5_conf += float(col[3])
                                               if col[1] == col[2]:
                                                   A5_acc += 1
                                       elif "upright" in col[9]:
                                           if "target_search" in col[10]:
                                               A6_trials += 1
                                               A6_conf += float(col[3])
                                               if col[1] == col[2]:
                                                   A6_acc += 1

                if A_trials != 0:
                    mean_A_acc = round(A_acc / A_trials, 5)
                    mean_A_conf = round(A_conf / A_trials, 5)
                    mAccA_values = np.append(mAccA_values, mean_A_acc)
                    mConfA_values = np.append(mConfA_values, mean_A_conf)
                    arr1 = np.append(arr1, np.array([[num + 1, float(A_acc / A_trials), float(A_conf / A_trials), A_trials]]), axis=0)
                if B_trials != 0:
                    mean_B_acc = round(B_acc / B_trials, 5)
                    mean_B_conf = round(B_conf / B_trials, 5)
                    mAccB_values = np.append(mAccB_values, mean_B_acc)
                    mConfB_values = np.append(mConfB_values, mean_B_conf)
                    arr2 = np.append(arr2, np.array([[num + 1, float(B_acc / B_trials), float(B_conf / B_trials), B_trials]]), axis=0)
                if C_trials != 0:
                    mean_C_acc = round(C_acc / C_trials, 5)
                    mean_C_conf = round(C_conf / C_trials, 5)
                    mAccC_values = np.append(mAccC_values, mean_C_acc)
                    mConfC_values = np.append(mConfC_values, mean_C_conf)
                    arr3 = np.append(arr3, np.array([[num + 1, float(C_acc / C_trials), float(C_conf / C_trials), C_trials]]), axis=0)
                if D_trials != 0:
                    mean_D_acc = round(D_acc / D_trials, 5)
                    mean_D_conf = round(D_conf / D_trials, 5)
                    mAccD_values = np.append(mAccD_values, mean_D_acc)
                    mConfD_values = np.append(mConfD_values, mean_D_conf)
                    arr4 = np.append(arr4, np.array([[num + 1, float(D_acc / D_trials), float(D_conf / D_trials), D_trials]]), axis=0)
                if A5_trials != 0:
                    mean_5_acc = round(A5_acc / A5_trials, 5)
                    mean_5_conf = round(A5_conf / A5_trials, 5)
                    mAcc5_values = np.append(mAcc5_values, mean_5_acc)
                    mConf5_values = np.append(mConf5_values, mean_5_conf)
                    arr5 = np.append(arr5, np.array([[num + 1, float(A5_acc / A5_trials), float(A5_conf / A5_trials), A5_trials]]), axis=0)
                if A6_trials != 0:
                    mean_6_acc = round(A6_acc / A6_trials, 5)
                    mean_6_conf = round(A6_conf / A6_trials, 5)
                    mAcc6_values = np.append(mAcc6_values, mean_6_acc)
                    mConf6_values = np.append(mConf6_values, mean_6_conf)
                    arr6 = np.append(arr6, np.array([[num + 1, float(A6_acc / A6_trials), float(A6_conf / A6_trials), A6_trials]]), axis=0)
                writefile.write("Subj: {}, Task 1 trials: {}, Task 2 trials: {}, Task 3 trials: {}, Task 4 trials: {} \n" .format(num + 1, A_trials, B_trials, C_trials, D_trials))
                writefile.write("Subj: {}, Task 5 trials: {}, Task 6 trials: {} \n" .format(num + 1, A5_trials, A6_trials))
                writefile.write("Mean task 1 acc: {}, Mean task 1 conf: {} \n" .format(mean_A_acc, mean_A_conf))
                writefile.write("Mean task 2 acc: {}, Mean task 2 conf: {} \n" .format(mean_B_acc, mean_B_conf))
                writefile.write("Mean task 3 acc: {}, Mean task 3 conf: {} \n" .format(mean_C_acc, mean_C_conf))
                writefile.write("Mean task 4 acc: {}, Mean task 4 conf: {} \n" .format(mean_D_acc, mean_D_conf))
                writefile.write("Mean task 5 acc: {}, Mean task 5 conf: {} \n" .format(mean_5_acc, mean_5_conf))
                writefile.write("Mean task 6 acc: {}, Mean task 6 conf: {} \n" .format(mean_6_acc, mean_6_conf))
                writefile.write("Total 1 acc: {}, Total 1 conf: {} \n" .format(A_acc, A_conf))
                writefile.write("Total 2 acc: {}, Total 2 conf: {} \n" .format(B_acc, B_conf))
                writefile.write("Total 3 acc: {}, Total 3 conf: {} \n" .format(C_acc, C_conf))
                writefile.write("Total 4 acc: {}, Total 4 conf: {} \n" .format(D_acc, D_conf))
                writefile.write("Total 5 acc: {}, Total 5 conf: {} \n" .format(A5_acc, A5_conf))
                writefile.write("Total 6 acc: {}, Total 6 conf: {} \n" .format(A6_acc, A6_conf))
            correlationA = np.corrcoef(mAccA_values, mConfA_values)
            correlationB = np.corrcoef(mAccB_values, mConfB_values)
            correlationC = np.corrcoef(mAccC_values, mConfC_values)
            correlationD = np.corrcoef(mAccD_values, mConfD_values)
            correlation5 = np.corrcoef(mAcc5_values, mConf5_values)
            correlation6 = np.corrcoef(mAcc6_values, mConf6_values)
            mdic1 = {'data': arr1}
            mdic2 = {'data': arr2}
            mdic3 = {'data': arr3}
            mdic4 = {'data': arr4}
            mdic5 = {'data': arr5}
            mdic6 = {'data': arr6}
            savemat(filename + "_C1" + ".mat", mdic1)
            savemat(filename + "_C2" + ".mat", mdic2)
            savemat(filename + "_C3" + ".mat", mdic3)
            savemat(filename + "_C4" + ".mat", mdic4)
            savemat(filename + "_C5" + ".mat", mdic5)
            savemat(filename + "_C6" + ".mat", mdic6)

                #8 conditions
        elif filename == 'data_Wierzchon_2014.csv':
            mdic1 = {}
            mdic2 = {}
            mdic3 = {}
            mdic4 = {}
            mdic5 = {}
            mdic6 = {}
            mdic7 = {}
            mdic8 = {}
            arr1 = np.empty((0, 4), float)
            arr2 = np.empty((0, 4), float)
            arr3 = np.empty((0, 4), float)
            arr4 = np.empty((0, 4), float)
            arr5 = np.empty((0, 4), float)
            arr6 = np.empty((0, 4), float)
            arr7 = np.empty((0, 4), float)
            arr8 = np.empty((0, 4), float)
            num_tasks = 8
            for num in range(subjects):
                A_acc = 0
                A_conf = 0
                B_acc = 0
                B_conf = 0
                C_acc = 0
                C_conf = 0
                D_acc = 0
                D_conf = 0
                A5_acc = 0
                A5_conf = 0
                A6_acc = 0
                A6_conf = 0
                A7_acc = 0
                A7_conf = 0
                A8_acc = 0
                A8_conf = 0
                A_trials = 0
                B_trials = 0
                C_trials = 0
                D_trials = 0
                A5_trials = 0
                A6_trials = 0
                A7_trials = 0
                A8_trials = 0
                mean_A_acc = 0
                mean_B_acc = 0
                mean_C_acc = 0
                mean_D_acc = 0
                mean_5_acc = 0
                mean_6_acc = 0
                mean_7_acc = 0
                mean_8_acc = 0
                mean_A_conf = 0
                mean_B_conf = 0
                mean_C_conf = 0
                mean_D_conf = 0
                mean_5_conf = 0
                mean_6_conf = 0
                mean_7_conf = 0
                mean_8_conf = 0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if not "NaN" in col[2]:
                           if not "NaN" in col[3]:
                               if "DS" in col[7]:
                                   if "PAS" in col[8]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A_acc += 1
                                   elif "P" in col[8]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == col[2]:
                                           B_acc += 1
                                   elif "O" in col[8]:
                                       C_trials += 1
                                       C_conf += float(col[3])
                                       if col[1] == col[2]:
                                           C_acc += 1
                                   elif "C" in col[8]:
                                       D_trials += 1
                                       D_conf += float(col[3])
                                       if col[1] == col[2]:
                                           D_acc += 1
                               elif "SD" in col[7]:
                                   if "PAS" in col[8]:
                                       A5_trials += 1
                                       A5_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A5_acc += 1
                                   elif "P" in col[8]:
                                       A6_trials += 1
                                       A6_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A6_acc += 1
                                   elif "O" in col[8]:
                                       A7_trials += 1
                                       A7_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A7_acc += 1
                                   elif "C" in col[8]:
                                       A8_trials += 1
                                       A8_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A8_acc += 1
                if A_trials != 0:
                    mean_A_acc = round(A_acc / A_trials, 5)
                    mean_A_conf = round(A_conf / A_trials, 5)
                    mAccA_values = np.append(mAccA_values, mean_A_acc)
                    mConfA_values = np.append(mConfA_values, mean_A_conf)
                    arr1 = np.append(arr1, np.array([[num + 1, float(A_acc / A_trials), float(A_conf / A_trials), A_trials]]), axis=0)
                if B_trials != 0:
                    mean_B_acc = round(B_acc / B_trials, 5)
                    mean_B_conf = round(B_conf / B_trials, 5)
                    mAccB_values = np.append(mAccB_values, mean_B_acc)
                    mConfB_values = np.append(mConfB_values, mean_B_conf)
                    arr2 = np.append(arr2, np.array([[num + 1, float(B_acc / B_trials), float(B_conf / B_trials), B_trials]]), axis=0)
                if C_trials != 0:
                    mean_C_acc = round(C_acc / C_trials, 5)
                    mean_C_conf = round(C_conf / C_trials, 5)
                    mAccC_values = np.append(mAccC_values, mean_C_acc)
                    mConfC_values = np.append(mConfC_values, mean_C_conf)
                    arr3 = np.append(arr3, np.array([[num + 1, float(C_acc / C_trials), float(C_conf / C_trials), C_trials]]), axis=0)
                if D_trials != 0:
                    mean_D_acc = round(D_acc / D_trials, 5)
                    mean_D_conf = round(D_conf / D_trials, 5)
                    mAccD_values = np.append(mAccD_values, mean_D_acc)
                    mConfD_values = np.append(mConfD_values, mean_D_conf)
                    arr4 = np.append(arr4, np.array([[num + 1, float(D_acc / D_trials), float(D_conf / D_trials), D_trials]]), axis=0)
                if A5_trials != 0:
                    mean_5_acc = round(A5_acc / A5_trials, 5)
                    mean_5_conf = round(A5_conf / A5_trials, 5)
                    mAcc5_values = np.append(mAcc5_values, mean_5_acc)
                    mConf5_values = np.append(mConf5_values, mean_5_conf)
                    arr5 = np.append(arr5, np.array([[num + 1, float(A5_acc / A5_trials), float(A5_conf / A5_trials), A5_trials]]), axis=0)
                if A6_trials != 0:
                    mean_6_acc = round(A6_acc / A6_trials, 5)
                    mean_6_conf = round(A6_conf / A6_trials, 5)
                    mAcc6_values = np.append(mAcc6_values, mean_6_acc)
                    mConf6_values = np.append(mConf6_values, mean_6_conf)
                    arr6 = np.append(arr6, np.array([[num + 1, float(A6_acc / A6_trials), float(A6_conf / A6_trials), A6_trials]]), axis=0)
                if A7_trials != 0:
                    mean_7_acc = round(A7_acc / A7_trials, 5)
                    mean_7_conf = round(A7_conf / A7_trials, 5)
                    mAcc7_values = np.append(mAcc7_values, mean_7_acc)
                    mConf7_values = np.append(mConf7_values, mean_7_conf)
                    arr7 = np.append(arr7, np.array([[num + 1, float(A7_acc / A7_trials), float(A7_conf / A7_trials), A7_trials]]), axis=0)
                if A8_trials != 0:
                    mean_8_acc = round(A8_acc / A8_trials, 5)
                    mean_8_conf = round(A8_conf / A8_trials, 5)
                    mAcc8_values = np.append(mAcc8_values, mean_8_acc)
                    mConf8_values = np.append(mConf8_values, mean_8_conf)
                    arr8 = np.append(arr8, np.array([[num + 1, float(A8_acc / A8_trials), float(A8_conf / A8_trials), A8_trials]]), axis=0)
                writefile.write("Subj: {}, Task 1 trials: {}, Task 2 trials: {}, Task 3 trials: {}, Task 4 trials: {} \n" .format(num + 1, A_trials, B_trials, C_trials, D_trials))
                writefile.write("Subj: {}, Task 5 trials: {}, Task 6 trials: {}, Task 7 trials: {}, Task 8 trials: {} \n" .format(num + 1, A5_trials, A6_trials, A7_trials, A8_trials))
                writefile.write("Mean task 1 acc: {}, Mean task 1 conf: {} \n" .format(mean_A_acc, mean_A_conf))
                writefile.write("Mean task 2 acc: {}, Mean task 2 conf: {} \n" .format(mean_B_acc, mean_B_conf))
                writefile.write("Mean task 3 acc: {}, Mean task 3 conf: {} \n" .format(mean_C_acc, mean_C_conf))
                writefile.write("Mean task 4 acc: {}, Mean task 4 conf: {} \n" .format(mean_D_acc, mean_D_conf))
                writefile.write("Mean task 5 acc: {}, Mean task 5 conf: {} \n" .format(mean_5_acc, mean_5_conf))
                writefile.write("Mean task 6 acc: {}, Mean task 6 conf: {} \n" .format(mean_6_acc, mean_6_conf))
                writefile.write("Mean task 7 acc: {}, Mean task 7 conf: {} \n" .format(mean_7_acc, mean_7_conf))
                writefile.write("Mean task 8 acc: {}, Mean task 8 conf: {} \n" .format(mean_8_acc, mean_8_conf))
                writefile.write("Total 1 acc: {}, Total 1 conf: {} \n" .format(A_acc, A_conf))
                writefile.write("Total 2 acc: {}, Total 2 conf: {} \n" .format(B_acc, B_conf))
                writefile.write("Total 3 acc: {}, Total 3 conf: {} \n" .format(C_acc, C_conf))
                writefile.write("Total 4 acc: {}, Total 4 conf: {} \n" .format(D_acc, D_conf))
                writefile.write("Total 5 acc: {}, Total 5 conf: {} \n" .format(A5_acc, A5_conf))
                writefile.write("Total 6 acc: {}, Total 6 conf: {} \n" .format(A6_acc, A6_conf))
                writefile.write("Total 7 acc: {}, Total 7 conf: {} \n" .format(A7_acc, A7_conf))
                writefile.write("Total 8 acc: {}, Total 8 conf: {} \n" .format(A8_acc, A8_conf))
            correlationA = np.corrcoef(mAccA_values, mConfA_values)
            correlationB = np.corrcoef(mAccB_values, mConfB_values)
            correlationC = np.corrcoef(mAccC_values, mConfC_values)
            correlationD = np.corrcoef(mAccD_values, mConfD_values)
            correlation5 = np.corrcoef(mAcc5_values, mConf5_values)
            correlation6 = np.corrcoef(mAcc6_values, mConf6_values)
            correlation7 = np.corrcoef(mAcc7_values, mConf7_values)
            correlation8 = np.corrcoef(mAcc8_values, mConf8_values)
            mdic1 = {'data': arr1}
            mdic2 = {'data': arr2}
            mdic3 = {'data': arr3}
            mdic4 = {'data': arr4}
            mdic5 = {'data': arr5}
            mdic6 = {'data': arr6}
            mdic7 = {'data': arr7}
            mdic8 = {'data': arr8}
            savemat(filename + "_C1" + ".mat", mdic1)
            savemat(filename + "_C2" + ".mat", mdic2)
            savemat(filename + "_C3" + ".mat", mdic3)
            savemat(filename + "_C4" + ".mat", mdic4)
            savemat(filename + "_C5" + ".mat", mdic5)
            savemat(filename + "_C6" + ".mat", mdic6)
            savemat(filename + "_C7" + ".mat", mdic7)
            savemat(filename + "_C8" + ".mat", mdic8)

        #10 conditions
        elif filename == 'data_Wierzchon_2012.csv':
            mdic1 = {}
            mdic2 = {}
            mdic3 = {}
            mdic4 = {}
            mdic5 = {}
            mdic6 = {}
            mdic7 = {}
            mdic8 = {}
            mdic9 = {}
            mdic10 = {}
            arr1 = np.empty((0, 4), float)
            arr2 = np.empty((0, 4), float)
            arr3 = np.empty((0, 4), float)
            arr4 = np.empty((0, 4), float)
            arr5 = np.empty((0, 4), float)
            arr6 = np.empty((0, 4), float)
            arr7 = np.empty((0, 4), float)
            arr8 = np.empty((0, 4), float)
            arr9 = np.empty((0, 4), float)
            arr10 = np.empty((0, 4), float)
            num_tasks = 10
            for num in range(subjects):
                A_acc = 0
                A_conf = 0
                B_acc = 0
                B_conf = 0
                C_acc = 0
                C_conf = 0
                D_acc = 0
                D_conf = 0
                A5_acc = 0
                A5_conf = 0
                A6_acc = 0
                A6_conf = 0
                A7_acc = 0
                A7_conf = 0
                A8_acc = 0
                A8_conf = 0
                A9_acc = 0
                A9_conf = 0
                A10_acc = 0
                A10_conf = 0
                A_trials = 0
                B_trials = 0
                C_trials = 0
                D_trials = 0
                A5_trials = 0
                A6_trials = 0
                A7_trials = 0
                A8_trials = 0
                A9_trials = 0
                A10_trials = 0
                mean_A_acc = 0
                mean_B_acc = 0
                mean_C_acc = 0
                mean_D_acc = 0
                mean_5_acc = 0
                mean_6_acc = 0
                mean_7_acc = 0
                mean_8_acc = 0
                mean_9_acc = 0
                mean_10_acc = 0
                mean_A_conf = 0
                mean_B_conf = 0
                mean_C_conf = 0
                mean_D_conf = 0
                mean_5_conf = 0
                mean_6_conf = 0
                mean_7_conf = 0
                mean_8_conf = 0
                mean_9_conf = 0
                mean_10_conf = 0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if not "NA" in col[2]:
                           if not "NA" in col[3]:
                               if "DS" in col[4]:
                                   if "PAS" in col[5]:
                                       A_trials += 1
                                       A_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A_acc += 1
                                   elif "P" in col[5]:
                                       B_trials += 1
                                       B_conf += float(col[3])
                                       if col[1] == col[2]:
                                           B_acc += 1
                                   elif "O" in col[5]:
                                       C_trials += 1
                                       C_conf += float(col[3])
                                       if col[1] == col[2]:
                                           C_acc += 1
                                   elif "C" in col[5]:
                                       D_trials += 1
                                       D_conf += float(col[3])
                                       if col[1] == col[2]:
                                           D_acc += 1
                                   elif "SD" in col[5]:
                                       A5_trials += 1
                                       A5_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A5_acc += 1
                               elif "SD" in col[4]:
                                   if "PAS" in col[5]:
                                       A6_trials += 1
                                       A6_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A6_acc += 1
                                   elif "P" in col[5]:
                                       A7_trials += 1
                                       A7_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A7_acc += 1
                                   elif "O" in col[5]:
                                       A8_trials += 1
                                       A8_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A8_acc += 1
                                   elif "C" in col[5]:
                                       A9_trials += 1
                                       A9_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A9_acc += 1
                                   elif "SD" in col[5]:
                                       A10_trials += 1
                                       A10_conf += float(col[3])
                                       if col[1] == col[2]:
                                           A10_acc += 1
                if A_trials != 0:
                    mean_A_acc = round(A_acc / A_trials, 5)
                    mean_A_conf = round(A_conf / A_trials, 5)
                    mAccA_values = np.append(mAccA_values, mean_A_acc)
                    mConfA_values = np.append(mConfA_values, mean_A_conf)
                    arr1 = np.append(arr1, np.array([[num + 1, float(A_acc / A_trials), float(A_conf / A_trials), A_trials]]), axis=0)
                if B_trials != 0:
                    mean_B_acc = round(B_acc / B_trials, 5)
                    mean_B_conf = round(B_conf / B_trials, 5)
                    mAccB_values = np.append(mAccB_values, mean_B_acc)
                    mConfB_values = np.append(mConfB_values, mean_B_conf)
                    arr2 = np.append(arr2, np.array([[num + 1, float(B_acc / B_trials), float(B_conf / B_trials), B_trials]]), axis=0)
                if C_trials != 0:
                    mean_C_acc = round(C_acc / C_trials, 5)
                    mean_C_conf = round(C_conf / C_trials, 5)
                    mAccC_values = np.append(mAccC_values, mean_C_acc)
                    mConfC_values = np.append(mConfC_values, mean_C_conf)
                    arr3 = np.append(arr3, np.array([[num + 1, float(C_acc / C_trials), float(C_conf / C_trials), C_trials]]), axis=0)
                if D_trials != 0:
                    mean_D_acc = round(D_acc / D_trials, 5)
                    mean_D_conf = round(D_conf / D_trials, 5)
                    mAccD_values = np.append(mAccD_values, mean_D_acc)
                    mConfD_values = np.append(mConfD_values, mean_D_conf)
                    arr4 = np.append(arr4, np.array([[num + 1, float(D_acc / D_trials), float(D_conf / D_trials), D_trials]]), axis=0)
                if A5_trials != 0:
                    mean_5_acc = round(A5_acc / A5_trials, 5)
                    mean_5_conf = round(A5_conf / A5_trials, 5)
                    mAcc5_values = np.append(mAcc5_values, mean_5_acc)
                    mConf5_values = np.append(mConf5_values, mean_5_conf)
                    arr5 = np.append(arr5, np.array([[num + 1, float(A5_acc / A5_trials), float(A5_conf / A5_trials), A5_trials]]), axis=0)
                if A6_trials != 0:
                    mean_6_acc = round(A6_acc / A6_trials, 5)
                    mean_6_conf = round(A6_conf / A6_trials, 5)
                    mAcc6_values = np.append(mAcc6_values, mean_6_acc)
                    mConf6_values = np.append(mConf6_values, mean_6_conf)
                    arr6 = np.append(arr6, np.array([[num + 1, float(A6_acc / A6_trials), float(A6_conf / A6_trials), A6_trials]]), axis=0)
                if A7_trials != 0:
                    mean_7_acc = round(A7_acc / A7_trials, 5)
                    mean_7_conf = round(A7_conf / A7_trials, 5)
                    mAcc7_values = np.append(mAcc7_values, mean_7_acc)
                    mConf7_values = np.append(mConf7_values, mean_7_conf)
                    arr7 = np.append(arr7, np.array([[num + 1, float(A7_acc / A7_trials), float(A7_conf / A7_trials), A7_trials]]), axis=0)
                if A8_trials != 0:
                    mean_8_acc = round(A8_acc / A8_trials, 5)
                    mean_8_conf = round(A8_conf / A8_trials, 5)
                    mAcc8_values = np.append(mAcc8_values, mean_8_acc)
                    mConf8_values = np.append(mConf8_values, mean_8_conf)
                    arr8 = np.append(arr8, np.array([[num + 1, float(A8_acc / A8_trials), float(A8_conf / A8_trials), A8_trials]]), axis=0)
                if A9_trials != 0:
                    mean_9_acc = round(A9_acc / A9_trials, 5)
                    mean_9_conf = round(A9_conf / A9_trials, 5)
                    mAcc9_values = np.append(mAcc9_values, mean_9_acc)
                    mConf9_values = np.append(mConf9_values, mean_9_conf)
                    arr9 = np.append(arr9, np.array([[num + 1, float(A9_acc / A9_trials), float(A9_conf / A9_trials), A9_trials]]), axis=0)
                if A10_trials != 0:
                    mean_10_acc = round(A10_acc / A10_trials, 5)
                    mean_10_conf = round(A10_conf / A10_trials, 5)
                    mAcc10_values = np.append(mAcc10_values, mean_10_acc)
                    mConf10_values = np.append(mConf10_values, mean_10_conf)
                    arr10 = np.append(arr10, np.array([[num + 1, float(A10_acc / A10_trials), float(A10_conf / A10_trials), A10_trials]]), axis=0)
                writefile.write("Subj: {}, Task 1 trials: {}, Task 2 trials: {}, Task 3 trials: {}, Task 4 trials: {} \n" .format(num + 1, A_trials, B_trials, C_trials, D_trials))
                writefile.write("Subj: {}, Task 5 trials: {}, Task 6 trials: {}, Task 7 trials: {}, Task 8 trials: {} \n" .format(num + 1, A5_trials, A6_trials, A7_trials, A8_trials))
                writefile.write("Subj: {}, Task 9 trials: {}, Task 10 trials: {} \n" .format(num + 1, A9_trials, A10_trials))
                writefile.write("Mean task 1 acc: {}, Mean task 1 conf: {} \n" .format(mean_A_acc, mean_A_conf))
                writefile.write("Mean task 2 acc: {}, Mean task 2 conf: {} \n" .format(mean_B_acc, mean_B_conf))
                writefile.write("Mean task 3 acc: {}, Mean task 3 conf: {} \n" .format(mean_C_acc, mean_C_conf))
                writefile.write("Mean task 4 acc: {}, Mean task 4 conf: {} \n" .format(mean_D_acc, mean_D_conf))
                writefile.write("Mean task 5 acc: {}, Mean task 5 conf: {} \n" .format(mean_5_acc, mean_5_conf))
                writefile.write("Mean task 6 acc: {}, Mean task 6 conf: {} \n" .format(mean_6_acc, mean_6_conf))
                writefile.write("Mean task 7 acc: {}, Mean task 7 conf: {} \n" .format(mean_7_acc, mean_7_conf))
                writefile.write("Mean task 8 acc: {}, Mean task 8 conf: {} \n" .format(mean_8_acc, mean_8_conf))
                writefile.write("Mean task 9 acc: {}, Mean task 9 conf: {} \n" .format(mean_9_acc, mean_9_conf))
                writefile.write("Mean task 10 acc: {}, Mean task 10 conf: {} \n" .format(mean_10_acc, mean_10_conf))
                writefile.write("Total 1 acc: {}, Total 1 conf: {} \n" .format(A_acc, A_conf))
                writefile.write("Total 2 acc: {}, Total 2 conf: {} \n" .format(B_acc, B_conf))
                writefile.write("Total 3 acc: {}, Total 3 conf: {} \n" .format(C_acc, C_conf))
                writefile.write("Total 4 acc: {}, Total 4 conf: {} \n" .format(D_acc, D_conf))
                writefile.write("Total 5 acc: {}, Total 5 conf: {} \n" .format(A5_acc, A5_conf))
                writefile.write("Total 6 acc: {}, Total 6 conf: {} \n" .format(A6_acc, A6_conf))
                writefile.write("Total 7 acc: {}, Total 7 conf: {} \n" .format(A7_acc, A7_conf))
                writefile.write("Total 8 acc: {}, Total 8 conf: {} \n" .format(A8_acc, A8_conf))
                writefile.write("Total 9 acc: {}, Total 9 conf: {} \n" .format(A9_acc, A9_conf))
                writefile.write("Total 10 acc: {}, Total 10 conf: {} \n" .format(A10_acc, A10_conf))
            correlationA = np.corrcoef(mAccA_values, mConfA_values)
            correlationB = np.corrcoef(mAccB_values, mConfB_values)
            correlationC = np.corrcoef(mAccC_values, mConfC_values)
            correlationD = np.corrcoef(mAccD_values, mConfD_values)
            correlation5 = np.corrcoef(mAcc5_values, mConf5_values)
            correlation6 = np.corrcoef(mAcc6_values, mConf6_values)
            correlation7 = np.corrcoef(mAcc7_values, mConf7_values)
            correlation8 = np.corrcoef(mAcc8_values, mConf8_values)
            correlation9 = np.corrcoef(mAcc9_values, mConf9_values)
            correlation10 = np.corrcoef(mAcc10_values, mConf10_values)
            mdic1 = {'data': arr1}
            mdic2 = {'data': arr2}
            mdic3 = {'data': arr3}
            mdic4 = {'data': arr4}
            mdic5 = {'data': arr5}
            mdic6 = {'data': arr6}
            mdic7 = {'data': arr7}
            mdic8 = {'data': arr8}
            mdic9 = {'data': arr9}
            mdic10 = {'data': arr10}
            savemat(filename + "_C1" + ".mat", mdic1)
            savemat(filename + "_C2" + ".mat", mdic2)
            savemat(filename + "_C3" + ".mat", mdic3)
            savemat(filename + "_C4" + ".mat", mdic4)
            savemat(filename + "_C5" + ".mat", mdic5)
            savemat(filename + "_C6" + ".mat", mdic6)
            savemat(filename + "_C7" + ".mat", mdic7)
            savemat(filename + "_C8" + ".mat", mdic8)
            savemat(filename + "_C9" + ".mat", mdic9)
            savemat(filename + "_C10" + ".mat", mdic10)

    #----------------------------------------
        #generic case, no cond separation
        #check cols to compute acc and conf
        #calc mean acc and conf
        #put this info into txt
        #calc r
        else:
            mdic1 = {}
            arr1 = np.empty((0, 4), float)
            for num in range(subjects):
                acc = 0
                trials = 0
                conf = 0.0
                subtrials = 0
                for line in linelist:
                    col = line.split(',')
                    if col[0] == str(num + 1):
                       if not "NaN" in col[3]:
                           if not "NaN" in col[2]:
                               if not "NA" in col[3]:
                                   if not "NA" in col[2]:
                                       if filename == 'data_Gallagher_2019_Exp1.csv':
                                           if int(col[1]) != 0:
                                               trials += 1
                                               conf += float(col[3])
                                               if int(col[1]) > 0:
                                                   if int(col[2]) > 0:
                                                       acc += 1
                                               else:
                                                   if int(col[2]) == 0:
                                                       acc += 1

                                       elif filename == 'data_Gallagher_2019_Exp2.csv':
                                           if int(col[1]) != 0:
                                               trials += 1
                                               conf += float(col[3])
                                               if int(col[1]) > 0:
                                                   if int(col[2]) > 0:
                                                       acc += 1
                                               else:
                                                   if int(col[2]) == 0:
                                                       acc += 1
                                       elif filename == 'data_Fallow_unpub_3.csv':
                                           trials += 1
                                           conf += float(col[5])
                                           if col[3] == col[4]:
                                               acc += 1
                                       elif filename == 'data_Gherman_2018.csv':
                                           if int(col[6]) != 1:
                                               trials += 1
                                               conf += float(col[3])
                                               if col[1] == col[2]:
                                                   acc += 1
                                       elif filename == 'data_Lempert_2015.csv':
                                           if (num + 1) != 3 and (num + 1) != 10 and (num + 1) != 11 and (num + 1) != 13 and (num + 1) != 32:
                                               trials += 1
                                               if trials > 36:
                                                   subtrials += 1
                                                   conf += float(col[3])
                                                   if col[1] == col[2]:
                                                       acc += 1
                                       elif filename == 'data_Lo_unpub.csv':
                                           if int(col[1]) != 0:
                                               trials += 1
                                               conf += float(col[3])
                                               acc += float(col[8])

                                       elif filename == 'data_Maniscalco_2017_expt1.csv':
                                           if (num + 1) != 3 and (num + 1) != 6 and (num + 1) != 9 and (num + 1) != 22:
                                               trials += 1
                                               conf += float(col[3])
                                               if col[1] == col[2]:
                                                   acc += 1
                                       elif filename == 'data_Maniscalco_2017_expt3.csv':
                                           if (num + 1) != 14:
                                               trials += 1
                                               conf += float(col[3])
                                               if col[1] == col[2]:
                                                   acc += 1
                                       elif filename == 'data_Maniscalco_2017_expt4.csv':
                                           if (num + 1) != 3 and (num + 1) != 4 and (num + 1) != 14 and (num + 1) != 25 and (num + 1) != 26 and (num + 1) != 32:
                                               trials += 1
                                               conf += float(col[3])
                                               if col[1] == col[2]:
                                                   acc += 1
                                       elif filename == 'data_OHora_2017.csv':
                                           if int(col[8]) != 1:
                                               trials += 1
                                               conf += float(col[3])
                                               if col[1] == col[2]:
                                                   acc += 1
                                       elif filename == 'data_Paulewicz_unpub4.csv':
                                           trials += 1
                                           conf += float(col[4])
                                           if col[1] == col[2]:
                                               acc += 1
                                       elif filename == 'data_Pereira_2018.csv':
                                           if float(col[3]) >= 0:
                                               trials += 1
                                               conf += float(col[3])
                                               if col[1] == col[2]:
                                                   acc += 1
                                       elif filename == 'data_Kreis_2019.csv':
                                           trials += 1
                                           conf += float(col[4])
                                           acc += abs(float(col[9]))
                                       elif filename == 'data_OHora_unpub_2.csv':
                                           if not "TIMEOUT" in col[2]:
                                               if not "TRUE" in col[9]:
                                                   if not "1" in col[8]:
                                                       trials += 1
                                                       conf += abs(float(col[3]))
                                                       acc += abs(float(col[7]))
                                       elif filename == 'data_Rausch_2018_Expt1.csv':
                                           if not "1" in col[7]:
                                               trials += 1
                                               conf += float(col[3])
                                               if col[1] == col[2]:
                                                   acc += 1
                                       elif filename == 'data_Rausch_2018_Expt2.csv':
                                           if not "1" in col[8]:
                                               trials += 1
                                               conf += float(col[3])
                                               if col[1] == col[2]:
                                                   acc += 1
                                       elif filename == 'data_Seow_2019.csv':
                                           trials += 1
                                           conf += float(col[15])
                                           acc += float(col[17])
                                       elif filename == 'data_Wang_2018.csv':
                                           trials += 1
                                           conf += float(col[3])
                                           if col[1] == '1' or col[1] == '2' or col[1] == '3':
                                               if col[2] == '1':
                                                   acc += 1
                                           elif col[1] == '5' or col[1] == '6' or col[1] == '7':
                                               if col[2] == '2':
                                                   acc += 1
                                       else:
                                           trials += 1
                                           conf += float(col[3])
                                           if filename == 'data_Double_2017.csv':
                                               acc += float(col[7])
                                           else:
                                               if col[1] == col[2]:                         #check this
                                                   acc += 1
                if trials != 0:
                    if filename == 'data_Lempert_2015.csv':
                        mean_acc = round(acc / subtrials, 5)
                        mean_conf = round(conf / subtrials, 5)
                        writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                        writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(subtrials, acc, conf))
                        mAcc_values = np.append(mAcc_values, mean_acc)
                        mConf_values = np.append(mConf_values, mean_conf)
                        arr1 = np.append(arr1, np.array([[num + 1, float(acc / subtrials), float(conf / subtrials), subtrials]]), axis=0)
                    else:
                        mean_acc = round(acc / trials, 5)
                        mean_conf = round(conf / trials, 5)
                        writefile.write("Subj: {}, Mean acc: {}, Mean conf: {} \n" .format(num + 1, mean_acc, mean_conf))
                        writefile.write("Trials: {}; Total acc: {}, Total conf: {} \n" .format(trials, acc, conf))
                        mAcc_values = np.append(mAcc_values, mean_acc)
                        mConf_values = np.append(mConf_values, mean_conf)
                        arr1 = np.append(arr1, np.array([[num + 1, float(acc / trials), float(conf / trials), trials]]), axis=0)
            correlationA = np.corrcoef(mAcc_values, mConf_values)
            mdic1 = {'data': arr1}
            savemat(filename + ".mat", mdic1)

        if num_tasks == 1:
            #generic code for all no sep cases---
            print(mAcc_values)
            print(mConf_values)
            print(correlationA)
            print(len(mAcc_values))

            #put r values into dictionary
            r_dict[filename] = [correlationA[0][1]]
            print(r_dict)
            #then put into excel
            worksheet.write(row, column, filename)
            worksheet.write(row, column + 1, correlationA[0][1])
            row += 1

        elif num_tasks == 2:
            print(mAccA_values)
            print(mConfA_values)
            print(correlationA)
            print(len(mAccA_values))

            print(mAccB_values)
            print(mConfB_values)
            print(correlationB)
            print(len(mAccB_values))

            #put r values into dictionary
            r_dict[filename] = [correlationA[0][1], correlationB[0][1]]
            print(r_dict)
            #then put into excel
            worksheet.write(row, column, filename)
            worksheet.write(row, column + 1, correlationA[0][1])
            worksheet.write(row, column + 2, correlationB[0][1])
            row += 1

        elif num_tasks == 3:
            print(mAccA_values)
            print(mConfA_values)
            print(correlationA)
            print(len(mAccA_values))

            print(mAccB_values)
            print(mConfB_values)
            print(correlationB)
            print(len(mAccB_values))

            print(mAccC_values)
            print(mConfC_values)
            print(correlationC)
            print(len(mAccC_values))
            #put r values into dictionary
            r_dict[filename] = [correlationA[0][1], correlationB[0][1], correlationC[0][1]]
            print(r_dict)
            #then put into excel
            worksheet.write(row, column, filename)
            worksheet.write(row, column + 1, correlationA[0][1])
            worksheet.write(row, column + 2, correlationB[0][1])
            worksheet.write(row, column + 3, correlationC[0][1])
            row += 1

        elif num_tasks == 4:
            print(mAccA_values)
            print(mConfA_values)
            print(correlationA)
            print(len(mAccA_values))

            print(mAccB_values)
            print(mConfB_values)
            print(correlationB)
            print(len(mAccB_values))

            print(mAccC_values)
            print(mConfC_values)
            print(correlationC)
            print(len(mAccC_values))

            print(mAccD_values)
            print(mConfD_values)
            print(correlationD)
            print(len(mAccD_values))
            #put r values into dictionary
            r_dict[filename] = [correlationA[0][1], correlationB[0][1], correlationC[0][1], correlationD[0][1]]
            print(r_dict)
            #then put into excel
            worksheet.write(row, column, filename)
            worksheet.write(row, column + 1, correlationA[0][1])
            worksheet.write(row, column + 2, correlationB[0][1])
            worksheet.write(row, column + 3, correlationC[0][1])
            worksheet.write(row, column + 4, correlationD[0][1])
            row += 1

        elif num_tasks == 6:
            print(mAccA_values)
            print(mConfA_values)
            print(correlationA)
            print(len(mAccA_values))

            print(mAccB_values)
            print(mConfB_values)
            print(correlationB)
            print(len(mAccB_values))

            print(mAccC_values)
            print(mConfC_values)
            print(correlationC)
            print(len(mAccC_values))

            print(mAccD_values)
            print(mConfD_values)
            print(correlationD)
            print(len(mAccD_values))

            print(mAcc5_values)
            print(mConf5_values)
            print(correlation5)
            print(len(mAcc5_values))

            print(mAcc6_values)
            print(mConf6_values)
            print(correlation6)
            print(len(mAcc6_values))

            #put r values into dictionary
            r_dict[filename] = [correlationA[0][1], correlationB[0][1], correlationC[0][1], correlationD[0][1], correlation5[0][1], correlation6[0][1]]
            print(r_dict)
            #then put into excel
            worksheet.write(row, column, filename)
            worksheet.write(row, column + 1, correlationA[0][1])
            worksheet.write(row, column + 2, correlationB[0][1])
            worksheet.write(row, column + 3, correlationC[0][1])
            worksheet.write(row, column + 4, correlationD[0][1])
            worksheet.write(row, column + 5, correlation5[0][1])
            worksheet.write(row, column + 6, correlation6[0][1])
            row += 1

        elif num_tasks == 8:
            print(mAccA_values)
            print(mConfA_values)
            print(correlationA)
            print(len(mAccA_values))

            print(mAccB_values)
            print(mConfB_values)
            print(correlationB)
            print(len(mAccB_values))

            print(mAccC_values)
            print(mConfC_values)
            print(correlationC)
            print(len(mAccC_values))

            print(mAccD_values)
            print(mConfD_values)
            print(correlationD)
            print(len(mAccD_values))

            print(mAcc5_values)
            print(mConf5_values)
            print(correlation5)
            print(len(mAcc5_values))

            print(mAcc6_values)
            print(mConf6_values)
            print(correlation6)
            print(len(mAcc6_values))

            print(mAcc7_values)
            print(mConf7_values)
            print(correlation7)
            print(len(mAcc7_values))

            print(mAcc8_values)
            print(mConf8_values)
            print(correlation8)
            print(len(mAcc8_values))

            print(len(mAcc8_values) + len(mAcc7_values) + len(mAcc6_values) + len(mAcc5_values) + len(mAccD_values) + len(mAccC_values) + len(mAccB_values) + len(mAccA_values))

            #put r values into dictionary
            r_dict[filename] = [correlationA[0][1], correlationB[0][1], correlationC[0][1], correlationD[0][1], correlation5[0][1], correlation6[0][1], correlation7[0][1], correlation8[0][1]]
            print(r_dict)
            #then put into excel
            worksheet.write(row, column, filename)
            worksheet.write(row, column + 1, correlationA[0][1])
            worksheet.write(row, column + 2, correlationB[0][1])
            worksheet.write(row, column + 3, correlationC[0][1])
            worksheet.write(row, column + 4, correlationD[0][1])
            worksheet.write(row, column + 5, correlation5[0][1])
            worksheet.write(row, column + 6, correlation6[0][1])
            worksheet.write(row, column + 7, correlation7[0][1])
            worksheet.write(row, column + 8, correlation8[0][1])
            row += 1

        elif num_tasks == 10:
            print(mAccA_values)
            print(mConfA_values)
            print(correlationA)
            print(len(mAccA_values))

            print(mAccB_values)
            print(mConfB_values)
            print(correlationB)
            print(len(mAccB_values))

            print(mAccC_values)
            print(mConfC_values)
            print(correlationC)
            print(len(mAccC_values))

            print(mAccD_values)
            print(mConfD_values)
            print(correlationD)
            print(len(mAccD_values))

            print(mAcc5_values)
            print(mConf5_values)
            print(correlation5)
            print(len(mAcc5_values))

            print(mAcc6_values)
            print(mConf6_values)
            print(correlation6)
            print(len(mAcc6_values))

            print(mAcc7_values)
            print(mConf7_values)
            print(correlation7)
            print(len(mAcc7_values))

            print(mAcc8_values)
            print(mConf8_values)
            print(correlation8)
            print(len(mAcc8_values))

            print(mAcc9_values)
            print(mConf9_values)
            print(correlation9)
            print(len(mAcc9_values))

            print(mAcc10_values)
            print(mConf10_values)
            print(correlation10)
            print(len(mAcc10_values))

            print(len(mAcc10_values) + len(mAcc9_values) + len(mAcc8_values) + len(mAcc7_values) + len(mAcc6_values) + len(mAcc5_values) + len(mAccD_values) + len(mAccC_values) + len(mAccB_values) + len(mAccA_values))

            #put r values into dictionary
            r_dict[filename] = [correlationA[0][1], correlationB[0][1], correlationC[0][1], correlationD[0][1], correlation5[0][1], correlation6[0][1], correlation7[0][1], correlation8[0][1], correlation9[0][1], correlation10[0][1]]
            print(r_dict)
            #then put into excel
            worksheet.write(row, column, filename)
            worksheet.write(row, column + 1, correlationA[0][1])
            worksheet.write(row, column + 2, correlationB[0][1])
            worksheet.write(row, column + 3, correlationC[0][1])
            worksheet.write(row, column + 4, correlationD[0][1])
            worksheet.write(row, column + 5, correlation5[0][1])
            worksheet.write(row, column + 6, correlation6[0][1])
            worksheet.write(row, column + 7, correlation7[0][1])
            worksheet.write(row, column + 8, correlation8[0][1])
            worksheet.write(row, column + 9, correlation9[0][1])
            worksheet.write(row, column + 10, correlation10[0][1])
            row += 1
        #-------------------------------------
        #delete file that was checked from list
        data_files = data_files[1:]


        myfile.close()
        writefile.close()
    workbook.close()
