from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from tabulate import tabulate
import pandas
import numpy

data = pandas.read_csv('dryBeanDataset.csv')
#No null values in this dataset

#replace classes
classes_map = {'SEKER': 0, 'BARBUNYA': 1, 'BOMBAY': 2, 
                "CALI": 3, "HOROZ": 4, "SIRA": 5, "DERMASON": 6}
classes_reverse_map = {0:'SEKER', 1:'BARBUNYA', 2:'BOMBAY', 
                3:"CALI", 4:"HOROZ", 5:"SIRA", 6:"DERMASON"}

data['Class'] = data['Class'].map(classes_map)

labels = numpy.array(data['Class'])
data = data.drop('Class', axis=1)

feature_list = list(data.columns)
data = numpy.array(data)

seed = 13
number_splits = 10
rf = RandomForestClassifier(n_estimators=1000, random_state=seed)
tenfold = KFold(n_splits=number_splits, shuffle=True, random_state=seed)

total_performance = {}
fold_performances = {}

fold_num = 0

for train_index, test_index in tenfold.split(data):
  fold_num += 1
  fold_performances[fold_num] = {}
  #Train_index is the list of indexes for our training data
  #Test_index is the list of indexes for our testing data
  train_features = [data[i] for i in train_index]
  train_labels = [labels[i] for i in train_index]
  test_features = [data[i] for i in test_index]
  test_labels = [labels[i] for i in test_index]
  
  rf = RandomForestClassifier(n_estimators=1000, random_state=seed)
  rf.fit(train_features, train_labels)
  predictions = rf.predict(test_features)

  #predicted value = row
  #actual value = column
  confusion_matrix = [[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]]

  for i in range(0, len(predictions)):
    confusion_matrix[predictions[i]][test_labels[i]] += 1
  
  print('----------FOLD ', str(fold_num), '----------')
  print('Confusion Matrix', end='\n\n')
  
  print('Actual Values'.rjust(10*(len(classes_map.keys())+4)//2))
  print(''.rjust(30), end='')
  for classification in classes_map.keys():
    print(str(classification).rjust(9), end=' ')
  print()
  print((10*(len(classes_map.keys())+3))*('-'))
  for i in range(len(confusion_matrix)):
    if i != len(classes_map.keys())//2:
      print(''.rjust(20), end='')
    else:
      print('Predicted Values'.rjust(20), end='')
    print(classes_reverse_map[i].rjust(9), end='|')
    row = confusion_matrix[i]
    for j in range(len(row)):
      val = row[j]
      if j != len(row)-1:
        print(str(val).rjust(8), end=', ')
      else:
        print(str(val).rjust(8), end='  ')
    print()
  
  for classif in range(0,7):
    tp = confusion_matrix[classif][classif]
    tn = 0
    fp = 0
    fn = 0
    for i in range(0,7):
      if  i == classif:
        for j in range(0,7):
          if j == classif:
            #This is tp
            continue
          else:
            fp += confusion_matrix[i][j]
      else:
        for j in range(0,7):
          if j == classif:
            fn += confusion_matrix[i][j]
          else:
            tn += confusion_matrix[i][j]
    
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    fpr = fp/(tn+fp)
    fnr = fn/(tp+fn)
    r = tp/(tp+fn)
    p = tp/(tp+fp)
    f1 = 2*(p*r)/(p+r)
    acc = (tp+tn)/(tp+fp+fn+tn)
    err = (fp+fn)/(tp+fp+fn+tn)
    bacc = (tpr+tnr)/2
    tss = (tp)/(tp+fn)-(fp)/(fp+tn)
    hss = (2*(tp*tn-fp*fn))/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn))
    
    fold_performances[fold_num][classes_reverse_map[classif]] = {}
    fold_performances[fold_num][classes_reverse_map[classif]]["TP"] = tp
    fold_performances[fold_num][classes_reverse_map[classif]]["TN"] = tn
    fold_performances[fold_num][classes_reverse_map[classif]]["FP"] = fp
    fold_performances[fold_num][classes_reverse_map[classif]]["FN"] = fn
    fold_performances[fold_num][classes_reverse_map[classif]]["FN"] = fn
    fold_performances[fold_num][classes_reverse_map[classif]]["TPR"] = tpr
    fold_performances[fold_num][classes_reverse_map[classif]]["TNR"] = tpr
    fold_performances[fold_num][classes_reverse_map[classif]]["FPR"] = fpr
    fold_performances[fold_num][classes_reverse_map[classif]]["FNR"] = fnr
    fold_performances[fold_num][classes_reverse_map[classif]]["r"] = r
    fold_performances[fold_num][classes_reverse_map[classif]]["p"] = p
    fold_performances[fold_num][classes_reverse_map[classif]]["f1"] = f1
    fold_performances[fold_num][classes_reverse_map[classif]]["Acc"] = acc
    fold_performances[fold_num][classes_reverse_map[classif]]["Err"] = err
    fold_performances[fold_num][classes_reverse_map[classif]]["BACC"] = bacc
    fold_performances[fold_num][classes_reverse_map[classif]]["TSS"] = tss
    fold_performances[fold_num][classes_reverse_map[classif]]["HSS"] = hss
    if fold_num == 1:
      total_performance[classes_reverse_map[classif]] = {}
      total_performance[classes_reverse_map[classif]]["TP"] = tp
      total_performance[classes_reverse_map[classif]]["TN"] = tn
      total_performance[classes_reverse_map[classif]]["FP"] = fp
      total_performance[classes_reverse_map[classif]]["FN"] = fn
      total_performance[classes_reverse_map[classif]]["TPR"] = tpr
      total_performance[classes_reverse_map[classif]]["TNR"] = tpr
      total_performance[classes_reverse_map[classif]]["FPR"] = fpr
      total_performance[classes_reverse_map[classif]]["FNR"] = fnr
      total_performance[classes_reverse_map[classif]]["r"] = r
      total_performance[classes_reverse_map[classif]]["p"] = p
      total_performance[classes_reverse_map[classif]]["f1"] = f1
      total_performance[classes_reverse_map[classif]]["Acc"] = acc
      total_performance[classes_reverse_map[classif]]["Err"] = err
      total_performance[classes_reverse_map[classif]]["BACC"] = bacc
      total_performance[classes_reverse_map[classif]]["TSS"] = tss
      total_performance[classes_reverse_map[classif]]["HSS"] = hss
    else:
      total_performance[classes_reverse_map[classif]]["TP"] += tp
      total_performance[classes_reverse_map[classif]]["TN"] += tn
      total_performance[classes_reverse_map[classif]]["FP"] += fp
      total_performance[classes_reverse_map[classif]]["FN"] += fn
      total_performance[classes_reverse_map[classif]]["TPR"] += tpr
      total_performance[classes_reverse_map[classif]]["TNR"] += tpr
      total_performance[classes_reverse_map[classif]]["FPR"] += fpr
      total_performance[classes_reverse_map[classif]]["FNR"] += fnr
      total_performance[classes_reverse_map[classif]]["r"] += r
      total_performance[classes_reverse_map[classif]]["p"] += p
      total_performance[classes_reverse_map[classif]]["f1"] += f1
      total_performance[classes_reverse_map[classif]]["Acc"] += acc
      total_performance[classes_reverse_map[classif]]["Err"] += err
      total_performance[classes_reverse_map[classif]]["BACC"] += bacc
      total_performance[classes_reverse_map[classif]]["TSS"] += tss
      total_performance[classes_reverse_map[classif]]["HSS"] = hss

  print()
  print_data = []
  for classif in fold_performances[fold_num].keys():
    curr_class_data = [classif]
    for parameter in fold_performances[fold_num][classif].keys():
      curr_class_data.append(fold_performances[fold_num][classif][parameter])
    print_data.append(curr_class_data)
  
  print(tabulate(print_data, headers=['Class', 'TP', 'TN', "FP", "FN", "TPR", "TNR",
                              "FPR", "FNR", "Recall", "Precision", "F1", "Accuracy",
                              "Error Rate", "BACC", "TSS", "HSS"]))
  print()

print('----------SUMMARY----------')
num_class = len(total_performance.keys())
total_performance['All Classes'] = {}
print_data = []
for classification in total_performance.keys():
  curr_class_data = [classification]
  if classification == 'All Classes': 
    for parameter in total_performance[classification].keys():
      curr_class_data.append(str(total_performance[classification][parameter] / (number_splits * num_class)))
  else:
    for parameter in total_performance[classification].keys():
      if parameter not in total_performance["All Classes"].keys():
        total_performance['All Classes'][parameter] = total_performance[classification][parameter]
      else:
        total_performance['All Classes'][parameter] += total_performance[classification][parameter]
      curr_class_data.append(str(total_performance[classification][parameter] / number_splits))
  print_data.append(curr_class_data)


print(tabulate(print_data, headers=['Class', 'TP', 'TN', "FP", "FN", "TPR", "TNR",
                            "FPR", "FNR", "Recall", "Precision", "F1", "Accuracy",
                            "Error Rate", "BACC", "TSS", "HSS"]))