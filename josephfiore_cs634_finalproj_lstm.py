from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.models import Sequential

data = pandas.read_csv('dryBeanDataset.csv')
#No null values in this dataset

#replace classes
classes_map = {'SEKER': 0, 'BARBUNYA': 1, 'BOMBAY': 2, 
                "CALI": 3, "HOROZ": 4, "SIRA": 5, "DERMASON": 6}
classes_reverse_map = {0:'SEKER', 1:'BARBUNYA', 2:'BOMBAY', 
                3:"CALI", 4:"HOROZ", 5:"SIRA", 6:"DERMASON"}

data['Class'] = data['Class'].map(classes_map)

#labels = numpy.array(data['Class'])
#data = data.drop('Class', axis=1)

feature_list = list(data.columns)
data = numpy.array(data)

seed = 13
number_splits = 10

tenfold = KFold(n_splits=number_splits, shuffle=True, random_state=seed)

total_performance = {}
fold_performances = {}

fold_num = 0

# parameters
time_steps = 1
inputs = 3
outputs = 1
learning_rate = 0.001
epochs = 500
neurons = 32

for train_index, test_index in tenfold.split(data):
  if(fold_num > 0):
    break
  fold_num += 1
  fold_performances[fold_num] = {}
  #Train_index is the list of indexes for our training data
  #Test_index is the list of indexes for our testing data
  train = numpy.array([data[i] for i in train_index])
  test = numpy.array([data[i] for i in test_index])
  
  scaler = MinMaxScaler(feature_range=(-1, 1))
  scaler = scaler.fit(train)
  train = train.reshape(train.shape[0], train.shape[1])
  train_scaled = scaler.transform(train)
  #print(train_scaled)
  #print(scaler.inverse_transform(train_scaled))
  test = test.reshape(test.shape[0], test.shape[1])
  test_scaled = scaler.transform(test)
  #print(test_scaled)
  #print(scaler.inverse_transform(test_scaled))
  
  
  train_x = train_scaled[:, 0:-1]
  batch_size = train_x.shape[0]
  train_y = train_scaled[:, -1]
  
  train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
  
  model = Sequential()
  model.add(LSTM(200, batch_input_shape=(1, 1, train_x.shape[2]), return_sequences=False, stateful=True))
  model.add(Dense(1, activation='tanh'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  print(model.summary())
  
  model.fit(train_x, train_y, epochs=50, batch_size=1, verbose=0, shuffle=False)
  
  test_x = test_scaled[:, 0:-1]
  test_y = test_scaled[:, -1]
  test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
  
  #predictions = model.predict(test_x)
  for test in test_x:
    pred = model.predict(test.reshape(1, 1, test.shape[1]), batch_size=1)
    print(test.reshape(1, 1, test.shape[1]))
    print(pred)
    long_test = numpy.concatenate((test[0], pred[0]))
    inverted = scaler.inverse_transform(long_test.reshape(1, -1))
    print(inverted)
                        
  #print(predictions)
  #print(len(predictions))
  #print(len(test_x))
  #inverted = scaler.inverse_transform(predictions)
  #print(inverted)
  """
  model = Sequential()
  model.add(LSTM(1, input_shape=(16,1)))
  model.add(Dense(1, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  model.fit(train_features, train_labels)
  
  predictions = model.predict(test_features)
  
  for i in range(len(predictions)):
    print(predictions[i], ", ", test_labels[i])
  """
