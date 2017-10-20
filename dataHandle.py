import numpy as np
import scipy.io
# import matplotlib.pyplot as plt
import dataNormalize as dataNorm

mat2014 = scipy.io.loadmat('/home/tairen/Texas_data/TexasNorthCenter_load_all2014.mat')

# read the 2013 & 2014 daily load from file
matdata2014 = mat2014.get('TexasNorthCenter_load_all2014')

# generate the time index for everyday load record [1,2,...,96,1,2,...,96,1,2,...]
index = [x%96 if x%96 != 0 else 96 for x in range(1,len(matdata2014)+1) ]
# transform to np.array
indArray = np.array(index).reshape((len(index),-1))
# combine two columns of data
data2014 = np.concatenate((indArray,matdata2014),axis=1)

# Pycharm multiple line comment: ctrl + /
# move code: tab or shift+tab
# plt.plot(range(len(data2014[1:960,1])),data2014[1:960,1])
# plt.show(block=True)
# plt.interactive(False)


def generator(data, backDays, preDays, min_index, max_index, batch_size=96):
    if max_index is None:
        max_index = len(data) - preDays - 1
    i = min_index + backDays
    # import pdb; pdb.set_trace() #pdb.set_trace()
    while True:
        if i + batch_size >= max_index:
            i = min_index + backDays
        rows = np.arange(i, min(i + batch_size, max_index))  # get the row number from data
        i += len(rows)
        # use 3D samples to store the selected data. 1D is batch_number, 2-3D is the sampled data
        samples = np.zeros((len(rows), backDays, data.shape[-1]))

        targets = np.zeros((len(rows),))  # intialize targets as zeros with length as batch length
        for j, row in enumerate(rows):
            # select the indices which is sampled by the interval: step from row: rows[j]-backDays to rows[j]
            indic = range(rows[j] - backDays, rows[j])
            samples[j] = data[indic]
            targets[j] = data[rows[j] + preDays][1]  # find next day samples as targets data
        yield samples, targets  # yield is the keyword to use generator
        # since I can't debug the generator, so I try to use return, and it works for simple debug
        # return samples, targets


backDays = 480
preDays = 96
batch_size = 96

# select the days for training and days for validation
trainStart = 336*96 + 1
trainStop = 356*96

validationStart = trainStop + 1
validationStop = 365*96

train_data,mean,std = dataNorm.dataNormalize(data2014,trainStart,trainStop+1,None,None)

train_gen = generator(train_data, backDays=backDays, preDays=preDays,
                      min_index=trainStart, max_index=trainStop, batch_size=batch_size)

train_data,mean,std = dataNorm.dataNormalize(data2014,validationStart,validationStop+1,std,mean)
val_gen = generator(train_data, backDays=backDays, preDays=preDays,
                    min_index=validationStart, max_index=validationStop, batch_size=batch_size)


# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (validationStop - validationStart - backDays) // batch_size
step_epochs = (trainStop - trainStart - backDays) // batch_size




