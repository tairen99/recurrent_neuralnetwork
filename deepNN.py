import dataHandle as dhandle
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

data = dhandle.data2014
backLen = dhandle.backDays
trainGen = dhandle.train_gen
step_epochs = dhandle.step_epochs
valGen = dhandle.val_gen
valSteps = dhandle.val_steps

# test normal neural network
# model = Sequential()
# model.add(layers.Flatten(input_shape=(backLen, data.shape[-1])))
# model.add(layers.Dense(40, activation='relu'))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(trainGen, steps_per_epoch=step_epochs, epochs=40,
#                               validation_data=valGen, validation_steps=valSteps)


model = Sequential()
model.add(layers.LSTM(32,dropout=0.2, recurrent_dropout=0.2,
                      input_shape=(None, data.shape[-1])))

# test the GRU case:
# model.add(layers.GRU(32, dropout=0.2,
#                      recurrent_dropout=0.2,input_shape=(None, data.shape[-1])))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(trainGen, steps_per_epoch=step_epochs, epochs=40,
                              validation_data=valGen, validation_steps=valSteps)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'r*', label='Training loss')
plt.plot(epochs, val_loss, 'b:', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show(block=True)
plt.interactive(False)
