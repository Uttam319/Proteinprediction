import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU, BatchNormalization, Flatten
from keras import optimizers, callbacks
from keras.regularizers import l2
import tensorflow as tf
import dataset

do_summary = True

LR = 0.0009 
drop_out = 0.38
batch_dim = 64
nn_epochs = 20
loss = 'categorical_crossentropy' # best standart
early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')
if dataset.filtered:
    filepath="CullPDB_Filtered-best.keras"
else:
    filepath="CullPDB6133-best.keras"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

def CNN_model():
    m = Sequential()
    m.add(Conv1D(128, 5, padding='same', activation='relu', input_shape=(dataset.cnn_width, dataset.amino_acid_residues)))
    m.add(BatchNormalization())
    m.add(Dropout(drop_out))
    m.add(Conv1D(128, 3, padding='same', activation='relu'))
    m.add(BatchNormalization()) 
    m.add(Dropout(drop_out))
    m.add(Conv1D(64, 3, padding='same', activation='relu'))
    m.add(BatchNormalization())  
    m.add(Dropout(drop_out))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(dataset.num_classes, activation = 'softmax'))
    opt = optimizers.Adam(learning_rate=LR)
    m.compile(optimizer=opt,
              loss=loss,
              metrics=['accuracy', 'mae'])
    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Drop out: " + str(drop_out))
        print("Batch dim: " + str(batch_dim))
        print("Number of epochs: " + str(nn_epochs))
        print("\nLoss: " + loss + "\n")

        m.summary()

    return m


if __name__ == '__main__':
    print("This script contains the model")
