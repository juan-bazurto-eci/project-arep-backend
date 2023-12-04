from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import concatenate, Input, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np

trainpath = './TrainImages/'
v_TrainPath = glob.glob(trainpath + '*')
print(len(v_TrainPath))  # Asegúrate de que esto retorne 120

df_test = pd.read_csv('./solution_stg1_release.csv', sep=';')
testpath = './TestImages/'
modelspath = './Models/'

print(df_test.head())

x_train = []
y_labels = []

for i in v_TrainPath:
    im = plt.imread(i)
    x_train.append(im)
    aux_split = i.split("TrainImages")[1][1:]
    im_class = aux_split.split('_')[0]
    if im_class == '001':
        y_labels.append(0)
    elif im_class == '010':
        y_labels.append(1)
    elif im_class == '100':
        y_labels.append(2)

x_test = []
y_labTest = []
for i in range(len(df_test)):
    x_test.append(plt.imread(testpath + df_test.iloc[i]['image_name']))
    if df_test.iloc[i]['Type_1'] == 1:
        y_labTest.append(0)
    if df_test.iloc[i]['Type_2'] == 1:
        y_labTest.append(1)
    if df_test.iloc[i]['Type_3'] == 1:
        y_labTest.append(2)

y_labels = np.array(y_labels)
y_train = to_categorical(y_labels, num_classes=3, dtype='float32')
print(y_train[0:120])

y_labTest = np.array(y_labTest)
y_test = to_categorical(y_labTest, num_classes=3, dtype='float32')
print(y_test[0:9])

x_train = np.array(x_train) / 255.
x_test = np.array(x_test) / 255.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


def Resnet50Model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    num_classes = 3
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    return model

model = Resnet50Model()
checkpointer = ModelCheckpoint(filepath=modelspath + 'ResNet50_8b10e.hdf5',
                               monitor='val_loss', verbose=1, save_best_only=True)

s_e = 10
s_bs = 8

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(x_train)

model.fit(datagen.flow(x_train, y_train, batch_size=s_bs), steps_per_epoch=len(x_train) / s_bs, epochs=s_e,
          validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=2)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.load_weights(modelspath + 'ResNet50_8b10e.hdf5')

y_pred_train = model.predict(x_train, verbose=1)
y_pred_test = model.predict(x_test, verbose=1)

print(confusion_matrix(np.argmax(y_train, axis=1), np.argmax(y_pred_train, axis=1)))
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred_test, axis=1)))

print(precision_recall_fscore_support(np.argmax(y_train, axis=1), np.argmax(y_pred_train, axis=1)))
print(precision_recall_fscore_support(np.argmax(y_test, axis=1), np.argmax(y_pred_test, axis=1)))
