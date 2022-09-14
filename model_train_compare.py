import os
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 전처리
from tensorflow.keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt



# 0~1 사이값으로 픽셀값을 변환
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir='/content/drive/MyDrive/Colab Notebooks/vision_project/moonimage/train'
val_dir='/content/drive/MyDrive/Colab Notebooks/vision_project/moonimage/validation'
test_dir='/content/drive/MyDrive/Colab Notebooks/vision_project/moonimage/test'

# 이미지 전처리
# 폴더에 있는 이미지를 전처리
train_generator = train_datagen.flow_from_directory(
    # 폴더명
    train_dir,
    # 이미지 크기를 동일한 크기로 변환
    target_size = (150, 150),
    # 한 번에 전처리할 이미지의 수
    batch_size = 20,
    # 라벨링 : binary 이진라벨링, categorical 다중라벨링
    # 라벨링 방법 : 폴더명의 첫 문자의 알파벳으로 0부터 부여
    class_mode = "categorical"
)

val_generator = val_datagen.flow_from_directory(
    # 폴더명
    val_dir,
    # 이미지 크기를 동일한 크기로 변환
    target_size = (150, 150),
    # 한 번에 전처리할 이미지의 수
    batch_size = 20,
    # 라벨링 : binary 이진라벨링, categorical 다중라벨링
    # 라벨링 방법 : 폴더명의 첫 문자의 알파벳으로 0부터 부여
    class_mode = "categorical"
)

test_generator = test_datagen.flow_from_directory(
    # 폴더명
    test_dir,
    # 이미지 크기를 동일한 크기로 변환
    target_size = (150, 150),
    # 한 번에 전처리할 이미지의 수
    batch_size = 20,
    # 라벨링 : binary 이진라벨링, categorical 다중라벨링
    # 라벨링 방법 : 폴더명의 첫 문자의 알파벳으로 0부터 부여
    class_mode = "categorical"
)


###### 일반적인 CNN ######

num_classes = 5
epochs = 25
input_shape = (150, 150, 3)

model1 = Sequential()

model1.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, padding="same", activation="relu"))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, padding="same", activation="relu"))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=input_shape, padding="same", activation="relu"))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(filters=256, kernel_size=(3, 3), input_shape=input_shape, padding="same", activation="relu"))
#model1.BatchNormal
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(units=512, activation="relu"))
model1.add(Dense(units=512, activation="relu"))
model1.add(Dense(units=5, activation="softmax"))

model1.summary()

op=optimizers.Adam(lr=0.0001)
model1.compile(loss="categorical_crossentropy", optimizer=op, metrics=["acc"])
es=EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model1.h5', monitor='val_acc', mode='min', save_best_only=True)

# generator : 제너레이터를 설정
h1 = model1.fit_generator(generator=train_generator, epochs=100, validation_data=test_generator,callbacks=[es,mc])

# 최종 결과 리포트
acc = h1.history['acc']
val_acc = h1.history['val_acc']
loss = h1.history['loss']
val_loss = h1.history['val_loss']
epochs = range(len(acc))


plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()
plt.clf()




###### VGG-16 ######

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

input_tensor = Input(shape=(150,150,3))
model = VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)
print(len(model.layers))

layer_dict = dict([(layer.name, layer) for layer in model.layers])
x = layer_dict['block5_pool'].output
# Cov2D Layer +
x = Conv2D(filters = 64, kernel_size=(3, 3), activation='relu')(x)
# MaxPooling2D Layer +
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(5, activation='softmax')(x)
new_model = Model(inputs = model.input, outputs = x)
new_model.summary()

for layer in new_model.layers[:19] :
    layer.trainable = False

op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model2.h5', monitor='val_loss', mode='min', save_best_only=True)
h2 = new_model.fit_generator(generator=train_generator, epochs=100, validation_data=test_generator,callbacks=[es,mc])

# 최종 결과 리포트
acc = h2.history['accuracy']
val_acc = h2.history['val_accuracy']
loss = h2.history['loss']
val_loss = h2.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()
plt.clf()



###### ResNet50 ######

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

input_tensor = Input(shape=(150,150,3))
model = ResNet50(weights='imagenet', include_top=False, input_tensor = input_tensor)
model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers])
x = layer_dict['conv5_block3_out'].output
# Cov2D Layer +
x = Conv2D(filters = 64, kernel_size=(3, 3), activation='relu')(x)
# MaxPooling2D Layer +
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(5, activation='softmax')(x)
new_model = Model(inputs = model.input, outputs = x)
print(len(model.layers))

for layer in new_model.layers[:175] :
    layer.trainable = False

op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model3.h5', monitor='val_loss', mode='min', save_best_only=True)
h3 = new_model.fit_generator(generator=train_generator, epochs=100, validation_data=test_generator,callbacks=[es,mc])

# 최종 결과 리포트
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

from matplotlib import pyplot as plt

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()
plt.clf()



###### VGG-19 ######

from tensorflow.keras.applications.vgg19 import VGG19

input_tensor = Input(shape=(150,150,3))
model = VGG19(weights='imagenet', include_top=False, input_tensor = input_tensor)
model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers])
x = layer_dict['block5_pool'].output
# Cov2D Layer +
x = Conv2D(filters = 64, kernel_size=(3, 3), activation='relu')(x)
# MaxPooling2D Layer +
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(5, activation='softmax')(x)
new_model = Model(inputs = model.input, outputs = x)
print(len(model.layers))

for layer in new_model.layers[:22] :
    layer.trainable = False
    
op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)
h4 = new_model.fit_generator(generator=train_generator, epochs=100, validation_data=test_generator,callbacks=[es,mc])

# 최종 결과 리포트
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

from matplotlib import pyplot as plt

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()



###### InceptionV3 ######

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D

input_tensor = Input(shape=(150,150,3))
model = InceptionV3(weights='imagenet', include_top=False, input_tensor = input_tensor)
model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers])
x = layer_dict['mixed10'].output
# Cov2D Layer +
x=GlobalAveragePooling2D()(x)
x = Dense(5, activation='softmax')(x)
new_model = Model(inputs = model.input, outputs = x)
print(len(model.layers))

for layer in new_model.layers[:311] :
    layer.trainable = False

op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)
h5 = new_model.fit_generator(generator=train_generator, epochs=200, validation_data=test_generator,callbacks=[es,mc])

# 최종 결과 리포트
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()
plt.clf()



###### DenseNet ######

from tensorflow.keras.applications.densenet import DenseNet121

input_tensor = Input(shape=(150,150,3))
model = DenseNet121(weights='imagenet', include_top=False, input_tensor = input_tensor)
model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers])
x = layer_dict['relu'].output
# Cov2D Layer +
x=GlobalAveragePooling2D()(x)
x = Dense(5, activation='softmax')(x)
new_model = Model(inputs = model.input, outputs = x)
print(len(model.layers))

for layer in new_model.layers[:427] :
    layer.trainable = False

op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)
h6 = new_model.fit_generator(generator=train_generator, epochs=200, validation_data=test_generator,callbacks=[es,mc])

# 최종 결과 리포트
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()



###### DenseNet ######

from tensorflow.keras.applications import MobileNetV3Large

input_shape = (224, 224, 3)
input_tensor = Input(shape=(150,150,3))
model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape = input_shape)
model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers])
x = layer_dict['multiply_240'].output
x=GlobalAveragePooling2D()(x)
x = Dense(5, activation='softmax')(x)
new_model = Model(inputs = model.input, outputs = x)
print(len(model.layers))

for layer in new_model.layers[:263] :
    layer.trainable = False

op=optimizers.Adam(lr=0.001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)

h7 = new_model.fit_generator(generator=train_generator, epochs=400, validation_data=test_generator,callbacks=[es,mc])

# 최종 결과 리포트
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()


###### NasNetLarge ######

from tensorflow.keras.applications.nasnet import NASNetLarge

input_tensor = Input(shape=(150,150,3))
model = NASNetLarge(weights='imagenet', include_top=False, input_tensor = input_tensor)
model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers])
x = layer_dict['activation_873'].output
x=GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(5, activation='softmax')(x)
new_model = Model(inputs = model.input, outputs = x)
print(len(model.layers))

for layer in new_model.layers[:1039] :
    layer.trainable = False

op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)
h8 = new_model.fit_generator(generator=train_generator, epochs=400, validation_data=test_generator,callbacks=[es,mc])

# 최종 결과 리포트
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

from matplotlib import pyplot as plt

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()
plt.clf()



###### EfficientNet ######

from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import tensorflow as tf

input_tensor = Input(shape=(150,150,3))
model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor = input_tensor)

model.trainable=False
model.summary()

x=GlobalAveragePooling2D(name='avg_pool')(model.output)
x=BatchNormalization()(x)

top_dropout_rate=0.2
x=Dropout(top_dropout_rate, name='top_dropout')(x)
outputs=Dense(5, activation='softmax', name='pred')(x)

new_model = Model(inputs = model.input, outputs = outputs, name='efficientnet')
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2)
new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']
              )
print(len(model.layers))
print(len(new_model.layers))

for layer in new_model.layers[:237] :
    layer.trainable = False

op=optimizers.Adam(learning_rate=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)
history = new_model.fit_generator(generator=train_generator, epochs=200, validation_data=test_generator,callbacks=[es,mc])



#### 여러 모델의 train acc 합친 plot ####

from matplotlib import pyplot as plt

acc1=h1.history['acc']
acc2=h2.history['accuracy']
acc3=h3.history['accuracy']
acc4=h4.history['accuracy']
acc5=h5.history['accuracy']
acc6=h6.history['accuracy']
acc7=h7.history['accuracy']
acc8=h8.history['accuracy']

epochs1 = range(len(acc1))
plt.plot(epochs1, acc1, 'r', label='SimpleCNN')

epochs2 = range(len(acc2))
plt.plot(epochs2, acc2, 'b', label='VGG-16')

epochs3 = range(len(acc3))
plt.plot(epochs3, acc3, 'orange', label='ResNet50')

epochs4 = range(len(acc4))
plt.plot(epochs4, acc4, 'm', label='VGG-19')

epochs5 = range(len(acc5))
plt.plot(epochs5, acc5, 'pink', label='InceptionV3')

epochs6 = range(len(acc6))
plt.plot(epochs6, acc6, 'purple',label='DenseNet121')

epochs7 = range(len(acc7))
plt.plot(epochs7, acc7, 'g',label='MobileNetV3Large')

epochs8 = range(len(acc8))
plt.plot(epochs8, acc8, 'gray', label='NasNetLarge')

plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Train accuracy', fontsize=13)

plt.legend()
plt.show()
plt.clf()

#### 여러 모델의 val acc 합친 plot ####

acc1=h1.history['val_acc']
acc2=h2.history['val_accuracy']
acc3=h3.history['val_accuracy']
acc4=h4.history['val_accuracy']
acc5=h5.history['val_accuracy']
acc6=h6.history['val_accuracy']
acc7=h7.history['val_accuracy']
acc8=h8.history['val_accuracy']

epochs1 = range(len(acc1))
plt.plot(epochs1, acc1, 'r', label='SimpleCNN')

epochs2 = range(len(acc2))
plt.plot(epochs2, acc2, 'b', label='VGG-16')

epochs3 = range(len(acc3))
plt.plot(epochs3, acc3, 'orange', label='ResNet50')

epochs4 = range(len(acc4))
plt.plot(epochs4, acc4, 'm', label='VGG-19')

epochs5 = range(len(acc5))
plt.plot(epochs5, acc5, 'pink', label='InceptionV3')

epochs6 = range(len(acc6))
plt.plot(epochs6, acc6, 'purple',label='DenseNet121')

epochs7 = range(len(acc7))
plt.plot(epochs7, acc7, 'g',label='MobileNetV3Large')

epochs8 = range(len(acc8))
plt.plot(epochs8, acc8, 'gray', label='NasNetLarge')

plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Test accuracy', fontsize=13)

plt.legend()
plt.show()
plt.clf()

#### 여러 모델의 val loss 합친 plot ####

acc1=h1.history['val_loss']
acc2=h2.history['val_loss']
acc3=h3.history['val_loss']
acc4=h4.history['val_loss']
acc5=h5.history['val_loss']
acc6=h6.history['val_loss']
acc7=h7.history['val_loss']
acc8=h8.history['val_loss']

epochs1 = range(len(acc1))
plt.plot(epochs1, acc1, 'r', label='SimpleCNN')

epochs2 = range(len(acc2))
plt.plot(epochs2, acc2, 'b', label='VGG-16')

epochs3 = range(len(acc3))
plt.plot(epochs3, acc3, 'orange', label='ResNet50')

epochs4 = range(len(acc4))
plt.plot(epochs4, acc4, 'm', label='VGG-19')

epochs5 = range(len(acc5))
plt.plot(epochs5, acc5, 'pink', label='InceptionV3')

epochs6 = range(len(acc6))
plt.plot(epochs6, acc6, 'purple',label='DenseNet121')

epochs7 = range(len(acc7))
plt.plot(epochs7, acc7, 'g',label='MobileNetV3Large')

epochs8 = range(len(acc8))
plt.plot(epochs8, acc8, 'gray', label='NasNetLarge')

plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Test loss', fontsize=13)

plt.legend()
plt.show()
plt.clf()

#### 여러 모델의 train loss 합친 plot ####

acc1=h1.history['loss']
acc2=h2.history['loss']
acc3=h3.history['loss']
acc4=h4.history['loss']
acc5=h5.history['loss']
acc6=h6.history['loss']
acc7=h7.history['loss']
acc8=h8.history['loss']

epochs1 = range(len(acc1))
plt.plot(epochs1, acc1, 'r', label='SimpleCNN')

epochs2 = range(len(acc2))
plt.plot(epochs2, acc2, 'b', label='VGG-16')

epochs3 = range(len(acc3))
plt.plot(epochs3, acc3, 'orange', label='ResNet50')

epochs4 = range(len(acc4))
plt.plot(epochs4, acc4, 'm', label='VGG-19')

epochs5 = range(len(acc5))
plt.plot(epochs5, acc5, 'pink', label='InceptionV3')

epochs6 = range(len(acc6))
plt.plot(epochs6, acc6, 'purple',label='DenseNet121')

epochs7 = range(len(acc7))
plt.plot(epochs7, acc7, 'g',label='MobileNetV3Large')

epochs8 = range(len(acc8))
plt.plot(epochs8, acc8, 'gray', label='NasNetLarge')

plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Train loss', fontsize=13)

plt.legend()
plt.show()
plt.clf()



#### 여러 모델을 결합한 모델 만들기 version 1 ####

from keras.models import Sequential
from keras.layers.merge import Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D

input_tensor = Input(shape=(150,150,3))
model1 = VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)
model2 = VGG19(weights='imagenet', include_top=False, input_tensor = input_tensor)

for layer in model1.layers:
    layer._name=layer._name+str('_16')
for layer in model2.layers:
    layer._name=layer._name+str('_19')
    
merge1=Concatenate()([model1.layers[-1].output, model2.layers[-1].output])

x=GlobalAveragePooling2D()(merge1)
x = Dropout(0.2)(x)
x = Dense(5, activation='softmax')(x)
new_model = Model(inputs = model1.input, outputs = x)
new_model.summary()

print(len(model1.layers))
print(len(model2.layers))

for layer in model1.layers[:19] :
    layer.trainable = False
for layer in model2.layers[:22] :
    layer.trainable = False

new_model.summary()

op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)

h9 = new_model.fit_generator(generator=train_generator, epochs=400, validation_data=test_generator,callbacks=[es,mc])



#### 여러 모델을 결합한 모델 만들기 version 2 ####

from keras.models import Sequential
from keras.layers.merge import Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation

input_tensor = Input(shape=(150,150,3))
model1 = VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)
model2 = VGG19(weights='imagenet', include_top=False, input_tensor = input_tensor)

for layer in model1.layers:
  layer._name=layer._name+str('_16')
for layer in model2.layers:
  layer._name=layer._name+str('_19')
  
x=Concatenate()([model1.layers[-1].output*0.6, model2.layers[-1].output*0.4])
x = Dense(units = 1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(units = 5, activation = 'softmax')(x)
new_model = Model(model1.input, x)

print(len(model1.layers))
print(len(model2.layers))

for layer in model1.layers[:19] :
    layer.trainable = False
for layer in model2.layers[:22] :
    layer.trainable = False

new_model.summary()

op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)
h10 = new_model.fit_generator(generator=train_generator, epochs=400, validation_data=test_generator,callbacks=[es,mc])



#### VGG16 + DenseNet 결합한 모델 만들기 ####

from tensorflow.keras.applications.densenet import DenseNet121
from keras.models import Sequential
from keras.layers.merge import Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation

input_tensor = Input(shape=(150,150,3))
model1 = VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)
model2 = DenseNet121(weights='imagenet', include_top=False, input_tensor = input_tensor)

x=Concatenate()([model1.layers[-1].output*0.8, model2.layers[-1].output*0.2])
x = Dense(units = 1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(units = 5, activation = 'softmax')(x)
new_model = Model(model1.input, x)

print(len(model1.layers))
print(len(model2.layers))

for layer in model1.layers[:19] :
    layer.trainable = False
for layer in model2.layers[:427] :
    layer.trainable = False

new_model.summary()

op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)
h10 = new_model.fit_generator(generator=train_generator, epochs=400, validation_data=test_generator,callbacks=[es,mc])



#### VGG19 + DenseNet 결합한 모델 만들기 ####

from tensorflow.keras.applications.densenet import DenseNet121
from keras.models import Sequential
from keras.layers.merge import Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation

input_tensor = Input(shape=(150,150,3))
model1 = VGG19(weights='imagenet', include_top=False, input_tensor = input_tensor)
model2 = DenseNet121(weights='imagenet', include_top=False, input_tensor = input_tensor)

x=Concatenate()([model1.layers[-1].output*0.9, model2.layers[-1].output*0.1])
x = Dense(units = 1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(units = 5, activation = 'softmax')(x)
new_model = Model(model1.input, x)

print(len(model1.layers))
print(len(model2.layers))

for layer in model1.layers[:22] :
    layer.trainable = False
for layer in model2.layers[:427] :
    layer.trainable = False

new_model.summary()

op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)
h10 = new_model.fit_generator(generator=train_generator, epochs=400, validation_data=test_generator,callbacks=[es,mc])



#### VGG16 + VGG19 + DenseNet 결합한 모델 만들기 ####

from tensorflow.keras.applications.densenet import DenseNet121
from keras.models import Sequential
from keras.layers.merge import Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation

input_tensor = Input(shape=(150,150,3))
model1 = VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)
model2 = VGG19(weights='imagenet', include_top=False, input_tensor = input_tensor)
model3 = DenseNet121(weights='imagenet', include_top=False, input_tensor = input_tensor)

for layer in model1.layers:
  layer._name=layer._name+str('_16')
for layer in model2.layers:
  layer._name=layer._name+str('_19')

x=Concatenate()([model1.layers[-1].output, model2.layers[-1].output,  model3.layers[-1].output])
x = Dense(units = 1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(units = 5, activation = 'softmax')(x)
new_model = Model(model1.input, x)

print(len(model1.layers))
print(len(model2.layers))
print(len(model3.layers))

for layer in model1.layers[:19] :
    layer.trainable = False
for layer in model2.layers[:22] :
    layer.trainable = False
for layer in model2.layers[:427] :
    layer.trainable = False

new_model.summary()

op=optimizers.Adam(lr=0.0001)

new_model.compile(loss='categorical_crossentropy',
                     optimizer=op,
                     metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc=ModelCheckpoint('best_model4.h5', monitor='val_loss', mode='min', save_best_only=True)
h10 = new_model.fit_generator(generator=train_generator, epochs=400, validation_data=test_generator,callbacks=[es,mc])
