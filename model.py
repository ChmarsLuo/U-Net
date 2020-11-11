import keras
from keras import layers
from keras.models import *
from keras.layers import *
from data import *
from keras.models import load_model
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

def U_NET(num_classes,input_shape=(256,256,3)):
    inputs = layers.Input(shape=input_shape)
    conv1_1 = layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",kernel_initializer="he_normal",activation="relu")(inputs)
    conv1_2 = layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",kernel_initializer="he_normal",activation="relu")(conv1_1)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv1_2)

    conv2_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(pool1)
    conv2_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(conv2_1)
    pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv2_2)

    conv3_1 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(pool2)
    conv3_2 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(conv3_1)
    pool3 = layers.MaxPooling2D(pool_size=(2,2))(conv3_2)

    conv4_1 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(pool3)
    conv4_2 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(conv4_1)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4_2)

    conv5_1 = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(pool4)
    conv5_2 = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(conv5_1)

    deconv6_up = layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",kernel_initializer="he_normal",activation="relu")(layers.UpSampling2D((2,2))(conv5_2))
    merge6 = layers.concatenate([conv4_2,deconv6_up],axis = 3)
    deconv6_1 = layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",kernel_initializer="he_normal",activation="relu")(merge6)
    deconv6_2 = layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",kernel_initializer="he_normal",activation="relu")(deconv6_1)

    deconv7_up = layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",kernel_initializer="he_normal",activation="relu")(layers.UpSampling2D((2,2))(deconv6_2))
    merge7 = layers.concatenate([conv3_2,deconv7_up],axis = 3)
    deconv7_1 = layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",kernel_initializer="he_normal",activation="relu")(merge7)
    deconv7_2 = layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",kernel_initializer="he_normal",activation="relu")(deconv7_1)

    deconv8_up = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(layers.UpSampling2D((2, 2))(deconv7_2))
    merge8 = layers.concatenate([conv2_2, deconv8_up],axis = 3)
    deconv8_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", activation="relu")(merge8)
    deconv8_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(deconv8_1)

    deconv9_up = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(layers.UpSampling2D((2, 2))(deconv8_2))
    merge9 = layers.concatenate([conv1_2, deconv9_up],axis = 3)
    deconv9_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(merge9)
    deconv9_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="relu")(deconv9_1)
    ###########num_classes的值根据有多少类别决定
    ###########激活函数sigmoid，因为labels是用one_hot编码
    outputs = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(deconv9_2)  # 我怀疑这个sigmoid激活函数是多余的，因为在后面的loss中用到的就是二进制交叉熵，包含了sigmoid

    #outputs = layers.Conv2D(filters=num_classes, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",activation="sigmoid")(conv10)


    model = Model(inputs=inputs,outputs=outputs)

    #model = U_netModel(2)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model
