from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Activation,
    UpSampling2D, BatchNormalization, Concatenate
)
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt

class UNET(Model):
    def __init__(self, org_shape, n_classes=1):
        ic = -1  # channels_last 가정

        def conv(x, n_f, mp_flag=True):
            if mp_flag:
                x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.05)(x)
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x

        def deconv_unet(x, e, n_f):
            x = UpSampling2D((2, 2))(x)
            x = Concatenate(axis=ic)([x, e])
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(n_f, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x

        inputs = Input(shape=org_shape)

        # encoder
        c1 = conv(inputs, 16, mp_flag=False)
        c2 = conv(c1, 32)
        encoded = conv(c2, 64)

        # decoder
        x = deconv_unet(encoded, c2, 32)
        x = deconv_unet(x, c1, 16)

        # binary segmentation: 1채널 + sigmoid
        outputs = Conv2D(n_classes, (1, 1), activation='sigmoid', padding='same')(x)

        super().__init__(inputs, outputs)
        # binary mask 라고 가정
        self.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )