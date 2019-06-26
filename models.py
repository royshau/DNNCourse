# from keras_vggface.vggface import VGGFace
from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply,add,\
    Subtract,Multiply,Concatenate,Dropout,GlobalMaxPool2D,GlobalAvgPool2D

from tensorflow.keras.backend import sqrt,sum,square,maximum
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
import warnings
import tensorflow as tf
RESNET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5'

'''VGGFace models for Keras.

# Notes:
- Resnet50 and VGG16  are modified architectures from Keras Application folder. [Keras](https://keras.io)

- Squeeze and excitation block is taken from  [Squeeze and Excitation Networks in
 Keras](https://github.com/titu1994/keras-squeeze-excite-network) and modified.

'''

from keras_vggface import utils


def resnet_identity_block(input_tensor, kernel_size, filters, stage, block,
                          bias=False):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, use_bias=bias,
               padding='same', name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block,
                      strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias,
                      name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(
        shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def RESNET50(include_top=True, weights='vggface',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=8631):

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)
    bn_axis = 3


    x = Conv2D(
        64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
        name='conv1/7x7_s2')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vggface_resnet50')

    # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_resnet50.h5',
                                    utils.RESNET50_WEIGHTS_PATH,
                                    cache_subdir=utils.VGGFACE_DIR)
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_resnet50.h5',RESNET50_WEIGHTS_PATH_NO_TOP)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model



def senet_se_block(input_tensor, stage, block, compress_rate=16, bias=False):
    conv1_down_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_down"
    conv1_up_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_up"

    num_channels = int(input_tensor.shape[-1])
    bottle_neck = int(num_channels // compress_rate)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, num_channels))(se)
    se = Conv2D(bottle_neck, (1, 1), use_bias=bias,
                name=conv1_down_name)(se)
    se = Activation('relu')(se)
    se = Conv2D(num_channels, (1, 1), use_bias=bias,
                name=conv1_up_name)(se)
    se = Activation('sigmoid')(se)

    x = input_tensor
    x = multiply([x, se])
    return x


def senet_conv_block(input_tensor, kernel_size, filters,
                     stage, block, bias=False, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = 3


    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, strides=strides,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    se = senet_se_block(x, stage=stage, block=block, bias=True)

    shortcut = Conv2D(filters3, (1, 1), use_bias=bias, strides=strides,
                      name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis,
                                  name=conv1_proj_name + "/bn")(shortcut)

    m = add([se, shortcut])
    m = Activation('relu')(m)
    return m


def senet_identity_block(input_tensor, kernel_size,
                         filters, stage, block, bias=False):
    filters1, filters2, filters3 = filters
    bn_axis = 3


    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    se = senet_se_block(x, stage=stage, block=block, bias=True)

    m = add([x, se])
    m = Activation('relu')(m)

    return m


def SENET50(include_top=True, weights='vggface',
            input_tensor=None, input_shape=None,
            pooling=None,
            classes=8631):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)
    bn_axis = 3

    x = Conv2D(
        64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
        name='conv1/7x7_s2')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = senet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = senet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = senet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = senet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = senet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = senet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = senet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = senet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = senet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = senet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = senet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vggface_senet50')
    SENET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_senet50.h5'

    # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_senet50.h5',
                                    utils.SENET50_WEIGHTS_PATH,
                                    cache_subdir=utils.VGGFACE_DIR)
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_senet50.h5',
                                    SENET50_WEIGHTS_PATH_NO_TOP)
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model

def classification_model(base_model=RESNET50(include_top=False, weights='vggface', input_shape=(None, None, 3))):
    input_1 = Input(shape=(None, None, 3))
    input_2 = Input(shape=(None, None, 3))

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    # x1_ = Reshape(target_shape=(7*7, 2048))(x1)
    # x2_ = Reshape(target_shape=(7*7, 2048))(x2)
    #
    # x_dot = Dot(axes=[2, 2], normalize=True)([x1_, x2_])
    # x_dot = Flatten()(x_dot)


    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x = Multiply()([x1, x2])
    x = Concatenate(axis=-1)([x, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model([input_1, input_2], out)

    return model


def triplet_loss(y_true, y_pred, alpha=400):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    #     print((y_pred[0]))
    N = y_pred.shape[1] // 3
    anchor = y_pred[:, 0:N]
    good = y_pred[:, N:N * 2]
    bad = y_pred[:, N * 2:N * 3]

    # distance between the anchor and the positive
    pos_dist = sqrt(sum(square(anchor - good), axis=1) + .01)

    # distance between the anchor and the negative
    neg_dist = sqrt(sum(square(anchor - bad), axis=1) + .01)

    # compute loss
    basic_loss = (pos_dist - neg_dist + alpha)
    loss = maximum(basic_loss, 0.0)

    return loss

def triplet_metric(y_true, y_pred):
    """

    """
    #     print((y_pred[0]))
    N = y_pred.shape[1] // 3
    anchor = y_pred[:, 0:N]
    good = y_pred[:, N:N * 2]
    bad = y_pred[:, N * 2:N * 3]

    # distance between the anchor and the positive
    pos_dist = sqrt(sum(square(anchor - good), axis=1) + .01)

    # distance between the anchor and the negative
    neg_dist = sqrt(sum(square(anchor - bad), axis=1) + .01)

    # compute loss
    basic_loss = (pos_dist - neg_dist)
    return basic_loss


def triplet_loss_model(base_model=RESNET50(include_top=False, weights='vggface', input_shape=(None, None, 3))):
    anchor = Input(shape=(None, None, 3))
    good = Input(shape=(None, None, 3))
    bad = Input(shape=(None, None, 3))

    anchor_emb = base_model(anchor)
    good_emb = base_model(good)
    bad_emb = base_model(bad)

    # x1_ = Reshape(target_shape=(7*7, 2048))(x1)
    # x2_ = Reshape(target_shape=(7*7, 2048))(x2)
    #
    # x_dot = Dot(axes=[2, 2], normalize=True)([x1_, x2_])
    # x_dot = Flatten()(x_dot)

    anchor_x = GlobalAvgPool2D()(anchor_emb)
    good_x = GlobalAvgPool2D()(good_emb)
    bad_x = GlobalAvgPool2D()(bad_emb)

    merged_embeddings = Concatenate(axis=-1)([anchor_x, good_x, bad_x])

    #
    # mse_good = Subtract()([anchor_x, good_x])
    # mse_good = Multiply()([mse_good, mse_good])
    #
    # mse_bad = Subtract()([anchor_x, bad_x])
    # mse_bad = Multiply()([mse_bad, mse_bad])
    #
    # pred_good = Dense(100, activation="relu")(mse_good)
    #
    # x = Dropout(0.01)(x)
    # out = Dense(1, activation="sigmoid")(x)
    model = Model([anchor, good,bad], merged_embeddings)

    return model

