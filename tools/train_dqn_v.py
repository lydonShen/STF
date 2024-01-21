import numpy as np
import argparse
import gym
import pdb
import os

from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam, SGD
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

from env.drl_env_v import ViewSearchEnv
from env.dqn import DQNAgent
from env.policy import BoltzmannQPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument("--config-file", default="configs/fcos/fcos_imprv_R_50_FPN_1x_DRL.yaml", metavar="FILE",)
    parser.add_argument("--weights", default="training_dir/fcos_imprv_R_50_FPN_1x_car/model_final.pth", metavar="FILE")
    parser.add_argument("--drl-weights", default="training_dir/ddqn", metavar="FILE")
    parser.add_argument("--min-image-size", type=int, default=800)
    parser.add_argument("--double", action="store_true", default=False)
    parser.add_argument("--dueling", action="store_true", default=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    # x = BatchNormalization(axis=3, name=bn_name)(x)
    return x
 
def Conv_Block(inpt, nb_filter, kernel_size, strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def get_model(env):
    inputs1 = Input(shape=env.feature_length1)
    x1 = Conv2d_BN(inputs1, nb_filter=64, kernel_size=(3, 3), padding='same')
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    x1 = Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
    x1 = Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
    x1 = Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    x1 = Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
    x1 = Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
    x1 = Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
    x1 = Flatten()(x1)
    output1 = Dense(32, activation='relu')(x1)
    model1 = Model(input=inputs1, output=output1)

    inputs2 = Input(shape=env.feature_length2)
    x2 = Conv2d_BN(inputs2, nb_filter=64, kernel_size=(3, 3), padding='same')
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
    x2 = Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
    x2 = Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
    x2 = Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
    x2 = Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
    x2 = Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
    x2 = Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
    x2 = Flatten()(x2)
    output2 = Dense(32, activation='relu')(x2)
    model2 = Model(input=inputs2, output=output2)

    inputs3 = Input(shape=env.feature_length3)
    x3 = Conv2d_BN(inputs3, nb_filter=64, kernel_size=(3, 3), padding='same')
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
    x3 = Conv_Block(x3, nb_filter=64, kernel_size=(3, 3))
    x3 = Conv_Block(x3, nb_filter=64, kernel_size=(3, 3))
    x3 = Conv_Block(x3, nb_filter=64, kernel_size=(3, 3))
    x3 = Flatten()(x3)
    output3 = Dense(32, activation='relu')(x3)
    model3 = Model(input=inputs3, output=output3)

    inputs4 = Input(shape=env.feature_length4)
    x4 = Conv2d_BN(inputs4, nb_filter=64, kernel_size=(3, 3), padding='same')
    x4 = Conv_Block(x4, nb_filter=64, kernel_size=(3, 3))
    x4 = Conv_Block(x4, nb_filter=64, kernel_size=(3, 3))
    x4 = Conv_Block(x4, nb_filter=64, kernel_size=(3, 3))
    x4 = Flatten()(x4)
    output4 = Dense(32, activation='relu')(x4)
    model4 = Model(input=inputs4, output=output4)

    inputs5 = Input(shape=env.feature_length5)
    x5 = Conv2d_BN(inputs5, nb_filter=64, kernel_size=(3, 3), padding='same')
    x5 = Conv_Block(x5, nb_filter=64, kernel_size=(3, 3))
    x5 = Conv_Block(x5, nb_filter=64, kernel_size=(3, 3))
    x5 = Conv_Block(x5, nb_filter=64, kernel_size=(3, 3))
    x5 = Flatten()(x5)
    output5 = Dense(32, activation='relu')(x5)
    model5 = Model(input=inputs5, output=output5)

    inputs6 = Input(shape=(1,) + env.feature_length6)
    x6 = Flatten()(inputs6)
    output6 = Dense(32, activation='relu')(x6)
    model6 = Model(input=inputs6, output=output6)

    inputs7 = Input(shape=(1,) + env.feature_length7)
    x7 = Flatten()(inputs7)
    output7 = Dense(32, activation='relu')(x7)
    model7 = Model(input=inputs7, output=output7)

    combined = concatenate([model1.output, model2.output, model3.output, model4.output, model5.output, model6.output, model7.output])
    x = Dense(128, activation='relu')(combined)
    output = Dense(env.action_space.n, activation='linear')(x)

    model = Model(input=[model1.input, model2.input, model3.input, model4.input, model5.input, model6.input, model7.input], output=output)
    print(model.summary())

    return model


if __name__ == '__main__':

    args = parse_args()
    env = ViewSearchEnv(args)
    model = get_model(env)

    memory = SequentialMemory(limit=20000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, policy=policy, memory=memory,
                   nb_steps_warmup=5000, gamma=.99, target_model_update=5000, env=env, 
                   enable_double_dqn=args.double, enable_dueling_network=args.dueling)
    dqn.compile(Adam(lr=0.001), metrics=['mae'])

    folder = os.path.exists(args.drl_weights)
    if not folder:
        os.makedirs(args.drl_weights)
    print("enable_double_dqn=%r, enable_dueling_network=%r" % (args.double, args.dueling))

    checkpoint_weights_filename = os.path.join(args.drl_weights, 'dqn_weights_{step}.h5f')
    log_filename = os.path.join(args.drl_weights, 'dqn_log.json')
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(dqn.env, callbacks=callbacks, nb_steps=130000, log_interval=5000)
