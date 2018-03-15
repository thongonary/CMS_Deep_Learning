import sys,types,os,glob
#os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
from six import string_types
#import site
#sys.path = [site.USER_BASE] + sys.path
if __package__ is None:
    sys.path.append(os.path.realpath("/nfshome/thong/CMS_Deep_Learning/"))


from CMS_Deep_Learning.postprocessing.analysistools import findsubsets
from CMS_Deep_Learning.layers.standardize import Standardize
from CMS_Deep_Learning.io import nb_samples_from_h5, gen_from_data
from CMS_Deep_Learning.preprocessing.pandas_to_numpy import PARTICLE_OBSERVS

from keras.models import Model
from keras.layers import Dense, Dropout, merge, Input, LSTM, Masking,GRU, Merge, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers

def build_GRU_HLF_model(name, input_width,out_width, depth,recurrent_width, lstm_activation="tanh", lstm_dropout = 0.0,
                     dropout=0.0, output_activation="softmax", single_list=False, l1_reg=0.0, **kargs):
    inputs = []
    a = Input(shape=(None , input_width), name="Particle Sequence")
    inputs.append(a)
    for i in range(depth):
        a = Masking(mask_value=0.0)(a)
        a = Standardize("/bigdata/shared/Delphes/np_datasets_3_5_2018/3_way/MaxLepDeltaR_des/std_stats.h5")(a)
        a = GRU(recurrent_width,
                 input_shape=(None, input_width),
                 dropout_W=lstm_dropout,
                 dropout_U=lstm_dropout,
                 activation=lstm_activation,
                 #implementation=2,
                 W_regularizer=regularizers.l1(l1_reg),
                 )(a)
        if(dropout > 0.0):
            a =  Dropout(dropout)(a)
    dense_out = Dense(recurrent_width, activation="relu")(a)
    aHLF = Input(shape=(14,), name = "HLF")
    inputs.append(aHLF)
    aHLF = Standardize("/bigdata/shared/Delphes/np_datasets_3_5_2018/3_way/MaxLepDeltaR_des/std_stats.h5", take_particles=False, take_HLF=True)(aHLF)
    aHLF = Dense(14, activation='relu')(aHLF)
    aHLF = Concatenate(axis=1)([dense_out, aHLF])
    if(dropout > 0.0):
        aHLF =  Dropout(dropout)(aHLF)
    dense_final = Dense(out_width, activation=output_activation)(aHLF)
    model = Model(input=inputs, output=dense_final, name=name)
    return model


DIR = '/bigdata/shared/Delphes/np_datasets_3_5_2018/3_way/'
train_dir = DIR +"MaxLepDeltaR_des/train"
val_dir = DIR +"MaxLepDeltaR_des/val"
print(glob.glob(train_dir))
train_samples = sum([nb_samples_from_h5(x) for x in glob.glob(train_dir+"/*.h5")])
val_samples = sum([nb_samples_from_h5(x) for x in glob.glob(val_dir+"/*.h5")])
print(train_samples, val_samples)


# training set:
train = gen_from_data(train_dir, batch_size=150, data_keys=[["Particles","HLF"], "Labels"])

# validation set:
val = gen_from_data(val_dir, batch_size=150, data_keys=[["Particles","HLF"], "Labels"])

earlyStopping = EarlyStopping(verbose=1, patience=8)
checkpoint = ModelCheckpoint(filepath='simple_GRU_HLF_dropout.h5', save_best_only=True, verbose=1)
reduceLR = ReduceLROnPlateau(patience=3, verbose=1)

model = build_GRU_HLF_model(name='GRU_HLF',
                         input_width=len(PARTICLE_OBSERVS),
                         out_width=3,
                         depth=1,
                         recurrent_width=50,
                         dropout=0.5,
                         lstm_activation='tanh'
                       )
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])

from keras.utils import plot_model
plot_model(model, to_file='gru_hlf.png', show_shapes=True, show_layer_names=False)
