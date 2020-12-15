import os, re, subprocess, pickle
import numpy as np 
import scipy 
import scipy.signal as signal
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras import layers
import pandas as pd 
import h5py 
from tqdm import tqdm
import sklearn.metrics

def DNN_model(dnn_sizes,input_shape,dropout=0.1):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))  
    model.add(layers.Flatten())
    for ii in range(len(dnn_sizes)-1):
        model.add(layers.Dense(dnn_sizes[ii], activation='relu'))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dnn_sizes[-1], activation='softmax'))
    return model

def CNN_model(filters,kernel_sizes,input_shape,pool_size,dnn_sizes,dropout=0.1):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    for ii in range(len(filters)):
        model.add(layers.Conv2D(filters[ii],kernel_sizes[ii], activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=pool_size))
        model.add(layers.Dropout(dropout))
    model.add(layers.Flatten())
    for ii in range(len(dnn_sizes)-1):
        model.add(layers.Dense(dnn_sizes[ii],activation="relu"))
    model.add(layers.Dense(dnn_sizes[-1],activation="softmax"))
    return model

def DSCNN_model(multipliers,kernel_sizes,input_shape,pool_size,dnn_sizes,dropout=0.1):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    for ii in range(len(kernel_sizes)):
        model.add(layers.DepthwiseConv2D(kernel_sizes[ii], depth_multiplier=multipliers[ii], activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=pool_size))
        model.add(layers.Dropout(dropout))
    model.add(layers.Flatten())
    for ii in range(len(dnn_sizes)-1):
        model.add(layers.Dense(dnn_sizes[ii],activation="relu"))
    model.add(layers.Dense(dnn_sizes[-1],activation="softmax"))
    return model

def representative_dataset():
  for data in testdata.batch(1).take(100):
    yield [data.astype(tf.float32)]

if __name__ == "__main__": 

    # load training data 
    DATASET_SIZE = 250980
    batch_size = 256 
    DATASET_SIZE //= batch_size
    train_size = int(0.8 * DATASET_SIZE)
    val_size = int(0.2 * DATASET_SIZE)
    waves = tf.data.experimental.load(
        "/home/timclements/CS249FINAL/data/waves/train",
        element_spec=(tf.TensorSpec(shape=(None, 80, 3, 1),
        dtype=tf.float32, 
        name=None),
        tf.TensorSpec(shape=(None, 3), 
        dtype=tf.float32, 
        name=None,))
    )
    waves = waves.shuffle(DATASET_SIZE)
    val_dataset = waves.take(val_size)
    train_dataset = waves.skip(val_size)

    # load test data 
    testdata = np.load("/home/timclements/CS249FINAL/data/test.npz")
    Xtest = testdata["Xtest"]
    Ytest = testdata["Ytest"]
    truth = np.argmax(Ytest,axis=-1)

    # directories for tflite and header files 
    DNNDIR = "/home/timclements/CS249FINAL/DNN"
    CNNDIR = "/home/timclements/CS249FINAL/CNN"
    DSCNNDIR = "/home/timclements/CS249FINAL/DSCNN"
    DNNMODELDIR = os.path.join(DNNDIR,"models")
    DNNHEADERDIR = os.path.join(DNNDIR,"headers")
    CNNMODELDIR = os.path.join(CNNDIR,"models")
    CNNHEADERDIR = os.path.join(CNNDIR,"headers")
    DSCNNMODELDIR = os.path.join(DSCNNDIR,"models")
    DSCNNHEADERDIR = os.path.join(DSCNNDIR,"headers")
    for DIR in [DNNMODELDIR,DNNHEADERDIR,CNNMODELDIR,CNNHEADERDIR,DSCNNMODELDIR,DSCNNHEADERDIR]:
        if not os.path.isdir(DIR): 
            os.makedirs(DIR)

    # create + train DNN models 
    input_shape = (80,3,1)
    model_sizes = [(2**ii,2**jj,3) for ii in range(3,7) for jj in range(3,ii+1)]
    models = {}
    for ii in range(len(model_sizes)):
        model_dict = {}
        model = DNN_model(model_sizes[ii],input_shape)
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
        history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)
        model_dict["params"] = model.count_params()
        model_dict["history"] = history.history
        model_dict["model_tf_size"] = model_sizes[ii]

        # get precision, recall, f1-score, support
        preds = model.predict(Xtest)
        outpred = np.argmax(preds,axis=-1)
        model_dict["report"] = sklearn.metrics.classification_report(
            truth,
            outpred,
            target_names=CLASSES,
            output_dict=True
        )

        # get confusion matrix 
        model_dict["confusion"] = tf.math.confusion_matrix(truth,outpred)

        # Convert the model to the TensorFlow Lite format without quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        model_name = os.path.join(DNNMODELDIR,"model_{}_{}_{}_{}.tflite".format(ii,*model_sizes[ii]))
        model_header = os.path.join(DNNHEADERDIR,"model_{}_{}_{}_{}.h".format(ii,*model_sizes[ii]))

        # Save the model to disk
        open(model_name, "wb").write(tflite_model)
        model_dict["model_size"] = os.path.getsize(model_name)

        # save header to disk 
        subprocess.call('echo "const unsigned char model[] __attribute__((aligned(4))) = {{"  > {}'.format(model_header),shell=True)
        subprocess.call("cat {} | xxd -i >> {}".format(model_name,model_header),shell=True)
        subprocess.call('echo "}};" >> {}'.format(model_header),shell=True)

        # get header size 
        model_dict["model_header_size"] = os.path.getsize(model_header)
        models[ii] = model_dict

    # write model history to disk 
    with open('/home/timclements/CS249FINAL/DNN_models.pkl', 'wb') as file:
        pickle.dump(models, file)

    ## train CNN models
    filter_sizes = [(2**ii,2**jj) for ii in range(1,6) for jj in range(ii,ii+3)]
    kernel_sizes = [[(3,1),(3,1)] for ii in range(len(filter_sizes))]
    pool_size = (3,1)
    dnn_sizes = (16,3)
    models = {}
    for ii in range(len(filter_sizes)):
        model_dict = {}
        model = CNN_model(filter_sizes[ii],kernel_sizes[ii],input_shape,pool_size,dnn_sizes)
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
        history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)
        model_dict["params"] = model.count_params()
        model_dict["history"] = history.history
        model_dict["filter_size"] = filter_sizes[ii]
        model_dict["kernel_size"] = kernel_sizes[ii]
        model_dict["pool_size"] = pool_size

        # get precision, recall, f1-score, support
        preds = model.predict(Xtest)
        outpred = np.argmax(preds,axis=-1)
        model_dict["report"] = sklearn.metrics.classification_report(
            truth,
            outpred,
            target_names=CLASSES,
            output_dict=True
        )

        # get confusion matrix 
        model_dict["confusion"] = tf.math.confusion_matrix(truth,outpred)

        # Convert the model to the TensorFlow Lite format without quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        model_name = os.path.join(CNNMODELDIR,"model_{}_{}_{}.tflite".format(ii,*filter_sizes[ii]))
        model_header = os.path.join(CNNHEADERDIR,"model_{}_{}_{}.h".format(ii,*filter_sizes[ii]))

        # Save the model to disk
        open(model_name, "wb").write(tflite_model)
        model_dict["model_size"] = os.path.getsize(model_name)

        # save header to disk 
        subprocess.call('echo "const unsigned char model[] __attribute__((aligned(4))) = {{"  > {}'.format(model_header),shell=True)
        subprocess.call("cat {} | xxd -i >> {}".format(model_name,model_header),shell=True)
        subprocess.call('echo "}};" >> {}'.format(model_header),shell=True)

        # get header size 
        model_dict["model_header_size"] = os.path.getsize(model_header)
        models[ii] = model_dict

    # write model history to disk 
    with open('/home/timclements/CS249FINAL/CNN_models.pkl', 'wb') as file:
        pickle.dump(models, file)

    ## train DS-CNN models
    multipliers = [(2**ii,2**jj) for ii in range(1,5) for jj in range(0,4)][:-1]
    kernel_sizes = [[(3,1),(3,1)] for ii in range(len(multipliers))]
    pool_size = (3,1)
    dnn_sizes = (16,3)
    models = {}
    for ii in range(len(filter_sizes)):
        model_dict = {}
        model = DSCNN_model(multipliers[ii],kernel_sizes[ii],input_shape,pool_size,dnn_sizes)
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
        history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)
        model_dict["params"] = model.count_params()
        model_dict["history"] = history.history
        model_dict["multipliers"] = multipliers[ii]
        model_dict["kernel_size"] = kernel_sizes[ii]
        model_dict["pool_size"] = pool_size

        # get precision, recall, f1-score, support
        preds = model.predict(Xtest)
        outpred = np.argmax(preds,axis=-1)
        model_dict["report"] = sklearn.metrics.classification_report(
            truth,
            outpred,
            target_names=CLASSES,
            output_dict=True
        )

        # get confusion matrix 
        model_dict["confusion"] = tf.math.confusion_matrix(truth,outpred)

        # Convert the model to the TensorFlow Lite format without quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        model_name = os.path.join(DSCNNMODELDIR,"model_{}_{}_{}.tflite".format(ii,*multipliers[ii]))
        model_header = os.path.join(DSCNNHEADERDIR,"model_{}_{}_{}.h".format(ii,*multipliers[ii]))

        # Save the model to disk
        open(model_name, "wb").write(tflite_model)
        model_dict["model_size"] = os.path.getsize(model_name)

        # save header to disk 
        subprocess.call('echo "const unsigned char model[] __attribute__((aligned(4))) = {{"  > {}'.format(model_header),shell=True)
        subprocess.call("cat {} | xxd -i >> {}".format(model_name,model_header),shell=True)
        subprocess.call('echo "}};" >> {}'.format(model_header),shell=True)

        # get header size 
        model_dict["model_header_size"] = os.path.getsize(model_header)
        models[ii] = model_dict

    # write model history to disk 
    with open('/home/timclements/CS249FINAL/DSCNN_models.pkl', 'wb') as file:
        pickle.dump(models, file)

    # train final model with DS-CNN
    multipliers = [4,2,2,2]
    kernel_sizes = [(3,1),(3,1),(3,1),(3,3)]
    pool_size = (2,1)
    dnn_sizes = (32,3)






# # CNN model  
# inputs = keras.Input(shape=(200,3,1))
# x = layers.Conv2D(filters=32, kernel_size=(5, 1), activation="relu")(inputs)
# x = layers.MaxPooling2D(pool_size=(3,1))(x)
# x = layers.Dropout(0.1)(x)
# x = layers.Conv2D(filters=32, kernel_size=(3, 1), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(3,1))(x)
# x = layers.Dropout(0.1)(x)
# x = layers.Conv2D(filters=32, kernel_size=(3, 1), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(3,1))(x)
# x = layers.Dropout(0.1)(x)
# x = layers.Flatten()(x)
# x = layers.Dense(16,activation="relu")(x)
# # Apply global average pooling to get flat feature vectors
# outputs = layers.Dense(3, activation="softmax")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
# history = model.fit(dataset, epochs=1, validation_data=val_dataset)

# # DS-CNN model 
# model = keras.Sequential()
# model.add(keras.Input(shape=(200,3,1)))  
# model.add(layers.Conv2D(16,(3,1), activation="relu"))
# model.add(layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(3,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(3,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(3,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.Flatten())
# model.add(layers.Dense(16,activation="relu"))
# model.add(layers.Dense(3,activation="softmax"))
# model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)
# # % validation after 100 epochs 
# # 3,563 parameters


# ### CNN model 16 - 16 - 32 - 32 - 64 - 64 - Dense(16) - Dense(3)
# model = keras.Sequential()
# model.add(keras.Input(shape=(80,3,1)))  
# model.add(layers.Conv2D(16,(3,1), activation="relu"))
# model.add(layers.Conv2D(16, (5, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(2,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.Conv2D(32, (5, 1), activation="relu"))
# model.add(layers.Conv2D(32, (5, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(3,3)))
# model.add(layers.Dropout(0.1))
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation="relu"))
# model.add(layers.Dense(3,activation="softmax"))
# model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)
# # 98.2% validation after 100 epochs 
# # 39,507 parameters 

# ### CNN model 16 - 16 - 32 - 32 - Dense(16) - Dense(3)
# model = keras.Sequential()
# model.add(keras.Input(shape=(200,3,1)))  
# model.add(layers.Conv2D(16,(5,1), activation="relu"))
# model.add(layers.Conv2D(16, (7, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(5,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.Conv2D(32, (7, 1), activation="relu"))
# model.add(layers.Conv2D(32, (7, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(3,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation="relu"))
# model.add(layers.Dense(3,activation="softmax"))
# model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)
# # 98.5% validation after 100 epochs 
# # 62,131 parameters 

# ### CNN model 16 - 16 - 32 - 32 - Dense(16) - Dense(3)
# model = keras.Sequential()
# model.add(keras.Input(shape=(200,3,1)))  
# model.add(layers.Conv2D(16,(5,1), activation="relu"))
# model.add(layers.Conv2D(16, (5, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(5,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.Conv2D(32, (3, 1), activation="relu"))
# model.add(layers.Conv2D(32, (3, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(3,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.Flatten())
# model.add(layers.Dense(16,activation="relu"))
# model.add(layers.Dense(3,activation="softmax"))
# model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)
# # 96.9% validation after 100 epochs 
# # 23,027 parameters 

# ### CNN model 16 - 16 - 32 - 32 - 64 - 64 - Dense(16) - Dense(3)
# model = keras.Sequential()
# model.add(keras.Input(shape=(200,3,1)))  
# model.add(layers.Conv2D(16,(3,1), activation="relu"))
# model.add(layers.Conv2D(16, (3, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(3,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.Conv2D(32, (3, 1), activation="relu"))
# model.add(layers.Conv2D(32, (3, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(3,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.Conv2D(64, (3, 1), activation="relu"))
# model.add(layers.Conv2D(64, (3, 1), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(3,1)))
# model.add(layers.Dropout(0.1))
# model.add(layers.Flatten())
# model.add(layers.Dense(16,activation="relu"))
# model.add(layers.Dense(3,activation="softmax"))
# model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)
# # ?% validation after 100 epochs 
# # 39,507 parameters 

# ### DNN model 45 - 35 - 25 - 15 - 3 
# model = keras.Sequential()
# model.add(layers.Flatten())
# model.add(layers.Dense(45, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(35, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(25, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(15, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(3, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset,epochs=100,validation_data=val_dataset)
# # 91% validation after 100 epochs 
# # 29,993 parameters 


# ### DNN model 64 - 16  - 3 
# model = keras.Sequential()
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(3, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset,epochs=100,validation_data=val_dataset)
# # 91% on validation after 100 epochs
# # 39,555 trainable parameters 

# ### DNN model 128 - 128  - 3 
# model = keras.Sequential()
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(3, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset,epochs=100,validation_data=val_dataset)
# # 97% on validation after 100 epochs
# # 93,827 trainable parameters 

# ### DNN model 256 - 128 - 64 - 32 - 16  - 3 
# model = keras.Sequential()
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(3, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset,epochs=100,validation_data=val_dataset)
# # 99% on validation after 100 epochs
# # 197,667 trainable parameters 

# ### DNN model 32 - 32 - 32 - 32 - 32  - 3 
# model = keras.Sequential()
# model.add(layers.Flatten())
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(3, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset,epochs=100,validation_data=val_dataset)
# # 87% on validation after 100 epochs
# # 23,555 trainable parameters 

# ### DNN model 512 - 256 - 3 
# model = keras.Sequential()
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(3, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae', 'accuracy'])
# history = model.fit(train_dataset,epochs=100,validation_data=val_dataset)
# # 99.7% on validation after 100 epochs
# # 439,811 trainable parameters 
