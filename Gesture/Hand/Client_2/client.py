import flwr as fl
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import keras

# print("Starting training .....")
def client():
    RANDOM_SEED = 42

    dataset = 'Gesture/Hand/Client_2/model/keypoint_classifier/keypoint.csv'
    model_save_path = 'Gesture/Hand/Client_2/model/keypoint_classifier/keypoint_classifier.hdf5'
    tflite_save_path = 'Gesture/Hand/Client_2/model/keypoint_classifier/keypoint_classifier.tflite'
    csv_path_label = 'Gesture/Hand/Client_2/model/keypoint_classifier/keypoint_classifier_label.csv'
    results = pd.read_csv(csv_path_label)
    NUM_CLASSES = len(results)+1

    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

    # Model building
    model = tf.keras.models.Sequential([
            tf.keras.layers.Input((21 * 2, )),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])

    model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)
    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    # Model compilation
    optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])



    class MnistClient(fl.client.NumPyClient):

        def get_parameters(self):
            return model.get_weights()

        def fit(self,parameters,config):
            model.set_weights(parameters)
            model.fit(X_train,y_train,epochs=1000,batch_size=128,validation_data=(X_test, y_test),callbacks=[cp_callback, es_callback])
            model_1 = tf.keras.models.load_model(model_save_path)
            predict_result = model_1.predict(np.array([X_test[0]]))
            print(np.squeeze(predict_result))
            print(np.argmax(np.squeeze(predict_result)))
            model_1.save(model_save_path, include_optimizer=False)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.experimental_new_converter = True
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_quantized_model = converter.convert()
            open(tflite_save_path, 'wb').write(tflite_quantized_model)
            interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
            interpreter.allocate_tensors()
            # Get I / O tensor
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))
            # Inference implementation
            interpreter.invoke()
            tflite_results = interpreter.get_tensor(output_details[0]['index'])
            print(np.squeeze(tflite_results))
            print(np.argmax(np.squeeze(tflite_results)))

            return model.get_weights() , len(X_train) , {}

        
        def evaluate(self,parameters,config):
            model.set_weights(parameters)
            val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
            print("val_loss is: " , val_loss , "val_acc is: " , val_acc)
            return val_loss , len(X_test)  , {"accuracy" : val_acc}

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=MnistClient(),
        grpc_max_message_length=1024*1024*1024
    )

# client()