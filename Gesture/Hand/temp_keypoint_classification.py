import csv
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import keras

def train_model():
    print("Starting training .....")
    RANDOM_SEED = 42

    dataset = 'Gesture/Hand/model/keypoint_classifier/keypoint.csv'
    model_save_path = 'Gesture/Hand/model/keypoint_classifier/keypoint_classifier.hdf5'
    tflite_save_path = 'Gesture/Hand/model/keypoint_classifier/keypoint_classifier.tflite'

    csv_path_label = 'Gesture/Hand/model/keypoint_classifier/keypoint_classifier_label.csv'
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
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, verbose=1, save_weights_only=False)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    # Model compilation
    # optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(
        optimizer='Adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    #Model Training
    model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[cp_callback, es_callback]
    )
    print("Finished Training")
    # Model evaluation
    val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)

    # Loading the saved model
    print("65")
    model = tf.keras.models.load_model(model_save_path)
    print("66")
    # Inference test
    predict_result = model.predict(np.array([X_test[0]]))
    print(np.squeeze(predict_result))
    print(np.argmax(np.squeeze(predict_result)))

    #Confusion Matrix
    # import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import confusion_matrix, classification_report

    # def print_confusion_matrix(y_true, y_pred, report=True):
    #     labels = sorted(list(set(y_true)))
    #     cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
        
    #     df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    
    #     fig, ax = plt.subplots(figsize=(7, 6))
    #     sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    #     ax.set_ylim(len(set(y_true)), 0)
    #     plt.show()
        
    #     if report:
    #         print('Classification Report')
    #         print(classification_report(y_test, y_pred))

    # Y_pred = model.predict(X_test)
    # y_pred = np.argmax(Y_pred, axis=1)

    # print_confusion_matrix(y_test, y_pred)

    #-----------------------------------------------------------
    # Convert to model for Tensorflow-Lite
    # Save as a model dedicated to inference
    print("102")
    model.save(model_save_path, include_optimizer=False)
    print("103")
    # Transform model (quantization)

    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_quantized_model = converter.convert()
    # tf.keras.backend.clear_session()

    # tf.keras.backend.set_learning_phase(False)
    # tf.keras.backend.clear_session()
    # tf.keras.backend.set_learning_phase(False)
    # keras_model = tf.keras.models.load_model(model_save_path)
    # keras_model.save('Gesture/Hand/model/keypoint_classifier/keypoint_classifier.pb',include_optimizer=False)
    # sess = tf.compat.v1.keras.backend.get_session()

    # converter = tf.lite.TFLiteConverter(keras_model.inputs, keras_model.outputs)
    # converter.convert()
    print("121")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()
    print("126")
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


train_model()