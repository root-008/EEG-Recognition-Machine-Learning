import createX_y
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,CSVLogger
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

def fit_ann_with_holdout(epochs,batch_size,early_stopping):
    X_train, X_val, X_test, y_train, y_val, y_test = createX_y.create_train_test_data_for_ann()
    
    l_encode = LabelEncoder()
    
    l_encode.fit(y_train)
    y_train = l_encode.transform(y_train)
    y_train = to_categorical(y_train)
    
    l_encode.fit(y_val)
    y_val = l_encode.transform(y_val)
    y_val = to_categorical(y_val)
    
    l_encode.fit(y_test)
    y_test = l_encode.transform(y_test)
    y_test = to_categorical(y_test)
    
    print(y_train.shape,y_val.shape,y_test.shape)
    
    # Create model
    ann = tf.keras.models.Sequential()

    ann.add(tf.keras.layers.BatchNormalization(input_shape=(178,)))
    ann.add(tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(0.2)))
    ann.add(tf.keras.layers.Dropout(0.4))
    ann.add(tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(0.2)))
    ann.add(tf.keras.layers.Dropout(0.4))
    ann.add(tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(0.2)))
    ann.add(tf.keras.layers.Dropout(0.4))
    ann.add(tf.keras.layers.Dense(178, activation=tf.keras.layers.LeakyReLU(0.2)))

    ann.add(tf.keras.layers.BatchNormalization())
    ann.add(tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(0.2)))
    ann.add(tf.keras.layers.Dropout(0.4))
    ann.add(tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(0.2)))
    ann.add(tf.keras.layers.Dropout(0.4))
    ann.add(tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(0.2)))
    ann.add(tf.keras.layers.Dropout(0.4))
    ann.add(tf.keras.layers.Dense(178, activation=tf.keras.layers.LeakyReLU(0.2)))

    ann.add(tf.keras.layers.BatchNormalization())
    ann.add(tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(0.2)))
    ann.add(tf.keras.layers.Dropout(0.4))
    ann.add(tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(0.2)))
    ann.add(tf.keras.layers.Dropout(0.4))
    ann.add(tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(0.2)))
    ann.add(tf.keras.layers.Dropout(0.4))
    ann.add(tf.keras.layers.Dense(178, activation=tf.keras.layers.LeakyReLU(0.2)))
    ann.add(tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(0.2)))

    # Output Layer
    ann.add(tf.keras.layers.Dense(5, activation='softmax'))

    # Compile model
    ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Early Stopping ve Model Checkpoint callbacks
    early_stopping = EarlyStopping(monitor='accuracy', patience=early_stopping, restore_best_weights=True)

    checkpoint_path = 'savedModels/weights.best.hdf5'
    checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True, monitor='accuracy')
    csv_logger = CSVLogger('savedModels/training_history.csv', separator=',', append=False)
    
    # Fit the model
    ann.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping, checkpoint_cb,csv_logger])

    #save model
    ann.save('savedModels/model_ann_holdout.keras')
    
    




