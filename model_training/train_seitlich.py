import tensorflow as tf
from tensorflow.keras import layers, models

def train_cnn(epoch, batch_size, image_size, num_classes, train_images, train_labels, val_images, val_labels):
    # define cnn
    cnn_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      padding='same', input_shape=(image_size[0], image_size[1], 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')])

    # compile cnn
    cnn_model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                              patience=10, min_lr=0.001)

    # fit cnn
    history = cnn_model.fit(train_images, train_labels,
                            batch_size=batch_size, epochs=epoch, validation_data=(val_images, val_labels), callbacks=[callback, reduce_lr])


    return cnn_model
