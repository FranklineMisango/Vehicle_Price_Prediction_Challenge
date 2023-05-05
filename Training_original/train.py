import tensorflow as tf
from keras.applications.resnet import ResNet152

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

img_width, img_height = 224, 224
num_channels = 3
train_data = 'data/train'
valid_data = 'data/valid'
num_classes = 196
num_train_samples = 6549
num_valid_samples = 1595
verbose = 1
batch_size = 16
num_epochs = 100000
patience = 50

if __name__ == '__main__':
    # build a classifier model
    if 'tf' in locals():
        model = tf.keras.Sequential([
            ResNet152(include_top=True, weights=None,
                      input_shape=(img_height, img_width, num_channels),
                      classes=num_classes)])
    else:
        model = ResNet152(include_top=True, weights=None,
                          input_shape=(img_height, img_width, num_channels),
                          classes=num_classes)

    # prepare data augmentation configuration
    train_data_gen = ImageDataGenerator(rotation_range=20.,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
    valid_data_gen = ImageDataGenerator()
    # callbacks
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True) if 'tf' in locals() else keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    log_file_path = 'logs/training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)
    trained_models_path = 'models/model'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr]

    # generators
    train_generator = train_data_gen.flow_from_directory(train_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical')
    valid_generator = valid_data_gen.flow_from_directory(valid_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical')

    # fine tune the model
    if 'tf' in locals():
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        model.fit(train_generator, steps_per_epoch=num_train_samples / batch_size,
                  validation_data=valid_generator,
                  validation_steps=num_valid_samples / batch_size,
                  epochs=num_epochs,
                  callbacks=callbacks,
                  verbose=verbose)
    else:
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit_generator(train_generator, steps_per_epoch=num_train_samples / batch_size,
                            validation_data=valid_generator,
                            validation_steps=num_valid_samples / batch_size,
                            epochs=num_epochs,
                            callbacks=callbacks,
                            verbose=verbose)
