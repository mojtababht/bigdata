import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D



def build_model_ffnn(input_size):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_size,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'true_negatives', 'true_positives', 'false_negatives', 'false_positives',
                           'precision', 'recall', 'f1_score'])
    return model


def train_and_evaluate_ffnn(X_train, y_train, X_test, y_test, epochs=20, batch_size=32, model=None):
    if model is None:
        model = build_model_ffnn(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    model.evaluate(X_test, y_test, verbose=0)
    return model, history


def build_model_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'true_negatives', 'true_positives', 'false_negatives', 'false_positives',
                           'precision', 'recall', 'f1_score'])
    return model


def train_and_evaluate_cnn(X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    model = build_model_cnn((X_train.shape[1], 1))

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.1,
              callbacks=[early_stopping, reduce_lr],
              verbose=1)

    model.evaluate(X_test, y_test, verbose=0)
    return model, history
