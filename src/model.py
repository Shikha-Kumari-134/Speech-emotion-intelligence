import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout,
    Bidirectional, LSTM, Dense
)
from tensorflow.keras.models import Model

# Added parameters for tuning: filters, lstm_units, and lr
def build_crnn_model(max_len, n_features, n_classes, filters=128, lstm_units=128, lr=1e-4):
    """
    CNN + BiLSTM model for Speech Emotion Recognition with tunable parameters.
    """
    inputs = Input(shape=(max_len, n_features))

    # 1. Convolutional Feature Extractor (Using Tuned Filters)
    x = Conv1D(filters, 5, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(filters * 2, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Dropout(0.3)(x)

    # 2. BiLSTM Sequence Modeler (Using Tuned LSTM Units)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(lstm_units))(x)
    x = Dropout(0.3)(x)

    # 3. Fully Connected Layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    # 4. Classification Layer
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model