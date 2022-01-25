import tensorflow as tf



input_layer=tf.keras.layers.Input((1024,2))
output_layer = tf.keras.layers.Conv1D(filters=4,padding='same',kernel_size=32)(input_layer)
model=tf.keras.models.Model(inputs = input_layer,outputs=output_layer)

model.summary()

