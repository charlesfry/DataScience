# import tensorflow
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

# set seed for reproducability
seed = 42
tf.random.set_seed = seed

from sklearn.datasets import load_iris
data = load_iris()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.33, random_state=seed)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(256, activation=tf.nn.relu,
                          kernel_regularizer=tf.keras.regularizers.l2(l=.01)),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)
print('\nEvaluating model:')
model.evaluate(X_test, y_test)
