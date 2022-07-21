from keras.layers import Dense
from keras.layers import Input, LeakyReLU
from keras.models import Model
from keras.datasets.mnist import load_data
from numpy import reshape
import matplotlib.pyplot as plt

(xtrain, _), (xtest, _) = load_data()
xtrain = xtrain.astype('float32') / 255.
xtest = xtest.astype('float32') / 255.
print(xtrain.shape, xtest.shape)  

input_size = xtrain.shape[1] * xtrain.shape[2]
latent_size = 16
x_train = xtrain.reshape((len(xtrain), input_size))
x_test = xtest.reshape((len(xtest), input_size))
print(x_train.shape)
print(x_test.shape)

# Encoder
enc_input = Input(shape=(input_size,))
enc_dense1 = Dense(units=256, activation="relu")(enc_input)
enc_activ1 = LeakyReLU()(enc_dense1)
enc_dense2 = Dense(units=latent_size)(enc_activ1)
enc_output = LeakyReLU()(enc_dense2)
encoder = Model(enc_input, enc_output)
encoder.summary()

# Decoder
dec_input = Input(shape=(latent_size,))
dec_dense1 = Dense(units=256, activation="relu")(dec_input)
dec_activ1 = LeakyReLU()(dec_dense1)
dec_dense2 = Dense(units=input_size, activation='sigmoid')(dec_activ1)
dec_output = LeakyReLU()(dec_dense2)
decoder = Model(dec_input, dec_output)
decoder.summary()

# Autoencoder
aen_input = Input(shape=(input_size,))
aen_enc_output = encoder(aen_input)
aen_dec_output = decoder(aen_enc_output)
aen = Model(aen_input, aen_dec_output)
aen.summary()

aen.compile(optimizer="rmsprop", loss="binary_crossentropy")
aen.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)

encoded_images = encoder.predict(x_test)
decoded_images = decoder.predict(encoded_images)
pred_images = reshape(decoded_images, newshape=(decoded_images.shape[0], 28, 28)) 
 
n = 10
plt.figure(figsize=(10, 2))
for i in range(n): 
 ax = plt.subplot(2, n, i + 1)
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
 plt.imshow(xtest[i].reshape(28, 28))
 plt.gray()
 
 ax = plt.subplot(2, n, i + 1 + n)
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
 plt.imshow(pred_images[i].reshape(28, 28))

plt.show()