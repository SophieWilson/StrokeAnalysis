
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

latent_dim = 200 # dimension of the latent space
n_samples = 46 # size of our dataset
n_classes = 2 #(outlier or not)
n_features = 89 # we use 89 features as that is the amount in the data

X = pd.read_csv('C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/equalled_data_vals.csv')
outlier_only = X[X.isoutlier == True]
fitter_only = X[X.isoutlier == False]
X = X.drop(X.columns[0], axis=1)
y = X['isoutlier']
# 0 is fitter, 1 is non-fitter/outlier
y = y.astype(int)
print('Size of our dataset:', len(X))
print('Number of features:', X.shape[1])
print('Classes:', set(y))
X = np.asarray(X).astype('float32')

# normalising the data to help with learning, this may not be necessary im not sure. 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_X = scaler.fit_transform(X)
outlier_only = scaler.fit_transform(outlier_only)
fitter_only = scaler.fit_transform(fitter_only)


# Some imports for the model layers
from keras.layers import Activation, Dropout, Flatten, Dense, Input, LeakyReLU
# Normalization layers
from keras.layers import BatchNormalization
# Merge layers
from keras.layers import concatenate, multiply
# Embedding Layers
from keras.layers import Embedding
# Keras models
from keras.models import Model, Sequential
# Keras optimizers
from keras.optimizers import Adam, RMSprop, SGD

def build_discriminator(optimizer=Adam(0.0002, 0.5)):
    '''
    Params:
        optimizer=Adam(0.0002, 0.5) - recommended values
    '''
    # features are input, label is first column of input
    features = Input(shape=(n_features,))
    label = Input(shape=(1,), dtype='int32')
    
    # Using an Embedding layer is recommended by the papers, this layer turns integers into dense vectors so that the labels have real values instead of dummy 01 from onehot encoding.
    label_embedding = Flatten()(Embedding(n_classes, n_features)(label))
    
    # Condition the discrimination of generated features, increasing the distinction between the two class labels (i think) 
    inputs = multiply([features, label_embedding])
    
    # This is the model. Dense has 512 neurons (potentially overfitting)
    x = Dense(512)(inputs)
    # Activation layer, this is a leaky non linearity (any negatives turned to 0), leaky means it keeps some negative values, making the model more flexible. ReLU may be better.
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    # Dropout randomly sets input to 0 with frequency of 0.4, prevents overfitting. Could lower the rate
    x = Dropout(0.4)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    # You use sigmoid as an output activation as it sets the bounds between 0-1, whereas ReLU does not provide an upper bound. Sigmoid over softmax as this is a binary classification. 
    valid = Dense(1, activation='sigmoid')(x)
    # Calling and compiling the model.
    model = Model([features, label], valid)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model

def build_generator():
    # Setting noise and label
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    
    # Using an Embedding layer is recommended by the papers, to flatten the textual information
    label_embedding = Flatten()(Embedding(n_classes, latent_dim)(label))
    
    # Condition the generation of features
    inputs = multiply([noise, label_embedding])
    
    x = Dense(256)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    # returns output close to 0 and standard deviation to 1, stabilises the inputs so its less impacted by the random weights initially. Accelerates learning speed.
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    # Using tanh instead of sigmoid as it gives better accuracy results. 
    features = Dense(n_features, activation='tanh')(x)
    
    model = Model([noise, label], features)
    model.summary()

    return model

def build_gan(generator, discriminator, optimizer=Adam(0.0002, 0.5)):
    '''
    Combines generator and discriminator so the input goes to generator which goes to discriminator and then outputs the GAN result. 
    Params:
        optimizer=Adam(0.0002, 0.5) - recommended values
    '''
    
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    # Building the generator and discriminator
    features = generator([noise, label])
    valid = discriminator([features, label])
    
    # We freeze the discriminator's layers since we're only 
    # interested in the generator and its learning
    discriminator.trainable = False
    
    model = Model([noise, label], valid)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    return model

discriminator = build_discriminator()
generator = build_generator()
gan = build_gan(generator, discriminator)

def get_random_batch(X, y, batch_size):
    '''
    Gets random batches of the data set to batch_size, used in training for each epoch.
    Params:
        X: numpy array - features
        y: numpy array - classes
        batch_size: Int
    '''
    idx = np.random.randint(0, len(X))
    
    X_batch = X[idx:idx+batch_size]
    y_batch = y[idx:idx+batch_size]
    
    return X_batch, y_batch

def train_gan(gan, generator, discriminator, 
              X, y, 
              n_epochs=1000, batch_size=32, 
              hist_every=10, log_every=100):
    '''
    Trains discriminator and generator separately in batches of size batch_size. 
        1. Discriminator is trained with real features from training data
        2. Discriminator is trained with fake features generated by the Generator
        3. GAN is trained, which will only change the Generator's weights.       
    Params:
        gan: GAN model
        generator: Generator model
        discriminator: Discriminator model
        X: numpy array - features
        y: numpy array - classes
        n_epochs: Int
        batch_size: Int
        hist_every: Int - will save the training loss and accuracy every hist_every epochs
        log_every: Int - will output the loss and accuracy every log_every epoch  
    Returns:
        loss_real_hist: List of Floats
        acc_real_hist: List of Floats
        loss_fake_hist: List of Floats
        acc_fake_hist: List of Floats
        loss_gan_hist: List of Floats
        acc_gan_hist: List of Floats
    '''
    
    half_batch = int(batch_size / 2)
    # setting output variables
    acc_real_hist = []
    acc_fake_hist = []
    acc_gan_hist = []
    loss_real_hist = []
    loss_fake_hist = []
    loss_gan_hist = []
    
    for epoch in range(n_epochs):
        # calling random batch function
        X_batch, labels = get_random_batch(X, y, batch_size)
        
        # train with real values
        y_real = np.ones((X_batch.shape[0], 1))
        loss_real, acc_real = discriminator.train_on_batch([X_batch, labels], y_real)
        
        # train with fake values
        # setting random noise within range
        noise = np.random.uniform(0, 1, (labels.shape[0], latent_dim))
        # predict fake values generator has made
        X_fake = generator.predict([noise, labels])
        # predict fake labels
        y_fake = np.zeros((X_fake.shape[0], 1))
        # calculate loss metrics
        loss_fake, acc_fake = discriminator.train_on_batch([X_fake, labels], y_fake)
        # training on gan labels and values
        y_gan = np.ones((labels.shape[0], 1))
        loss_gan, acc_gan = gan.train_on_batch([noise, labels], y_gan)
        # only recording metrics every 10 epochs
        if (epoch+1) % hist_every == 0:
            acc_real_hist.append(acc_real)
            acc_fake_hist.append(acc_fake)
            acc_gan_hist.append(acc_gan)
            loss_real_hist.append(loss_real)
            loss_fake_hist.append(loss_fake)
            loss_gan_hist.append(loss_gan)
        # concatenating the correct format for printing
        if (epoch+1) % log_every == 0:
            lr = 'loss real: {:.3f}'.format(loss_real)
            ar = 'acc real: {:.3f}'.format(acc_real)
            lf = 'loss fake: {:.3f}'.format(loss_fake)
            af = 'acc fake: {:.3f}'.format(acc_fake)
            lg = 'loss gan: {:.3f}'.format(loss_gan)
            ag = 'acc gan: {:.3f}'.format(acc_gan)
            # printing every 10 epochs
            print('{}, {} | {}, {} | {}, {}'.format(lr, ar, lf, af, lg, ag))
        
    return loss_real_hist, acc_real_hist, loss_fake_hist, acc_fake_hist, loss_gan_hist, acc_gan_hist
# printing metrics
loss_real_hist, acc_real_hist, \
loss_fake_hist, acc_fake_hist, \
loss_gan_hist, acc_gan_hist = train_gan(gan, generator, discriminator, scaled_X, y)

# some plots to show the GAN learning output
# This shows loss, so plotting error or training loss. Values close to 0 indicate the training set was learnt perfectly.
ax, fig = plt.subplots(figsize=(15, 6))
plt.plot(loss_real_hist)
plt.plot(loss_fake_hist)
plt.plot(loss_gan_hist)
plt.title('Training loss over time')
plt.legend(['Loss real', 'Loss fake', 'Loss GAN'])

# This is plotting accuracy over time, so the amount of correct classifications/total classifications, accuracy is lower for gan than real/fake.
ax, fig = plt.subplots(figsize=(15, 6))
plt.plot(acc_real_hist)
plt.plot(acc_fake_hist)
plt.plot(acc_gan_hist)
plt.title('Training accuracy over time')
plt.legend(['Acc real', 'Acc fake', 'Acc GAN'])


def generate_samples(class_for, n_samples=40):
    '''
    Generates new random realistic features using the trained generator.
    Params:
        class_for: Int - features for this class
        n_samples: Int - how many samples to generate
    '''
    
    noise = np.random.uniform(0, 1, (n_samples, latent_dim))
    label = np.full((n_samples,), fill_value=class_for)
    return generator.predict([noise, label])


def visualize_fake_features(fake_features, figsize=(15, 6), color='r'):
    ax, fig = plt.subplots(figsize=figsize)
    
    # Let's plot our dataset to compare
    for i in range(n_classes):
        plt.scatter(scaled_X[:, 0][np.where(y==i)], scaled_X[:, 1][np.where(y==i)])

    plt.scatter(fake_features[:, 0], fake_features[:, 1], c=color)
    plt.title('Real and fake features')
    plt.legend(['Class 0', 'Class 1', 'Fake'])


    

# Generating fake samples
features_class_0 = generate_samples(0)
features_class_1 = generate_samples(1)

# visualising in 2D, not representitive. 
#visualize_fake_features(features_class_0)
#visualize_fake_features(features_class_1)

# Visualising in PCA 
from sklearn.decomposition import PCA
# Visualising the data
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_X)
pca_class1 = pca.fit_transform(features_class_1)
pca_class0 = pca.fit_transform(features_class_0)

# plot dataset
plt.figure()
plt.scatter(pca_result[:, 0], pca_result[:,1], c=y) # pca 0 on x and pca 1 on y
plt.title('PCA of dataset')

# plotting dataset plus fakes
plt.figure()
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y)
plt.scatter(pca_class1[:, 0], pca_class1[:, 1])
plt.scatter(pca_class0[:, 0], pca_class0[:, 1])
plt.title('PCA of dataset with fake samples included')
plt.legend(['', 'Fake Fitters', 'Fake Non-fitters'])

# plotting class 1 (nonfitters)
pca_nonfitter = pca.fit_transform(outlier_only)
plt.figure()
plt.scatter(pca_class1[:,0], pca_class1[:,1])
plt.scatter(pca_nonfitter[:,0], pca_nonfitter[:, 1])
plt.title('PCA of only non-fitter class')
plt.legend(['fake', 'real'])

# plotting class 0 (fitters)
pca_fitter = pca.fit_transform(fitter_only)
plt.figure()
plt.scatter(pca_fitter[:,0], pca_fitter[:, 1])
plt.scatter(pca_class0[:,0], pca_class0[:,1])
plt.title('PCA of only fitter class')
plt.legend(['real', 'fake' ])


plt.show()




