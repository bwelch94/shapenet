########################################################
############### ShapeNet image classifier ##############
## Builds supervised model to classify shapes as either
## circles or squares. Then removes classification 
## layer, replaces it with 2-cluster K-means
## unsupervised clustering algorithm. K-means should 
## classify the shapes as well as the supervised 
## classifier. At least that's the goal...
########################################################

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob



def supervised_fit(n_epochs=10,train_dir='trainset_bw',
  validation_dir='testset_bw',savepath=None):

  #######################################################
  ############# Image Pre-processing ####################
  #set up data generators to read training and validation
  #image data. Copied from image_classification_part1
  #######################################################

  #training circle and square directories
  train_circle_dir = os.path.join(train_dir, 'circle')
  train_square_dir = os.path.join(train_dir, 'rectangle')

  #validation circle/square directories
  validation_circle_dir = os.path.join(validation_dir, 'circle')
  validation_square_dir = os.path.join(validation_dir, 'rectangle')

  #rescale images to 0-1 scale
  train_datagen = ImageDataGenerator(rescale=1./255.) 
  test_datagen = ImageDataGenerator(rescale=1./255.)

  # Flow training images in batches of 20 using train_datagen generator
  train_generator = train_datagen.flow_from_directory(
          train_dir,  
          target_size=(144, 144), 
          batch_size=10, shuffle=True,
          # Since we use binary_crossentropy loss, we need binary labels
          class_mode='binary')

  # Flow validation images in batches of 20 using test_datagen generator
  validation_generator = test_datagen.flow_from_directory(
          validation_dir,
          target_size=(144, 144),
          batch_size=10, shuffle=True,
          class_mode='binary')


  #######################################################
  ############ Model definition #########################
  #define CNN to recognize images. copied from 
  #image_classification_part1.ipynb
  #######################################################

  # Our input feature map is nxnx4: nxn for the image pixels, and 3 for
  # the three color channels: R, G, B
  img_input = layers.Input(shape=(144, 144, 3))

  # First convolution extracts 16 filters that are 3x3
  # Convolution is followed by max-pooling layer with a 2x2 window
  x = layers.Conv2D(16, 3, activation='relu')(img_input)
  x = layers.MaxPooling2D(2)(x)
  #x = layers.BatchNormalization()(x)

  # Second convolution extracts 32 filters that are 3x3
  # Convolution is followed by max-pooling layer with a 2x2 window
  x = layers.Conv2D(32, 3, activation='relu')(x)
  x  = layers.MaxPooling2D(2)(x)
  #x = layers.BatchNormalization()(x)

  # Third convolution extracts 64 filters that are 3x3
  # Convolution is followed by max-pooling layer with a 2x2 window
  x = layers.Conv2D(64, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  #x = layers.BatchNormalization()(x)

  # Flatten feature map to a 1-dim tensor so we can add fully connected layers
  x = layers.Flatten()(x)

  # Create a fully connected layer with ReLU activation and 512 hidden units
  x = layers.Dense(512, activation='relu')(x)

  # Create output layer with a single node and sigmoid activation
  output = layers.Dense(1, activation='sigmoid')(x)

  # Create model:
  # input = input feature map
  # output = input feature map + stacked convolution/maxpooling layers + fully 
  # connected layer + sigmoid output layer
  model = Model(img_input, output)



  print(model.summary())



  #compile model.
  #use binary crossentropy loss because this is a binary problem
  #optimize using Adam optimizer. Automates learning rate tuning
  model.compile(loss='binary_crossentropy',
               optimizer=Adam(lr=0.001),
                metrics=['acc'])



  #Train the model! This may take a while...
  history = model.fit_generator(
        train_generator,
        steps_per_epoch=400,  # 4000 images = batch_size * steps
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=200,  # 2000 images = batch_size * steps
        verbose=2)

  if savepath is not None:
    save_model(model,savepath)


  return history,model



def make_plot(history,figname=None):
  # Retrieve a list of accuracy results on training and test data
  # sets for each training epoch
  acc = history.history['acc']
  val_acc = history.history['val_acc']

  # Retrieve a list of list results on training and test data
  # sets for each training epoch
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  # Get number of epochs
  epochs = range(len(acc))

  # Plot training and validation accuracy per epoch
  fix,(ax1,ax2)=plt.subplots(1,2)
  ax1.plot(epochs, acc,label='train')
  ax1.plot(epochs, val_acc,label='test')
  ax1.legend(loc='best')
  ax1.set_title('Training and validation accuracy')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Accuracy')

  # Plot training and validation loss per epoch
  ax2.plot(epochs, loss,label='train')
  ax2.plot(epochs, val_loss,label='test')
  ax2.legend(loc='best')
  ax2.set_title('Training and validation loss')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Loss')

  plt.tight_layout()
  if figname==None:
    plt.show()
  else:
    plt.savefig(figname)


def unsupervised_fit(model,predict_directory='predictset'):
  #######################################################
  ########## Unsupervised clustering part ###############
  #             hold on to your butts... 
  #######################################################
  pred_datagen = ImageDataGenerator(rescale=1./255.)
  pred_generator = pred_datagen.flow_from_directory(
        predict_directory,
        target_size=(144,144),
        batch_size=10,shuffle=False,
        class_mode=None)

  # get last layer of trained model
  last_layer = model.get_layer(index=8)
  last_output = last_layer.output

  # define new model without classification layer
  headless = Model(model.input,last_output)

  headless.compile(loss='binary_crossentropy',
                optimizer=Adam(lr=0.001),
                metrics=['acc'])

  # run model on prediction data
  headless_output = headless.predict_generator(
        pred_generator, steps=10)  # 100 images = steps * batch_size

  # define unsupervised K-means clustering algorithm
  estimator = KMeans(n_clusters=2,n_jobs=-1)

  # do clustering step, predict labels
  estimator.fit(headless_output)
  labels = estimator.labels_

  # print outputs of clustering
  imdir = os.path.join(predict_directory,'images')
  predict_files = glob.glob(os.path.join(imdir,'*.jpg'))
  c=[]
  s=[]
  for i in range(len(labels)):
    shapename = predict_files[i][-10:-4]
    print(labels[i],shapename)
    if shapename == 'circle':
      c.append(labels[i])
    else:
      s.append(labels[i])

  c=np.array(c)
  s=np.array(s)
  cstring = 'Circle: N_0:%.0f, N_1:%.0f'%(len(c[c==0]),len(c[c==1]))
  sstring = 'Square: N_0:%.0f, N_1:%.0f'%(len(s[s==0]),len(s[s==1]))
  print(cstring)
  print(sstring)
  return

def nn_find(model,predict_directory='predictset_bw'):
  #######################################################
  ########## Unsupervised clustering part ###############
  #    But this time, finding nearest neighbors 
  # Input data is 100 unique images, 10 duplicates
  # leading to a total of 110 images
  #######################################################
  pred_datagen = ImageDataGenerator(rescale=1./255.)
  pred_generator = pred_datagen.flow_from_directory(
        predict_directory,
        target_size=(144,144),
        batch_size=10,shuffle=False,
        class_mode=None)

  # get last layer of trained model
  last_layer = model.get_layer(index=8)
  last_output = last_layer.output

  # define new model without classification layer
  headless = Model(model.input,last_output)

  headless.compile(loss='binary_crossentropy',
                optimizer=Adam(lr=0.001),
                metrics=['acc'])

  # run model on prediction data
  headless_output = headless.predict_generator(
        pred_generator, steps=11)  # 110 images = steps * batch_size

  # KDTree 
  nbrs = NearestNeighbors(n_neighbors=3,algorithm='brute').fit(headless_output)

  # find nearest neighbor pairs for each point
  distances, indices = nbrs.kneighbors(headless_output)

  # get image files
  imdir = os.path.join(predict_directory,'images')
  predict_files = glob.glob(os.path.join(imdir,'*.jpg'))

  for i in indices:
    print(predict_files[i[0]],predict_files[i[1]],predict_files[i[2]])

  return(indices)


def nike(n_epochs=10,figname=None):
  history,model = supervised_fit(n_epochs)
  make_plot(history,figname)
  #unsupervised_fit(model)
  nn_find(model)
  return

