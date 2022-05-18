
#--- imports needed for retrieving data
import zipfile
import tarfile
import os
import pathlib
import numpy as np

#---- imports needed to visualize images:
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import plot_model

#--- Imports used for callbacks:
from datetime import datetime
from tensorflow.keras import callbacks
    
    
def unzip_files(file_name):
  """
    Will extract provided zip or tar.gz file.
  """
  #---Determine File Type
  if '.zip' in file_name.lower()[-4:]:
    zip_ref = zipfile.ZipFile(file_name)
  elif '.tar.gz' in file_name.lower()[-7:]:
    zip_ref = tarfile.open(file_name)
  else:
    print("File must be a '.zip' or '.tar.gz', file has not been extracted.")
    return None
    
  #---Extract the downloaded file:   
  zip_ref.extractall()
  zip_ref.close()
  print('Done!')

def count_files(folder_name):
  """
    Displays to the user the number of files inside a folder.
  """
  #---- Review number of image files in folders:
  for dirpath, dirnames, filenames in os.walk(folder_name):
    nfiles = len(filenames)
    if nfiles:
      print(f'There are {nfiles} images in "{dirpath}".')

def get_class_names(folder_name):
  """
    Return an array containing class names
  """
  data_dir = pathlib.Path(folder_name)
  class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
  print(class_names)
  return class_names 

#-------------------------------------Helper Functions: make_subfolder, move_files
def make_subfolders(root,class_names):
  """
    Will create a directory structure:
      root/
        class_a/
        class_b/
        ...
  """
  if not os.path.isdir(root):
    os.mkdir(root)
  
  for class_name in class_names:
    subfolder = f'{root}/{class_name}'
    if not os.path.isdir(subfolder):
      os.mkdir(subfolder)
      

def move_files(file_names,move_from_dir,move_to_dir,class_names):
  """
    ***NOTE: Must have run 'make_subfolder' function first! ***
    Will move image files from source directory into specified subdirectories. 
    ex: 
      move files specified in this directory structure:
        food-101/
          |_images/
              |_class_a/
              |_class_b/
              |_...

      into files in this directory structure:
        food-101/
          |_test/
              |_class_a/
              |_class_b/
              |_...
          |_train/
              |_class_a/
              |_class_b/
              |_...
  """
  file_img_names = open(file_names,'r')
  for image_name in file_img_names.readlines():
    #----- Setup:
    image_name = image_name.replace('\n','')                            #removes the new-line character ('\n') from image name 
    class_name = image_name.split('/')[0]

    source_filename = f'{move_from_dir}/{image_name}.jpg'               #Get current full path to image
    destination_filename = f'{move_to_dir}/{image_name}.jpg'            #Make new destination path for image
    destination_folder = f'{move_to_dir}/{class_name}'                  #Will be used to check if destination subfolder exists

    #----- Move Files
    source_exists = os.path.isfile(source_filename)
    destination_exists = os.path.isdir(destination_folder)

    if (source_exists and destination_exists):                          #double check that files and folders exist
      os.replace(source_filename, destination_filename)                 #Move image file     
    else:                                                               #report error to user
      print(f'source file exists {source_exists}: {source_filename}')
      print(f'destination folder exists {destination_exists}: {destination_folder}')
  file_img_names.close()                                                #close/terminate resource
  print('done!')


#-------------------------------------Helper Functions: view_random_class_images

def _get_random_images0(folder):
  print(f'target folder: {folder}')
  return rnd.choices(os.listdir(folder),k=9)

def _show_image0(folder,file,subplot):
    """
      Handles displaying an image on a specified subplot
    """
    img = mpimg.imread(f'{folder}/{file}')
    ax = plt.subplot(subplot)
    ax.imshow(img)  
    return ax,img

def _format_fig0(figure,class_name):
  """
    Formats subplots to reduce whitespace
  """
  figure.suptitle(f'Images of {class_name}',
                  fontsize=15,
                  y=1.0)
  figure.tight_layout(pad=2.10,
                      h_pad=1.10,
                      w_pad=0.25)

def _format_ax0(ax,idx,shape):
    """
      helper function for displaying images
      hides axes ticks to reduce cluttering
    """
    #--- Format axes
    ax.set_title(f'shape: {shape}')

    #--- Keep Axes ticks along edges, delete others.
    if (idx % 3):
      ax.set_yticks([])
    if (idx < 6):
      ax.set_xticks([])

def view_random_class_images(target_dir,target_class):
  
  #Setup the target directory:
  target_folder = f'{target_dir}/{target_class}'

  #get 9 random image names:
  random_images = _get_random_images0(target_folder)

  #Setup for showing images
  images = []                 #return a list of 9 images
  fig = plt.figure(figsize=(9,9))

  #Show random images
  for ax_idx, img_name in enumerate(random_images):

    ax,img = _show_image0(folder=target_folder,
                         file=img_name,
                         subplot=331+ax_idx)
    #-Format:
    _format_ax0(ax=ax,
               idx=ax_idx,
               shape=img.shape)
    
    #-Append image to list 
    images.append(img)
  
  #--- Add a Title
  _format_fig0(figure=fig,
              class_name=target_class)

#----------------------------------------------Helper Functions: view_random_augmented_images, 

def _get_random_images1(images,labels):
  batch_size = images.shape[0]
  random_indexes = rnd.sample(range(0,batch_size),k=9)
  r_imgs = images[random_indexes,...]
  r_labels = labels[random_indexes]
  return r_imgs,r_labels

def _show_image1(image,subplot):
    """
      Handles displaying an image on a specified subplot
    """
    ax = plt.subplot(subplot)
    ax.imshow(image)  
    return ax

def _format_fig1(figure):
  """
    Formats subplots to reduce whitespace
  """
  figure.suptitle(f'Images of Augmented Data',
                  fontsize=12,
                  y=0.99)
  figure.tight_layout(pad=2,
                      h_pad=0.90,
                      w_pad=0.25)


def _format_ax1(ax,idx,label,names):
    """
      helper function for displaying images
      hides axes ticks to reduce cluttering
    """
    #--- Format axes
    label_idx = np.where(label==1)[0][0]
    text = names[label_idx]
    ax.set_title(f'{text}')

    #--- Keep Axes ticks along edges, delete others.
    if (idx % 3):
      ax.set_yticks([])
    if (idx < 6):
      ax.set_xticks([])

def view_random_augmented_images(data,class_names):
  
  #Setup the target directory:
  target_data, target_labels = data.next()

  #get 9 random image names:
  images,labels = _get_random_images1(target_data,target_labels)

  #Setup for showing images
  fig = plt.figure(figsize=(9,9))

  #Show random images
  for ax_idx, img in enumerate(zip(images,labels)):
    image, label = img
    ax = _show_image1(image=image,
                     subplot=331 + ax_idx)
    #-Format:
    _format_ax1(ax=ax,
               idx=ax_idx,
               label=label,
               names=class_names)
  
  #--- Add a Title
  _format_fig1(figure=fig)
  
#--------------------------------------------------Helper Functions: create_tensorboard_callback

 def create_tensorboard_callback(dir_name, experiment_name):
  #---Create Log Directory:
  time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")  #yymmdd_HHMMSS
  log_dir = f'{dir_name}/{experiment_name}/{time_stamp}'

  #--Init TB callback:
  tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)
  
  #---Return output:
  print(f'Set TensorBoard log files to: {log_dir}')
  return tensorboard_callback
  
def upload_tensorboard(log_dir,experiment_name,description):
    eval(f'!tensorboard dev upload \
      --logdir {log_dir} \
      --name {experiment_name} \
      --description {description} \
      --one_shot')   

def Show_Is_Trainable(model):
  """
    will display a table showing the model's layer names, 
    and the layer's trainable status.
  """
  fmt0 = len(f'{len(model.layers)}')
  fmtw = max([len(layer.name) for layer in model.layers]) + 10
  for i, layer in enumerate(model.layers):
    fmtsp = " "*(fmtw-len(layer.name))
    laynum = f'{i}'.rjust(fmt0,'0')
    print(laynum,layer.name,f'{fmtsp} {layer.trainable}')  

#----------------------------------------Helper functions: Show_Model, Training_Plot, BuildCompileFit, ContinueTraining

def Show_Model(model,name):
    """
      Will provide a visual representation of model,
      showing layers, and input/output shapes. 
    """
    file_name = f'{name}.png'
    plot = plot_model(model,
                      to_file=file_name,
                      show_layer_activations=True,
                      show_shapes=True)
    
    plt.figure(figsize=(8,8))
    image = mpimg.imread(file_name)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])  

def _format_training_plot(fig,axL,axR):
  axL.set_title('Metrics')
  axR.set_title('Losses')

  axL.set_ylabel('Accuracy/Other')
  axR.set_ylabel('Loss')
  
  axL.set_xlabel('epochs')
  axR.set_xlabel('epochs')

  axL.legend()
  axR.legend()

  fig.suptitle('Training Results',fontsize=15,y=1.0)
  fig.tight_layout(pad=2.0,w_pad=1)

def Training_Plot(history):
  """
    Will plot results from model training:
      loss,accuracy vs epochs
  """
  fig = plt.figure(figsize=(15,4))
  axL = plt.subplot(121)
  axR = plt.subplot(122)
  for key,y in history.history.items():
    x = range(0,len(y))
    if 'loss' in key:
      axR.plot(x,y,label=key)
    else:
      axL.plot(x,y,label=key)

  _format_training_plot(fig,axL,axR)


def BuildCompileFit(trn_data,val_data,layers,loss,optimizer,callbacks,metrics,
                    rndSeed,epochs,validation_percent=1,verbose=0,show_model=False,model_name='Model'):
  
  #--- Set Random Seed
  tf.random.set_seed(rndSeed)

  #--- Build Model
  model = tf.keras.Sequential(layers)

  #--- Compile Model
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)
  
  if show_model:
     Show_Model(model,model_name)
  
  
  train_steps = len(trn_data)
  valid_steps = max(1,
                   int(validation_percent*len(val_data)))
  
  
  #--- Fit Model:
  history=model.fit(trn_data,
                    epochs=epochs,
                    steps_per_epoch=train_steps ,
                    validation_steps=valid_steps,
                    callbacks=callbacks,
                    validation_data=val_data,
                    validation_steps=len(val_data),
                    verbose=verbose)


  #--- Evaluate Model
  if not verbose:
    print('\nEvaluation: ',model.evaluate(trn_data),'\n')              

  #------------------View Loss/Training Curve:
  Training_Plot(history)

  return model,history


def ContinueTraining(trn_data,val_data,model,callbacks,epochs,verbose=0):
  """
    Used to avoid recompiling a new model, 
    to continue after some initial training. 
  """

  #--- Fit Model:
  history=model.fit(trn_data,
                    epochs=epochs,
                    steps_per_epoch=len(trn_data),
                    callbacks=callbacks,
                    validation_data=val_data,
                    validation_steps=len(val_data),
                    verbose=verbose)

  #--- Evaluate Model
  if not verbose:
    print('\nEvaluation: ',model.evaluate(trn_data),'\n')              

  #------------------View Loss/Training Curve:
  Training_Plot(history)

  return model, history 
  
  
  
#-------------show_image_cnn_filter_view, get_batch, get_outputs

def _make_figure(layer_name,depth,shape):
  """
    Create a matplotlib figure and add a title
  """
  fig = plt.figure(figsize=(18,6),tight_layout={'pad':1.5,'h_pad':1.5,'w_pad':1.5})
  fig.suptitle(f'Layer {layer_name}, Depth {depth}, Shape {shape}',
               fontsize=15,color='red',
               x=0.36,
               y=1.025)

  return fig


def _make_axes(figure):
  """
    Will set up layout of axes on figure:
    Left: 10 axes for CNN filters, 
    Right: 4 axes (combined) for original image

           Left                 Right       
    fax0 fax1 fax2 fax3 fax4 | Img Img
    - - - - - - - - - - - - -| 
    fax5 fax6 fax7 fax8 fax9 | Img Img

  """
  gs = GridSpec(2,7,figure=figure)
  Laxes = []
  Rax = figure.add_subplot(gs[:,5:])

  for row in range(0,2):
    for col in range(0,5):
      ax = figure.add_subplot(gs[row,col])
      Laxes.append(ax)

  return Laxes,Rax  


def _display_original_image(image,ax,label):
  """
    Displays input image for reference. 
  """
  ax.imshow(image)
  ax.set_xticks([])
  ax.set_yticks([])
  text = 'Steak' if int(label) else 'Pizza'
  ax.set_title(text,fontsize=20)    

def _format_filter_ax(ax,idx):
  """
    reduce clutter on filter image axes by removing x-axis, and y-axis ticks. 
  """
  ax.set_title(f'filter {idx}',fontsize=10)
  ax.set_xticks([]) 
  ax.set_yticks([])

def _get_min_max(output):
  """
    used to provide consistant color scaling accross filter images.
    finds the global min, max values for all filter views. 
  """
  if 'dense' not in output.layer_name:
    min = tf.reduce_min(output).numpy()
    max = tf.reduce_max(output).numpy()
  else:
    min = None
    max = None
  return min,max

def _get_dense_layer_view(outputs,layers,layer_idx):
    """
      CNNs & Maxpools have a 'Filter View' however,
      Dense layers have a 'Neuron View' and need
      to be calculated differently. 
    """
    #--Search for the closest Non-Flattened layer
    idx = layer_idx
    while idx > 0:
      idx -= 1
      if len(outputs[idx].shape) > 2:
        reshape_dense_view = outputs[idx].shape
        break

    #--Calculate Dense Layer Neuron View:
    dense_weights = tf.squeeze(layers[layer_idx].weights[0])
    output = np.reshape(outputs[layer_idx-1] * dense_weights,reshape_dense_view)
    output = tf.constant(output)

    #--Store information to be held for axes & figure formatting
    output.layer_name = outputs[layer_idx].layer_name
    output.layer_idx = layer_idx

    return output

def show_image_cnn_filter_view(outputs,images,labels,layers,layer_idx,img_idx,cmap):
  """
    Displays the results of each filter after processing an image,
    along with the original image.
  """
  #----Setup:
  output = outputs[layer_idx]
  image = images[img_idx,:,:,:]
  im_shape = output.shape[1:-1]
  is_dense = False

  if 'dense' in output.layer_name:
    output = _get_dense_layer_view(outputs,layers,layer_idx)
    im_shape = output.shape[1:-1]

  fig = _make_figure(layer_name=output.layer_name,
                     depth=output.layer_idx,
                     shape=im_shape)
  
  Laxes,Rax = _make_axes(fig)

  #------------- Display Right Side Image
  _display_original_image(image=image,
                          ax=Rax,
                          label=labels[img_idx])
  
  #------------- Display Left Side Images:
  #--Get Min,Max values for consistent color scaling
  min_val,max_val = _get_min_max(output)
  
  #--Display filter 'view'
  for filter_idx,ax in enumerate(Laxes):
    filter_view = output[img_idx,:,:,filter_idx]
    img = ax.imshow(filter_view,
                    cmap=cmap,
                    vmin=min_val,
                    vmax=max_val)
    
    _format_filter_ax(ax=ax,
                      idx=filter_idx)



def get_batch(data_batches,batch_idx):
  """
    Retrieves a specific batch from the ImageDataGenerator
    Args:
        data_batches: ImageDataGenerator instance (ex: train_data, test_data)
           batch_idx: the batch index (0, num of batches - 1)
    
    Returns:
           batch_images: the batch x-data tensor
           batch_labels: the batch y-data tensor
  """
  
  data_batches.reset()
  for bidx, data in enumerate(data_batches):
    batch_images, batch_labels = data

    if bidx == batch_idx:
      break

  """
  #--Rest total_batches_seen, this is a factor in generating batches.
     Otherwise a new random set of images make up the same 'batch'
     and passing the same batch_idx will result in a different image. 
  """
  data_batches.total_batches_seen -= batch_idx + 1 #<--- This allows repeatability!  
  
  return batch_images, batch_labels


def get_outputs(batch_images,layers):
  """
    Will save the outputs of each layer. 
  """
  #-- Setup
  outputs = []
  output = batch_images
  for Lidx,layer in enumerate(layers):
    output = layer(output)         #-- Replaces previous layer output, with current layer output.   
    output.layer_name = layer.name #-- Label Layer Output
    output.layer_idx = Lidx
    outputs.append(output)         #-- Add to list
#    if Lidx == layer_idx:
#     break

  return outputs  