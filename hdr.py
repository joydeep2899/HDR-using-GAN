import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers as tfl
from PIL import Image
import matplotlib.pyplot as plt
from utils import EV
from mprelu import MPReLU
from loss import loss
#img =cv2.imread("test.jpeg")
print(tfl)
#original_img=np.asarray(img)
#print(original_img)

#hYPER parameters 

epsilon=0.00005





MPReLU=MPReLU()








class hdrGAN:
    def __init__(self,inputimg):
       
       
        self.img=inputimg
        self.input_dim=self.img.shape
    def gplus(self,input_dim,output_dim):
          # with tf.compat.v1.VariableScope(reuse=False,name="generator") as scope: 
           
            model=tf.keras.Sequential()
       ############################################################ down sampling layers ############################
       
           # model.add(tfl.InputLayer(input_shape=input_dim,dtype=tf.int16))
            
           # model.add(tfl.Conv2D(filters=3,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
               # first block 
            
            model.add(tfl.Conv2D(filters=64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())

            #second block

            model.add(tfl.Conv2D(filters=128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block2'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())

            #third  block

            model.add(tfl.Conv2D(filters=256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block3'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())



            #fourth block

            model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block4'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())
             
            #fifth block

            model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block5'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())
            
         

         ############################################################ UP sampling layers ############################

            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2D(filters=2*512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block6'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())
               

            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2D(filters=2*256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block7'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())


            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2D(filters=2*128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block8'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())

            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2DTranspose(filters=2*64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block9'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())
            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2DTranspose(filters=2*3,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block10'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())  
            


           
            model.add(tfl.Conv2DTranspose(filters=3,kernel_size=(10,10),strides=(2,2),padding='same',name='deconvolution layer11',data_format="channels_last"))
           # model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())  
            # model.add(tfl.Flatten())
            
            return model  


    def gminus(self,input_dim,output_dim):
          # with tf.compat.v1.VariableScope(reuse=False,name="generator") as scope: 
           
            model=tf.keras.Sequential()
       ############################################################ down sampling layers ############################
       
           # model.add(tfl.InputLayer(input_shape=input_dim,dtype=tf.int16))
            
           # model.add(tfl.Conv2D(filters=3,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
               # first block 
            
            model.add(tfl.Conv2D(filters=64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())

            #second block

            model.add(tfl.Conv2D(filters=128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block2'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())

            #third  block

            model.add(tfl.Conv2D(filters=256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block3'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())



            #fourth block

            model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block4'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())
             
            #fifth block

            model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block5'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())
            
         

         ############################################################ UP sampling layers ############################

            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2D(filters=2*512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block6'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())
               

            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2D(filters=2*256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block7'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())


            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2D(filters=2*128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block8'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())

            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2DTranspose(filters=2*64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block9'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())
            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2DTranspose(filters=2*3,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block10'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())  
            


           
            model.add(tfl.Conv2DTranspose(filters=3,kernel_size=(10,10),strides=(2,2),padding='same',name='deconvolution layer11',data_format="channels_last"))
           # model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(MPReLU())  
            # model.add(tfl.Flatten())
            
            return model  
  

    def dplus(self,input_dim,output_dim):
                      # with tf.compat.v1.VariableScope(reuse=False,name="discriminator") as scope: 
                      inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
                      tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

                      model=tf.keras.Sequential()       
                     
                     # model.add(tfl.InputLayer(input_shape=input_dim,dtype=tf.int16))
                      model.add(tf.keras.layers.concatenate([inp, tar]))

                      model=tf.keras.Sequential()
                      model.add(tfl.Conv2D(filters=6,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block0'))
                      
                      # first block 
                        
                      model.add(tfl.Conv2D(filters=64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
                      model.add(tfl.PReLU())

                      #second block

                      model.add(tfl.Conv2D(filters=128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block2'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(tfl.PReLU())

                      #third  block

                      model.add(tfl.Conv2D(filters=256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block3'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(tfl.PReLU())



                      #fourth block

                      model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block4'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(tfl.PReLU())
                      
                      #fifth block

                      model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block5',activation=tf.keras.activations.sigmoid))
                      model.add(tfl.PReLU())
                      
                      
                      model(inputs=[inp,tar],outputs=last) 
            
            
    def dminus(self,input_dim,output_dim):
                      initializer = tf.random_normal_initializer(0., 0.02)

                      inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
                      tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

                      model=tf.keras.Sequential()       
                     
                     # model.add(tfl.InputLayer(input_shape=input_dim,dtype=tf.int16))
                      model.add(tf.keras.layers.concatenate([inp, tar]))
                      

                      model.add(tfl.Conv2D(filters=6,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block0'))
                      
                      # first block 
                      
                      model.add(tfl.Conv2D(filters=64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
                      model.add(MPReLU())

                      #second block

                      model.add(tfl.Conv2D(filters=128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block2'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(MPReLU())

                      #third  block

                      model.add(tfl.Conv2D(filters=256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block3'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(MPReLU())



                      #fourth block

                      model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block4'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(MPReLU())
                      
                      #fifth block

                      model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block5',activation=tf.keras.activations.sigmoid))
                      model.add(MPReLU())

                      return model(inputs=[inp,tar],outputs=last)         


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)








img=np.array(Image.open('./images/ev'+str(0)+'.jpg'),dtype=np.int16)

#img=np.array(Image.open('./test.jpeg'),dtype=np.float32)
i,j,k=img.shape[0],img.shape[1],img.shape[2]

img=tf.image.resize(img,(256,256))
img=np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
tar=img
GAN=hdrGAN(img)        
print(img.shape)

gplus=GAN.gplus(img.shape,img.shape)
gminus=GAN.gminus(img.shape,img.shape)
dplus=GAN.dplus(img.shape,img.shape)
dminus=GAN.dminus(img.shape,img.shape)

loss=loss(GAN.gplus,GAN.gminus,GAN.dplus,GAN.dminus)



#### entering training 


gplus_output=gplus(img,training=True)
dplus_output=dplus([img,tar],training=True)

plus_loss=loss.ganloss(img,tar,'gplus')
gplus_loss=plus_loss[0]
dplus_loss=plus_loss[1]

minus_loss=loss.ganloss(img,tar,'gminus')
gminus_loss=minus_loss[0]
dminus_loss=minus_loss[1]

gen_tape=tf.GradientTape()
disc_tape=tf.GradientTape()

gplus_graident=gen_tape.gradient(gplus_loss,gplus.trainable_variables)
dplus_gradient=disc_tape.gradient(dplus_loss,dplus.trainable_variables)

generator_optimizer.apply_gradients(zip(gplus_graident,gplus.trainable_variables))
discriminator_optimizer.apply_gradients(zip(dplus_gradient,dplus.trainable_variables))



#generator=np.reshape(generator,(1,256,256,3))
"""
img=generator[0]*256
#img=tf.dtypes.cast(img, tf.int16)
print(img)
imgplot = plt.imshow(img)
plt.show()

#tf.image.encode_png(img)
print(generator.shape)
        """                 
  ###########################image dataset  https://www2.cs.sfu.ca/~colour/data/funt_hdr/          #########33333333  
  ######################   https://www.scss.tcd.ie/Emin.Zerman/databaseInformations/_hdrImgVidList.html   ##################             
