import numpy as np
import math

import tensorflow as tf
input_dim=(128,128,3)

# #########    https://www.tensorflow.org/tutorials/generative/pix2pix ######
############   https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py        ###########
#https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

class loss :

    def __init__(self,gplus,gminus,dplus,dminus):
        self.Dplus=dplus
        self.Dminus=dminus
        self.Gplus=gplus
        self.Gminus=gminus
       
  
    def l1loss(self,x,y,gloss):
         if(gloss=='gplus'):
           G=self.Gplus
         elif(gloss=='gminus'):
           G=self.Gminus

            
         l1loss= np.subtract(y,G(x))
         if(l1loss < 0):
           l1loss=-l1loss
      
         return l1loss


    def ganloss(self,x,y,gloss):
         concat=tf.keras.layers.concatenate
         if(gloss=='gplus'):
           G=self.Gplus
           D=self.Dplus
         elif(gloss=='gminus'):
           G=self.Gminus
           D=self.Dminus
        
         Gloss=np.square(D.predict(concat([G.predict(x),x])) - 1 ) 
         Dloss= 0.5*(np.square(D.predict(concat([y,x]))-1) ) + 0.5*(np.square(D.predict(concat([G.predict(x),x])) -1))
          
         total_loss=Gloss
          
         
         return Gloss,Dloss,total_loss
    
        

     
            
           
