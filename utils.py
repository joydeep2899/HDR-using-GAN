import numpy as np 
import math
from PIL import Image
import matplotlib.pyplot as plt


import tensorflow as tf

def weights(pixel_values):
#the weighting function for calculating weights from pixels
 zmin,zmax=0,255

 z=np.random.randint(zmin,zmax)


 zmid= (zmin+zmax)//2

 if (z < zmid):
    w=z-zmin
 else:

    w=zmax-z
 return w    

def sample_pixels(images):
     # sampling the pixels from images 
      zmin,zmax=0,255

      num_images=len(images)
      num_intensities=zmax-zmin+1

      intensites=np.zeros((num_intensities,num_images,3),dtype=np.int64)
     # intensites=[]
      mid_img=images[math.ceil(num_images/2)]
      #print(mid_img.shape)
      for i in range(zmin,zmax+1):
        rows,cols=mid_img.shape[0],mid_img.shape[1]
        #print('rows : {}  cols {}'.format(rows,cols))
        if((rows)!=0):
            idxi=np.random.randint((rows))
            idxj=np.random.randint((cols))
            for j in range(num_images):
                image=images[j]
               # print(i,j)
                pixel=image[idxi][idxj]
                #print(pixel)
                for k in range(3):
                 intensites[i][j][k]=pixel[k]
                 
      return intensites

def compute_response_curve(intensites,log_exposure_times,weighting_func,smooth_lambda ): 
# get the response curve 
    zmin,zmax=0,255
    num_images=len(log_exposure_times)
    num_samples=len(intensites)
    intensity_range=zmax-zmin
    mat_A=np.zeros((num_samples*num_images+intensity_range,num_samples+intensity_range+1),dtype=np.int64)
    mat_B=np.zeros((mat_A.shape[0],1),dtype=np.int64)
    #data-fitting 
    k=0 
    
    for i in range(num_samples):
        for j in range(num_images):
            zij=intensites[i][j]
            wij=weighting_func(zij)
            
            mat_A[k,zij]=wij
            mat_A[k,(intensity_range+1)+i]=-wij
            mat_B[k,0]= wij*log_exposure_times[j]
            k+=1
    

    for zk in range(zmin+1,zmax):
            wzk=weighting_func(zk)
            
            mat_A[k,zk-1]=wzk*smooth_lambda
            mat_A[k,zk]=-2*wzk*smooth_lambda
            mat_A[k,zk+1]=wzk*smooth_lambda
            k+=1




    mat_Ainv=np.linalg.pinv(mat_A)

    x=np.dot(mat_Ainv,mat_B)
    g=x[0:intensity_range+1]

    return g[:,0]



def radiance_map(images,compute_response_curve,weighting_func,log_exposure_times):
 #get the radiance map from the response curve
  num_images=len(log_exposure_times)
  img_shape =images[0].shape
  image_radiance_map =np.zeros(img_shape,dtype=np.float64)
  
  #for i in range(img_shape[0]):
   #   for j in  range( img_shape[1]):
  for  i in range(256):
      for j in range(256): 
         # g=compute_response_curve(images[k][i][j] for k in range(num_images))
         # w=weighting_func(images[k][i][j] for k in range(num_images))
          g = np.array([response_curve[images[k][i][j]] for k in range(num_images) ])
         # g=compute_response_curve
          w=  np.array([weighting_func(images[k][i][j] for k in range(num_images)) ])

          weight_sum=np.sum(w)
         # weight_sum=0
          if (weight_sum > 0): 
                #how do I KNOW WHICH LOG EXPOSURE TIME TO SUBTRACT
                image_radiance_map[i][j]=  np.sum(np.dot(w[0],np.divide((g - log_exposure_times[0] ),weight_sum)))
          else :
               
               image_radiance_map[i][j]=np.sum(g[num_images//2] - log_exposure_times[num_images//2])            
  return image_radiance_map  


def local_tone_mapping(image,gamma):
     # gamma is an 2d array of the same shape as of image and contains the gamma values for each pixels
     #http://cs.brown.edu/courses/cs129/results/proj5/njooma/
     #https://en.wikipedia.org/wiki/Tone_mapping

     ### tone mapping is done using gamma compression########


     ##### local tone mapping methods gradient domain high dynamic compression #######
     rows,cols=image.shape[0],image.shape[1]
     assert image.shape==gamma.shape
     for i in range(rows):
         for j in range(cols):

             image[i][j]=math.pow(image[i][j],gamma[i][j])

     # each pixel is tone mapped differently accoring to location      
     # 
########### CALCULATING THE EV values at a given ISO >>> N=stop number at f/stop number ex: if aperture=f/4 then N=4 and t=exposure time or shutter speed######  
#   EV=log2(N^2/t)
#   more info -- https://www.scantips.com/lights/evchart.html    https://en.wikipedia.org/wiki/Exposure_value
##############################################

def EV(N,T,ISO):
     n=N
     t=T
     iso=ISO

     EV = math.log(math.pow(N,2)/T,2)-math.log((iso/100),2)
     EV=round(EV,3)
     return EV


def mean_squared_error(i1,i2):
    m1,n1=i1.shape
    m2,n2=i2.shape
    if((m1==m2) and (n1==n2)):
       m=m1
       n=n1
    sum=0
    diff=0
    for i in range(m):
        for j in range(n):
             diff=i1[i,j]-i2[i,j]
             diff_sq=diff*diff
             sum=sum+diff_sq
    mse=sum/(m*n)
    return mse
def psnr(mean_squared_error,i1,i2):
    R=1
    mse=mean_squared_error(i1,i2)
    psnr=10*math.log((R*R)/mse,10)  
    return psnr  

"""


imglist=[]        
for i in range(5): 
 img=np.array(Image.open('./images/ev'+str(i)+'.jpg'))
 #np.reshape(img,(256,256,3))
 imglist.append(img)
log_exp=[1/17,1/10,1/10,1/33,1/33]

#print(len(imglist))
imgstack=np.stack(imglist)
#print(imgstack)             
response_curve=compute_response_curve(sample_pixels(imgstack),log_exp,weights,0.6)             

radiance_map=radiance_map(imgstack,response_curve,weights,log_exp)

print("response curve {}\n radiance map {}".format(response_curve,radiance_map))

print('response curve len {}  radiance map len {}  '.format(len(response_curve),len(radiance_map)))



imgplot = plt.imshow(radiance_map)
plt.show()
"""
#print(EV(1,1,100))
#f= open('./images/ev0.jpg','rb')
#a=f.read()      
#print(a[:500])      
### ord function   https://stackoverflow.com/questions/5851607/python3-parsing-jpeg-dimension-info   parsing file struct

         
### 11 1/500  1/400 10 8 1/250
      
         







