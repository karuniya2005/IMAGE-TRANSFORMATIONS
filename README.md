# IMAGE-TRANSFORMATIONS

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required
- Anaconda - Python 3.7
- OpenCV
- NumPy
- Matplotlib

## Algorithm

### Step 1:
Import the required libraries such as OpenCV, NumPy, and Matplotlib.

### Step 2:
Read the input image using `cv2.imread()` and display it.

### Step 3:
Define and apply transformation techniques:
- **Translation** – Shift the image in x and y directions using a translation matrix.  
- **Scaling** – Resize the image using `cv2.resize()` or a scaling matrix.  
- **Shearing** – Apply shearing using an affine matrix and `cv2.warpAffine()`.  
- **Reflection** – Flip the image using `cv2.flip()` or reflection matrices.  
- **Rotation** – Rotate the image using `cv2.getRotationMatrix2D()` and `cv2.warpAffine()`.  
- **Cropping** – Extract a region of interest by slicing the image array.  

### Step 4:
Store the output of each transformed image.

### Step 5:
Display the original and transformed images using Matplotlib for comparison.


## Program:
```
Developed By:KARUNIYA M
Register Number:212223240068

Original Imagae:

import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('q1.png')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
plt.title("Original Image")  
plt.axis('off') 


i)Image Translation

tx, ty = 100, 50  
M_translation = np.float32([[1, 0, tx], [0, 1, ty]]) 
translated_image = cv2.warpAffine(image, M_translation, (image.shape[1], image.shape[0]))
plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))  # Display the translated image
plt.title("Translated Image")  
plt.axis('off')


ii) Image Scaling

fx, fy = 5.0, 2.0 
scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title("Scaled Image")  
plt.axis('off')



iii)Image shearing

shear_matrix = np.float32([[1, 0.5, 0], [0.5, 1, 0]]) 
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
plt.imshow(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB))  # Display the sheared image
plt.title("Sheared Image")
plt.axis('off')


iv)Image Reflection

reflected_image = cv2.flip(image, 2)
plt.imshow(cv2.cvtColor(reflected_image, cv2.COLOR_BGR2RGB))  
plt.title("Reflected Image") 
plt.axis('off')




v)Image Rotation


(height, width) = image.shape[:2] 
angle = 45  
center = (width // 2, height // 2)
M_rotation = cv2.getRotationMatrix2D(center, angle, 1)
rotated_image = cv2.warpAffine(image, M_rotation, (width, height))
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))  
plt.title("Rotated Image") 
plt.axis('off')



vi)Image Cropping


x, y, w, h = 100, 100, 200, 150 
cropped_image = image[y:y+h, x:x+w]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))  
plt.title("Cropped Image")  
plt.axis('off')


```
## Output:
### Original Image

<img width="384" height="470" alt="image" src="https://github.com/user-attachments/assets/e30e7d6a-c4c8-4917-b5d6-7c85e76586be" />


### i)Image Translation

<img width="387" height="457" alt="image" src="https://github.com/user-attachments/assets/63f09198-4e95-42f9-8c51-901d889c3f15" />


### ii) Image Scaling

<img width="609" height="343" alt="image" src="https://github.com/user-attachments/assets/e9b92717-4503-4016-96de-fbf2b0d7413b" />


### iii)Image shearing

<img width="424" height="451" alt="image" src="https://github.com/user-attachments/assets/7ef4089f-ca29-4ee0-9f0c-7ddd77f7001f" />



### iv)Image Reflection

<img width="369" height="462" alt="image" src="https://github.com/user-attachments/assets/9594b6cf-f087-45fb-9fe7-1e9e38e7aa62" />


### v)Image Rotation

<img width="344" height="463" alt="image" src="https://github.com/user-attachments/assets/39256893-766b-429f-a16d-1e3dd762ccfa" />


### vi)Image Cropping

<img width="598" height="460" alt="image" src="https://github.com/user-attachments/assets/2fcbeab6-a0a4-4c4d-aa5d-bc92020427ad" />




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
