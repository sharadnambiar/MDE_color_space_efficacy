#Code to perform image segmentation on an input image using K means algorithm to create Ground Truth (GT)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2
import torch
from PIL import Image
from skimage.measure import compare_ssim

%matplotlib inline

imdir, filename = (r"<DIR>/01.png", "image.png")


# Reading image from the image ,Channel identifier : Blue 0 , Green 1 , Red 2
image = cv2.imread(imdir)
 
    
figure(figsize=(15, 10), dpi=80)

# Change color to RGB (from BGR) and new Channel identifier : Red 0 , Green 1 , Blue 2
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
    
plt.subplot(2,1,1)   
plt.imshow(image)
plt.title("Input image")



# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)

print(pixel_vals.shape)

#the below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
#becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# then perform k-means clustering wit h number of clusters defined as 3
#also random centres are initially choosed for k-means clustering
k = 3
# k=4
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]



# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))

plt.subplot(2,1,2)   
plt.imshow(segmented_image)
plt.title("Segmented_image")

print(segmented_image.shape)

segmented_image_exp=np.copy(segmented_image)

#Loading MiDas

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
    
    
 #Code to generate depth map using Midas SoA model for segmented image


input_image_seg = transform(segmented_image).to(device)

with torch.no_grad():
    prediction = midas(input_image_seg)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=segmented_image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output_seg_img = prediction.cpu().numpy()


figure(figsize=(15, 10), dpi=80)
print("Depth Map ")
plt.imshow(output_seg_img)


#Code to generate depth map using Midas SoA model for original image

image = cv2.imread(imdir)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_image = transform(image).to(device)

with torch.no_grad():
    prediction = midas(input_image)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output_ori_img = prediction.cpu().numpy()


figure(figsize=(15, 10), dpi=80)
print("Depth Map ")
plt.imshow(output_ori_img)

#Code to compare two images using  SSIM metrics (Structural Similarity Index)
image_ssim = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
segmented_image_ssim = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

score_ori_dep = compare_ssim(image_ssim, output_ori_img,GPU=False,Full=True)
print("SSIM (Original image vs Depth map): ",score_ori_dep)

score_seg_dep = compare_ssim(segmented_image_ssim, output_seg_img,GPU=False,Full=True)
print("SSIM (Segmented image vs Depth map): ",score_seg_dep)



#Experiments

#Code to use a segmented image , make two channels zero and then create a depth map for it 
#Channel identifier : Red 0 , Green 1 , Blue 2
#Channels made zero are Blue and Green  


img_red = np.copy(segmented_image_exp)

#Making channel Green (1) zero
img_red[:,:,1] = np.zeros([img_red.shape[0], img_red.shape[1]])

#Making channel Blue (2) zero
img_red[:,:,2] = np.zeros([img_red.shape[0], img_red.shape[1]])

input_batch_red = transform(img_red).to(device)


with torch.no_grad():
    prediction = midas(input_batch_red)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_red.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output_red = prediction.cpu().numpy()

print(output_red.shape)

print("Depth Map for image with only Red channel")


plt.subplot(2,1,1)
plt.title("GT -Red channel")
plt.imshow(img_red)
plt.subplot(2,1,2)
plt.title("Output depth map")
plt.imshow(output_red)

#Code to use a segmented image , make two channels zero and then create a depth map for it 
#Channel identifier : Red 0 , Green 1 , Blue 2
#Channels made zero are Blue and Red  




img_green = np.copy(segmented_image_exp)

#Making channel Red (2) zero
img_green[:,:,2] = np.zeros([img_green.shape[0], img_green.shape[1]])

#Making channel Blue (0) zero
img_green[:,:,0] = np.zeros([img_green.shape[0], img_green.shape[1]])

input_batch_green = transform(img_green).to(device)

with torch.no_grad():
    prediction = midas(input_batch_green)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_green.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output_green = prediction.cpu().numpy()

print("Depth Map for image with only Green channel")



plt.subplot(2,1,1)
plt.title("GT -Blue channel")
plt.imshow(img_green)
plt.subplot(2,1,2)
plt.title("Output depth map")
plt.imshow(output_green)



#Code to use a segmented image , make two channels zero and then create a depth map for it 
#Channel identifier : Red 0 , Green 1 , Blue 2
#Channels made zero are Green and Red  

img_blue = np.copy(segmented_image_exp)


#Making channel Red (0) zero
img_blue[:,:,0] = np.zeros([img_blue.shape[0], img_blue.shape[1]])

#Making channel Green (1) zero
img_blue[:,:,1] = np.zeros([img_blue.shape[0], img_blue.shape[1]])

input_batch_blue = transform(img_blue).to(device)

with torch.no_grad():
    prediction = midas(input_batch_blue)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_blue.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output_blue = prediction.cpu().numpy()

print("Depth Map for image with only Blue channel")

plt.subplot(2,1,1)
plt.title("GT -Blue channel")
plt.imshow(img_blue)
plt.subplot(2,1,2)
plt.title("Output depth map")
plt.imshow(output_blue)

image_red_ssim = cv2.cvtColor(img_red, cv2.COLOR_RGB2GRAY)
image_green_ssim = cv2.cvtColor(img_green, cv2.COLOR_RGB2GRAY)
image_blue_ssim = cv2.cvtColor(img_blue, cv2.COLOR_RGB2GRAY)

score_seg_red = compare_ssim(image_red_ssim, output_red,GPU=False,Full=True)
print("SSIM (GT - Red Channel vs Depth map): ",score_seg_red)

score_seg_green = compare_ssim(image_green_ssim, output_green,GPU=False,Full=True)
print("SSIM (GT- Green Channel vs Depth map): ",score_seg_green)

score_seg_blue = compare_ssim(image_blue_ssim, output_blue,GPU=False,Full=True)
print("SSIM (GT - Blue Channel vs Depth map): ",score_seg_blue)

#Code to plot images and corresponding depth maps


fig, ax = plt.subplots(3, 2, figsize=(16, 8))
fig.tight_layout()
plt.subplot(2,2,1)
plt.title("Original Image ",)
plt.imshow(image)
plt.subplot(2,2,2)
plt.title("Output depth map (Original Image)")
plt.imshow(output_ori_img)
plt.subplot(2,2,3)
plt.title("Ground Truth ",)
plt.imshow(segmented_image)
plt.subplot(2,2,4)
plt.title("Output depth map (Ground Truth)")
plt.imshow(output_seg_img)


fig, ax = plt.subplots(3, 2, figsize=(16, 8))
fig.tight_layout()
plt.subplot(3,2,1)
plt.title("GT - Red channel",)
plt.imshow(img_red)
plt.subplot(3,2,2)
plt.title("Output depth map ")
plt.imshow(output_red)
plt.subplot(3,2,3)
plt.title("GT - Green channel",)
plt.imshow(img_green)
plt.subplot(3,2,4)
plt.title("Output depth map ")
plt.imshow(output_green)
plt.subplot(3,2,5)
plt.title("GT - Blue channel ",)
plt.imshow(img_blue)
plt.subplot(3,2,6)
plt.title("Output depth map ")
plt.imshow(output_blue)



#Experiments with differnet colour space

#LUV

img_luv = np.copy(segmented_image_exp)

#converting the image to new LUV colour space

img_luv=cv2.cvtColor(img_luv, cv2.COLOR_RGB2Luv)



input_batch_luv = transform(img_luv).to(device)


with torch.no_grad():
    prediction = midas(input_batch_luv)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_luv.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output_luv = prediction.cpu().numpy()

print(output_luv.shape)

print("Depth Map for image with LUV colour space")
plt.imshow(output_luv)


#YCrCb
img_ycc = np.copy(segmented_image_exp)

#converting the image to new LUV colour space

img_ycc=cv2.cvtColor(img_ycc, cv2.COLOR_RGB2YCrCb)



input_batch_ycc = transform(img_ycc).to(device)


with torch.no_grad():
    prediction = midas(input_batch_ycc)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_ycc.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output_ycc = prediction.cpu().numpy()

print(output_ycc.shape)

print("Depth Map for image with ycc colour space")
plt.imshow(output_ycc)



#YUV
img_yuv = np.copy(segmented_image_exp)

#converting the image to new LUV colour space

img_yuv=cv2.cvtColor(img_yuv, cv2.COLOR_RGB2YUV)



input_batch_yuv = transform(img_yuv).to(device)


with torch.no_grad():
    prediction = midas(input_batch_yuv)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_yuv.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output_yuv = prediction.cpu().numpy()

print(output_yuv.shape)

print("Depth Map for image with yuv colour space")
plt.imshow(output_yuv)


#LAB
img_lab = np.copy(segmented_image_exp)

#converting the image to new LUV colour space

img_lab=cv2.cvtColor(img_lab, cv2.COLOR_RGB2Lab)



input_batch_lab = transform(img_lab).to(device)


with torch.no_grad():
    prediction = midas(input_batch_lab)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_lab.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output_lab = prediction.cpu().numpy()

print(output_lab.shape)

print("Depth Map for image with lab colour space")
plt.imshow(output_lab)



fig, ax = plt.subplots(3, 2, figsize=(20,15))
fig.tight_layout()
plt.subplot(4,2,1)
plt.title("GT - LUV colour space ",)
plt.imshow(img_luv)
plt.subplot(4,2,2)
plt.title("Output depth map ")
plt.imshow(output_luv)
plt.subplot(4,2,3)
plt.title("GT - YCC colour space ",)
plt.imshow(img_ycc)
plt.subplot(4,2,4)
plt.title("Output depth map ")
plt.imshow(output_ycc)
plt.subplot(4,2,5)
plt.title("GT - YUV colour space ",)
plt.imshow(img_yuv)
plt.subplot(4,2,6)
plt.title("Output depth map ")
plt.imshow(output_yuv)
plt.subplot(4,2,7)
plt.title("GT - LAB colour space ",)
plt.imshow(img_lab)
plt.subplot(4,2,8)
plt.title("Output depth map ")
plt.imshow(output_lab)



image_luv_ssim = cv2.cvtColor(img_luv, cv2.COLOR_LUV2RGB)
image_luv_ssim = cv2.cvtColor(image_luv_ssim, cv2.COLOR_RGB2GRAY)

score_seg_luv = compare_ssim(image_luv_ssim, output_luv,GPU=False,Full=True)
print("SSIM (GT - LUV vs Depth map): ",score_seg_luv)


image_ycc_ssim = cv2.cvtColor(img_ycc, cv2.COLOR_YCrCb2RGB)
image_ycc_ssim = cv2.cvtColor(image_ycc_ssim, cv2.COLOR_RGB2GRAY)

score_seg_ycc = compare_ssim(image_ycc_ssim, output_ycc,GPU=False,Full=True)
print("SSIM (GT - YCC vs Depth map): ",score_seg_ycc)


image_yuv_ssim = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
image_yuv_ssim = cv2.cvtColor(image_yuv_ssim, cv2.COLOR_RGB2GRAY)

score_seg_yuv = compare_ssim(image_yuv_ssim, output_yuv,GPU=False,Full=True)
print("SSIM (GT - YUV vs Depth map): ",score_seg_yuv)


image_lab_ssim = cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)
image_lab_ssim = cv2.cvtColor(image_lab_ssim, cv2.COLOR_RGB2GRAY)

score_seg_lab = compare_ssim(image_lab_ssim, output_lab,GPU=False,Full=True)
print("SSIM (GT - LAB vs Depth map): ",score_seg_lab)





