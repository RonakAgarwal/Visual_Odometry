# %%
import os
import cv2
from UndistortImage import UndistortImage
import numpy as np
from ReadCameraModel import ReadCameraModel
import matplotlib.pyplot as plt

# Read camera model
fx, fy, cx, cy, _, LUT = ReadCameraModel("./Oxford_dataset_reduced/model")

# Compute intrinsic matrix K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])


prev_R = np.eye(3)
prev_T = np.zeros((3, 1))
rotations = [prev_R]
translations = [prev_T]

print("Intrinsic Matrix K:")
print(K)

# %%
orb = cv2.ORB_create()

image_folder = './Oxford_dataset_reduced/images/'

image_files = sorted(os.listdir(image_folder))

rotations = []
translations = []
essentials = []

ticker1 = 0

# Load and process images
for i in range(50):
    ticker1 += 1
    print("ticker 1", ticker1)

    # Load images
    imported_image = image_folder + image_files[i]
    imported_image_next = image_folder + image_files[i+1]
    #imported_image = image_files[i]
    #imported_image_next = image_files[i+1]

    # Load Bayer pattern encoded images
    bayer_image = cv2.imread(imported_image, flags=-1)
    bayer_image_next = cv2.imread(imported_image_next, flags=-1)

    # Demosaic images
    color_img = cv2.cvtColor(bayer_image, cv2.COLOR_BayerGR2BGR)
    color_img_next = cv2.cvtColor(bayer_image_next, cv2.COLOR_BayerGR2BGR)

    # Undistort images using provided script and LUT
    undistorted_img = UndistortImage(color_img, LUT)
    undistorted_img_next = UndistortImage(color_img_next, LUT)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(undistorted_img, None)
    kp2, des2 = sift.detectAndCompute(undistorted_img_next, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = sorted(bf.match(des2, des1), key=lambda x: x.distance)[:500]
    for m in matches:
        print(m.distance)

    
    src_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    fundamental_matrix, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

    E = np.dot(np.dot(K.T, fundamental_matrix), K)
    essentials.append(E)

    _, R, T, _ = cv2.recoverPose(E, src_pts, dst_pts, K)

    # Append the current rotation and translation to the lists
    rotations.append(R)
    translations.append(T)

    
print(essentials)
print(rotations)
print(translations)



ticker2 = 0
# Initialize camera center positions
camera_centers = [(0, 0, 0)]

u = np.identity(4)

# Compute camera center positions
for i in range(50):
    # Compute the transformation matrix T between camera i and camera i+1
    T = np.vstack([np.hstack([rotations[i], translations[i]]), [0, 0, 0, 1]])

    u = np.dot(T, u)
    
    # Compute the inverse of the transformation matrix T^-1
    T_inv = np.linalg.inv(u)
    
    # Multiply the camera center position of camera i with the inverse transformation matrix T^-1
    camera_center_i_minus_1 = np.dot(T_inv, np.append(camera_centers[i], 1))
    
    # Append the camera center position to the list of camera center positions
    camera_centers.append(camera_center_i_minus_1[:3])

# Convert camera center positions to numpy array
camera_centers = np.array(camera_centers)

# Plot camera center positions in 2D
plt.figure(figsize=(8, 6))
plt.plot(camera_centers[:, 0], camera_centers[:, 1])
plt.title('Camera Center Positions (2D)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
plt.savefig('plot50.png')

