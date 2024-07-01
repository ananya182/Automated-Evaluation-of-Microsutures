import cv2
import numpy as np
import canny_edge_detector as ced
import task2
import task3
import os
import pandas as pd

def threshold_filter(image, threshold_value):
    return np.where(image <= threshold_value, 0, 255).astype(np.uint8)

def dilate(image, kernel):

    rows, cols = image.shape
    krows, kcols = kernel.shape

    padded_image = np.pad(image, ((krows//2, krows//2), (kcols//2, kcols//2)), mode='constant')

    dilated_image = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i+krows, j:j+kcols]
            max_value = np.max(region * kernel)
            dilated_image[i, j] = max_value

    return dilated_image

def main(img_dir, output_csv):
    image_files = os.listdir(img_dir)
    image_paths=[]
    for i in image_files:
        image_paths.append(os.path.join(img_dir, i))

    preprocessed_imgs = preprocess_images(image_paths)
    imgs_final = detect_edges(preprocessed_imgs)
    imgs_final=postprocess_images(imgs_final)

    results=[]

    for i in range(len(image_files)):
        matrix = imgs_final[i]
        d={}
        num_components, centroid,leftmostpt = count_connected_components(matrix)
        d["image name"]=image_files[i]
        d["number of sutures"]=num_components
        mean,var=task2.inter_suture_distance(centroid)
        height = imgs_final[i].shape[0]
        d["mean inter suture spacing"]=mean/height
        d["variance of inter suture spacing"]=var/(height**2)  
        d["mean suture angle wrt x-axis"],d["variance of suture angle wrt x-axis"]=task3.find_angle_mean_var(centroid,leftmostpt)
        results.append(d)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)


def preprocess_images(image_paths):
    imgs=[]
    for i in image_paths:
        original_image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

        kernel_size = (1, 13)  
        kernel = np.ones(kernel_size, np.uint8)

        dilated_image = dilate(original_image, kernel)

        threshold_value = 90
        threshold_image = threshold_filter(dilated_image, threshold_value)

        kernel_size = (1, 9)  
        kernel = np.ones(kernel_size, np.uint8)
        threshold_image = dilate(threshold_image, kernel)

        imgs.append(threshold_image)

    return imgs

def detect_edges(image_list):
    detector = ced.cannyedgedetector(image_list, sigma=6.5, kernel_size=5, lowthreshold=0.9, highthreshold=0.9, weak_pixel=150)
    imgs_final = detector.detect()
    return imgs_final

def postprocess_images(imgs_final):

    for i in range(len(imgs_final)):
        kernel_size=(1,120)
        kernel = np.ones(kernel_size, np.uint8)
        imgs_final[i] = dilate(imgs_final[i].astype(np.uint8), kernel)
        kernel_size=(6,1)
        kernel = np.ones(kernel_size, np.uint8)
        imgs_final[i] = dilate(imgs_final[i], kernel)
    
    return imgs_final
    

def count_connected_components(matrix):
    rows, cols = len(matrix), len(matrix[0])
    visited = [[False] * cols for _ in range(rows)]

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and matrix[x][y] == 255 and not visited[x][y]

    def dfs(x, y):
        stack = [(x, y)]
        component_size = 0
        sumx=0
        sumy=0
        leftx=0
        lefty=100000000

        while stack:
            current_x, current_y = stack.pop()
            if not visited[current_x][current_y]:
                visited[current_x][current_y] = True
                component_size += 1
                sumx+=current_x
                sumy+=current_y
                if(current_y<lefty):
                    leftx=current_x
                    lefty=current_y

                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if is_valid(current_x + i, current_y + j):
                            stack.append((current_x + i, current_y + j))

        return component_size, sumx, sumy, leftx, lefty

    components = []
    
    centroid={}
    leftmostpt={}
    start=0
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 255 and not visited[i][j]:
                component_size,sumx,sumy,leftx,lefty = dfs(i, j)
                components.append(component_size)
                centroid[start]=(sumx/component_size,sumy/component_size)
                leftmostpt[start]=(leftx,lefty)
                start+=1

    return len(components), centroid, leftmostpt


