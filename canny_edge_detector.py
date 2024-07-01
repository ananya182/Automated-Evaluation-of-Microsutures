import numpy as np

class cannyedgedetector:
    def __init__(self, imgs, sigma, weak_pixel,kernel_size, lowthreshold, highthreshold,strong_pixel=255):
        self.imgs = imgs
        self.imgs_output = []
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowthreshold = lowthreshold
        self.highthreshold = highthreshold

    def create_sobel(self, img):
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = self.convolve(img, kernel_x)
        Iy = self.convolve(img, kernel_y)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)

    def convolve(self, image, kernel):
        h,w = image.shape
        a,b=kernel.shape
        padded_img = np.pad(image,((a, a), (b, b)), mode='constant')
        flippedkernel = np.flip(kernel)
        result = np.array([[(flippedkernel*padded_img[y:y+a,x:x+b]).sum() for x in range(w)] for y in range(h)],'uint8')
        return result
    
    def create_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1,M-1):
            for j in range(1,N-1):
                q = 255
                r = 255

                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0


        return Z

    def threshold(self, img):

        highthreshold = img.max() * self.highthreshold
        lowthreshold = highthreshold * self.lowthreshold

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highthreshold)

        weak_i, weak_j = np.where((img <= highthreshold) & (img >= lowthreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res

    def hysteresis(self, img):

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                        
        return img
    
    def detect(self):
        for i, img in enumerate(self.imgs):
            self.img_smoothed = self.convolve(img, self.create_kernel(self.kernel_size, self.sigma))
            self.gradient_matrix, self.theta_matrix = self.create_sobel(self.img_smoothed)
            self.nonmaxsupimg = self.non_max_suppression(self.gradient_matrix, self.theta_matrix)
            self.threshold_img = self.threshold(self.nonmaxsupimg)
            img_final = self.hysteresis(self.threshold_img)
            self.imgs_output.append(img_final)

        return self.imgs_output

