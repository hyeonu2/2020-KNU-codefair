import RPi.GPIO as gp
import os
import cv2
import sys
import time
import numpy as np
import cv2
# from matplotlib import pyplot as plt
from PIL import Image
import math

gp.setwarnings(False)
gp.setmode(gp.BOARD)

gp.setup(7, gp.OUT)
gp.setup(11, gp.OUT)
gp.setup(12, gp.OUT)

gp.setup(15, gp.OUT)
gp.setup(16, gp.OUT)
gp.setup(21, gp.OUT)
gp.setup(22, gp.OUT)

gp.output(11, True)
gp.output(12, True)
gp.output(15, True)
gp.output(16, True)
gp.output(21, True)
gp.output(22, True)

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.10

def find_matrix(im1,im2):
    
    # Convert images to grayscale
#     im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#     im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im1Gray = im1
    im2Gray = im2
    
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

     # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    
    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("1_matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    
    return h

def Warping (im1,im2,h) :
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    
    return im1Reg    #이것은 이미지1를 이미지2에 맞게 warping한것이다

def singleScaleRetinex(img,sigma):
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imGray = np.double(imGray)
    row,col = imGray.shape
    retinex = np.zeros((row,col))
    
    retinex[:,:] = np.log10(imGray[:,:]) - np.log10(cv2.GaussianBlur(imGray[:,:],(11,11),sigma))
    im = cv2.normalize(retinex,None,0.0,1.0,cv2.NORM_MINMAX, cv2.CV_64F)
    im = im * 255
    im = np.uint8(im)
    return im


def split_wt(wt_img):
#    row, col = wt_img.shape[:2]
    row = len(wt_img)
    col = len(wt_img[0])
    row2 = int(row/2)
    col2 = int(col/2)
   
    LL = np.zeros((int(row/2), int(col/2)))
    LH = np.zeros((int(row/2), int(col/2)))
    HL = np.zeros((int(row/2), int(col/2)))
    HH = np.zeros((int(row/2), int(col/2)))
   
    for x in range(0,col2):
        for y in range(0, row2):
            LL[y][x] = wt_img[y][x]
           
    for x in range(0,col2):
        for y in range(row2, row):
            LH[y-row2][x] = wt_img[y][x]
           
    for x in range(col2, col):
        for y in range(0, row2):
            HL[y][x-col2] = wt_img[y][x]
           
    for x in range(col2, col):
        for y in range(row2, row):
            HH[y-row2][x-col2] = wt_img[y][x]
           
   
    return LL, LH, HL, HH

def combine_wt(LL, LH, HL, HH):
#     row, col = LL.shape[:2]
     row = len(LL)
     col = len(LL[0])  
     row2= row * 2
     col2 = col *2
     
     wt_img = np.zeros((row2,col2))
     
     for x in range(0,col):
        for y in range(0, row):
            wt_img[y][x] = LL[y][x]
           
     for x in range(0,col):
        for y in range(row, row2):
            wt_img[y][x] = LH[y - row][x]
           
     for x in range(col, col2):
        for y in range(0, row):
            wt_img[y][x] = HL[y][x-col]
           
     for x in range(col, col2):
        for y in range(row, row2):
           wt_img[y][x] =  HH[y-row][x-col]
     
     return wt_img
     
     


   
def fwt97_2d(m, nlevels=1):
    ''' Perform the CDF 9/7 transform on a 2D matrix signal m.
    nlevel is the desired number of times to recursively transform the
    signal. '''

    w = len(m[0])
    w1 = int(w/2)
    h = len(m)
    h1 = int(h/2)
   
    m = fwt97(m, w, h) # cols
    m = fwt97(m, h, w) # rows
   
    LL,LH,HL,HH = split_wt(m)
   
    LL1 = fwt97(LL, w1, h1) # rows  
    LL1 = fwt97(LL1, h1, w1) # rows

   
    wt_img = combine_wt(LL1,LH,HL,HH)
   
    return wt_img
       
# 이것을 다음 한 번 더 돌릴때는 ll만 넣어서 돌리고 7장 이미지 만들어서 4장 처리하고 하나 만들어 진거를 또 4장씩 처리.        
     
def iwt97_2d(m, nlevels=1):
    ''' Inverse CDF 9/7 transform on a 2D matrix signal m.
        nlevels must be the same as the nlevels used to perform the fwt.
    '''

    w = len(m[0])
    h = len(m)
    w1 = int(w/2)
    h1 = int(h/2)

    # Find starting size of m:
#    for i in range(nlevels-1):
#        w = int(w/2)
#        h = int(h/2)
    LL,LH,HL,HH = split_wt(m)

    LL1 = iwt97(LL, w1, h1) # rows
    LL1 = iwt97(LL1, h1, w1) # cols
   
    wt_img = combine_wt(LL1,LH,HL,HH)
   
   
    wt_img1 = iwt97(wt_img, w, h) # rows
    wt_img1 = iwt97(wt_img1, h, w) # cols


    return wt_img1



def fwt97(s, width, height):
    ''' Forward Cohen-Daubechies-Feauveau 9 tap / 7 tap wavelet transform
    performed on all columns of the 2D n*n matrix signal s via lifting.
    The returned result is s, the modified input matrix.
    The highpass and lowpass results are stored on the left half and right
    half of s respectively, after the matrix is transposed. '''

    # 9/7 Coefficients:
    a1 = -1.586134342
    a2 = -0.05298011854
    a3 = 0.8829110762
    a4 = 0.4435068522

    # Scale coeff:
    k1 = 0.81289306611596146 # 1/1.230174104914
    k2 = 0.61508705245700002 # 1.230174104914/2
    # Another k used by P. Getreuer is 1.1496043988602418

    for col in range(width): # Do the 1D transform on all cols:
        ''' Core 1D lifting process in this loop. '''
        ''' Lifting is done on the cols. '''

        # Predict 1. y1
        for row in range(1, height-1, 2):
            s[row][col] += a1 * (s[row-1][col] + s[row+1][col])
        s[height-1][col] += 2 * a1 * s[height-2][col] # Symmetric extension

        # Update 1. y0
        for row in range(2, height, 2):
            s[row][col] += a2 * (s[row-1][col] + s[row+1][col])
        s[0][col] +=  2 * a2 * s[1][col] # Symmetric extension

        # Predict 2.
        for row in range(1, height-1, 2):
            s[row][col] += a3 * (s[row-1][col] + s[row+1][col])
        s[height-1][col] += 2 * a3 * s[height-2][col]

        # Update 2.
        for row in range(2, height, 2):
            s[row][col] += a4 * (s[row-1][col] + s[row+1][col])
        s[0][col] += 2 * a4 * s[1][col]

    # de-interleave
    temp_bank = [[0]*height for i in range(width)]
    for row in range(height):
        for col in range(width):
            # k1 and k2 scale the vals
            # simultaneously transpose the matrix when deinterleaving
            if row % 2 == 0: # even
                r1 =int(row/2)
                temp_bank[col][r1] = k1 * s[row][col]
            else:            # odd
                r2 = int(row/2 + height/2)
                temp_bank[col][r2] = k2 * s[row][col]

    # write temp_bank to s:
    s2 = [[0]*height for i in range(width)]
    for row in range(height):
        for col in range(width):
            s2[col][row] = temp_bank[col][row]

    return s2


def iwt97(s, width, height):
    ''' Inverse CDF 9/7. '''

    # 9/7 inverse coefficients:
    a1 = 1.586134342
    a2 = 0.05298011854
    a3 = -0.8829110762
    a4 = -0.4435068522

    # Inverse scale coeffs:
    k1 = 1.230174104914
    k2 = 1.6257861322319229

    # Interleave:
    temp_bank = [[0]*height for i in range(width)]
    w1 = int(width/2)
    for col in range(w1):
        for row in range(height):
            # k1 and k2 scale the vals
            # simultaneously transpose the matrix when interleaving
            temp_bank[col * 2][row] = k1 * s[row][col]
            temp_bank[col * 2 + 1][row] = k2 * s[row][col + w1]

    # write temp_bank to s:
    s2 = [[0]*height for i in range(width)]
    for row in range(height):
        for col in range(width):
            s2[col][row] = temp_bank[col][row]


    for col in range(height): # Do the 1D transform on all cols:
        ''' Perform the inverse 1D transform. '''

        # Inverse update 2.
        for row in range(2, width, 2):
            s2[row][col] += a4 * (s2[row-1][col] + s2[row+1][col])
        s2[0][col] += 2 * a4 * s2[1][col]

        # Inverse predict 2.
        for row in range(1, width-1, 2):
            s2[row][col] += a3 * (s2[row-1][col] + s2[row+1][col])
        s2[width-1][col] += 2 * a3 * s2[width-2][col]

        # Inverse update 1.
        for row in range(2, width, 2):
            s2[row][col] += a2 * (s2[row-1][col] + s2[row+1][col])
        s2[0][col] +=  2 * a2 * s2[1][col] # Symmetric extension

        # Inverse predict 1.
        for row in range(1, width-1, 2):
            s2[row][col] += a1 * (s2[row-1][col] + s2[row+1][col])
        s2[width-1][col] += 2 * a1 * s2[width-2][col] # Symmetric extension

    return s2


def seq_to_img(m, pix):
    ''' Copy matrix m to pixel buffer pix.
    Assumes m has the same number of rows and cols as pix. '''
    for row in range(len(m)):
        for col in range(len(m[row])):
            pix[col,row] = m[row][col]
#---------------------------------------------------------------
#------------- WaveletFusion function --------------------------
#---------------------------------------------------------------
def  WaveletFusion(C1, C2):
    #return None
   # input: C1, C2 = h by w matrix, wavelet transform of a image channel
    # h by w image height and width respectively
    # output: return the h by w array, fusion of C1 and C2

    # 1. create a new matrix C.
    matrix_C = C1
# rows = len(C1)
# cols = len(C1[0])    
# for i in range(0,rows):
# for j in range(0, cols):
# if abs(C1[i][j]) > abs(C2[i][j]):
# matrix_C[i][j] = C1[i][j]
# else:
# matrix_C[i][j] = C2[i][j]    
   
   
   
    C1_bi = cv2.normalize(C1,None,0.0,1.0,cv2.NORM_MINMAX, cv2.CV_32F)
    C2_bi = cv2.normalize(C2,None,0.0,1.0,cv2.NORM_MINMAX, cv2.CV_32F)
# C1_bi = np.uint8(C1_bi*255)
# C2_bi = np.uint8(C2_bi*255)
    C1_bi = cv2.GaussianBlur(C1_bi,(31,31),0)
    C2_bi = cv2.GaussianBlur(C2_bi,(31,31),0)
# C1_bi = cv2.bilateralFilter(C1_bi,15,75,75)
# C2_bi = cv2.bilateralFilter(C2_bi,15,75,75)    
    # C(i,j) = C1(i,j), if absolute(C1(i,j)) > absolute(C2(i,j))
    # else  C(i,j) = C2(i,j)
    rows = len(C1)
    cols = len(C1[0])

    for i in range(0,rows):
        for j in range(0, cols):
# if abs(C1[i][j]) > abs(C2[i][j]):
            matrix_C[i][j] = (C1_bi[i][j]*C1[i][j])/(C1_bi[i][j]+C2_bi[i][j]) + (C2_bi[i][j]*C2[i][j])/(C1_bi[i][j]+C2_bi[i][j])
# else:
# matrix_C[i][j] = C2[i][j]

    r2 = int(rows/2)
    c2 = int(cols/2)            
    for i in range(0,r2):
        for j in range(0, c2):
# if abs(C1[i][j]) > abs(C2[i][j]):
            matrix_C[i][j] = (C1_bi[i][j]*C1[i][j])/(C1_bi[i][j]+C2_bi[i][j]) + (C2_bi[i][j]*C2[i][j])/(C1_bi[i][j]+C2_bi[i][j])
# else:
# matrix_C[i][j] = C2[i][j]


# matrix_C = C1*W + C2*(1-W)

    # 2. Top left sub-matrix elements of C (125 by 125) will be the
    # average of C1 and C2 (Top left sub-matrix)
# for i in range(0,125):
# for j in range(0,125):
# matrix_C[i][j] = (C1[i][j]+ C2[i][j]) /
    r1 = int(rows/4)
    c1 = int(cols/4)
    for i in range(0,r1):
        for j in range(0,c1):
            matrix_C[i][j] = (C1[i][j]+ C2[i][j]) / 2


    # 3. return the C matrix
    # --------Write your code-------
    return matrix_C

def minMaxHold(img):
    img = cv2.max(0,img)
    img = cv2.min(255,img)
    img = np.uint8(img)
    return img

def capture(cam):

    cap = cv2.VideoCapture(cam)  #비디오 캡쳐는 영상을 뽑는것 아닌가?
    

    
    while True:
        gp.output(7, False)
        gp.output(11, False)
        gp.output(12, True)   
        
        ret1,frame1 = cap.read()
        ret1,frame1 = cap.read()
        ret1,frame1 = cap.read()

        
        gp.output(7, True)
        gp.output(11, False)
        gp.output(12, True)
        
        ret2,frame2 = cap.read()
        ret2,frame2 = cap.read()
        ret2,frame2 = cap.read()
        
        cv2.imshow('test',frame1)
        cv2.imshow('test2',frame2)        
        
        if cv2.waitKey(1) == ord('q'):      #1ms 대기한 후 입력이 없으면 다시 영상을 찍음. 느리다고 느낌.
            break
        
    cv2.imwrite('1_NIR.jpg',frame1)
    cv2.imwrite('1_VIS.jpg',frame2)
                        

    frame1_rtx = singleScaleRetinex(frame1,50)
    frame2_rtx = singleScaleRetinex(frame2,50)
    
    h = find_matrix(frame1_rtx,frame2_rtx)  #find h-matrix
    
    

    im1Reg = Warping(frame1,frame2,h)
        
#     cv2.imshow('test',frame1)
#     cv2.imwrite('im1.jpg',frame1)
#     cv2.imshow('test2',frame2)
#     cv2.imwrite('im2.jpg',frame2)
#     cv2.imshow('result',im1Reg)
    cv2.imwrite('1_WRP.jpg',im1Reg)    
   


        

#         result = cv2.hconcat([frame1,frame2]) #가로로 두개 붙이는 함수.
#         cv2.imshow('test3',result)


       
#     cv2.waitKey(0)      #1ms 대기한 후 입력이 없으면 다시 영상을 찍음. 느리다고 느낌.

       
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    
    
    
    
    i2c = "i2cset -y 1 0x70 0x00 0x04"
    os.system(i2c)
    gp.output(7, False)
    gp.output(11, False)
    gp.output(12, True)
       
    capture(0)    
        
    
# Read and show the original image
    im1 = Image.open("1_VIS.jpg")
       
    # im1 = im1.resize((680,680))
    print(im1.format, im1.size, im1.mode)
    # print len(im1.getbands())
    h, w = im1.size
    im2 = Image.open("1_WRP.jpg")
    # im2 = im2.resize((680,680))

    im_color = cv2.imread('1_VIS.jpg')
    im_Lum = cv2.imread('1_WRP.jpg')    
    # Create an image buffer object for fast access.
    pix1 = im1.load()
    pix2 = im2.load()
    im1_channels = im1.split()
    im2_channels  = im2.split()

    # Convert the 2d image to a 1d sequence:
    im1_matrix = []
    im2_matrix = []
    for i in range(0,3):
        im1_matrix.append(list(im1_channels[i].getdata()))
        im2_matrix.append(list(im2_channels[i].getdata()))

    # Convert the 1d sequence to a 2d matrix.
    # Each sublist represents a row. Access is done via m[row][col].
    for ind in range(0,3):
        im1_matrix[ind] = [im1_matrix[ind][i:i+im1.size[0]] for i in range(0, len(im1_matrix[ind]), im1.size[0])]
        im2_matrix[ind] = [im2_matrix[ind][i:i+im2.size[0]] for i in range(0, len(im2_matrix[ind]), im2.size[0])]

    #Luminance 채널로 만듦
    im1_Lum = 0.3*np.array(im1_matrix[0]) + 0.6*np.array(im1_matrix[1]) + 0.1*np.array(im1_matrix[2])
    im2_Lum = 0.3*np.array(im2_matrix[0]) + 0.6*np.array(im2_matrix[1]) + 0.1*np.array(im2_matrix[2])
    #--------------------------------------------------------
    final_im_channels = np.zeros((w,h), dtype='float64')

#-----------------------------------------------------------
# 1. call fwt97_2d funtion to get wavelet signal for a image channel
# 2. convert the type as numpy array
# 3. call WaveletFusion to fuse two channels
# 2. call iwt97_2d function to get actual image channel
# 3. put it in final_im_channels array #final_im_channels[:,:,i] = channel
#------------------------------------------------------------

    im1_signal = fwt97_2d(im1_Lum)
    im2_signal = fwt97_2d(im2_Lum)
    #
    im1_signal = np.array(im1_signal)
    im2_signal = np.array(im2_signal)
    # cv2.imshow('wavelet_1', im1_signal)
    # cv2.imshow('wavelet_2', im2_signal)

    fused_matrix = WaveletFusion(im1_signal, im2_signal)
    actual_channel = iwt97_2d(fused_matrix)
    final_im_channels[:,:] = actual_channel

    #----------------------------------------------------------------------
    # This code will show the images
    #-----------------------------------------------------------------------
    im_final = np.zeros((w,h), dtype='float64')
    im_final = final_im_channels
    im_final = cv2.normalize(im_final,None,0.0,1.0,cv2.NORM_MINMAX, cv2.CV_64F)
    # im_final[:,:,1] = final_im_channels[:,:,1]
    # im_final[:,:,2] = final_im_channels[:,:,0]

    Lum = im_final
# Lum = 0.1*im_final[:,:,0]+0.6*im_final[:,:,1]+0.3*im_final[:,:,2]  
   
    im_final = np.uint8(im_final*255)    
# im_final = cv2.max(0,im_final)
# im_final = cv2.min(255,im_final)
   
# im1_Wav = np.zeros((w,h),np.uint8)
# im1_Wav =
# im2_Wav = np.zeros((w,h),np.uint8)
#    
# im_final[1] = cv2.max(0,im_final[1])
# im_final[1] = cv2.min(255,im_final[1])
#    
# im_final[2] = cv2.max(0,im_final[2])
# im_final[2] = cv2.min(255,im_final[2])

# im_final = minMaxHold(im_final)    

# im_final = cv2.cvtColor(im_final,cv2.COLOR_BGR2LAB)
# Lum = im_final[:,:,0]
   
   

# im_color = np.zeros((w,h,3), dtype='int64')
# im_color[:,:,0] = im1[:,:,2]
# im_color[:,:,1] = im1[:,:,1]
# im_color[:,:,2] = im1[:,:,0]


##image fusion
    Lab = np.zeros((w,h,3))
    Labcom = cv2.cvtColor(im_color,cv2.COLOR_BGR2LAB)
    Lcom,acom,bcom = cv2.split(Labcom)
       
    Lcom = Lcom / 255.0
       
    Lcom = Lcom * 99.0 +1.0

    rad = Lum
       
    # rad = cv2.normalize(rad,None,0.0,1.0,cv2.NORM_MINMAX, cv2.CV_64F)


    ratio = 0.8 * 100 * rad / (Lcom+0.0001)
    # ratio = 1
       
    La = cv2.GaussianBlur(Lcom,(31,31),30)
    La = cv2.normalize(La,None,0.0,1.0,cv2.NORM_MINMAX, cv2.CV_64F)
    # ratio_c = ratio*pow(La,0.2) + (1-pow(La,0.2))
       
    aa = minMaxHold((np.double(acom) - 128.0)*ratio + 128.0)
    bb = minMaxHold((np.double(bcom) - 128.0)*ratio + 128.0)

    Lab[:,:,1] = aa
    Lab[:,:,2] = bb
       
    Lab[:,:,0] = rad * 255        
       
       
    Lab = np.uint8(Lab)
    Lab = cv2.cvtColor(Lab,cv2.COLOR_LAB2BGR)
    HSV = cv2.cvtColor(Lab,cv2.COLOR_BGR2HSV)

    Hcom,Scom,Vcom = cv2.split(HSV)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    Vratio = cv2.filter2D(Vcom, -1, kernel)

    V2 = Vcom + 0.4*Vratio
    V2 = cv2.normalize(V2,None,0.0,1.0,cv2.NORM_MINMAX, cv2.CV_64F)    
           

    HSV[:,:,2] = V2 * 255

    HSV = np.uint8(HSV)
    im_sharp = cv2.cvtColor(HSV,cv2.COLOR_HSV2BGR)  




    # cv2.imwrite('final5.jpg',im_final)
    #plt.subplot(121),plt.imshow(im1, cmap = plt.get_cmap('brg'), vmin = 0, vmax = 255),plt.title('im1')
    #plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(im2),plt.title('im2')
    #plt.xticks([]), plt.yticks([])


    # im_final = int(((im_final-im_final.min)*255)/(im_final.max - im_final.min))
#    cv2.imshow('final_Lab', Lab)
    cv2.imshow('final_sharpening', im_sharp)
#    cv2.imwrite('20091502_nosharpening.jpg', Lab)
#    cv2.imwrite('20091502_sharpening.jpg', im_sharp)
       
    # cv2.imshow('wavelet_1', im1_signal)
    # cv2.imshow('wavelet_2', im2_signal)
    # cv2.imshow('mix', im_final)

       
    # cv2.imwrite('img_detail_no_weight_2.jpg', im_final)  
    # cv2.imwrite('img_WV_RGB_2.jpg', im1_signal)  
    # cv2.imwrite('img_WV_NIR_2.jpg', im2_signal)
    # cv2.imwrite('img_mix_2.jpg', Lab)    
    # cv2.imwrite('img_weight_2.jpg', output_i)
    #
    # cv2.imwrite('img_fsued_2depth_weight_less_12.jpg',Lab)
    # cv2.imwrite('im1_ras.jpg',Lab)
    # cv2.imwrite('img_fsued_LAB_2.jpg',Lab)
    # cv2.imshow('final', im_final)
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
