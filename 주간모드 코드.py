
import numpy as np  
import cv2


def blending(imgLow, imgHigh, imgDark, imgBright):

#    if cv2.mean(imgLow) >= cv2.mean(imgHigh):    
#        low = np.double(imgHigh)
#        high = np.double(imgLow)    
#    else:
    low = np.double(imgLow) #lat_low
    high = np.double(imgHigh) #lat_high
    Dmap = np.double(imgDark)
    Bmap = np.double(imgBright)
#    low = (Llow + Hhigh)/2
#    high = (Llow + Hhigh)/2
    
    rad = np.zeros((row,col,3))
    Map = np.ones((row,col,3))
#    if 0:
#        w = cv2.GaussianBlur(low[:,:,1],(11,11),100)
#        minLow,maxLow,minLowLoc,maxLowLoc = cv2.minMaxLoc(w)
    #    minHigh,maxHigh,minHighLoc,maxHighLoc = cv2.minMaxLoc(high[:,:,1])
#        w = (w-minLow) /(maxLow-minLow)
#        w = 2*np.power(w,1)
    #    g = 2
    #    w = g*w
    #    w = g*(high[:,:,1] /maxHigh)
#        w = cv2.min(1,w)
#        for i in range(0,3):
#            rad[:,:,i] = w*low[:,:,i]  + (1-w)*0.3*high[:,:,i]
#            rad[:,:,i] = cv2.min(255,rad[:,:,i])
#            rad[:,:,i] = cv2.max(0,rad[:,:,i])
            

    #Bmap[:,:,1] = pow(Bmap[:,:,1]/255, 1)*255
    #Dmap[:,:,1] = pow(Dmap[:,:,1]/255, 1)*255
    Bmap = np.uint8(Bmap)
    Dmap = np.uint8(Dmap)
    w_h = cv2.bilateralFilter(Bmap[:,:,1],9,50,50)
    w_l = cv2.bilateralFilter(Dmap[:,:,1],9,50,50)
    w_h = np.double(w_h)
    w_l = np.double(w_l)
    minLow,maxLow,minLowLoc,maxLowLoc = cv2.minMaxLoc(w_l)
    minHigh,maxHigh,minHighLoc,maxHighLoc = cv2.minMaxLoc(w_h) #for normalizing function
    w_l = (w_l-minLow) /(maxLow-minLow)
    w_h = (w_h-minHigh) /(maxHigh-minHigh)
#    w_h = 1/(1 + np.exp(-(10*w_h-5)))
#    w_l = 1/(1 + np.exp(-(10*w_l-5)))
    w_h = pow(w_h, 1.)
    w_l = pow(w_l, 1.)
#    w_l = 2*np.power(w_l,1)
#    w_l = cv2.min(1,w_l)    
    #Mmap = pow(w_h*w_l, 0.4)
    g = 1
    Map = 1 + w_h - w_l
#    w = g*w
#    w = g*(high[:,:,1] /maxHigh)
    for i in range(0,3):
        #weight 함수로 high, low blur 모두 사용. user factor 설계 
        rad[:,:,i] = (w_h/Map)*1.2*low[:,:,i] + ((1-w_l)/Map)*0.8*high[:,:,i]
        #rad[:,:,i] = Mmap*low[:,:,i] + (1-Mmap)*high[:,:,i]
        #minLrad,maxLrad,minLradLoc,maxLradLoc = cv2.minMaxLoc(rad[:,:,i])
        #if Vmax < maxLrad:
        #    Vmax = maxLrad
        rad[:,:,i] = cv2.min(255,rad[:,:,i])
        rad[:,:,i] = cv2.max(0,rad[:,:,i])

    #rad = (rad/Vmax)*255    
    rad = np.uint8(rad)
    return rad

def pyramidBlending(src,dest,mask,net):
  
  height,width,_ = src.size()
  
  h = int(max(2**math.ceil(math.log(height,2)),2**math.ceil(math.log(width,2))))
  w = h
  
  src1 = src.cpu().numpy()
  dest1 = dest.cpu().numpy()
  mask1 = mask.cpu().numpy()
  
  src1 = cv2.resize(src1, (w, h), interpolation=cv2.INTER_LINEAR)
  dest1 = cv2.resize(dest1, (w, h), interpolation=cv2.INTER_LINEAR)
  mask1 = cv2.resize(mask1, (w, h), interpolation=cv2.INTER_NEAREST)
  
  
  comp = src1
  comp[mask1==1] = dest1[mask1==1]
  
  src_G =[]
  dest_G = []
  mask_G = []
  
  t = h

def pad_replicate(image, kernel_row, kernel_col):
    img_row, img_col = image.shape[:2]
    a = (kernel_row-1)/2
    b = (kernel_col-1)/2
    new_row = img_row + 2*a
    new_col = img_col + 2*b
    
    if len(image.shape) == 3:
        new_img = np.zeros((new_row, new_col, 3), dtype=np.float32)
    else:
        new_img = np.zeros((new_row, new_col), dtype=np.float32)
        
    for i in range(new_row):
        for j in range(new_col):
            if i < a:
                if j < b:
                    new_img[i][j] = image[0][0]
                elif j > (new_col-b-1):
                    new_img[i][j] = image[0][img_col-1]
                else:
                    new_img[i][j] = image[0][j-b]
            elif i > (new_row-a-1):
                if j < b:
                    new_img[i][j] = image[img_row-1][0]
                elif j > (new_col-b-1):
                    new_img[i][j] = image[img_row-1][img_col-1]
                else:
                    new_img[i][j] = image[img_row-1][j-b]
            else:
                if j < b:
                    new_img[i][j] = image[i-a][0]
                elif j > (new_col-b-1):
                    new_img[i][j] = image[i-a][img_col-1]
                else:
                    new_img[i][j] = image[i-a][j-b]
                    
    return new_img

def Mean_Image(image):
    img_height, img_width = image.shape
    sum_value = 0.0
    for y in range(0,img_height):
        for x in range(0,img_width):
            sum_value += np.float64(image[y,x])
    
    mean_value = (sum_value)/(float(img_width*img_height))
    
    return mean_value

def Calc_Gamma(mean):
   if (110 >= mean):
            gamma= (1/30)*(110-mean) + 1   #(3-1)/(60)*(110-mean) + 1
            gamma = 1/gamma
            if (gamma < 0.3): 
                gamma=0.3
   elif (mean > 140):
            gamma = (1/20)*(mean-140) + 1 # ((2-1)/20)*(mean-140)+1    
            if (gamma > 2): 
                gamma=2
   else:
       gamma=1
   return gamma 

def sortMatrix(img,col,row):
    sMat = np.reshape(img,col*row)
    return np.sort(sMat)

def minMaxHold(img):
    img = cv2.max(0,img)
    img = cv2.min(255,img)
    img = np.uint8(img)
    return img

def lat(image, imageR, ratio, highlow):
    image = np.double(image)
    imageR = np.double(imageR)
    #image = pow(image/255, 2.2)*255
    Lum = 0.1*image[:,:,0]+0.6*image[:,:,1]+0.3*image[:,:,2]
    LumR = 0.1*imageR[:,:,0]+0.6*imageR[:,:,1]+0.3*imageR[:,:,2]
    LanR = cv2.GaussianBlur(LumR,(21,21),10)
    # color space change RGB to Lab
    image = np.uint8(image)
    Lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    # color space split Lab to L, a, b channels
    L,a,b = cv2.split(Lab)
    # convert uint8 type to double type
    L_double = np.double(Lum)
    # find min and max value of input L
    minLL,maxLL,minLoc,maxLoc = cv2.minMaxLoc(L_double)
    minLLR,maxLLR,minLoc,maxLoc = cv2.minMaxLoc(LanR)
    # rearrnage the scale of L 0.01 to 1 
    L_double = (L_double - minLL)/(maxLL - minLL) * 0.99 + 0.01
    LanR = (LanR - minLLR)/(maxLLR - minLLR)
    
    #L_double = pow(np.double(L_double), 2.2)
    # refresh intital value msr 
    msr = 0
    LanR1 = np.zeros([row,col,3])
    La_max = 1000
    weighting = [50,100,100]
    # star mlat 
    for ssr_num in range(0,1):
        La = cv2.GaussianBlur(L_double,(151,151),weighting[ssr_num])
        minLLa,maxLLa,minLoc,maxLoc = cv2.minMaxLoc(La)
        #Lan = (La - minLLa)/(maxLLa - minLLa)
###############################################################################
        Lan = La/maxLLa            #max 1
        Lan_br = (La/maxLLa)*La_max  #max 1000 nt --> real La
###############################################################################
        #La = Lmax*ratio*Lan
        La_min = La_max*minLLa/maxLLa
    #La = cv2.bilateralFilter(L_double,9,50,50)
        #minLa = ratio*minLa
        #maxLa = ratio*maxLL
        
###############################################################################
        rv_max = 0.444+0.045*np.log(10000+0.6034)
        rv_min = 0.444+0.045*np.log(La_min+0.6034)
        rv_range = rv_max - rv_min
        rv = 0.444+0.045*np.log(La+0.6034)
        #Lan = 1/(np.exp(-10*(Lan-0.5))+1)
###############################################################################
        log_L_min = 6.84*np.exp(0.12*np.log10(Lan_br)) - 8.36
        L_min = pow(10,log_L_min)
        log_L_max = 1.88*np.exp(0.23*np.log10(Lan_br)) - 0.12
        L_max = pow(10,log_L_max)
        
        rv_new = 0.173*np.exp(0.31*np.log10(Lan_br)) - 0.329
        
        brightness_La = 9.9 * pow((Lan_br/(L_max-L_min)),rv_new) + 0.1 #La 에서 brightness function
        minBLa,maxBLa,minLoc,maxLoc = cv2.minMaxLoc(brightness_La)
        #brightness_La_min = 9.9 * pow((minLLa/(la_max-la_min)),rv_new) + 0.1
        #brightness_La_max = 9.9 * pow((maxLLa/(la_max-la_min)),rv_new) + 0.1
###############################################################################       
        if highlow >= 1:
            #rv1= (1-LanR) + LanR*rv_max/rv
            #rv1= (1-Lan)*0.4+(Lan)*rv/rv_min #user factor 수식화가 필요 18.08.28.
            tone_set = (1-LanR) + 0.65*LanR*(minBLa/brightness_La)
            #tone_set = (1-LanR) + LanR*(brightness_La/minBLa)
            #rv1= rv/rv_max
            LanR = np.uint8(Lan*255)
            LanR1[:,:,0] = LanR
            LanR1[:,:,1] = LanR
            LanR1[:,:,2] = LanR    
            cv2.imwrite('LanR_High.jpg',LanR1)
        else:
            #rv1= Lan*1 + (1-Lan)*rv/rv_max
            #rv1= LanR+(1-LanR)*rv/rv_max
            #rv1= (Lan)*1.1+(1-Lan)*rv/rv_max #user factor 수식화가 필요 18.08.28.
            tone_set = LanR + 2.0*(1-LanR)*(maxBLa/brightness_La)
            #tone_set = LanR + (1-LanR)*(brightness_La/maxBLa)
            #rv1= rv/rv_max
            LanR = np.uint8(Lan*255)
            LanR1[:,:,0] = LanR
            LanR1[:,:,1] = LanR
            LanR1[:,:,2] = LanR 
            cv2.imwrite('LanR_Low.jpg',LanR1)
        #rv_m = 0.444+0.045*np.log(50+0.6034)
############################################################################### 
        #ssr = 99.9*pow(L_double,tone_set) + 0.1
        ssr = L_double*tone_set
        ssr = cv2.min(1.0,ssr)
############################################################################### 
        msr = msr + ssr/1
    
    # remove 1 and 99 pixels
    sort_msr = sortMatrix(msr,col,row)
    th_low = sort_msr[int(col*row*0.01)]
    th_up = sort_msr[int(col*row*0.99)]
    msr = cv2.min(th_up,msr)
    msr = cv2.max(th_low,msr)
    """
    # match the scale of msr to input L (minLL and maxLL)
    if (maxLL - minLL) != 0:
        msr = (msr - th_low)/(th_up-th_low)*(maxLL - minLL) + minLL
    else:
        msr = np.zeros((row,col))
        
    """
    min_msr,max_msr,min_msrLoc,max_msrLoc = cv2.minMaxLoc(msr)
    # msr = (msr - min_msr)/(max_msr-min_msr)*(maxLL - minLL) + minLL
    msr = (msr - min_msr)/(max_msr-min_msr)
    #msr = pow(np.double(msr), 0.4545)
    # chromiance compensation 
    ratio = msr/(L_double+0.0001)
    #ratio = 1
    Lab[:,:,1] = minMaxHold((np.double(a) - 128.0) * ratio+ 128.0)
    Lab[:,:,2] = minMaxHold((np.double(b) - 128.0) * ratio + 128.0)

    # convert double type to uint8 type
#    msr = np.uint8(msr*255)
    
    msr = msr*255
    # insert mlat image to Lab image
    
    Lab[:,:,0] = msr
    
    # color space change lab to RGB
    mlat = cv2.cvtColor(Lab,cv2.COLOR_LAB2BGR)
    return mlat


global row,col

scene1 = cv2.imread("data/high/high (7).jpg",cv2.IMREAD_COLOR)
scene1 = cv2.resize(scene1,None,fx=0.5,fy=0.5)
scene2 = cv2.imread("data/low/low (7).jpg",cv2.IMREAD_COLOR)
scene2 = cv2.resize(scene2,None,fx=0.5,fy=0.5)
#cv2.imshow('bright',scene1)
#cv2.imshow('dark',scene2)



row, col, ch = scene1.shape
#print row, col, ch

mlat_b = lat(scene1,scene2,1,1)
mlat_d = lat(scene2,scene1,0.06,0)

#cv2.imshow( 'lat_dark', mlat_d)
#cv2.imshow( 'lat_bright', mlat_b)
cv2.imwrite('lat_dark.jpg',mlat_d)
cv2.imwrite('lat_bright.jpg',mlat_b)

image = blending(mlat_d, mlat_b, scene2, scene1) 
shrink_image = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_AREA)

#image = np.uint8(image)    
#cv2.imshow('radiance',image)


###when testing erasing '#'
cv2.imshow( 'lat', shrink_image)
cv2.imwrite('out_son.jpg',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

            

  


