import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
# ============================================
# library untuk mengambil atau get request image
from PIL import Image
import requests
from io import BytesIO

class ImageEnhancement:
    def __init__(self, Path : str, Smoothing : bool = False, Gray_Scale : bool = True, 
                 Equalize_Image : bool = False, split_image : bool = True, PlotShow : bool = True,
                 Rescale : float = None) -> None:
        """
        Path : str = berisikan path file gambar yang ingin dibuka
        Smoothing : bool = yaitu untuk membuat gambar tampak lebih halus (Optional)
        Gray_Scale : bool = dengan ini image automatis akan menjadi gray scale image
        Equalize_Image: bool = membuat image menjadi lebih baik yang meiliki kontras lebih tinggi dan nilai normalisasi
        split_image : bool = side by side raw image and new image
        PlotShow : bool = menampilkan perbandingan histogram antara
        Rescale : bool = Scalling image range(0.0 - 1.0)
        """
        self.Path = Path
        self.Smoothing = Smoothing
        self.Gray_Scale = Gray_Scale
        self.Equalize_Image = Equalize_Image
        self.split_image = split_image
        self.PlotShow = PlotShow
        self.Rescale = Rescale
        self.Tr = True # Format image

    # method ini untuk membatasi file, hanya image yang dapat dimasukan
    def __Format(self) -> None:
        # because PIL Image is RGB we convert to BGR to tranfer to cv2
        # response = requests.get(path)
        # img = Image.open(BytesIO(response.content))
        # open_cv_image = np.array(img) 
        # open_cv_image = open_cv_image[:, :, ::-1].copy() 

        list = ('.png', '.jpge', '.jpg')
        self.Tr = True
        if self.Path.endswith(list):
            self.Tr = False
            return self.Tr
        
        if self.Tr :
            warnings.simplefilter('error', UserWarning)
            warnings.warn("Please Put Image File")

    # methos ini digunakan untuk rescale image
    def __ScaleImage(self, images):
        ranges = [x / 10 for x in range(0, 10)]
        if self.Rescale in ranges:
            Scale_image = self.Rescale
            width = int(images.shape[1] * Scale_image)
            height = int(images.shape[0] * Scale_image)
            dimension = (width, height)
            images = cv2.resize(images, dimension, cv2.INTER_AREA)
        else:
            warnings.simplefilter('error', UserWarning)
            warnings.warn("Please insert 0.0 - 1.0")
        return images

    # method ini membuat tampak gambar lebih halus untuk filteringnya disini creator menggunakan gaussian blur
    def __Smooth(self, Image):
        shp = (Image.shape[0], Image.shape[1])
        npOnes = np.ones(shp, np.float32)
        for i in range(0, Image.shape[0]):
            for j in range(0, Image.shape[1]):
                Image[i][j] = (Image[i][j] - (npOnes[i][j] * (1.0/9.0))) + 255.0

        gaussian_blur = cv2.GaussianBlur(src=Image, ksize=(3,3), sigmaX=0, sigmaY=1)
        return gaussian_blur

    # method ini digunakan untuk menconvert gambar rgb menjadi gray image
    def __GreyScaleImage(self, Image):
        Gray_img = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        return Gray_img

    # method ini digunakan untuk menequalize image rgb
    def __BGRimages(self, Image):
        # terdiri dari 3 array warna Red, Green, Blue
        # jika gambarnya berwarna atau RGB
        # pisah kan 3 warna tersebut
        Red_Image, Green_Image, Blue_Image, *Alpha = cv2.split(Image)
        Red_Image = cv2.equalizeHist(Red_Image)
        Green_Image = cv2.equalizeHist(Green_Image)
        Blue_Image = cv2.equalizeHist(Blue_Image)
        # menggabungkan kan 3 warna tersebut
        image = cv2.merge((Red_Image,Green_Image,Blue_Image))
        return image

    # method ini digunakan untuk menequalize image rgb dan gray image
    def __EqualizeImage(self, image):
        if len(image.shape) == 3 :
            image = self.__BGRimages(image) # equalize RGB Image
            # print(len(image.shape))
        else:
            image = cv2.equalizeHist(image)
        # image = len(image.shape)
        # print(image)
        return image 
    
    # method ini digunakan untuk menampilakn plot dari gambar yaitu intensitas pixel dari gambar tersebut
    def __ShowPlot(self, new_image, oldimage):
        fig, (ax1,ax2) = plt.subplots(1,2, figsize = (7,4))
        plt.title("Perbandingan Histogram Gambar")
        # before equalize
        hist,bins = np.histogram(oldimage.flatten(),256,[0,256])
        cdf = hist.cumsum()
        normalisasi = cdf * (hist.max()/ cdf.max())
        ax1.plot(normalisasi, color = 'black')
        ax1.hist(oldimage.flatten(),256,[0,256], color = 'r')
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Quantity")
        ax1.set_title("Image Before Equalization")
        ax1.legend(('cdf','histogram'))
        # after equalize
        hist,bins = np.histogram(new_image.flatten(),256,[0,256])
        cdf = hist.cumsum()
        normalisasi = cdf * (hist.max()/ cdf.max())
        ax2.plot(normalisasi, color = 'black')
        ax2.hist(new_image.flatten(),256,[0,256], color = 'r')
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Quantity")
        ax2.set_title("Image After Equalization")
        ax2.legend(('cdf','histogram'))
        plt.show()

    # fungsi ini guna untuk menampilkan gambar sebelum dan sesudah preprocessing image
    def __shw(self, Old, New_Image):
        teks1 = "Before"
        teks2 = "After"
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 0)
        thickness = 2
        if self.Gray_Scale:  
            New_Image = cv2.cvtColor(New_Image, cv2.COLOR_GRAY2BGR)
            res = cv2.hconcat([Old, New_Image])
            w = res.shape[1]
            h = res.shape[0]
            fontScale = self.Rescale
            mx = np.max(h)
            my = np.min(h)
            wx = np.max(w)
            wy = np.min(w) 
            #  Using cv2.putText() method for adding teks on image
            image = cv2.putText(res, teks1, (int(w*(wx/wy) * (.34)),int(h*(mx/my) * (0.05))), font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            image = cv2.putText(res, teks2, (int(w*(wx/wy) * (.88)),int(h*(mx/my) * (0.05))), font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow("image Show", image)
        else:
            res = cv2.hconcat([Old, New_Image])
            w = res.shape[1]
            h = res.shape[0]
            fontScale = self.Rescale
            mx = np.max(h)
            my = np.min(h)
            wx = np.max(w)
            wy = np.min(w) 
            #  Using cv2.putText() method for adding teks on image
            image = cv2.putText(res, teks1, (int(w*(wx/wy) * (.34)),int(h*(mx/my) * (0.05))), font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            image = cv2.putText(res, teks2, (int(w*(wx/wy) * (.88)),int(h*(mx/my) * (0.05))), font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow("image Show", image)

    # main method
    def show(self) -> None:
        Tr = self.__Format()
        if Tr == False:
            if self.Path.startswith("https:"):
                # because PIL Image is RGB we convert to BGR to tranfer to cv2
                response = requests.get(self.Path)
                img = Image.open(BytesIO(response.content))
                images = np.array(img) 
                New_Image = images[:, :, ::-1].copy() 
                OldImage = New_Image.copy() 

                if self.Rescale:
                    New_Image = self.__ScaleImage(New_Image)
                    OldImage = self.__ScaleImage(OldImage)
            else:
                New_Image = cv2.imread(self.Path)
                OldImage = cv2.imread(self.Path)
                if self.Rescale:
                    New_Image = self.__ScaleImage(New_Image)
                    OldImage = self.__ScaleImage(OldImage)

            if self.Smoothing:
                New_Image = self.__Smooth(New_Image)

            if self.Gray_Scale == True and len(New_Image.shape) >= 3:
                New_Image = self.__GreyScaleImage(New_Image)
                if self.Equalize_Image:
                    New_Image = self.__EqualizeImage(New_Image)
            else:
                if self.Equalize_Image:
                    New_Image = self.__EqualizeImage(New_Image)
            
            if self.split_image :
                # img = np.hstack((OldImage, New_Image))
                # self.__shw("Old Image", OldImage)
                self.__shw(OldImage, New_Image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                self.__shw("New", New_Image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if self.PlotShow:
                self.__ShowPlot(New_Image, OldImage)


if __name__ == "__main__":
    # ----- Dummy Image Computer File ---------
    # path = r"D:\Dokumen kuliah\[KULIAH] PELAJAAN BINA NUSANTARA\semester 5\Computer Vision\LAB\Session 2\Case 2\flowers.jpg" your file put in here

    # ----- Dummy Image on Internet ---------
    # path = 'https://as1.ftcdn.net/v2/jpg/02/99/96/14/1000_F_299961418_4u9ebmBi7542aVW8LbroeVSoUPiROkgI.jpg'
    # path = 'https://as2.ftcdn.net/v2/jpg/04/60/51/01/1000_F_460510116_NLB2FDELQMz6crIEB3vENfrNbxZvsdBc.jpg'
    # path = 'https://as1.ftcdn.net/v2/jpg/03/32/70/36/1000_F_332703631_eCep9lVzH5YlyIxFEpwFGHmO3n0Pp3uK.jpg'
    # path = 'https://st2.depositphotos.com/1005145/7740/i/450/depositphotos_77409796-stock-photo-mountain-flowers-in-a-sunny.jpg'
    path = 'https://st2.depositphotos.com/4009139/6138/i/600/depositphotos_61383523-stock-photo-awesome-vintage-woman-looking-away.jpg'

    ImageEnhancement(path, Smoothing=True, Gray_Scale=False, Equalize_Image=True, Rescale = 0.6, PlotShow=True).show()
    
