import hashlib
import random
import cv2
import numpy as np


class ImgTransform:

    ''' Augmentation selection for spatio temporal crops '''

    def __init__(self, img, hash, config):
        self.config = config
        self.img = img
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.hash = self.gen_hash(hash)
        random.seed(hash)
        self.crop_size = random.randrange(50, int(self.height - 20))
        self.x = random.randrange(10, self.width - self.crop_size + 10)
        self.y = random.randrange(10, self.height - self.crop_size + 10)
        self.flip_val = random.randrange(-1, 1)

    def crop(self, img):
        img = img[self.y:self.y + self.crop_size,
                  self.x:self.x + self.crop_size, ]
        return img

    def noise(self, img, amount):
        gaussian_noise = np.zeros_like(img)
        gaussian_noise = cv2.randn(gaussian_noise, 0, amount)
        img = cv2.add(img, gaussian_noise, dtype=cv2.CV_8UC3)
        return img

    def gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def flip(self, img):

        img = cv2.flip(img, self.flip_val)
        return img

    def blur(self, img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        return img

    def gen_hash(self, tbh):
        hash_object = hashlib.md5(tbh.encode())
        return hash_object

    def transform_with_prob(self, img):
        img = self.crop(img)
        trans_prob = self.config["transform_prob"].get()
        if random.random() < trans_prob:
            img = self.gray(img)
            img = self.flip(img)
        img = self.blur(img)
        img = self.noise(img, int(trans_prob * 50))
        return img



