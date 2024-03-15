from PIL import Image
import numpy as np
import os
import codder
import random
import csv


def create_images():
    k = 1
    for image_name in os.listdir('./Dataset'):
        if image_name[-3:] == 'png':
            img = Image.open(f'./Dataset/{image_name}').convert('RGB')
            image = np.asarray(img)
            n = int(min(image.shape[0], image.shape[1]) // 32)
            print(min(image.shape[0], image.shape[1]), image_name)
            print(n)
            for y in range(n):
                for x in range(n):
                    image_crop = img.crop((32 * x, 32 * y, 32 * (x + 1), 32 * (y + 1)))
                    image_crop.save(f'./Datapack/{k}.png')
                    k += 1
                    if k == 24000:
                        return 0


def create_shifr_pic():
    shetchik = [0, 0]
    for image_name in os.listdir('./Datapack'):
        if image_name[-3:] == 'png':
            image = np.asarray(Image.open(f'./Datapack/{image_name}').convert('RGB'))
            bit = random.randint(0, 1)
            (p0, p1) = codder.create_Patterns()
            if bit == 1:
                pattern = p1
            else:
                pattern = p0
            img_shifr = codder.template_x32(pattern, 10, image)
            if bit == 1:
                shetchik[1] += 1
            else:
                shetchik[0] += 1
            Image.fromarray(np.uint8(img_shifr)).convert('RGB').save(f'./bit_{bit}/bit{bit}_{shetchik[bit]}.png')
            if shetchik[0] + shetchik[0] >= 24000:
                return 0


def read():
    with open('cripto_base.csv', "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            print(row[0].shape)


create_shifr_pic()
