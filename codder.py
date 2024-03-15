import numpy as np
from PIL import Image
import torch
import net
import cv2
import torchvision.transforms.functional as TF


def input_image(dir_way, shifring: bool = False, idx: int = -1):
    """
    Загрузка и первичная обработка картинки
    :param dir_way: путь к файлу
    :param shifring: флаг формата подготовки в зависимости от задачи
    :param idx: иттерационный параметр для сохранения промежуточных результатов
    :return: andarray картинка подходящая под параметры встраивания или извлечения, её размеры
    """
    image = Image.open(dir_way).convert('RGB')
    (H, W) = image.size
    if shifring:
        image = image.resize((64 * (H // 64), 64 * (W // 64)))
        if idx + 1:
            image.save(f'prepaired{idx + 1}.png')
        return np.asarray(image), image.size
    else:
        image = image.crop((0, 0, int(H / 2), int(W / 2)))
        return image, image.size


def string_to_bytes(
        string,
        alph: str = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдеёжзийклмнопрстуфхцчшщьыъэюя -,.'
):
    """
    Функция перевода текста в двоичную последовательность
    :param string: текст
    :param alph: алфавит
    :return: ndarray - двочиная последовательность битов
    """
    result = ''
    for i in string:
        num = alph.index(i)
        num_bin = bin(num)[2:]
        while len(num_bin) < 8:
            num_bin = '0' + num_bin
        result += num_bin
    if result[-1] == '0':
        result = result[:-1] + '1'
        result = '1' + result
    else:
        result = '0' + result
    return np.array(list(map(int, list(result))))


def bytes_to_string(
        bytes,
        alph: str = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдеёжзийклмнопрстуфхцчшщьыъэюя -,.'
):
    """
    Функция перевода битовой двоичной строки в текст
    :param bytes: двоичная строка
    :param alph: алфавит
    :return: текст
    """
    text = ''
    for i in range(0, len(bytes), 8):
        num = int(bytes[i:i + 8], 2)
        text += alph[num]
    return text


def message_create(H: int, W: int):
    """
    Ввод сообщения (int)
    :param H: длина изображения обложки
    :param W: ширина изображения обложки
    :return: Двоичная запись сообщения в формате ndarray
    """
    print('Max val:', 2 ** int((H / 64) * (W / 64) - 2))
    mes = int(input('input mes\n'))
    mod = (mes + 1) % 2
    while mes >= 2 ** int((H / 64) * (W / 64) - 2):
        mes = int(input('input mes\n'))
    mes = format(mes, 'b')
    a = [mod] + [int(i) for i in list(mes)]
    if mod == 1:
        a[-1] = mod
    return np.array(a)


def create_Patterns():
    """
    Паттерны битов 0/1 в формате ndarray
    :return: кортеж паттернов встраивания
    """

    pattern_0 = np.zeros((32, 32), 'int8')
    pattern_0[1:7, 1:7] = np.ones((6, 6))
    pattern_0[25:31, 25:31] = np.ones((6, 6))
    pattern_0[10:22, 10:22] = np.ones((12, 12))

    pattern_1 = np.zeros((32, 32), 'int8')
    pattern_1[1:7, 25:31] = -np.ones((6, 6))
    pattern_1[25:31, 1:7] = -np.ones((6, 6))
    pattern_1[10:22, 10:22] = -np.ones((12, 12))

    return (pattern_0, pattern_1)


def mes_to_P_mf(mes, H_w, W_w):
    """
    Конвертация сообщения в четверть-шаблон P_mf
    :param mes: Сообщение в формате битовового списка
    :param H_w: длина встраиваемого блока, равна H/2
    :param W_w: ширина встраиваемого блока, равна W/2
    :return: ndarray размерами H/2 x W/2
    """

    (pattern_0, pattern_1) = create_Patterns()
    P_mf = np.zeros([int(H_w * W_w / 1024), 32, 32], 'int8')
    for i in np.where(mes == 1)[0]:
        P_mf[i] = pattern_1
    for i in np.where(mes == 0)[0]:
        P_mf[i] = pattern_0
    for i in range(mes.shape[0], P_mf.shape[0]):
        P_mf[i] = pattern_0
    P = np.zeros([W_w, H_w])
    for y in range(int(W_w / 32)):
        for x in range(int(H_w / 32)):
            P[32 * y:32 * (y + 1), 32 * x:32 * (x + 1)] = P_mf[int(H_w * y / 32) + x]
    return P


def template(P_mf, beta, Img):
    """
    Встраивание 4 четверть-шаблонов P_mf в синий канал изображения
    :param P_mf: шаблон H/2 x W/2 встраивания
    :param beta: коэффицент встраивания
    :param Img: изображение HxW
    :return: ndarray изображение после встраивания
    """
    P_rashir = np.hstack([P_mf, P_mf])
    P_rashir = np.vstack([P_rashir, P_rashir])
    P_rashir = P_rashir.reshape(P_rashir.shape[0], P_rashir.shape[1], 1)
    P_full = np.zeros((Img.shape), dtype='int8')
    P_full[::, ::, 2:] = P_rashir
    Img = Img + beta * P_full
    return Img


def template_x32(pattern, beta, Img):
    """
    Функция встраивания патерна в изображение 32х32
    :param pattern: шаблон встраивания
    :param beta: коэффицент встраивания
    :param Img: ndarray 32х32х3
    :return:
    """
    P_full = np.zeros((Img.shape), dtype='int8')
    P_full[::, ::, 2:] = pattern.reshape(32, 32, 1)
    Img = Img + beta * P_full
    return Img


def shifr(im_dir, saving_dir, betta: int = 10, mes_num: bool = True, idx: int = -1):
    """
    функция внедрения сообщения в изображение
    :param im_dir: путь к изображению-обложке
    :param saving_dir: путь к сохраняемому изображению
    :param betta: коэффицент встраивания
    :param mes_num: флаг формата сообщения
    :param idx: иттерационный параметр для сохранения промежуточных результатов
    :return: Файл в необходимой дирректори.
    """
    (img, im_size) = input_image(im_dir, shifring=True, idx=idx)
    (H, W) = im_size
    if mes_num:
        mes = message_create(H, W)
    else:
        alph = str(input('Input alph or press [1] to use russian alph\n'))
        if alph == '1':
            mes = string_to_bytes(str(input('Input message\n')))
        else:
            mes = string_to_bytes(str(input('Input message\n')), alph)
    if idx + 1:
        zer = np.zeros(int(H * W / 4096), 'int8')
        zer[:mes.shape[0]] = mes
        np.save(f'mes{idx + 1}', zer)
    print('Message:', mes)
    P_mf = mes_to_P_mf(mes, int(H / 2), int(W / 2))
    image_tempated = template(P_mf, betta, img)
    result = Image.fromarray(np.uint8(image_tempated)).convert('RGB')
    result.save(saving_dir)
    return None


def deshifr(im_dir, idx: int = -1):
    """
    функция извлечения ЦВЗ из изображения
    :param im_dir: путь и название изображения
    :param idx: иттерационный параметр для сохранения промежуточных результатов
    :return: ЦВЗ
    """
    model = net.ClassificatorNet(3, 64, 2)
    model.load_state_dict(torch.load('Materials/Model.pth'))
    model.train(False)

    (img, im_size) = input_image(im_dir, idx=idx)
    (H, W) = im_size
    message = ''
    for y in range(int(W / 32)):
        for x in range(int(H / 32)):
            image_iter = img.crop((32 * x, 32 * y, 32 * (x + 1), 32 * (y + 1)))
            image_iter.save('iteration.png')

            image = cv2.imread('iteration.png', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = TF.to_tensor(image).unsqueeze(0)
            pred = model(image)
            if pred[0][0] > pred[0][1]:
                message += '1'
            else:
                message += '0'

    return message


def main():
    if bool(int(input("encoding - press [1]\ndecoding - press [0]\n"))):
        if bool(int(input("Message - text?\n[1] - yes\n[0] - no\n"))):
            flag = False
        else:
            flag = True
        name_pic = str(input('Input image name\n'))
        shifr(name_pic, f'{name_pic[:-4]}2.png', mes_num=flag)
    else:
        name_pic = str(input('Input image name\n'))
        alph = str(input('Input alph or press [1] to use russian alph\n'))
        mes = deshifr(name_pic)

        mes = mes.rstrip('0')
        if mes[0] == '0':
            mes = mes[1:]
            while '0000000000000000' in mes:
                mes = mes[:-1] + '0'
                mes = mes.rstrip('0')

        else:
            mes = mes[1:]
            mes = mes[:-1] + '0'
            while '0000000000000000' in mes:
                mes = mes.rstrip('0')
                mes = mes[:-1] + '0'

        if alph == '1':
            text = bytes_to_string(deshifr(name_pic))
        else:
            text = bytes_to_string(deshifr(name_pic), alph)

        print('Decoded message', text)
        if bool(int(input('Write in file?\n[1] - yes\n[0] - no\n'))):
            file = open(f"{name_pic[:-4]}.txt", 'w')
            file.write(text)
            file.close()


if __name__ == "__main__":
    main()
