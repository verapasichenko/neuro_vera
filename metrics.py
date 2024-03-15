from PIL import Image, ImageEnhance
from math import log10
from skimage.metrics import structural_similarity as ssim
import numpy as np
import codder


def add_speckle_noise(image, mean=0, var=0.01):
    row, col, channels = image.shape
    gauss = np.random.normal(mean, var ** 0.5, (row, col, channels))
    noisy = image + image * gauss
    return np.clip(noisy, 0, 255)


def MSE(source_img_path, final_img_path):
    source_img = np.asarray(Image.open(source_img_path).convert('RGB'))
    final_img = np.asarray(Image.open(final_img_path).convert('RGB'))
    return np.square(np.subtract(source_img, final_img)).mean()


def RMSE(source_img_path, final_img_path):
    return np.sqrt(MSE(source_img_path, final_img_path))


def PSNR(source_img_path, final_img_path):
    return 20 * log10(255 / RMSE(source_img_path, final_img_path))


def BER(pred, label):
    Ber = 0
    for i in range(label.shape[0]):
        if pred[i] == label[i]:
            Ber += 1
    return 1 - (Ber / label.shape[0])


def indicators(i):
    # pic_dir = str(input('Введите имя изначального файла\n'))
    pic_dir = f'./images/{i + 1}.png'
    codder.shifr(i, pic_dir, f'saved{i + 1}.png', 10, False)
    psnr = PSNR(f'prepaired{i + 1}.png', f'saved{i + 1}.png')

    pic = np.asarray(Image.open(f'prepaired{i + 1}.png').convert('RGB'))
    pic_ch = np.asarray(Image.open(f'saved{i + 1}.png').convert('RGB'))
    SSIM = ssim(pic, pic_ch, channel_axis=2)

    mes = np.load(f'mes{i + 1}.npy')
    pred_mes = np.array(list(map(int, list(codder.deshifr(f'saved{i + 1}.png', i)))))
    ber = BER(pred_mes, mes)
    print('PSNR = %.4f\nSSIM = %.4f\nBER = %.4f' % (psnr, SSIM, ber))


def indicators_yark(i):
    original = np.asarray(Image.open(f'prepaired{i + 1}.png').convert('RGB'))
    changed = Image.open(f'saved{i + 1}.png').convert('RGB')
    changed = ImageEnhance.Brightness(changed).enhance(1.5)
    changed.save(f'yark{i + 1}.png')
    changed = np.asarray(changed)
    psnr = PSNR(f'prepaired{i + 1}.png', f'yark{i + 1}.png')

    print(f"{i + 1}.png")

    SSIM = ssim(original, changed, channel_axis=2)
    mes = np.load(f'mes{i + 1}.npy')
    pred_mes = np.array(list(map(int, list(codder.deshifr(f'yark{i + 1}.png', i)))))
    ber = BER(pred_mes, mes)
    print('PSNR = %.4f\nSSIM = %.4f\nBER = %.4f' % (psnr, SSIM, ber))


def indicators_mashtab(i):
    original = np.asarray(Image.open(f'prepaired{i + 1}.png').convert('RGB'))
    changed = Image.open(f'saved{i + 1}.png').convert('RGB')
    (H, W) = changed.size
    changed = changed.resize(((H // 2), (W // 2)))
    changed = changed.resize((H, W))
    changed.save(f'mashtab{i + 1}.png')
    changed = np.asarray(changed)
    psnr = PSNR(f'prepaired{i + 1}.png', f'mashtab{i + 1}.png')

    print(f"{i + 1}.png")

    SSIM = ssim(original, changed, channel_axis=2)
    mes = np.load(f'mes{i + 1}.npy')
    pred_mes = np.array(list(map(int, list(codder.deshifr(f'mashtab{i + 1}.png', i)))))
    ber = BER(pred_mes, mes)
    print('PSNR = %.4f\nSSIM = %.4f\nBER = %.4f' % (psnr, SSIM, ber))


def indicators_yark(i):
    original = np.asarray(Image.open(f'prepaired{i + 1}.png').convert('RGB'))
    changed = Image.open(f'saved{i + 1}.png').convert('RGB')
    changed = ImageEnhance.Brightness(changed).enhance(1.5)
    changed.save(f'yark{i + 1}.png')
    changed = np.asarray(changed)
    psnr = PSNR(f'prepaired{i + 1}.png', f'yark{i + 1}.png')

    print(f"{i + 1}.png")

    SSIM = ssim(original, changed, channel_axis=2)
    mes = np.load(f'mes{i + 1}.npy')
    pred_mes = np.array(list(map(int, list(codder.deshifr(f'yark{i + 1}.png', i)))))
    ber = BER(pred_mes, mes)
    print('PSNR = %.4f\nSSIM = %.4f\nBER = %.4f' % (psnr, SSIM, ber))


def crop_center(pil_img, crop_width: int, crop_height: int) -> Image:
    """
    Функция для обрезки изображения по центру.
    """
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def indicators_povorot(i):
    original = np.asarray(Image.open(f'prepaired{i + 1}.png').convert('RGB'))
    changed = Image.open(f'saved{i + 1}.png').convert('RGB')
    (H, W) = changed.size
    changed = changed.rotate(60, expand=True)
    changed = changed.rotate(60, expand=True)
    changed = changed.rotate(60, expand=True)
    changed = changed.rotate(60, expand=True)
    changed = changed.rotate(60, expand=True)
    changed = changed.rotate(60, expand=True)
    changed = crop_center(changed, H, W)
    changed.save(f'reversed{i + 1}.png')
    changed = np.asarray(changed)
    psnr = PSNR(f'prepaired{i + 1}.png', f'reversed{i + 1}.png')

    print(f"{i + 1}.png")

    SSIM = ssim(original, changed, channel_axis=2)
    mes = np.load(f'mes{i + 1}.npy')
    pred_mes = np.array(list(map(int, list(codder.deshifr(f'reversed{i + 1}.png', i)))))
    ber = BER(pred_mes, mes)
    print('PSNR = %.4f\nSSIM = %.4f\nBER = %.4f' % (psnr, SSIM, ber))


def indicators_shum(i):
    original = np.asarray(Image.open(f'prepaired{i + 1}.png').convert('RGB'))
    changed = Image.open(f'saved{i + 1}.png').convert('RGB')
    changed = np.asarray(changed)
    changed = add_speckle_noise(changed)
    changed = Image.fromarray(np.uint8(changed)).convert('RGB')

    changed.save(f'shum{i + 1}.png')
    changed = np.asarray(changed)
    psnr = PSNR(f'prepaired{i + 1}.png', f'shum{i + 1}.png')

    print(f"{i + 1}.png")

    SSIM = ssim(original, changed, channel_axis=2)
    mes = np.load(f'mes{i + 1}.npy')
    pred_mes = np.array(list(map(int, list(codder.deshifr(f'shum{i + 1}.png', i)))))
    ber = BER(pred_mes, mes)
    print('PSNR = %.4f\nSSIM = %.4f\nBER = %.4f' % (psnr, SSIM, ber))


def indicators_szatie(i):
    original = np.asarray(Image.open(f'prepaired{i + 1}.png').convert('RGB'))
    changed = Image.open(f'saved{i + 1}.png').convert('RGB')
    changed.save(f'szat90_{i + 1}.jpg', quality=90)

    changed = np.asarray(Image.open(f'szat90_{i + 1}.jpg').convert('RGB'))

    psnr = PSNR(f'prepaired{i + 1}.png', f'szat90_{i + 1}.jpg')
    print(f"{i + 1}.png")

    SSIM = ssim(original, changed, channel_axis=2)
    mes = np.load(f'mes{i + 1}.npy')
    pred_mes = np.array(list(map(int, list(codder.deshifr(f'szat90_{i + 1}.jpg', i)))))
    ber = BER(pred_mes, mes)
    print('PSNR = %.4f\nSSIM = %.4f\nBER = %.4f' % (psnr, SSIM, ber))


if __name__ == "__main__":
    for i in range(0, 30):
        try:
            indicators_szatie(i)
        except:
            continue
