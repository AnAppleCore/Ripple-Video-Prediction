import os
import cv2
import random
import numpy as np
import taichi as ti

# basic parameter default settings
dx = 0.02
dt = 0.02
kappa = 2
depth = 4
gamma = 0.
eta = 1.333
margin = 200
frame_cnt = 100
light_color = 1
shape = (256, 256)

# taichi initilization with cuda (could be replaced by GPU)
ti.init(arch=ti.cuda)
height = ti.field(dtype=float, shape=shape)
velocity = ti.field(dtype=float, shape=shape)
pixels = ti.field(dtype=float, shape=(*shape, 3))
acceleration = ti.field(dtype=float, shape=shape)
background = ti.field(dtype=float, shape=(*shape, 3))

# taichi functions and kernel
@ti.kernel
def reset():
    for i, j in height:
        t = i // 32 + j // 32
        background[i, j, 1] = (t * 0.5) % 1.0
        height[i, j] = 0
        velocity[i, j] = 0
        acceleration[i, j] = 0


@ti.func
def laplacian(i, j):
    return (-4 * height[i, j] + height[i, j - 1] + height[i, j + 1] +
            height[i + 1, j] + height[i - 1, j]) / (4 * dx**2)


@ti.func
def gradient(i, j):
    return ti.Vector([
        height[i + 1, j] - height[i - 1, j],
        height[i, j + 1] - height[i, j - 1]
    ]) * (0.5 / dx)


@ti.func
def take_linear(i, j, c):
    if i < 0:
        i = -i
    elif i > shape[0] - 1:
        i = 2 * shape[0] - 2 - i
    if j < 0:
        j = -j
    elif j > shape[1] - 1:
        j = 2 * shape[1] - 2 - j
    m, n = int(i), int(j)
    i, j = i - m, j - n
    ret = 0.0
    if 0 <= i < shape[0] and 0 <= i < shape[1]:
        ret = (i * j * background[m + 1, n + 1, c] + 
               (1 - i) * j * background[m, n + 1, c] + i *
               (1 - j) * background[m + 1, n, c] + (1 - i) *
               (1 - j) * background[m, n, c])
    return ret


@ti.kernel
def touch_at(hurt: ti.f32, x: ti.f32, y: ti.f32):
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):

        r2 = (i - x)**2 + (j - y)**2
        # height[i, j] = height[i, j] + hurt * ti.exp(-0.02 * r2)
        r = ti.sqrt(r2)
        if r < 30:
            height[i, j] = height[i, j] + hurt * (ti.cos(r / 30 * 3.1415926) + 1)


@ti.kernel
def update():
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        acceleration[i, j] = kappa * laplacian(i, j) - gamma * velocity[i, j]

    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        velocity[i, j] = velocity[i, j] + acceleration[i, j] * dt
        height[i, j] = height[i, j] + velocity[i, j] * dt


@ti.kernel
def paint():
    for i, j, c in pixels:
        g = gradient(i, j)
        # https://www.jianshu.com/p/66a40b06b436
        cos_i = 1 / ti.sqrt(1 + g.norm_sqr())
        cos_o = ti.sqrt(1 - (1 - (cos_i)**2) * (1 / eta**2))
        fr = pow(1 - cos_i, 2)
        coh = cos_o * depth
        g = g * coh
        k, m = g[0], g[1]
        color = take_linear(i + k, j + m, c)
        pixels[i, j, c] = (1 - fr) * color + fr * light_color
        # pixels[i, j, c] = fr * light_color


def RippleGenerator(background_img, img_save_path):
    '''
        Water Ripple video generator using taichi. Store the result video in output directory
        @param:
        `background_img`: background image with pixel value in (0, 1)
        `framc_cnt`: number of frames in the generated wave
        @return:
        `frames`: list contains all the frames of the video, each frame is of dtype np.uint8
        `heights`: list containes the height masks of all frames, each height mask is of dtype np.float32
    '''

    # initialization
    frames, heights = [], []
    gui = ti.GUI(img_save_path, shape)
    background_img = background_img / 255. # convert to (0, 1)
    background.from_numpy(background_img)
    writer = cv2.VideoWriter(img_save_path+'/ripple_raw.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, shape[::-1])

    t = 0
    origin_touched = False
    while t < frame_cnt + margin:
        t += 1

        # capture the position of the wave origin in GUI
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                gui.running = False
            elif e.key == 'r':
                reset()
            elif e.key == ti.GUI.LMB:
                x, y = e.pos
                origin_touched = True
                touch_at(3, x * shape[0], y * shape[1])

        if not origin_touched:
            x = random.random()
            y = random.random()
            origin_touched = True
            touch_at(3, x * shape[0], y * shape[1])

        # update height and paint image
        update()
        paint()

        frame = (pixels.to_numpy() * 255).astype(np.uint8)
        if t > margin:
            height_mask = height.to_numpy().astype(np.float32)
            heights.append(height_mask)
            frames.append(frame)
            writer.write(frame)
            if (t-margin) % 10 == 0:
                print('[{:>4}/{:>4}]\t{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}'.format(t-margin, frame_cnt, frame.shape, frame.max(), frame.min(), height_mask.shape, height_mask.max(), height_mask.min()))

        # visualization
        cv2.imshow("show", frame)
        gui.set_image(pixels)
        gui.show()

    cv2.destroyAllWindows()
    writer.release()
    reset()

    # store the results
    frames = np.array(frames)
    heights = np.array(heights)
    # print(frames.shape, frames.dtype)
    # print(heights.shape, heights.dtype)
    np.save(img_save_path+'/frames.npy', frames)
    np.save(img_save_path+'/heights.npy', heights)


def RippleVideoPreparation(img_dir='./images', save_path='./output'):
    '''
        Generate the ripple videos, frames and height masks, store them as dataset (both train and val)
        @param:
        `save_path`: directory where to save data
        `img_dir`: directory where the base images are stored
    '''
    for img_name in os.listdir(img_dir):
        img = cv2.imread(img_dir + '/' + img_name)
        img = cv2.resize(img, shape[::-1])
        img_save_path = save_path + '/videos/'+img_name.split('.')[0]
        if not os.path.isdir(img_save_path):
            os.makedirs(img_save_path+'/train')
            os.makedirs(img_save_path+'/val')
        print('------Ripple video over image: {} generation------'.format(img_name))
        RippleGenerator(img, img_save_path+'/train')
        RippleGenerator(img, img_save_path+'/val')
    print('Ripple dataset generated from {} is stored in {}'.format(img_dir, save_path))


def main():
    '''
        Safely execute this script to create the dataset with default setting
    '''
    RippleVideoPreparation(img_dir='./images', save_path='./output')


if __name__=='__main__':
    main()
