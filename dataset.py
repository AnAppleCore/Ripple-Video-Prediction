import cv2
import numpy as np

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

class RippleVideoDataset(Dataset):
    '''
        `RippleVideoDataset(img_dir, img_name, frame_cnt, img_shape)`:
        Stores the ripple video frames and height masks generated by the taichi wave generator script over a single image
        @param:
        `img_name`: name of the basic image
        `data_dir`: directory where the frames and height masks are stored
        `frame_cnt`: number of frames generated of the ripple video
        `img_shape`: shape of the video
        `usage`: wether used as train set or val set
        Notice that frame count and image shape should match the default setting in `ripple.py`
    '''
    def __init__(self, img_name='0.png', data_dir='./output', frame_cnt=100, img_shape=(256, 256), is_train=True):
        self.img_name = img_name
        self.data_dir = data_dir
        self.frame_cnt = frame_cnt
        self.img_shape = img_shape

        self.img_path = data_dir + '/videos/' + img_name.split('.')[0]
        self.img_path += '/train' if is_train else '/val'
        self.frames = np.load(self.img_path+'/frames.npy') # np.uint8
        self.heights = np.load(self.img_path+'/heights.npy') # np.float32

        assert self.frame_cnt == self.frames.shape[0], 'frames num error'
        assert self.frame_cnt == self.heights.shape[0], 'height masks num error'
        assert self.frames[0].shape == (*self.img_shape, 3), 'frame shape error'
        assert self.heights[0].shape == (self.img_shape), 'height shape error'

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index):
        # convert the input data and label to np.array of dtype np.float32
        frame = self.frames[index].astype(np.float32)
        height = self.heights[index].astype(np.float32)

        frame = TF.to_tensor(frame)
        height = torch.from_numpy(height)
        return frame, height


def get_dataloader(args):

    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = args.pin_memory

    train_dataset = RippleVideoDataset(args.img_name, args.save_path, args.frame_cnt, args.img_shape, is_train=True)
    val_dataset = RippleVideoDataset(args.img_name, args.save_path, args.frame_cnt, args.img_shape, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dataloader, val_dataloader


def main():
    # test RippleVodeoDataset
    dataset = RippleVideoDataset(is_train=False)
    print('Dataset length', len(dataset))
    frame, height = dataset[99]
    print('Return shape', frame.shape, height.shape)


if __name__ == '__main__':
    main()
