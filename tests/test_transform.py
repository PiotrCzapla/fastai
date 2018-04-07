import numpy as np

import unittest

from fastai.transforms import *

t_rand_img128x128x1 = np.random.uniform(size=[128,128,1])
t_rand_img128x128x3 = np.random.uniform(size=[128,128,3])
#
# # as per https://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray
# def rolling_window(a, size):
#     shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
#     strides = a.strides + (a. strides[-1],)
#     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class TestTransform(unittest.TestCase):

    def test_random_crop(self):
        rndCrop = RandomCrop(23)
        self._assert_randomness_works(rndCrop, t_rand_img128x128x1)

        y = t_rand_img128x128x1
        t = rndCrop.determ()
        y2 = t(y, TfmType.PIXEL)
        self.assertEqual(y2.shape[0], 23)
        self.assertEqual(y2.shape[1], 23)

        #TODO find y2 in y using rolling window or convolution

    def _assert_randomness_works(self, transf, y, tfmtype=TfmType.PIXEL ):
        rand_crop = RandomCrop(23)

        y = t_rand_img128x128x1
        t = transf.determ()

        y2 = t(y, tfmtype)
        y3 = t(y, tfmtype)
        np.testing.assert_equal(y2, y3, "The transformation should be deterministic after determ")


        t2 = transf.determ()
        y4 = t2(y, tfmtype)
        if np.all(y4 == y3): # in case we are unlucky and determ() shuffles exactly the same values we give it another try
            t2 = transf.determ()
            y4 = t2(y, tfmtype)

        self.assertFalse(np.all(y4 == y3), "Two distinct calls to determ should return 2 different transformations if it is random")


stats = inception_stats
tfm_norm = Normalize(*stats, tfm_y=TfmType.COORD)
tfm_denorm = Denormalize(*stats)

buggy_offset = 2  # This is a bug in the current transform_coord, I will fix it in the next commit

class TestTransforms(unittest.TestCase):

    def test_transforms_works_with_coords(self): # test of backward compatible behavior
        sz = 16
        transforms = image_gen(tfm_norm, tfm_denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=CropType.NO,
                  tfm_y=TfmType.COORD, sz_y=sz, pad_mode=cv2.BORDER_REFLECT, apply_transform=None)

        x, y = transforms(t_rand_img128x128x3, np.array([0,0,128,128, 0,0,64,64]))
        bbs = partition(y, 4)
        self.assertEqual(x.shape[0], 3, "The image was converted from NHWC to NCHW (channle first pytorch format)")

        h,w = x.shape[1:]
        np.testing.assert_equal(bbs[0], [0, 0, h-buggy_offset, w-buggy_offset], "The outer bounding box was converted correctly")
        np.testing.assert_equal(bbs[1], [0, 0, h/2-buggy_offset, w/2-buggy_offset], "The inner bounding box was converted correctly")

    def test_it_is_possible_to_transform_mask_n_coords(self):
        def mask_n_coord_apply(t, x, labels):
            bbs,m =labels
            return t(x, TfmType.PIXEL, is_y=False), t(bbs, TfmType.COORD, is_y=True), t(m, TfmType.MASK, is_y=True)

        sz=4
        transforms = image_gen(tfm_norm, tfm_denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=CropType.NO,
                               tfm_y=TfmType.COORD, sz_y=sz, pad_mode=cv2.BORDER_REFLECT, apply_transform=mask_n_coord_apply)

        mask = np.ones([128,128,1], dtype=np.float32)
        mask[0:64,0:64,:] = 2

        x, bb, m = transforms(t_rand_img128x128x3, [np.array([0, 0, 128, 128, 0, 0, 64, 64]), mask])
        bbs = partition(bb, 4)
        self.assertEqual(x.shape[0], 3, "The image was converted from NHWC to NCHW (channle first pytorch format)")

        h, w = x.shape[1:]

        np.testing.assert_equal(bbs[0], [0, 0, h - buggy_offset, w - buggy_offset],
                                "The outer bounding box was converted correctly")
        np.testing.assert_equal(bbs[1], [0, 0, h / 2 - buggy_offset, w / 2 - buggy_offset],
                                "The inner bounding box was converted correctly")

        self.assertEqual(x.shape[1:], m.shape[1:], "Mask should have the same shape as img")

        em = np.array([[[2., 2., 1., 1.],
                        [2., 2., 1., 1.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.]]])

        np.testing.assert_equal(m, em, "Mask should be scaled correctly")

if __name__ == '__main__':
    unittest.main()