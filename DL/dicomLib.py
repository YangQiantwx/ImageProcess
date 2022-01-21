import numpy as np
import pydicom
import matplotlib.image
import os
import cv2


# for testing if the slice is correct
from matplotlib import pyplot as plt


def generate_Xray_pic(file_path: str, patient_id: str):
    # read the dicom file
    ds = pydicom.dcmread(file_path)

    # find the shape of your pixel data
    shape = ds.pixel_array.shape
    # get the half of the x dimension. For the 
    # y dimension use shape[0]
    half_x = int(shape[1] / 2)

    right_part = ds.pixel_array[:, :half_x]
    left_part = ds.pixel_array[:, half_x:]

    # create directory for the specific patient and save l-r images into it
    os.mkdir(os.path.join('processed', patient_id))
    path_to_right_image = os.path.join('processed', patient_id, 'right.jpg')
    path_to_left_image = os.path.join('processed', patient_id, 'left.jpg')

    matplotlib.image.imsave(path_to_right_image, right_part)
    matplotlib.image.imsave(path_to_left_image, left_part)

    print('FINISHED PATIENT: ' + patient_id)


def get_dicom_file_series_description(file_path: str):
    ds = pydicom.dcmread(file_path)
    return ds.SeriesDescription


def convert_from_dicom_to_jpg(file_path: str, jpg_save_path: str):
    # ds = pydicom.dcmread(file_path)
    # matplotlib.image.imsave(jpg_save_path, ds.pixel_array)


    # # 下面是将对应的dicom格式的图片转成jpg
    # ds_array = sitk.ReadImage(file_path)  # 读取dicom文件的相关信息
    # img_array = sitk.GetArrayFromImage(ds_array)  # 获取array
    # # SimpleITK读取的图像数据的坐标顺序为zyx，即从多少张切片到单张切片的宽和高，此处我们读取单张，因此img_array的shape
    # # 类似于 （1，height，width）的形式
    # shape = img_array.shape
    # img_array = np.reshape(img_array, (shape[1], shape[2]))  # 获取array中的height和width
    # high = np.max(img_array)
    # low = np.min(img_array)
    #
    # # 转换成jpg文件并保存到对应的路径
    # lungwin = np.array([low * 1., high * 1.])
    # newimg = (img_array - lungwin[0]) / (lungwin[1] - lungwin[0])  # 归一化
    # newimg = (newimg * 255).astype('uint8')  # 将像素值扩展到[0,255]
    # cv2.imwrite(jpg_save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


    ds = pydicom.dcmread(file_path)  # 读取.dcm文件
    img = ds.pixel_array  # 提取图像信息
    # scipy.misc.imsave(jpg_save_path, img)
    cv2.imwrite(jpg_save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



