from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow.compat.v1 as tf
import random
import sys
import cv2
import pydicom as dicom
import os
from pathlib import Path
from imageio.v2 import imread
import pandas as pd
import re

import subprocess
import time
from shutil import rmtree
from argparse import ArgumentParser
from echocv3.echoanalysis_tools import output_imgdict

import echocv3.nets.vgg as network 

tf.disable_eager_execution()


def dicom2jpg(path, dicomfile, out_dir):
    '''
    uses python GDCM to convert a compressed dicom into jpg images
    :param path: path to dicom file
    :param dicomfile: name of dicom file
    :param out_dir: output directory to put jpgs
    '''
    # TODO: move over to pathlib
    ds = dicom.dcmread(os.path.join(path, dicomfile))
    if dicomfile[-4:] == '.dcm':
        dicomfile = dicomfile.replace('.dcm', '.jpg')
    else:
        dicomfile = dicomfile + '.jpg'

    try:
        pixel_array_numpy = ds.pixel_array
        counter = 0
        for img_array in pixel_array_numpy:
            cv2.imwrite(os.path.join(out_dir, dicomfile.replace('.jpg', '-{}.jpg'.format(counter))), img_array)
            counter += 1
    except AttributeError:
        print(dicomfile + " failed: no Pixel Data")
        pass
    return


def extract_jpg_single_dicom(dicom_directory, out_directory, filename, min_frames=10):
    '''
    Functional to convert dicom to jpg
    :param: dicom_directory: directory of dicoms
    :param out_directory: directory to place jpgs (expect this to be dicom_directory/image/)
    :param filename: name of dicom file
    :min_frames: minimum number of frames to sample
    '''

    print(filename, "trying")
    # time.sleep(2)
    ds = dicom.dcmread(filename, force=True)
    framedict = output_imgdict(ds)

    # some quick error handling
    if framedict == "Only Single Frame":
        print("Single Frame DICOM skipped: {}".format(filename))
        return
    if framedict == "General Failure":
        print("DICOM skipped, likely has an attribute missing: {}".format(filename))
        return

    y = len(framedict.keys()) - 1
    try:
        # ds = dicom.dcmread(filename)
        # framedict = output_imgdict(ds)
        # y = len(framedict.keys()) - 1
        if y > min_frames:
            m = random.sample(range(0, y), min_frames)
            for n in m:
                targetimage = framedict[n][:]
                outfile = str(out_directory/(filename.name + '-{}.jpg'.format(n)))
                cv2.imwrite(outfile, cv2.resize(targetimage, (224, 224)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            print("Too few frames: {}".format(filename))

    except (IOError, EOFError, KeyError) as e:
        print(out_directory + "\t" + filename + "\t" +
              "error", e)
    return None


def extract_imgs_from_dicoms(dicom_directory, out_directory, filenames=None):
    """
    Extracts jpg images from DCM files in the given directory

    @param dicom_directory: folder with DCM files of echos
    @param out_directory: destination folder to where converted jpg files are placed
    @param target: destination folder to where converted jpg files are placed
    """
    if filenames is None:
        filenames = [x for x in dicom_directory.iterdir()]

    for filename in filenames[:]:
        if filename == 'image': # skip if its the temp directory we've made called "image/"
            pass
        else:
            extract_jpg_single_dicom(dicom_directory, out_directory, filename)
    return 1


def classify(directory, feature_dim, label_dim, model_name):
    """
    Classifies jpg echo images in given directory

    @param directory: folder with jpg echo images for classification
    """
    if type(model_name) == type(Path()):
        model_name = model_name.as_posix()
        
    imagedict = {}
    predictions = {}
    for filename in directory.iterdir():
        if "jpg" in filename.as_posix():
            image = imread(directory/filename).astype('uint8').flatten()
            imagedict[filename.as_posix()] = [image.reshape((224, 224, 1))]

    tf.reset_default_graph()

    # GPU budget
    # TODO: allocate this per number of processes we spin up
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    model = network.Network(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    for filename in imagedict:
        predictions[filename] = np.around(model.probabilities(sess, imagedict[filename]), decimals=3)

    return predictions




def viewclass(dicomdir = "/Users/jameswilkinson/Documents/FeinbergData/2022-05-22/dicoms/",
         batch_size=100,
         model_name="view_23_e5_class_11-Mar-2018",
         model_path=Path('./echo_deeplearning/models/'), output_dir = '', patientid = None, **kwargs):

    '''
    TODO: rewrite for use with pooled resources

    :param dicomdir: directory containing dicoms
    :param batch_size: number of dicoms to process in one go. If less than the number of dicoms in the
                    dicomdir, then the output results file will be writen multiple times, once after each
                    batch runs.
    :param model_name: name of the model. Try downloading the model prefix'd "view_23_e5_class_11-Mar-2018"
                from https://www.dropbox.com/sh/0tkcf7e0ljgs0b8/AACBnNiXZ7PetYeCcvb-Z9MSa?dl=0. You will need
                three files, one with each of the extensions ['.data-00000-of-00001', '.index', '.meta'].
                All three files should be inside the directory at model_path
    :param model_path: path to the model
    :return: None, but writes results to a csv in ./results/
    '''

    model_pathname = model_path/model_name

    infile = open(os.path.join(os.getcwd(),'echocv3', "viewclasses_" + model_name + ".txt"))
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]

    feature_dim = 1
    label_dim = len(views)

    out = pd.DataFrame(index=None, columns=['patientid', 'dicom_location', 'study', 'model'] + ["prob_{}".format(v) for v in views])
    
    
    x = time.time()
    if len(output_dir) > 0:
        temp_image_directory = kwargs.get('build_path', Path(output_dir)/'image/')
        print('creating path '+ (Path('/data/tom/MESA_Echos/temp')/'image/').as_posix())
    else:
        temp_image_directory = kwargs.get('build_path', dicomdir/'image/')
    if temp_image_directory.exists():
        rmtree(temp_image_directory)
    if not temp_image_directory.exists():
        temp_image_directory.mkdir(parents=True, exist_ok=True)

    RemainingDicoms = [x for x in dicomdir.iterdir()]
    while len(RemainingDicoms) > 0: # we want to process the dicoms in batches versus all at once
        dicoms = RemainingDicoms[:min(batch_size, len(RemainingDicoms))] # filenames of dicoms in current batch

        if len(dicoms) < len(RemainingDicoms): # if theres any left, trim down the stack
            RemainingDicoms = RemainingDicoms[batch_size:] # remove these from the stack
        else:
            RemainingDicoms = [] # if we've used all the dicoms, set list to an empty one

        # 1) extract jpg images from dicoms into temp_image_directory
        extract_imgs_from_dicoms(dicomdir, temp_image_directory, filenames=dicoms)

        # 2) generate predictions
        predictions = classify(temp_image_directory, feature_dim, label_dim, model_pathname)

        # 3) write to the results, and save as csv
        predictprobdict = {}
        for imagename in predictions.keys():
            prefix = re.split('-[0-9]+.jpg', imagename)[0] # name of dicom file (not incl. the frame number)
            if prefix not in predictprobdict:
                predictprobdict[prefix] = []
            predictprobdict[prefix].append(predictions[imagename][0])
        for prefix in predictprobdict.keys():
            predictprobmean = np.mean(predictprobdict[prefix], axis=0)
            predictprobdict[prefix] = predictprobmean # replace with mean of all predictions
            fulldata_list = [patientid, dicomdir.as_posix(), prefix, model_name] + list(predictprobmean)
            out.loc[len(out) + 1] = fulldata_list

        # _dicompathtemp = os.path.normpath(dicomdir)
        output_file_name = 'results_' + patientid + '_'.join(dicomdir.parents[0].as_posix()[:dicomdir.parents[0].as_posix().rfind('/')].split('/')) + '.csv' #why does .parents not work here?
        print("Predictions for {} with {} \n {}".format(dicomdir, model_name, out))
        out.to_csv(Path(output_dir)/output_file_name, index=False)

        # 4) empty the tmp directory of jpgs
        for f in temp_image_directory.iterdir():
            f.unlink()

    y = time.time()
    print("time:  " + str(y - x) + " seconds for " + str(len(predictprobdict.keys())) + " videos")
    rmtree(temp_image_directory)
    if len(output_dir) > 0:
        return out
    return None


if __name__ == '__main__':
    # dicomdir needs to be a directory containing dicoms. There could be multiple of these, just loop through if needed
    # batch_size limits the number of dicoms that are processed in one go. This can help out if a directory has hundreds
    #    of dicoms, which could take days to process. batch_size means the code will write and save a user-accessible
    #    output on-the-fly, processing in smaller batches as it goes.

    # # Hyperparams
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--dicomdir",
        help="Location of dicoms",
        default="/data2/NMEcho/echo_testing/dicoms/",
        type=Path)
    parser.add_argument(
        "-g", "--gpu",
        default="0",
        help="cuda device to use")
    parser.add_argument(
        "-m", "--model",
        help="Which model name to use")
    parser.add_argument(
        "-M", "--model_path",
        default="/data2/NMEcho/jtw_echo2/models/",
        type=Path)
    args = parser.parse_args()
    # print(args)
    dicomdir = args.dicomdir
    model = args.model

    #import vgg as network

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    viewclass(dicomdir=args.dicomdir, 
        model_path=args.model_path, 
        batch_size=10)