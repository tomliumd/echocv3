import argparse
from ast import arg
from distutils.command.build import build
import os
from doctest import OutputChecker
# from black import out
from datetime import datetime
from pathlib import Path
import distutils.util
import py_compile
import re
import pickle
import gzip
import json
from shutil import rmtree
from itertools import islice
import pandas as pd
from tqdm import tqdm
from loguru import logger
import time

from predict_viewclass_v2 import extract_imgs_from_dicoms

import multiprocessing

multiprocessing.freeze_support()
from multiprocessing import Manager, Process, Pool, Queue

BATCH_SIZE = 100
queue_main = Queue()

def take(iter, n):
    # get the first n elements from iterable
    return list(islice(iter, n))


def run_preprocess(args_dict: dict, inputpath: Path, build_path: Path = None):
    """Create individual jpegs from DICOMs in a directory.

    Args:
        args_dict (dict): arguments
        inputpath (Path): path to DICOMs
        build_path (Path, optional): output location for jpegs. Defaults to None.
    """
    verbose = args_dict.verbose
    if verbose:
        print("RUNNING")
    logger.info(build_path)
    
    if not build_path.exists():
        build_path.mkdir(parents=True)
    
    it = inputpath.iterdir()
    while (batch := take(it, BATCH_SIZE)):
        
        if args_dict.verbose:
            print(batch)
        extract_imgs_from_dicoms(args_dict.input, build_path, filenames=batch)
        # for x in batch: queue_main.put(x) # use if classifying in addition to preprocessing


def main(args_dict):
    if args_dict.clean:
        args_dict.log_to.unlink()
    logger.remove()
    logger.add(args_dict.log_to, enqueue=True)

    args_dict.build.mkdir(exist_ok=True)

    if args_dict.batches:

        # This assumes each directory below args_dict.input contains only DICOMS, no further structure
        # TODO: access handling substructure
        batch_indexes = [x for x in args_dict.input.iterdir()]
        num_batches = len(batch_indexes)
        logger.info(f"num_batches: {num_batches}")

        num_threads = multiprocessing.cpu_count()
        if args_dict.processes > 0:
            num_threads = min(args_dict.processes, num_threads)

        if args_dict.clean:
            rmtree(args_dict.build)
            
        build_products = [args_dict.build/f"process{i}" for i in range(num_threads)]
        for dir in build_products:
            dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=num_batches) as pbar:
            with Pool(processes=num_threads) as pool:
                
                def callback(*args):
                    pbar.update()
                    pass
                    
                results = [pool.apply_async(
                    run_preprocess,
                    args=(
                        args_dict,
                        batch,
                        build_products[i % len(build_products)]
                        # if we want everything in one dir, the changing the line above does it
                        # just give each thread the same build path
                        # shouldn't have to worry about name conflicts (TM)
                    ),
                    callback=callback
                )
                for i, batch in enumerate(batch_indexes)]
                results = [r.get() for r in results]
    else:
        pass

    logger.info("DONE")


if __name__ == "__main__":
    help_str = """ Rahul Deo predict echo view classifier """
    ap = argparse.ArgumentParser(description=help_str)
    ap.add_argument(
        "-i", "--input",
        default="/data2/NMEcho/echo_testing/dicoms/without/",
        help="Path to the directory or the file that contains the PHI note, the default is /data2/nea914/echo_temp/dicoms",
        type=Path,
    )
    ap.add_argument(
        "-b", "--batches",
        help="If true, the input directory contains multiple batches. Each batch is a directory with notes.",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    ap.add_argument(
        "-p", "--processes",
        default=0,
        help="Number of cpu processes to run in parallel, the default is 0, which means use all available cpus",
        type=int,
    )
    # TODO: not in use if only doing preprocessing
    ap.add_argument(
        "-g", "--gpu",
        default="0",
        help="Cuda device to use for classification. (Currently only works with a single device)",
        type=str,
    )
    # TODO: not in use if only doing preprocessing
    ap.add_argument(
        "-o", "--output",
        default="./results",
        help="Path to the directory to save the PHI-reduced notes in, the default is ./results",
        type=Path,
    )
    ap.add_argument(
        "-l", "--log-to",
        default="./logs/file.log",
        help="Path to file in which to save the logs, the default is ./logs/file.log",
        type=Path,
    )
    ap.add_argument(
        "-v", "--verbose",
        default=False,
        help="When verbose is true, will emit messages about script progress",
        action=argparse.BooleanOptionalAction,
    )
    # TODO: not in use if only doing preprocessing
    ap.add_argument(
        "-M", "--model_path",
        default="/data2/NMEcho/jtw_echo2/models/",
        help="Location of model checkpoints",
        type=Path
    )
    ap.add_argument(
        "-B", "--build",
        default="./build/",
        help="Location for intermediate build files (e.g. echo movies)",
        type=Path
    )
    ap.add_argument(
        "-c", "--clean",
        default=False,
        help="Recreate all frames for classification",
        action=argparse.BooleanOptionalAction
    )

    args_dict = ap.parse_args()
    print(args_dict)
    # run_pv(args_dict, args_dict.input, args_dict.output)
    start = time.time()
    main(args_dict)
    end = time.time()
    print("time: ", str(end - start))
    # 3 processes were able to process 3 batches of 100 DICOMs (300 total) in ~230 seconds (1.25 miuntes / 100 DICOMs)