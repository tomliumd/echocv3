import argparse
import os
from doctest import OutputChecker
# from black import out
from datetime import datetime
from pathlib import Path
import distutils.util
import re
import pickle
import gzip
import json
from tqdm import tqdm
from loguru import logger

import multiprocessing

multiprocessing.freeze_support()
from multiprocessing import Manager, Process, Pool

def run_pv(args_dict, inputpath: Path, outputpath: Path, prod: bool = True, build_path=None):
    from predict_viewclass_v2 import viewclass, classify, extract_imgs_from_dicoms
    
    run_eval = args_dict.run_eval
    verbose = args_dict.verbose

    if prod:
        run_eval = False
        verbose = False

        pv_config = {
            "verbose": verbose,
            "run_eval": run_eval,
            "dicomdir": inputpath,
            "foutpath": outputpath,
            "build_path": build_path,
            "model_path": args_dict.model_path,
        }

    else:
        pv_config = {
            "verbose": args_dict.verbose,
            "run_eval": args_dict.run_eval,
            "dicomdir": inputpath,
            "foutpath": outputpath,
            "build_path": build_path,
            "model_path": args_dict.model_path,
        }

        if verbose:
            print("RUNNING")
    logger.info(build_path)

    # RemainingDicoms = os.listdir(pv_config['dicomdir'])
    # while len(RemainingDicoms) > 0: # we want to process the dicoms in batches versus all at once
    #     dicoms = RemainingDicoms[:min(pv_config['batch_size'], len(RemainingDicoms))] # filenames of dicoms in current batch

    #     if len(dicoms) < len(RemainingDicoms): # if theres any left, trim down the stack
    #         RemainingDicoms = RemainingDicoms[pv_config['batch_size']:] # remove these from the stack
    #     else:
    #         RemainingDicoms = [] # if we've used all the dicoms, set list to an empty one

    #     # 1) extract jpg images from dicoms into temp_image_directory
    #     extract_imgs_from_dicoms(pv_config['dicomdir'], pv_config['build_path'], filenames=dicoms)

    #     # 2) generate predictions
    #     #
    #     predictions = classify(pv_config['build_path'], label_dim=1, feature_dim=23, model_name=pv_config['model_path'])

    #     # 3) write to the results, and save as csv
    #     predictprobdict = {}
    #     for imagename in predictions.keys():
    #         prefix = re.split('-[0-9]+.jpg', imagename)[0] # name of dicom file (not incl. the frame number)
    #         if prefix not in predictprobdict:
    #             predictprobdict[prefix] = []
    #         predictprobdict[prefix].append(predictions[imagename][0])
    #     for prefix in predictprobdict.keys():
    #         predictprobmean = np.mean(predictprobdict[prefix], axis=0)
    #         predictprobdict[prefix] = predictprobmean # replace with mean of all predictions
    #         fulldata_list = [prefix, model_name] + list(predictprobmean)
    #         out.loc[len(out) + 1] = fulldata_list

    #     _dicompathtemp = os.path.normpath(dicomdir)
    #     output_file_name = 'results_' + '_'.join(_dicompathtemp.split(os.sep)[-2:]) + '.csv'
    #     print("Predictions for {} with {} \n {}".format(dicomdir, model_name, out))
    #     out.to_csv(output_file_name, index=False)

    #     # 4) empty the tmp directory of jpgs
    #     for f in os.listdir(temp_image_directory):
    #         os.remove(os.path.join(temp_image_directory, f))
    viewclass(**pv_config)


def main(args_dict):
    logger.remove()
    logger.add(args_dict.log_to, enqueue=True)

    Path(args_dict.build).mkdir(exist_ok=True)

    if args_dict.batches:

        batch_indexes = os.listdir(args_dict.input)
        num_batches = len(batch_indexes)
        logger.info(f"num_batches: {num_batches}")

        num_threads = multiprocessing.cpu_count()
        if args_dict.processes > 0:
            num_threads = min(args_dict.processes, num_threads)

        build_products = [os.path.join(args_dict.build, f"process{i}") for i in range(num_threads)]

        for dir in build_products:
            Path(dir).mkdir(parents=True, exist_ok=True)

        with tqdm(total=num_batches) as pbar:
            with Pool(processes=num_threads) as pool:
                
                def callback(*args):
                    pbar.update()
                    pass
                    
                results = [pool.apply_async(
                    run_pv,
                    args=(
                        args_dict,
                        f"{args_dict.input}/{batch}/",
                        args_dict.output,
                        args_dict.prod,
                        build_products[i % len(build_products)]
                    ),
                    callback=callback
                )
                for i, batch in enumerate(batch_indexes)]
                results = [r.get() for r in results]
    else:
        pass
        # predict_view(
        #     args = args_dict,
        #     inputpath = args_dict.input,
        #     outputpath = args_dict.output,
        #     prod = args_dict.prod,
        # )

    logger.info("DONE")


if __name__ == "__main__":
    help_str = """ Rahul Deo predict echo view classifier """
    ap = argparse.ArgumentParser(description=help_str)
    ap.add_argument(
        "-i",
        "--input",
        default="/data2/nea914_echo_temp/dicoms",
        help="Path to the directory or the file that contains the PHI note, the default is /data2/nea914/echo_temp/dicoms",
        type=str,
    )
    ap.add_argument(
        "-b",
        "--batches",
        action="store_true",
        help="If true, the input directory contains multiple batches. Each batch is a directory with notes.",
    )
    ap.add_argument(
        "-p",
        "--processes",
        default=0,
        help="Number of cpu processes to run in parallel, the default is 0, which means use all available cpus",
        type=int,
    )
    ap.add_argument(
        "-g",
        "--gpu",
        default="0",
        help="Cuda device to use for classification. (Currently only works with a single device)",
        type=str,
    )
    ap.add_argument(
        "-o",
        "--output",
        default="./results",
        help="Path to the directory to save the PHI-reduced notes in, the default is ./results",
        type=str,
    )
    ap.add_argument(
        "-l",
        "--log-to",
        default="./logs/file.log",
        help="Path to file in which to save the logs, the default is ./logs/file.log",
        type=str,
    )
    # ap.add_argument(
    #     "-f",
    #     "--filters",
    #     default="./configs/integration_1.json",
    #     help="Path to our config file, the default is ./configs/integration_1.json",
    #     type=str,
    # )
    # ap.add_argument(
    #     "-x",
    #     "--xml",
    #     default="./data/phi_notes.json",
    #     help="Path to the json file that contains all xml data",
    #     type=str,
    # )
    # ap.add_argument(
    #     "-c",
    #     "--coords",
    #     default="./data/coordinates.json",
    #     help="Path to the json file that contains the coordinate map data",
    #     type=str,
    # )
    # ap.add_argument(
    #     "--eval_output",
    #     default="./data/phi/",
    #     help="Path to the directory that the detailed eval files will be outputted to",
    #     type=str,
    # )
    ap.add_argument(
        "-v",
        "--verbose",
        default=True,
        help="When verbose is true, will emit messages about script progress",
        type=lambda x: bool(distutils.util.strtobool(x)),
    )
    ap.add_argument(
        "-e",
        "--run_eval",
        default=True,
        help="When run_eval is true, will run our eval script and emit summarized results to terminal",
        type=lambda x: bool(distutils.util.strtobool(x)),
    )
    ap.add_argument(
        "--prod",
        default=False,
        help="When prod is true, this will run the script with output in i2b2 xml format without running the eval script",
        type=lambda x: bool(distutils.util.strtobool(x)),
    )
    ap.add_argument(
        "-M",
        "--model_path",
        default="/data2/jtw_echo2/models/",
        help="Location of model checkpoints",
        type=str
    )
    ap.add_argument(
        "-B",
        "--build",
        default="./build/",
        help="Location for intermediate build files (e.g. echo movies)",
        type=str
    )

    args_dict = ap.parse_args()
    print(args_dict)
    # run_pv(args_dict, args_dict.input, args_dict.output)
    main(args_dict)