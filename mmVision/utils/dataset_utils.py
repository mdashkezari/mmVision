
import os
import pandas as pd


def classes(dir_path):
    """
    Walk through a dataset directory `dir_path` and return the class names,
    the number of images per class, and the total number of images.
    
    Parameters
    -------------
    dir_path: str
        Path to the dataset directory

    Returns
    -------------
    class_df: pandas.DataFrame
        Dataframe containing class names, dir path to each class, and number of images per class.
    total_samples: int
        Total number of images
    """
    class_names, class_dirs, class_samples, total_samples = [], [], [], 0
    for dirpath, subdirnames, filenames in os.walk(dir_path):
        classname = os.path.basename(dirpath)
        if len(classname) < 1: continue
        class_names.append(classname)
        class_dirs.append(dirpath)
        class_samples.append(len(filenames))
        total_samples += len(filenames)
    class_df = pd.DataFrame({
                            "class": class_names,
                            "dir": class_dirs,
                            "samples": class_samples,                            
                            })    
    return class_df, total_samples