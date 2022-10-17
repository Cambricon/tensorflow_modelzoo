from abc import abstractmethod
import tensorflow as tf
import os

def fetch_files(data_dir,filename_pattern):
    if data_dir == None:
        return []
    files = tf.io.gfile.glob(os.path.join(data_dir, filename_pattern))
    if files == []:
        raise ValueError('Can not find any files in {} with '
                         'pattern "{}"'.format(data_dir, filename_pattern))
    return files

class RawData:
    def __init__(self):
        self.imageData=None
        self.image_label=None 
        self.image_height=None
        self.image_width=None
        self.id=None
        self.areas=None
        self.bboxes=None
        self.groundtruth_is_crowd=None
        self.bbox_labels=None
        self.instance_masks=None
        self.instance_masks_png=None
        self.text=None
    
class DatasetBase:
    """
    the interface of DataSet
    """
    def __init__(self,folder_path=None):
        self.folder_path = folder_path
    
    def fetch_validation(self):
        return fetch_files(self.folder_path,"validation*")
    
    def valid_dataset_path(self):
        if os.path.isdir(self.folder_path):
            return True
        else:
            return False
    
    def set_dataset_folder(self,folder_path):
        self.folder_path=folder_path
        flag=self.valid_dataset_path()
        if not flag:
            raise ValueError("dataset folder:{} is not valid!".format(folder_path))
    
    def set_batchsize(self,batchsize):
        pass
    
    def initialize(self):
        pass
    
    @abstractmethod
    def decode(self,record)-> RawData:
        pass
