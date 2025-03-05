import os
import sys
import logging
from enum import Enum

class LogType(Enum):
    TRAIN = 'train'
    FEAT = 'feat'
    INDEX = 'index'
    QUERY = 'query'


def setup_logging(log_file_name, log_type=LogType.TRAIN):
    log_format = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, force=True)

    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_file_path))
    print(current_file_path, parent_dir)
    
    log_dir = os.path.join(parent_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if log_type != LogType.TRAIN:
        log_dir = os.path.join(log_dir, log_type.value)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    log_file_path = os.path.join(log_dir, f'{log_file_name}.log')
            
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
