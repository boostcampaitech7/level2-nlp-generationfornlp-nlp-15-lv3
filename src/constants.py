import os

# 현재 파일의 절대 경로 (constants.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# constants.py 파일 바로 위의 디렉토리 경로
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

CONFIG_DIR = os.path.join(BASE_DIR, 'configs')
TRAIN_CONFIG = os.path.join(CONFIG_DIR, 'config.yaml')