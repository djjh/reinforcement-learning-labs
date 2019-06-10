import coloredlogs
import logging
import os

from os.path import splitext, basename
from time import strftime, gmtime

# Basic Configuration
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
coloredlogs.install(level='DEBUG')

# Library specific logging levels.
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('Python').setLevel(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

def get_expermiment_logging_directory(script_filepath):
    experiment_name = splitext(basename(script_filepath))[0]
    timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    return "log/{}/summaries/{}".format(experiment_name, timestamp)
