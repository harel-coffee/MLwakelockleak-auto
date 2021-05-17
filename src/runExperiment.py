import analysis
import MLAlgo
import warnings
warnings.filterwarnings("ignore")
from graphs import *

clean_dir = r'\Dataset\CleanApps'
leak_dir = r'\Dataset\LeakApps'
clean_out_dir = r'\Dataset\CleanAppsOut'
leak_out_dir = r'\Dataset\LeakAppsOut'
compressed_path = r'\Dataset\CompressedFiles'

 ##/////////////Step-1: Extracting FCG///////////////////////////
# process_dir(leak_dir, leak_out_dir, mode='FCG')
# process_dir(clean_dir, clean_out_dir, mode='FCG')
##/////////////Step-2: Labelling and Hashing/////////////////////
# a = analysis.Analysis([leak_out_dir, clean_out_dir], labels=[1, 0])
# a.save_data(compressed_path, 'X_test.npz', 'Y_test.npz')
##///////////Step-3: Applying Machine Learning Algorithms////////
MLAlgo.load_data_plot(compressed_path, 'X_test.npz', 'Y_test.npz')
