from .third_party.src.main import prepare, train, test
from .third_party.src.preprocess import preprocess
import os


def preprocessing(save_dir, out_ext, dp_preprocess_txcm, corenlp_folder):
    datasets = ["/test", "/dev", "/train"]
    for dataset in datasets:
        if not os.path.exists(save_dir+out_ext+dataset+"/plain"):
            os.mkdir(save_dir+out_ext+dataset+"/plain")
        if not os.path.exists(save_dir+out_ext+dataset+"/xml"):
            os.mkdir(save_dir+out_ext+dataset+"/xml")
        if not os.path.exists(save_dir+out_ext+dataset+"/conll"):
            os.mkdir(save_dir+out_ext+dataset+"/conll")
        if not os.path.exists(save_dir+out_ext+dataset+"/merge"):
            os.mkdir(save_dir+out_ext+dataset+"/merge")
        if not os.path.exists(save_dir+out_ext+dataset+"/brackets"):
            os.mkdir(save_dir+out_ext+dataset+"/brackets")
        preprocess(save_dir+out_ext+dataset+"/edus", 
                   save_dir+out_ext+dataset+"/sentences", 
                   save_dir+out_ext+dataset+"/plain", 
                   save_dir+out_ext+dataset+"/xml", 
                   save_dir+out_ext+dataset+"/conll", 
                   save_dir+out_ext+dataset+"/merge",
                   dp_preprocess_txcm, corenlp_folder)

def preparation(save_dir, out_ext, dp_ext):
    # No need to prepare dev or test set, only required for training
    # First parameter is to the *.dis files
    prepare(save_dir+out_ext+"/train/discourse", save_dir+dp_ext+'/data_helper.bin')

def training(save_dir, out_ext, dp_ext):
    train(save_dir+out_ext+"/train/discourse", save_dir+dp_ext+'/model.gz', save_dir+dp_ext+'/data_helper.bin')

def testing(save_dir, dp_ext, dp_rst_test_data, dp_instr_test_data, measures):
    if not dp_rst_test_data == "None":
        print("\nRunning evaluation on", dp_rst_test_data)
        if 'orig' in measures:
            print("\t Running on the original parseval metric")
            test(dp_rst_test_data, save_dir+dp_ext+'/model.gz', 'orig')
        if 'rst' in measures:
            print("\t Running on the RST parseval metric")
            test(dp_rst_test_data, save_dir+dp_ext+'/model.gz', 'rst')
    if not dp_instr_test_data == "None":
        print("\nRunning evaluation on", dp_instr_test_data, '\n')
        if 'orig' in measures:
            print("\t Running on the original parseval metric")
            test(dp_instr_test_data, save_dir+dp_ext+'/model.gz', 'orig')
        if 'rst' in measures:
            print("\t Running on the RST parseval metric")
            test(dp_instr_test_data, save_dir+dp_ext+'/model.gz', 'rst')
