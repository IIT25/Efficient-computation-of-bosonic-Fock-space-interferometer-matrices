
import pickle
import numpy as np
def write_pickle(file_path, parameters):
    with open(file_path, 'wb') as outp:
        pickle.dump(parameters, outp, pickle.HIGHEST_PROTOCOL)

def read_pickle(file_path):
    file_path = str(file_path)
    with open(file_path, 'rb') as inp:
        parameters = pickle.load(inp)
        return parameters

def read_file(file_name):
    print(str(file_name[-4:]))
    if str(file_name[-4:]) == ".pkl":   
         parameters = read_pickle(file_name)   
         parameters_np = np.array(parameters, dtype=np.complex128)    
         return parameters_np
          