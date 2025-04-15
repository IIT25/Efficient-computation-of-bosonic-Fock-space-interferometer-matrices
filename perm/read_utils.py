
import pickle

def write_pickle(file_path, parameters):
    with open(file_path, 'wb') as outp:
        pickle.dump(parameters, outp, pickle.HIGHEST_PROTOCOL)

def read_pickle(file_path, mode):
    with open(file_path, 'rb') as inp:
        parameters = pickle.load(inp)
        if mode == "stanadard":
            return parameters
        elif mode == "fwd":
            return parameters
        elif mode == "bwd":
            return parameters