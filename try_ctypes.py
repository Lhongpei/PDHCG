import ctypes, os
lib_path = os.path.abspath('PDHCG_sysimage.so')
lib = ctypes.CDLL(lib_path)