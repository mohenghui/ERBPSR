import os
def makedir(c_path, file_flag=False):
    if file_flag:
        c_path = os.path.dirname(c_path)
    if not os.path.exists(c_path):
        father_dir = os.path.dirname(c_path)
        if not os.path.exists(father_dir):
            makedir(father_dir)
        os.mkdir(c_path)