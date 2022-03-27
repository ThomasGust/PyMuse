import os

def abc2wav(abc_file):
    path_to_tool = os.path.join('bin', 'abc2wav')
    cmd = "{} {}".format(path_to_tool, abc_file)
    return os.system(cmd)