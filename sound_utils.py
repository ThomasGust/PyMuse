import os
from music21 import converter

def abc2wav(abc_file):
    path_to_tool = os.path.join('bin', 'abc2wav')
    cmd = "{} {}".format(path_to_tool, abc_file)
    return os.system(cmd)

def abc2midi(abc_file):
  path_to_tool = os.path.join('bin', 'abc2midi')
  cmd = "{} {}".format(path_to_tool, abc_file)
  return os.system(cmd)

def abc2midipy(in_path, out_path=None):
  s = converter.parse(in_path)
  if out_path is not None:
    s.write("midi", f"{out_path}.mid")
    return s
  else:
    return s