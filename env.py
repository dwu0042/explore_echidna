import configparser
import pathlib

parser = configparser.ConfigParser()
parser.read("env.ini")
env = {k: pathlib.Path(v) for k,v in parser['filetree'].items()}

subdirs = [
    "notebooks",
    "outputs",
    "report",
    "scripts",
]
for subdir in subdirs:
    env[subdir] = env['root'] / subdir