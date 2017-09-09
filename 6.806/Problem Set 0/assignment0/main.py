import sys
import os

script_name_argument_index = 1
script_name = sys.argv[script_name_argument_index]

if script_name == '--test_1':
    os.system('python ./test_1.py')
elif script_name == '--test_2':
    os.system('python ./test_2.py')
else:
    raise Exception('provided script name not supported')