#!/usr/bin/env python3
import os
import sys

# Override input and file dialog for automated testing
os.environ['TEST_MODE'] = '1'
os.environ['TEST_FILE'] = '/home/ravindran/worksapce/college/paper/US3.txt'
os.environ['TEST_NUMBER'] = '1'

# Mock tkinter before import
import sys
from unittest.mock import MagicMock

sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()

# Import and patch
import tkinter as tk
from tkinter import filedialog

# Create mock objects
mock_root = MagicMock()
mock_top = MagicMock()
tk.Tk = MagicMock(return_value=mock_root)
tk.Toplevel = MagicMock(return_value=mock_top)
filedialog.askopenfilename = MagicMock(return_value=os.environ['TEST_FILE'])

# Patch input
original_input = input
def patched_input(prompt):
    print(prompt + os.environ['TEST_NUMBER'])
    return os.environ['TEST_NUMBER']

__builtins__.input = patched_input

# Now run the actual script
print("Starting test run...\n")
exec(compile(open('rlfullcode.py').read(), 'rlfullcode.py', 'exec'))
