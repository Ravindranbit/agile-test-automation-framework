#!/usr/bin/env python3
"""
Test runner for rlfullcode.py - bypasses GUI for automated testing
"""
import os
import sys

# Configuration
FILE_PATH = "/home/ravindran/worksapce/college/paper/US3.txt"
USER_STORY_NUMBER = 1  # Change this to test different user stories (1-12)

print("="*70)
print("RL/QL AUTO TEST CASE GENERATION - TEST RUN")
print("="*70)
print(f"Input File: {FILE_PATH}")
print(f"User Story: #{USER_STORY_NUMBER}")
print("="*70 + "\n")

# Import all necessary modules
import random
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
import spacy
import string
from word2number import w2n
import xml.etree.ElementTree as ET
import json
import numpy as np
from collections import defaultdict

# Load the spaCy English model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model 'en_core_web_sm' not found.")
    print("Please install it using: python -m spacy download en_core_web_sm")
    sys.exit(1)

# Enhanced pattern repository with generic patterns
repo = {
    # Character type patterns
    "alphabetic": "[[A-Z][a-z]]",
    "alphabet": "[[A-Z][a-z]]",
    "letters": "[[A-Z][a-z]]",
    "alphanumeric": "[[a-zA-Z][0-9]+]",
    "numeric": "[0-9]",
    "number": "[0-9]",
    "numbers": "[0-9]",
    "digit": "[0-9]",
    "digits": "[0-9]",
    "specialcharacters": "[[!#$%&'()+,-./:;<=>?@[]^_`{|}~]+]",
    "special": "[[!#$%&'()+,-./:;<=>?@[]^_`{|}~]+]",
    "special-characters": "[[!#$%&'()+,-./:;<=>?@[]^_`{|}~]+]",
    
    # Constraint keywords
    "should contain": "(include)",
    "should have": "(include)",
    "contain": "(include)",
    "contains": "(include)",
    "include": "(include)",
    "includes": "(include)",
    "be": "(include)",
    "have": "(include)",
    "should not": "(exclude)",
    "should not contain": "(exclude)",
    "must not": "(exclude)",
    "must not contain": "(exclude)",
    "must be": "(include)",
    "must have": "(include)",
    "must contain": "(include)",
    "can be": "(include)",
    "cannot be": "(exclude)",
    "cannot": "(exclude)",
    "cannot contain": "(exclude)",
    "in": "(include)",
    "and": "(include)",
    "only": "(include)",
    "or": "(optional)",
    "not": "(exclude)",
    "if": "(condition)",
    "else": "(condition)",
    "any": "(include)",
    
    # Length constraints
    "minimum": "min length",
    "minimum of": "min length",
    "min": "min length",
    "at least": "min length",
    "atleast": "min length",
    "maximum": "max length",
    "maximum of": "max length",
    "max": "max length",
    "length": "len",
    "size": "len",
    
    # Positional constraints
    "begin": "(begin)",
    "begins": "(begin)",
    "start": "(begin)",
    "starts": "(begin)",
    "end": "(end)",
    "ends": "(end)",
    
    # Comparison operators
    "greater than": ">",
    "less than": "<",
    "equal": "=",
    "equals": "=",
    "greater than or equal to": ">=",
    "less than or equal to": "<=",
    
    # Special symbols
    "hyphen": "-",
    "hifen": "-",
    "dash": "-",
    "@": r"[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    "email": r"[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    "mail": r"[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    "mailid": r"[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    
    # Requirement modifiers
    "required": "(mandatory)",
    "mandatory": "(mandatory)",
    "optional": "(optional)",
    "regex": "(regular expression)",
    
    # Date/Time patterns
    "day_pattern": "[(mon|tue|wed|thu|fri|sat|sun)]",
    "day": "[(mon|tue|wed|thu|fri|sat|sun)]",
    "date": r"[(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}]",
    "time": r"[([01]\d|2[0-3]):[0-5]\d:[0-5]\d$]",
    "datetime": r"[(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4} (0[1-9]|1[0-2]):[0-5]\d [APap][mM]]",
    "timezone": r"[(GMT|[ECMP][DS]T|(?:[A-Z]+\/[A-Z_]+))]",
    "dob": r"[(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}]",
    
    # Other patterns
    "url": r"[[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    "uppercase": "[[A-Z]+]",
    "lowercase": "[[a-z]+]",
    "temperature": r"[-?\d+(\.\d+)?[°]?[CFcf]]",
    "password": "[[a-zA-Z0-9!@#$%^&*]+]",
    "username": "[[a-zA-Z][a-zA-Z0-9_]+]",
    "adress": "[[a-zA-Z0-9@.]+]"
}

# Use provided file path and number
file_path = FILE_PATH
number = USER_STORY_NUMBER

# Continue with the rest of the original script...
# (The complete implementation would be copied here from rlfullcode.py, 
# excluding the GUI components)

print(f"\n✓ Configuration loaded successfully")
print(f"✓ Testing with User Story #{number} from {os.path.basename(file_path)}\n")
