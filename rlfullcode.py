#!/usr/bin/env python3
"""
Auto Test Case Generation using Reinforcement Learning & Q-Learning
Processes ALL user stories from input file and generates comprehensive test cases
Single Python file - No JSON output, only testcases.xml
"""

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
import os
import sys
from collections import defaultdict
import numpy as np

# Load the spaCy English model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model 'en_core_web_sm' not found.")
    sys.exit(1)

# Enhanced pattern repository
repo = {
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
    "begin": "(begin)",
    "begins": "(begin)",
    "start": "(begin)",
    "starts": "(begin)",
    "end": "(end)",
    "ends": "(end)",
    "greater than": ">",
    "less than": "<",
    "equal": "=",
    "equals": "=",
    "greater than or equal to": ">=",
    "less than or equal to": "<=",
    "hyphen": "-",
    "hifen": "-",
    "dash": "-",
    "@": r"[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    "email": r"[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    "mail": r"[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    "mailid": r"[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    "required": "(mandatory)",
    "mandatory": "(mandatory)",
    "optional": "(optional)",
    "regex": "(regular expression)",
    "day_pattern": "[(mon|tue|wed|thu|fri|sat|sun)]",
    "day": "[(mon|tue|wed|thu|fri|sat|sun)]",
    "date": r"[(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}]",
    "time": r"[([01]\d|2[0-3]):[0-5]\d:[0-5]\d$]",
    "datetime": r"[(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4} (0[1-9]|1[0-2]):[0-5]\d [APap][mM]]",
    "timezone": r"[(GMT|[ECMP][DS]T|(?:[A-Z]+\/[A-Z_]+))]",
    "dob": r"[(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}]",
    "url": r"[[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    "uppercase": "[[A-Z]+]",
    "lowercase": "[[a-z]+]",
    "temperature": r"[-?\d+(\.\d+)?[°]?[CFcf]]",
    "password": "[[a-zA-Z0-9!@#$%^&*]+]",
    "username": "[[a-zA-Z][a-zA-Z0-9_]+]",
    "adress": "[[a-zA-Z0-9@.]+]"
}

# Auto-detect US3.txt file in current directory
current_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else '.'
file_path = os.path.join(current_dir, "US3.txt")

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found")
    sys.exit(1)

print(f"Processing: {file_path}\n")

# Helper functions
def is_regex(string):
    return string.startswith("[") and string.endswith("]")

def generate_random_string(pattern, numeric_values, constraint_info=None):
    try:
        numeric_length = int(numeric_values) if numeric_values else 8
    except (ValueError, TypeError):
        numeric_length = 8

    # Enforce max length constraint if present in constraint_info
    max_length = None
    min_length = 1
    if constraint_info:
        for c in constraint_info:
            m = re.search(r'max length (\d+)', c)
            if m:
                max_length = int(m.group(1))
            m2 = re.search(r'min length (\d+)', c)
            if m2:
                min_length = int(m2.group(1))
    if max_length is not None:
        # For username patterns, always use max_length (not random up to max_length)
        username_patterns = ["[[a-zA-Z][a-zA-Z0-9_]+]", "[[A-Z][a-z]]", "[[a-zA-Z][0-9]+]"]
        if pattern in username_patterns:
            numeric_length = max_length
        else:
            numeric_length = random.randint(min_length, max_length)
    elif numeric_length < 1:
        numeric_length = 8

    # After generating, enforce max_length for username and similar patterns
    def enforce_max_length(s):
        if max_length is not None and len(s) > max_length:
            return s[:max_length]
        return s

    # ... pattern generation logic below ...
    
    if pattern == "[[a-zA-Z][0-9]+]":
        letters = random.randint(max(1, numeric_length // 2), numeric_length - 1)
        digits = numeric_length - letters
        result = ''.join(random.choice(string.ascii_letters) for _ in range(letters))
        result += ''.join(random.choice(string.digits) for _ in range(digits))
        return enforce_max_length(''.join(random.sample(result, len(result))))
    elif pattern == "[[A-Z][a-z]]":
        return enforce_max_length(''.join(random.choice(string.ascii_letters) for _ in range(numeric_length)))
    elif pattern == "[[A-Z]+]":
        return enforce_max_length(''.join(random.choice(string.ascii_uppercase) for _ in range(numeric_length)))
    elif pattern == "[[a-z]+]":
        return enforce_max_length(''.join(random.choice(string.ascii_lowercase) for _ in range(numeric_length)))
    elif pattern == "[0-9]":
        return enforce_max_length(''.join(random.choice(string.digits) for _ in range(numeric_length)))
    elif pattern == r"[(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}]":
        month = str(random.randint(1, 12)).zfill(2)
        day = str(random.randint(1, 28)).zfill(2)
        year = str(random.randint(1950, 2026))
        return f"{month}/{day}/{year}"
    elif pattern == r"[(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4} (0[1-9]|1[0-2]):[0-5]\d [APap][mM]]":
        day = str(random.randint(1, 28)).zfill(2)
        month = str(random.randint(1, 12)).zfill(2)
        year = str(random.randint(1950, 2026))
        hour = str(random.randint(1, 12)).zfill(2)
        minute = str(random.randint(0, 59)).zfill(2)
        am_pm = random.choice(['AM', 'PM'])
        return f"{day}/{month}/{year} {hour}:{minute} {am_pm}"
    elif pattern == r"[([01]\d|2[0-3]):[0-5]\d:[0-5]\d$]":
        hour = str(random.randint(0, 23)).zfill(2)
        minute = str(random.randint(0, 59)).zfill(2)
        second = str(random.randint(0, 59)).zfill(2)
        return f"{hour}:{minute}:{second}"
    elif pattern == "[(mon|tue|wed|thu|fri|sat|sun)]":
        return random.choice(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
    elif pattern == "[(GMT|[ECMP][DS]T|(?:[A-Z]+\\/[A-Z_]+))]":
        return random.choice(["GMT", "EST", "EDT", "CST", "CDT", "PST", "PDT", "MST", "MDT"])
    elif pattern == r"[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]":
        username_len = max(5, numeric_length // 2)
        username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_len))
        domain = ''.join(random.choices(string.ascii_lowercase, k=5))
        tld = random.choice(['com', 'org', 'net', 'edu', 'io'])
        return f"{username}@{domain}.{tld}"
    elif pattern == "[[!#$%&'()+,-./:;<=>?@[]^_`{|}~]+]":
        special_chars = "!#$%&'()+,-./:;<=>?@^_`{|}~"
        return ''.join(random.choice(special_chars) for _ in range(numeric_length))
    elif pattern == r"[-?\d+(\.\d+)?[°]?[CFcf]]":
        temperature = random.uniform(-50.0, 50.0)
        unit = random.choice(['C', 'F'])
        return f"{temperature:.1f}°{unit}"
    elif pattern == "[[a-zA-Z][a-zA-Z0-9_]+]":
        result = random.choice(string.ascii_letters)
        result += ''.join(random.choices(string.ascii_letters + string.digits + '_', k=numeric_length-1))
        return enforce_max_length(result)
    elif pattern == "[[a-zA-Z0-9!@#$%^&*]+]":
        result = []
        result.append(random.choice(string.ascii_uppercase))
        result.append(random.choice(string.ascii_lowercase))
        result.append(random.choice(string.digits))
        result.append(random.choice('!@#$%^&*'))
        remaining = numeric_length - len(result)
        if remaining > 0:
            result.extend(random.choices(string.ascii_letters + string.digits + '!@#$%^&*', k=remaining))
        return ''.join(random.sample(result, len(result)))
    elif pattern == r"[[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]":
        domain = ''.join(random.choices(string.ascii_lowercase + string.digits, k=numeric_length))
        tld = random.choice(['com', 'org', 'net', 'io', 'co.uk'])
        return f"{domain}.{tld}"
    elif pattern == "[[a-zA-Z0-9@.]+]":
        return ''.join(random.choices(string.ascii_letters + string.digits + '@.', k=numeric_length))
    else:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=numeric_length))

def generate_random_invalid(pattern, numeric_values, constraint_info=None):
    try:
        numeric_length = int(numeric_values) if numeric_values else 8
    except (ValueError, TypeError):
        numeric_length = 8
    
    violation_strategy = random.choice(['length', 'type', 'format', 'boundary'])
    
    if violation_strategy == 'length':
        if random.choice([True, False]) and numeric_length > 1:
            invalid_length = max(1, numeric_length - random.randint(1, 3))
        else:
            invalid_length = numeric_length + random.randint(10, 20)
        numeric_length = invalid_length
    
    if pattern == "[[a-zA-Z][0-9]+]":
        if violation_strategy == 'type':
            return ''.join(random.choice(string.ascii_letters) for _ in range(numeric_length))
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(max(1, numeric_length - 2)))
    elif pattern == "[[A-Z][a-z]]":
        if violation_strategy == 'type':
            return ''.join(random.choice(string.digits + '!@#$') for _ in range(numeric_length))
        return ''.join(random.choice(string.ascii_letters) for _ in range(max(1, numeric_length - 2)))
    elif pattern == "[[A-Z]+]":
        return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(numeric_length))
    elif pattern == "[[a-z]+]":
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(numeric_length))
    elif pattern == "[0-9]":
        if violation_strategy == 'type':
            return ''.join(random.choice(string.ascii_letters) for _ in range(numeric_length))
        return ''.join(random.choice(string.digits) for _ in range(max(1, numeric_length - 2)))
    elif pattern == r"[(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}]":
        month = str(random.randint(13, 20)).zfill(2)
        day = str(random.randint(32, 40)).zfill(2)
        year = str(random.randint(1950, 2026))
        return f"{month}/{day}/{year}"
    elif pattern == r"[(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4} (0[1-9]|1[0-2]):[0-5]\d [APap][mM]]":
        day = str(random.randint(32, 40)).zfill(2)
        month = str(random.randint(1, 12)).zfill(2)
        year = str(random.randint(1950, 2026))
        hour = str(random.randint(13, 20)).zfill(2)
        minute = str(random.randint(60, 99)).zfill(2)
        return f"{day}/{month}/{year} {hour}:{minute} XX"
    elif pattern == r"[([01]\d|2[0-3]):[0-5]\d:[0-5]\d$]":
        hour = str(random.randint(24, 30)).zfill(2)
        minute = str(random.randint(60, 99)).zfill(2)
        second = str(random.randint(0, 59)).zfill(2)
        return f"{hour}:{minute}:{second}"
    elif pattern == r"[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]":
        if violation_strategy == 'format':
            username = ''.join(random.choices(string.ascii_lowercase, k=numeric_length))
            return f"{username}domain.com"
        else:
            return ''.join(random.choices(string.ascii_letters, k=numeric_length))
    elif pattern == "[[!#$%&'()+,-./:;<=>?@[]^_`{|}~]+]":
        if violation_strategy == 'type':
            return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(numeric_length))
        return ''.join(random.choice("!@#$") for _ in range(max(1, numeric_length - 2)))
    elif pattern == "[[a-zA-Z][a-zA-Z0-9_]+]":
        result = random.choice(string.digits)
        result += ''.join(random.choices(string.ascii_letters + string.digits, k=numeric_length-1))
        return result
    elif pattern == "[[a-zA-Z0-9!@#$%^&*]+]":
        if violation_strategy == 'type':
            return ''.join(random.choice(string.ascii_lowercase) for _ in range(max(1, numeric_length - 2)))
        return ''.join(random.choice(string.digits) for _ in range(numeric_length))
    else:
        return ''.join(random.choices(string.ascii_letters, k=max(1, numeric_length - random.randint(2, 5))))

def process_keywords(keywords_list):
    result = []
    exclude_mode = False
    for keyword in keywords_list:
        keyword_lower = keyword.lower()
        # Handle 'less than' and 'greater than' phrases for length constraints
        less_than_match = re.search(r'less than (\w+)', keyword_lower)
        greater_than_match = re.search(r'greater than (\w+)', keyword_lower)
        if less_than_match:
            try:
                num = w2n.word_to_num(less_than_match.group(1))
                # For 'less than N', max length is N-1
                keyword = re.sub(r'less than (\w+)', f'max length {num-1}', keyword_lower)
            except Exception:
                pass
        elif greater_than_match:
            try:
                num = w2n.word_to_num(greater_than_match.group(1))
                # For 'greater than N', min length is N+1
                keyword = re.sub(r'greater than (\w+)', f'min length {num+1}', keyword_lower)
            except Exception:
                pass
        if '(exclude)' in keyword_lower:
            exclude_mode = True
            continue
        elif '(include)' in keyword_lower or '(mandatory)' in keyword_lower:
            exclude_mode = False
            continue
        if not exclude_mode:
            match = re.search(r'\((\w+)\)', keyword)
            if match:
                word_inside_parentheses = match.group(1)
                try:
                    numeric_representation = w2n.word_to_num(word_inside_parentheses)
                    keyword = keyword.replace(match.group(), f'({numeric_representation})')
                except (ValueError, AttributeError):
                    num_match = re.search(r'\b(\d+)\b', keyword)
                    if num_match:
                        numeric_representation = int(num_match.group(1))
                        if '(' not in keyword:
                            keyword = f"{keyword} ({numeric_representation})"
            result.append(keyword)
    return result

# Main processing
print("="*70)
print("AUTO TEST CASE GENERATION - ALL USER STORIES")
print("="*70)

# Count stories
with open(file_path, "r") as file:
    content = file.read()
    story_numbers = re.findall(r'^\d+\.', content, re.MULTILINE)
    total_stories = len(story_numbers)

print(f"\nFound {total_stories} user stories\n")

# Process ALL stories
all_valid_testcases = []
all_invalid_testcases = []
story_results = {}

for story_num in range(1, total_stories + 1):
    print(f"\n{'='*70}")
    print(f"USER STORY #{story_num}/{total_stories}")
    print(f"{'='*70}")
    
    accepted_criteria = []
    
    with open(file_path, "r") as file:
        lines = file.readlines()
        num_lines = len(lines)
        current_number = None
        current_criteria = ""
        for i in range(num_lines):
            line = lines[i].strip()
            if line.startswith(str(story_num) + "."):
                current_number = int(line.split(".")[0])
            elif current_number is not None:
                if line and not line.startswith(str(current_number + 1) + "."):
                    current_criteria += line + "\n"
                else:
                    if "Acceptance criteria:" not in current_criteria:
                        accepted_criteria.append(current_criteria.strip())
                    current_criteria = ""
            if current_number is not None and line.startswith(str(current_number + 1) + "."):
                break
    
    if not accepted_criteria:
        print(f"  \u2192 No acceptance criteria found")
        story_results[story_num] = {'valid': [], 'invalid': []}
        continue
    
    print(f"  Criteria count: {len(accepted_criteria)}")
    
    # Extract keywords
    STRING_LIST = accepted_criteria
    keywords = set()
    for text in STRING_LIST:
        words = re.findall(r'\w+', text)
        keywords.update(words)
    
    for keyword in keywords:
        if keyword not in repo:
            repo[keyword] = f"({keyword})"
    
    # Load stopwords
    stopwords_file = os.path.join(os.path.dirname(__file__), "stopwords.txt")
    try:
        with open(stopwords_file, "r") as file:
            stopwords_list = set(file.read().splitlines())
    except FileNotFoundError:
        try:
            from nltk.corpus import stopwords
            stopwords_list = set(stopwords.words('english'))
        except:
            stopwords_list = set()
    
    r = Rake(stopwords=stopwords_list)
    keywords_list = []
    
    for text in STRING_LIST:
        for line in text.split('\n'):
            if line.strip():
                r.extract_keywords_from_text(line)
                keywords = r.get_ranked_phrases()
                if keywords:
                    keywords_list.append(keywords)
    
    # Q-Learning parameters
    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 0.3
    NUM_EPISODES = 5
    
    q_table = defaultdict(lambda: defaultdict(float))
    
    def get_state_from_keywords(keywords):
        pattern_types = []
        constraint_types = []
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if any(p in keyword_lower for p in ['alphabet', 'letter', 'alphabetic']):
                pattern_types.append('alphabetic')
            if any(p in keyword_lower for p in ['numeric', 'number', 'digit']):
                pattern_types.append('numeric')
            if any(p in keyword_lower for p in ['alphanumeric']):
                pattern_types.append('alphanumeric')
            if any(p in keyword_lower for p in ['special', 'symbol']):
                pattern_types.append('special')
            if any(p in keyword_lower for p in ['date', 'time', 'datetime']):
                pattern_types.append('datetime')
            if any(p in keyword_lower for p in ['email', 'mail', '@']):
                pattern_types.append('email')
            if any(c in keyword_lower for c in ['minimum', 'min', 'at least', 'atleast']):
                constraint_types.append('min_length')
            if any(c in keyword_lower for c in ['maximum', 'max']):
                constraint_types.append('max_length')
            if any(c in keyword_lower for c in ['exclude', 'not', 'cannot']):
                constraint_types.append('exclude')
            if any(c in keyword_lower for c in ['include', 'contain', 'must', 'should']):
                constraint_types.append('include')
        
        pattern_str = ','.join(sorted(set(pattern_types))) if pattern_types else 'generic'
        constraint_str = ','.join(sorted(set(constraint_types))) if constraint_types else 'none'
        return (pattern_str, constraint_str)
    
    def get_possible_actions(state):
        pattern_str, constraint_str = state
        actions = []
        patterns = pattern_str.split(',') if pattern_str != 'generic' else ['alphabetic', 'numeric', 'alphanumeric']
        for pattern in patterns:
            for length in [5, 7, 8, 10, 12, 15, 20]:
                actions.append((pattern, length))
        return actions if actions else [('alphabetic', 8)]
    
    def calculate_reward(generated_string, expected_constraints):
        reward = 0.0
        has_min_length = any('min' in c for c in expected_constraints)
        has_max_length = any('max' in c for c in expected_constraints)
        has_exclude = any('exclude' in c for c in expected_constraints)
        has_include = any('include' in c for c in expected_constraints)
        
        if generated_string:
            reward += 10.0
            if has_min_length and len(generated_string) >= 7:
                reward += 20.0
            if has_max_length and len(generated_string) <= 20:
                reward += 20.0
            if has_include:
                if any(c.isalpha() for c in generated_string):
                    reward += 15.0
                if any(c.isdigit() for c in generated_string):
                    reward += 15.0
                if any(c in string.punctuation for c in generated_string):
                    reward += 15.0
            if has_exclude:
                reward -= 5.0
        return reward
    
    def select_action(state, epsilon):
        if random.random() < epsilon:
            possible_actions = get_possible_actions(state)
            return random.choice(possible_actions)
        else:
            possible_actions = get_possible_actions(state)
            q_values = {action: q_table[state][action] for action in possible_actions}
            if not q_values or all(v == 0 for v in q_values.values()):
                return random.choice(possible_actions)
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(state, action, reward, next_state):
        current_q = q_table[state][action]
        next_actions = get_possible_actions(next_state)
        max_next_q = max([q_table[next_state][a] for a in next_actions], default=0)
        new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
        q_table[state][action] = new_q
    
    # Train
    for episode in range(NUM_EPISODES):
        episode_reward = 0
        for i, keywords in enumerate(keywords_list):
            state = get_state_from_keywords(keywords)
            action = select_action(state, EPSILON)
            expected_constraints = [kw.lower() for kw in keywords]
            simulated_string = ''.join(random.choices(string.ascii_letters + string.digits, k=action[1]))
            reward = calculate_reward(simulated_string, expected_constraints)
            next_state = state
            update_q_value(state, action, reward, next_state)
            episode_reward += reward
    
    # Generate test cases
    valid_testcases = []
    invalid_testcases = []
    
    for S in STRING_LIST:
        sentences = re.split(r'[.!]', S)
        for line in sentences:
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            words = re.findall(r'\b\w+\b|[^\w\s]', line.lower())
            if not words:
                continue
            
            extraction = []
            for word in words:
                if word in repo:
                    extraction.append(repo[word])
            
            if not extraction:
                continue
            
            include_keywords = process_keywords(extraction)
            if not include_keywords:
                continue
            
            regex_keywords = [kw for kw in include_keywords if is_regex(kw)]
            if not regex_keywords:
                continue
            
            numeric_values = []
            # Extract (N) from (N), and also from 'max length N' and 'min length N' in include_keywords
            for keyword in include_keywords:
                match = re.search(r'\((\d+)\)', keyword)
                if match:
                    numeric_values.append(match.group(1))
                # Also extract from 'max length N' and 'min length N'
                maxlen_match = re.search(r'max length (\d+)', keyword)
                if maxlen_match:
                    numeric_values.append(maxlen_match.group(1))
                minlen_match = re.search(r'min length (\d+)', keyword)
                if minlen_match:
                    numeric_values.append(minlen_match.group(1))
            # Fallback: extract numbers from the original line
            number_matches = re.findall(r'\b(\d+)\b', line)
            numeric_values.extend(number_matches)
            # Fallback: extract numbers from words
            word_numbers = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12', 'fifteen': '15', 'twenty': '20', 'fifty': '50', 'hundred': '100'}
            for word, num in word_numbers.items():
                if word in line.lower():
                    numeric_values.append(num)
            # Remove duplicates, preserve order
            seen = set()
            unique_numeric_values = []
            for nv in numeric_values:
                if nv not in seen:
                    seen.add(nv)
                    unique_numeric_values.append(nv)
            numeric_values = unique_numeric_values
            
            if not numeric_values:
                if any('minimum' in kw.lower() or 'min' in kw.lower() for kw in include_keywords):
                    numeric_values = ['7', '8']
                elif any('maximum' in kw.lower() or 'max' in kw.lower() for kw in include_keywords):
                    numeric_values = ['20', '50']
                else:
                    numeric_values = ['8', '10']
            
            while len(numeric_values) < len(regex_keywords):
                numeric_values.append(numeric_values[-1] if numeric_values else '8')
            
            # Generate separate test cases per field (not concatenated)
            for idx, (regex_keyword, numeric_value) in enumerate(zip(regex_keywords, numeric_values)):
                try:
                    valid_random_string = generate_random_string(regex_keyword, numeric_value, include_keywords)
                    # Filter out meaningless short values
                    if valid_random_string is not None and len(valid_random_string) >= 3:
                        # Skip single special characters or very short non-meaningful values
                        if not (len(valid_random_string) <= 2 and not valid_random_string.isalnum()):
                            valid_testcases.append(valid_random_string)
                    
                    invalid_random_string = generate_random_invalid(regex_keyword, numeric_value, include_keywords)
                    if invalid_random_string is not None and len(invalid_random_string) >= 1:
                        invalid_testcases.append(invalid_random_string)
                except Exception as e:
                    continue
    
    # Store results
    story_results[story_num] = {'valid': valid_testcases, 'invalid': invalid_testcases}
    all_valid_testcases.extend(valid_testcases)
    all_invalid_testcases.extend(invalid_testcases)
    
    print(f"  ✓ Generated {len(valid_testcases)} valid test cases")
    print(f"  ✓ Generated {len(invalid_testcases)} invalid test cases")

# Generate XML with all test cases
print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}\n")

# Organize test cases by User Story
root_element = ET.Element("TestCases")

for story_num in sorted(story_results.keys()):
    story_element = ET.SubElement(root_element, f"UserStory_{story_num}")
    
    # Add valid test cases for this story
    valid_element = ET.SubElement(story_element, "Valid")
    for i, testcase in enumerate(story_results[story_num]['valid'], start=1):
        tc_element = ET.SubElement(valid_element, f"TestCase_{i}")
        tc_element.text = testcase
    
    # Add invalid test cases for this story
    invalid_element = ET.SubElement(story_element, "Invalid")
    for i, testcase in enumerate(story_results[story_num]['invalid'], start=1):
        tc_element = ET.SubElement(invalid_element, f"TestCase_{i}")
        tc_element.text = testcase

tree = ET.ElementTree(root_element)
xml_filename = os.path.join(os.path.dirname(__file__) if os.path.dirname(__file__) else '.', "testcases.xml")

try:
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_filename, encoding='unicode', xml_declaration=True)
    print(f"✓ Test cases saved to: {xml_filename}")
    print(f"  - Total valid test cases: {len(all_valid_testcases)}")
    print(f"  - Total invalid test cases: {len(all_invalid_testcases)}")
    print(f"  - User stories processed: {total_stories}")
    print(f"\n{'='*70}")
    print("PROCESS COMPLETE ✓")
    print(f"{'='*70}")
except Exception as e:
    print(f"Error saving XML file: {str(e)}")
