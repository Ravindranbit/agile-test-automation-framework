# Real-Time Test Generation using Natural Language Processing and Reinforcement Learning in Agile Development

![Research Status](https://img.shields.io/badge/Research-Implementation-red)
![Field](https://img.shields.io/badge/Domain-NLP%20%7C%20Reinforcement%20Learning-blue)
![Agile](https://img.shields.io/badge/Agile-DevOps-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)

## Abstract

This repository contains the official implementation of a novel framework for **automated test case generation** in Agile environments. By combining **Natural Language Processing (NLP)** to interpret evolving User Stories and **Reinforcement Learning (RL)** with **Q-Learning** to optimize test data generation, the system produces both valid and invalid test cases in real-time. This approach minimizes manual intervention and adapts dynamically to the rapid release cycles inherent in Agile development.

---

## Key Features

- **Automated Test Case Generation**: Generates both valid and invalid test cases from natural language User Stories
- **NLP-Powered Parsing**: Uses spaCy and RAKE for semantic extraction of constraints from acceptance criteria
- **Q-Learning Optimization**: Employs reinforcement learning to optimize test data generation strategies
- **Pattern Recognition**: Extensive pattern repository for common data types (email, date, time, phone, etc.)
- **Constraint Handling**: Supports min/max length, inclusion/exclusion, and format constraints
- **XML Output**: Generates structured test cases in XML format for easy integration with test frameworks

---

##  System Architecture

The framework operates through a modular pipeline that bridges the gap between human-readable requirements and machine-executable test data.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                    |
│                     User Stories (Plain Text Format)                        │
│                                                                             │
│  "As a user, I want to login to the system.                                 │
│   Acceptance Criteria:                                                      │
│   - The password must be alphanumeric with length of eight."                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NLP PROCESSING LAYER                              │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   Tokenization  │───▶│ Keyword Extract │───▶│ Pattern Mapping │          │
│  │    (spaCy)      │    │    (RAKE)       │    │  (Repository)   │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│                                                                             │
│  Extracts: Field names, Data types, Constraints, Length requirements        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REINFORCEMENT LEARNING ENGINE                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Q-Learning Agent                             │    │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐           │    │
│  │  │  State  │───▶│ Action  │───▶│ Reward  │───▶│ Update  │           │    │
│  │  │ (Pattern│    │(Generate│    │(Validate│    │Q-Table  │           │    │
│  │  │  Type)  │    │  Data)  │    │ Output) │    │         │           │    │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Parameters: α=0.1 (learning rate), γ=0.9 (discount), ε=0.3 (exploration)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEST GENERATION LAYER                             │
│                                                                             │
│  ┌──────────────────────┐         ┌──────────────────────┐                  │
│  │    Valid Test Cases  │         │  Invalid Test Cases  │                  │
│  │  ─────────────────── │         │  ─────────────────── │                  │
│  │  • Constraint-       │         │  • Length violations │                  │
│  │    compliant data    │         │  • Type mismatches   │                  │
│  │  • Format-correct    │         │  • Format errors     │                  │
│  │  • Boundary values   │         │  • Boundary exceeded │                  │
│  └──────────────────────┘         └──────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER                                   │
│                          testcases.xml (Structured)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## User Story Format

The system processes User Stories written in a specific format. Each User Story must follow this structure:

### Basic Structure

```
[Story Number]. As a [role], I want to [action].

Acceptance Criteria:

[Constraint 1]
[Constraint 2]
...
```

### Example User Stories

#### Example 1: Login System
```
1. As a developer, I want to login to the system.

Acceptance Criteria:

The password must be less than six characters.
The username accepts alphanumeric with length of eight.
```

#### Example 2: Account Management
```
2. As a user, I want to deactivate my account.

Acceptance Criteria:

The description is required.
The zip code should be greater than nine characters.
The account number must have a minimum of three digit characters.
The email must be valid format.
```

#### Example 3: File Upload
```
3. As a user, I want to upload a file.

Acceptance Criteria:

The file name must be alphanumeric.
The file size should be less than 10 MB.
The extension must be alphabetic and maximum of four characters.
```

#### Example 4: Registration Form
```
4. As a new user, I want to register an account.

Acceptance Criteria:

The username must be alphanumeric with minimum of six characters.
The password should contain special characters and alphabetic.
The email must be valid format.
The phone number must be numeric with exactly ten digits.
The date of birth must be in the format MM/DD/YYYY.
```

---

##  Supported Constraints & Keywords

### Data Type Keywords

| Keyword | Pattern Generated | Example Output |
|---------|-------------------|----------------|
| `alphabetic`, `letters`, `alphabet` | `[A-Za-z]` | `AbCdEf` |
| `numeric`, `number`, `digits` | `[0-9]` | `123456` |
| `alphanumeric` | `[A-Za-z0-9]` | `Ab12Cd` |
| `special characters`, `symbols` | `[!@#$%^&*]` | `!@#$%^` |
| `email`, `mail`, `mailid` | Email format | `user@domain.com` |
| `date` | `MM/DD/YYYY` | `01/15/2024` |
| `time` | `HH:MM:SS` | `14:30:45` |
| `datetime` | `DD/MM/YYYY HH:MM AM/PM` | `15/01/2024 02:30 PM` |
| `url` | URL format | `example.com` |
| `password` | Mixed characters | `P@ss1word` |
| `username` | Username format | `user_123` |

### Length Constraint Keywords

| Keyword | Interpretation | Example |
|---------|----------------|---------|
| `minimum`, `min`, `at least` | Minimum length | "minimum of 5 characters" → length ≥ 5 |
| `maximum`, `max` | Maximum length | "maximum of 10 characters" → length ≤ 10 |
| `exactly` | Exact length | "exactly six characters" → length = 6 |
| `less than` | Less than N | "less than 10" → length ≤ 9 |
| `greater than` | Greater than N | "greater than 5" → length ≥ 6 |

### Inclusion/Exclusion Keywords

| Keyword | Action |
|---------|--------|
| `must contain`, `should have`, `include` | Include pattern |
| `must not`, `cannot`, `should not` | Exclude pattern |
| `required`, `mandatory` | Field is required |
| `optional` | Field is optional |

### Number Words Supported
The system understands written numbers:
- `one` → 1, `two` → 2, `three` → 3, `four` → 4, `five` → 5
- `six` → 6, `seven` → 7, `eight` → 8, `nine` → 9, `ten` → 10
- `eleven` → 11, `twelve` → 12, `fifteen` → 15, `twenty` → 20

---

##  Q-Learning Reward Function

The Reinforcement Learning agent uses a custom reward function to optimize test case generation:

**R(s, a) = w₁ · I(valid) + w₂ · I(min_length) + w₃ · I(max_length) + w₄ · TypeMatch - w₅ · I(exclude)**

Where:
- **w₁ = 10.0** : Base reward for generating valid string
- **w₂ = 20.0** : Reward for meeting minimum length constraint
- **w₃ = 20.0** : Reward for meeting maximum length constraint
- **w₄ = 15.0** : Reward for matching required character types (alpha, numeric, special)
- **w₅ = 5.0** : Penalty for violating exclusion constraints

(I = Indicator function, evaluates to 1 if condition is true, 0 otherwise)

### Q-Learning Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| α (Alpha) | 0.1 | Learning rate - how much new information overrides old |
| γ (Gamma) | 0.9 | Discount factor - importance of future rewards |
| ε (Epsilon) | 0.3 | Exploration rate - probability of random action |
| Episodes | 1000 | Number of training iterations |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Programming Language** | Python 3.10+ |
| **NLP Library** | spaCy (en_core_web_sm), RAKE-NLTK |
| **Tokenization** | NLTK |
| **Word-to-Number** | word2number |
| **Output Format** | XML (ElementTree) |
| **Reinforcement Learning** | Custom Q-Learning implementation |

---

##  Project Structure

```
├── rlfullcode.py              # Main RL/Q-Learning test generation engine
├── US3.txt                    # Input file containing User Stories
├── testcases.xml              # Generated test cases output (XML format)
├── learned_patterns.json      # Stored learned patterns from RL agent
├── learned_patterns_before.json # Backup of previous learned patterns
├── stopwords.txt              # Custom stopwords for NLP processing
├── test_runner.py             # Test runner for automated testing
├── run_test.py                # Script to run tests without GUI
└── README.md                  # This documentation file
```

---

##  Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Virtual Environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ravindranbit/agile-test-automation-framework.git
   cd agile-test-automation-framework
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # OR
   .\venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install spacy nltk rake-nltk word2number numpy
   ```

4. **Download spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"
   ```

### Running the System

1. **Prepare your User Stories**
   
   Create or edit `US3.txt` with your User Stories following the format described above.

2. **Run the test case generator**
   ```bash
   python rlfullcode.py
   ```

3. **View generated test cases**
   
   The output will be saved to `testcases.xml` in the same directory.

---

##  Output Format

The system generates test cases in XML format, organized by User Story:

```xml
<?xml version='1.0' encoding='utf-8'?>
<TestCases>
  <UserStory_1>
    <Valid>
      <TestCase_1>8Zd&amp;6e</TestCase_1>
      <TestCase_2>y1OF9r53</TestCase_2>
    </Valid>
    <Invalid>
      <TestCase_1>146425</TestCase_1>
      <TestCase_2>vpGCO</TestCase_2>
    </Invalid>
  </UserStory_1>
  <UserStory_2>
    <Valid>
      <TestCase_1>389</TestCase_1>
      <TestCase_2>g9gxn@uursg.net</TestCase_2>
    </Valid>
    <Invalid>
      <TestCase_1>2</TestCase_1>
      <TestCase_2>LRUJPvUqzRzFOoDuLwYirHJx</TestCase_2>
    </Invalid>
  </UserStory_2>
</TestCases>
```

---

## Sample Results

Processing 500 synthetic User Stories, the system typically generates:

| Metric | Value |
|--------|-------|
| Total User Stories Processed | 500 |
| Valid Test Cases Generated | ~3,000-5,000 |
| Invalid Test Cases Generated | ~3,000-5,000 |
| Processing Time | ~30-60 seconds |

---

## How It Works

### Step 1: Parse User Stories
The system reads the input file and splits it into individual User Stories based on the numbering pattern.

### Step 2: Extract Acceptance Criteria
Each User Story's acceptance criteria are parsed to identify:
- Field names (password, username, email, etc.)
- Data type requirements (alphabetic, numeric, etc.)
- Constraints (min/max length, format requirements)

### Step 3: NLP Processing
- **Tokenization**: Break sentences into tokens using spaCy
- **Keyword Extraction**: Use RAKE algorithm to extract key phrases
- **Pattern Mapping**: Map extracted keywords to regex patterns from the repository

### Step 4: Q-Learning Optimization
The RL agent learns optimal generation strategies by:
1. Observing the current state (pattern type + constraints)
2. Selecting an action (pattern + length combination)
3. Generating test data and calculating reward
4. Updating Q-table for future decisions

### Step 5: Generate Test Cases
For each constraint:
- **Valid Test Cases**: Generated to satisfy all specified constraints
- **Invalid Test Cases**: Generated with deliberate violations (wrong length, wrong type, wrong format)

### Step 6: Output Results
All generated test cases are organized by User Story and saved to XML format.

---

