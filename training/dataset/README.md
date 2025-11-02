# Dataset

## Download Instructions

The training dataset is **not included** in this repository due to its size (~350 MB).

### Option 1: Download from Kaggle

1. Visit: [Python Code Instruction Dataset](https://www.kaggle.com/datasets/thedevastator/python-code-instruction-dataset)
2. Download `train.csv`
3. Convert to JSON format or use the preprocessing script
4. Place as `kaggle_python_dataset.json` in this directory

### Option 2: Use Your Own Dataset

Create a JSON file with the following format:

```json
[
  {
    "instruction": "Write a function to add two numbers",
    "input": "",
    "output": "def add(a, b):\n    return a + b"
  },
  {
    "instruction": "Create a function to calculate factorial",
    "input": "n = 5",
    "output": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)"
  }
]
```

### Required Fields

- `instruction`: Description of what the code should do
- `input`: Optional input context
- `output`: The actual Python code

### Dataset Statistics (Kaggle dataset)

- Total samples: 18,612
- Language: Python only
- Size: ~350 MB
- Format: JSON

## After Download

```bash
# Verify dataset is in correct location
ls training/dataset/kaggle_python_dataset.json

# Start training
cd training
python train_model.py
```
