# AI 100 Final Project - GenAI Debugging System

## Overview
This project builds a CNN image classifier on Fashion-MNIST using PyTorch, then intentionally introduces 10 bugs to study how AI systems fail and how GenAI (Claude) can serve as a Socratic reasoning partner during debugging.

## Authors
- Devin Myers
- Vina Dang

## Repository Contents
```
AI100FinalProject/
  code/
    model.py                  # Working CNN classifier (91% accuracy)
    bug_cases/
      bug01_wrong_loss.py     # MSELoss instead of CrossEntropyLoss
      bug02_high_lr.py        # Learning rate 10.0 instead of 0.001
      bug03_wrong_num_classes.py  # 5 output classes instead of 10
      bug04_no_normalization.py   # Removed data normalization
      bug05_shuffled_labels.py    # Randomly permuted training labels
      bug06_no_relu.py        # Removed all ReLU activations
      bug07_wrong_input_channels.py # 3 input channels instead of 1
      bug08_no_grad_zero.py   # Removed optimizer.zero_grad()
      bug09_no_eval_mode.py   # Missing model.eval() at test time
      bug10_sigmoid_output.py # Added sigmoid before CrossEntropyLoss
  genai_bug_cases.xlsx        # Spreadsheet with all 10 cases
  AI 100 Final Project Report.pdf  # PDF report
```

## How to Run

### Requirements
```bash
pip install torch torchvision matplotlib
```

### Run the working model
```bash
cd code
python model.py
```
Expected output: ~91% test accuracy after 5 epochs.

### Run any bug case
```bash
cd code
python bug_cases/bug01_wrong_loss.py
```
Each bug case script demonstrates a specific failure mode.

## Bug Case Summary

| Case | Bug | Type |
|------|-----|------|
| 1 | Wrong loss function (MSELoss) | Crash |
| 2 | Learning rate too high (10.0) | Silent degradation |
| 3 | Wrong number of output classes (5) | Crash |
| 4 | No data normalization | Silent degradation |
| 5 | Shuffled training labels | Silent degradation |
| 6 | Removed ReLU activations | Silent degradation |
| 7 | Wrong input channels (3 vs 1) | Crash |
| 8 | Missing optimizer.zero_grad() | Silent degradation |
| 9 | Missing model.eval() at test time | Subtle inconsistency |
| 10 | Sigmoid on output layer | Silent degradation |
