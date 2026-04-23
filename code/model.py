"""
AI Final Project – GenAI Debugging System
Author: Vina Dang

This script is a simple placeholder model used for demonstrating
how small changes in code can introduce bugs and how GenAI can help
reason through debugging steps.
"""

def buggy_addition(a, b):
    """
    Intentional bug:
    This function is supposed to add two numbers,
    but it incorrectly subtracts them.
    """
    return a - b   # <-- intentional bug for the project


def analyze_output():
    """
    Runs the buggy function and prints results.
    Students use this output to reflect on debugging.
    """
    x = 10
    y = 5

    correct = x + y
    buggy = buggy_addition(x, y)

    print("Correct result:", correct)
    print("Buggy result:", buggy)
    print("\nReflection:")
    print("- Why does the buggy result differ?")
    print("- What part of the code caused the issue?")
    print("- How would you fix it?")
    print("- What debugging steps would you take?")


if __name__ == "__main__":
    analyze_output()

