#!/usr/bin/env python3
"""
Efficient Fibonacci number calculator using memoization (dynamic programming).
"""

import sys

def fibonacci_memo(n, memo={}):
    """
    Calculate the nth Fibonacci number using memoization.
    This approach has O(n) time complexity.
    """
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

def fibonacci_iterative(n):
    """
    Calculate the nth Fibonacci number using an iterative approach.
    This approach has O(n) time complexity and O(1) space complexity.
    """
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

def fibonacci_matrix(n):
    """
    Calculate the nth Fibonacci number using matrix exponentiation.
    This approach has O(log n) time complexity.
    """
    def matrix_multiply(A, B):
        a = A[0][0] * B[0][0] + A[0][1] * B[1][0]
        b = A[0][0] * B[0][1] + A[0][1] * B[1][1]
        c = A[1][0] * B[0][0] + A[1][1] * B[1][0]
        d = A[1][0] * B[0][1] + A[1][1] * B[1][1]
        return [[a, b], [c, d]]
    
    def matrix_power(A, n):
        if n == 1:
            return A
        if n % 2 == 0:
            return matrix_power(matrix_multiply(A, A), n // 2)
        else:
            return matrix_multiply(A, matrix_power(matrix_multiply(A, A), (n - 1) // 2))
    
    if n == 0:
        return 0
    
    return matrix_power([[1, 1], [1, 0]], n)[0][1]

if __name__ == "__main__":
    # Check if a command line argument is provided
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Error: '{sys.argv[1]}' is not a valid integer")
            sys.exit(1)
    else:
        # If no argument provided, prompt the user
        try:
            n = int(input("Enter a number to calculate its Fibonacci value: "))
        except ValueError:
            print("Error: Invalid input. Please enter an integer.")
            sys.exit(1)
    
    print(f"Fibonacci({n}) using memoization = {fibonacci_memo(n)}")
    print(f"Fibonacci({n}) using iteration = {fibonacci_iterative(n)}")
    print(f"Fibonacci({n}) using matrix exponentiation = {fibonacci_matrix(n)}")