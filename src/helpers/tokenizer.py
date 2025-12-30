#!/bin/bash

import sys

class Tokenizer:
    @staticmethod
    def data_imputation():
        pass

    @staticmethod
    def tokenize():
        pass

def usage():
    print("Usage:")
    print("  ./tokenizer [FILE]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        exit(1)

    file = sys.argv[1]
