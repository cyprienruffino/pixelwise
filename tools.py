# -*- coding: utf-8 -*-
import os
import sys


def create_dir(folder):
    '''
    creates a folder, if necessary
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)



if __name__ == "__main__":
    print("this is just a library.")
