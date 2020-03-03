# Get access to parent directory
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

# Import own modules
from attention.modules import SoftAttention, HardAttention
from decoder import *
from encoder import *

# Further imports
# ...

def main():
    pass


if __name__ == '__main__':
    main()
