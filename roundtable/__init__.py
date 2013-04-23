'''
roundtable.py

Pure Python implementations of a table
Includes features such as:
- access by column name or column index
- sort by multiple columns
- select subset of rows as a new view of the data

Author: Jim Kitchen
Created: 2012-09-12
'''
__version__ = 0.2

from ._table import Table
from ._lookuptable import LookupTable, DuplicateKeyError