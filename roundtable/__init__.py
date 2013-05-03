'''
RoundTable is a collection of table-like containers.  King Arthur's favorite!

*Table*
    Think of this as a list of namedtuples, except the elements of the namedtuples
    can be edited.  The table is initialized with a list of column header names.
    Appending to the table adds new rows.  Rows are created using a list (positional)
    or dict (column names).  Unspecified column cells default to `None`.
    
    Other list methods (`insert`, `pop`, `remove`, `extend`, `reverse`) work as expected.
    `sort` allows a string `key` to indicate a sort-by-column.  Search methods
    (`index`, `count`, `__contains__`) accept either a Row object or a list or dict.
    
    Additional methods useful for tables:
    * `take(indexes_or_func)` returns a new table based on passed in indexes
      Alternatively, applies a function to every row in the table and builds
      a new table out of rows that evaluate to True
    * `column(colname)` returns an iterator over values in a column
    
*LookupTable*
    A lookup function or column is added to a basic Table.  The lookup function is similar
    to a hash function, taking a Row object and returning a value.  If a column is provided,
    the value in that column is the "lookup" value.  A dict keeps track of these "lookup"
    values, allowing fast lookups on all seach methods (`index`, `count`, `__contains__`).
    These search methods accept either a Row object (which is run through the lookup function)
    or the "lookup" value.  The "lookup" values do not need to be unique.

Example
-------
    
    from datetime import datetime, date, timedelta
    from roundtable import Table
    # Create empty table with column headers
    tbl = Table(['Timestamp', 'Event', 'Root Cause', 'Due Date'])
    # Add rows to the table
    tbl.append((datetime(2013,1,2,12,30), 'Error code 129',
                'Short on board', date(2013,1,8)))
    tbl.append({'Event': 'Pairwise testing',
                'Due Date': date(2013,1,7)}) # other columns default to None
    # Build a sorted list of tasks due in the next week
    task_list = tbl.take(lambda row: row['Due Date'] - date.today() < timedelta(days=7))
    task_list.sort(col='Due Date', reverse=True)

    
Rows can be accessed in a table:
- By index :             Returns a Row object        mytable[0]; mytable[-1]
- By slice :             Returns a Table view        mytable[0:10:2]

Cells can be accessed:
- By index (from Row) :  Returns a value             mytable[0][0]; mytable[0][-1]
           (from Table)                              mytable[0, 0]; mytable[0, -1]
- By name (from Row) :   Returns a value             mytable[0]['Col2']
          (from Table)                               mytable[0, 'Col2']

As a special case, if a column name matches ^[A-Z][A-Za-z0-9_]*$
    (i.e. it is a valid Python variable name and starts with a capital letter)
    then an attribute will be added to the Row object allowing access.
    The requirement to start with a capital letter avoids conflicts with
    other attributes and functions of the Row object.
    
    Cell access (from Row) : Returns a value         mytable[0].Col2

Data can be accessed column-wise through the .column(index_or_name) method.
    This returns an iterator to access items in the specified column.

Table object can be transformed into:
- NumPy Array :      mytable.as_array()
- Pandas DataFrame:  mytable.as_dataframe()

Author: Jim Kitchen
Created: 2012-09-12
'''
__version__ = '0.4'

from ._table import Table
from ._lookuptable import LookupTable