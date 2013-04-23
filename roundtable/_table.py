'''
_table.py

Author: Jim Kitchen
Created: 2012-09-12
'''
import re as _re
import itertools as _itertools
import functools as _functools
import operator as _operator

class RowFactory(object):
    cache = {}
    count = {}

    @staticmethod
    def build(headers):
        if headers not in RowFactory.cache:
            RowFactory.cache[headers] = RowFactory._build(headers)
            RowFactory.count[headers] = 0
        RowFactory.count[headers] += 1
        return RowFactory.cache[headers]
    
    @staticmethod
    def _build(headers_):
        @_functools.total_ordering
        class Row(object):
            __slots__ = ['_col%d' % i for i in range(len(headers_))]
            _mapper = {}
            for i, h in enumerate(headers_):
                _mapper[i] = __slots__[i] # reference by index
                _mapper[-i-1] = __slots__[-i-1] # reference by negative index
                _mapper[h] = __slots__[i] # reference by header string
            del i, h
            headers = headers_
            
            def __init__(self, dict_or_iterable):
                if isinstance(dict_or_iterable, dict):
                    # Make shallow copy of dict to avoid modifing passed in dict
                    d = dict_or_iterable.copy()
                    # Fill values from dict, fill missing headers with None
                    for h in self.headers:
                        self[h] = d.pop(h, None)
                    # If any keys remain in d, try to set them (will cause KeyError)
                    for key in d:
                        self[key] = d[key]
                else:
                    # Fill values, fill missing indexes with None
                    # Iterable longer than Row will cause IndexError
                    if not hasattr(dict_or_iterable, '__len__'):
                        dict_or_iterable = list(dict_or_iterable)
                    itemcount = max(len(self), len(dict_or_iterable))
                    for i, item in _itertools.izip_longest(xrange(itemcount), dict_or_iterable):
                        self[i] = item
            
            def __str__(self):
                return repr(self)
            
            def __repr__(self):
                vals = tuple(getattr(self, s) for s in self.__slots__)
                return 'Row(%s)' % str(vals)
            
            def __len__(self):
                return len(self.__slots__)
            
            def __getitem__(self, key):
                try:
                    return getattr(self, self._mapper[key])
                except KeyError:
                    if isinstance(key, int):
                        raise IndexError('Row index out of range')
                    else:
                        raise KeyError("Row has no column named '%s'" % str(key))
            
            def __setitem__(self, key, value):
                try:
                    return setattr(self, self._mapper[key], value)
                except KeyError:
                    if isinstance(key, int):
                        raise IndexError('Row index out of range')
                    else:
                        raise KeyError("Row has no column named '%s'" % str(key))
            
            def __delitem__(self, key):
                try:
                    setattr(self, self._mapper[key], None)
                except KeyError:
                    if isinstance(key, int):
                        raise IndexError('Row index out of range')
                    else:
                        raise KeyError("Row has no column named '%s'" % str(key))
            
            def __eq__(self, other):
                try:
                    return tuple(self) == tuple(other)
                except TypeError:
                    return NotImplemented
            
            def __lt__(self, other):
                try:
                    return tuple(self) < tuple(other)
                except TypeError:
                    return NotImplemented
        # Add upper-case property accessors
        for h in headers_:
            if _re.match(r'[A-Z][A-Za-z0-9_]*$', h):
                attr = Row._mapper[h]
                setattr(Row, h, property(lambda self, attr=attr: getattr(self, attr)))
        return Row
    
    @staticmethod
    def remove(headers):
        '''
        Decreases the reference count of the headers
        Once the reference count reaches zero, removes the Row
            associated with the headers from the cache
        '''
        RowFactory.count[headers] -= 1
        if RowFactory.count[headers] <= 0:
            del RowFactory.cache[headers]
            del RowFactory.count[headers]

class Table(object):
    '''
    Pure Python implementation of a table with string column names
    The table behaves very similar to a list of namedtuple, but allowing
        items within the namedtuple to be modified
    
    The Table is created with a list of column names
        These are fixed and cannot change after creating the Table
    Data is added and removed by row(s) using list-like syntax
        (append, extend, insert, pop, __getitem__, __delitem__).
        Values provided must be either an iterable or a dict of column names.
        Partial rows are allowed (missing cells will contain None).
    Table rows can be modified using reverse, sort, sort_by_col
    Views into the Table can be created using copy() or take().
        Modification of existing rows in a view will modify the
        original Table, but rearranging, adding, or removing rows
        will not affect the original Table.
        
    Rows can be accessed:
    - By index :            Returns a Row object     mytable[0]; mytable[-1]
    - By slice :            Returns a Table view     mytable[0:10:2]
    
    Cells can be accessed:
    - By index (from Row) :  Returns a value             mytable[0][0]; mytable[0][-1]
               (from Table)                              mytable[0, 0]; mytable[0, -1]
    - By name (from Row) :   Returns a value             mytable[0]['Col2']
              (from Table)                               mytable[0, 'Col2']
    
    As a special case, if a column name matches [A-Z][A-Za-z0-9_]*$
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
    '''
    def __init__(self, headers):
        '''
        headers : strings or int
            strings -> the column headers names
            int -> the desired number of column headers, filled with default names
        '''
        if isinstance(headers, int):
            headers = ['c%d' % i for i in range(headers)]
        self.headers = tuple(headers)
        assert tuple(map(str, self.headers)) == self.headers
        self._rows = []
        self.Row = RowFactory.build(self.headers)
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return 'Table<%d Rows>' % len(self._rows)
    
    def __len__(self):
        return len(self._rows)
    
    def __getitem__(self, index):
        if isinstance(index, int): # table[0] -> Row
            return self._rows[index]
        elif isinstance(index, tuple): # table[0,0]; table[0, 'Col2'] -> value
            if len(index) != 2:
                raise TypeError('Table index by tuple must be 2-tuple indicating (row, column)')
            return self._rows[index[0]][index[1]]
        elif isinstance(index, slice): # table[0:10:2] -> new Table
            r = Table(self.headers)
            r._rows = self._rows[index]
            return r
        else:
            raise TypeError('Table index must be int, slice, or 2-tuple')
    
    def __setitem__(self, index, rowvalue):
        if isinstance(index, int):
            if not isinstance(rowvalue, self.Row):
                rowvalue = self.Row(rowvalue)
            self._rows[index] = rowvalue
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise TypeError('Table index by tuple must be 2-tuple indicating (row, column)')
            self._rows[index[0]][index[1]] = rowvalue
        elif isinstance(index, slice):
            if not isinstance(rowvalue, Table):
                raise TypeError('setitem using slice must be passed a Table')
            if rowvalue.Row is not self.Row:
                raise TypeError('setitem using slice must have the same Row for both Tables')
            self._rows[index] = rowvalue._rows
        else:
            raise TypeError('Table index must be int, slice, or 2-tuple')
    
    def __delitem__(self, index):
        if isinstance(index, int): # table[0] -> remove Row
            del self._rows[index]
        elif isinstance(index, tuple): # table[0,0]; table[0, 'Col2'] -> cell value to None
            if len(index) != 2:
                raise TypeError('Table index by tuple must be 2-tuple indicating (row, column)')
            del self._rows[index[0]][index[1]]
        elif isinstance(index, slice): # table[0:10:2] -> remove Rows
            del self._rows[index]
        else:
            raise TypeError('Table index must be int, slice, or 2-tuple')
    
    def __eq__(self, other):
        try:
            return (self.headers == other.headers) and (self._rows == other._rows)
        except AttributeError:
            return NotImplemented
    
    def __del__(self):
        RowFactory.remove(self.headers)
    
    def __getstate__(self):
        # Eliminate Row class, convert to list of tuples
        d = self.__dict__.copy()
        del d['Row']
        d['_rows'] = map(tuple, self._rows)
        return d
    
    def __setstate__(self, d):
        # Recreate Row class
        self.__dict__ = d
        self.Row = RowFactory.build(self.headers)
        self._rows = map(self.Row, self._rows)
    
    def append(self, row):
        '''
        row can be: Row object, dict, or iterable
        '''
        if not isinstance(row, self.Row):
            row = self.Row(row)
        self._rows.append(row)
    
    def extend(self, rowlist):
        '''
        rowlist must be iterable of: Row object, dict, or iterable
        '''
        for row in rowlist:
            self.append(row)
    
    def insert(self, index, row):
        '''
        row can be: Row object, dict, or iterable
        '''
        if not isinstance(row, self.Row):
            row = self.Row(row)
        self._rows.insert(index, row)
    
    def copy(self):
        '''
        Returns a shallow copy of the table
        This can be used to get slices of the table via .filter()
            ex. mytable.copy().filter(filter_func)
        
        This method is faster than using copy.copy because it avoids
            pickling and unpickling the data
        For a deep copy, use copy.deepcopy
        '''
        r = Table(self.headers)
        r._rows = self._rows[:]
        return r
    
    def pop(self, index=-1):
        '''
        Remove index row from the table and return it
        '''
        return self._rows.pop(index)
    
    def reverse(self):
        self._rows.reverse()
        return self
    
    def sort(self, key=None, reverse=False):
        '''
        key is a function called on each Row in the Table
        '''
        self._rows.sort(key=key, reverse=reverse)
        return self
    
    def sort_by_col(self, col, reverse=False):
        '''
        Sort the rows by column or columns
        col must be a column header string, column index, or list of string/index
        
        This function is syntactic sugar for sort(key=operator.itemgetter(col))
        '''
        if isinstance(col, (int, basestring)):
            col = [col]
        self._rows.sort(key=_operator.itemgetter(*col), reverse=reverse)
        return self
    
    def take(self, func_or_indexes):
        '''
        Returns a new Table comprised of rows contained in indexes
        or where func(row) returns True
        '''
        r = Table(self.headers)
        if callable(func_or_indexes):
            r._rows = filter(func_or_indexes, self._rows)
        else:
            r._rows = [self._rows[i] for i in func_or_indexes]
        return r
    
    def column(self, index_or_name):
        '''
        Builds and returns a generator of items in the specified column
        '''
        # Access first row to raise error for invalid index_or_name
        if self:
            self[0][index_or_name]
        return (row[index_or_name] for row in self)
    
    def as_array(self, dtype=None):
        '''
        Return a NumPy array containing the data in the table
        The new array is a shallow copy of the data, not a view
            Changes will not propagate between the two objects
            unless an embedded mutable object is modified
        
        dtype : np.dtype, default None
            Data type to force, otherwise infer
        '''
        try:
            import numpy
        except ImportError:
            raise ImportError('as_array() requires numpy')
        return numpy.array(self._rows, dtype=dtype)
    
    def as_dataframe(self, index=None, dtype=None):
        '''
        Return a pandas DataFrame containing the headers and data in the table
        The Table column headers are passed to DataFrame for column headers
        The new array is a shallow copy of the data, not a view
            Changes will not propagate between the two objects
            unless an embedded mutable object is modified
        
        index: pandas Index or array-like
            defaults to np.arange(n)
        dtype : np.dtype, default None
            Data type to force, otherwise infer
        '''
        try:
            import pandas
        except ImportError:
            raise ImportError('as_dataframe() requires pandas')
        return pandas.DataFrame(self._rows, columns=self.headers,
                                index=index, dtype=dtype)
