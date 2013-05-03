'''
_lookuptable.py

Author: Jim Kitchen
Created: 2012-09-12
'''
import operator, collections, copy, itertools
from ._table import Table

try:
    basestring = basestring # py 2.x
    range = xrange
except NameError:
    basestring = str # py 3.x

try:
    zip_longest = itertools.zip_longest # py 3.x
except AttributeError:
    zip_longest = itertools.izip_longest # py 2.x

def row_upgrade(RowClass, lookup, callback):
    '''
    RowClass is a Row class from a regular Table
    lookup is a function called on self
    callback is called whenever the values in self change
    
    Callback arguments are (self, prev_lookup_value, new_lookup_value)
    This allows the LookupTable to update its index properly
    '''
    class Row(RowClass):
        _lookup_func = lookup
        _callback = callback
        
        def __init__(self, dict_or_iterable):
            if isinstance(dict_or_iterable, dict):
                # Make shallow copy of dict to avoid modifing passed in dict
                d = dict_or_iterable.copy()
                # Fill values from dict, fill missing headers with None
                for h in self.headers:
                    super(Row, self).__setitem__(h, d.pop(h, None))
                # If any keys remain in d, try to set them (will cause KeyError)
                for key in d:
                    super(Row, self).__setitem__(key, d[key])
            else:
                # Fill values, fill missing indexes with None
                # Iterable longer than Row will cause IndexError
                if not hasattr(dict_or_iterable, '__len__'):
                    dict_or_iterable = list(dict_or_iterable)
                itemcount = max(len(self), len(dict_or_iterable))
                for i, item in zip_longest(range(itemcount), dict_or_iterable):
                    super(Row, self).__setitem__(i, item)
        
        def __setitem__(self, key, value):
            print('LookupRow setitem')
            prev_lookup_value = self._lookup()
            super(Row, self).__setitem__(key, value)
            Row._callback(self, prev_lookup_value)
        
        def __delitem__(self, key):
            prev_lookup_value = self._lookup()
            super(Row, self).__delitem__(key)
            Row._callback(self, prev_lookup_value)
        
        def __setattr__(self, name, value):
            prev_lookup_value = self._lookup()
            super(Row, self).__setattr__(name, value)
            Row._callback(self, prev_lookup_value)
        
        def __delattr__(self, name):
            prev_lookup_value = self._lookup()
            super(Row, self).__delattr__(name)
            Row._callback(self, prev_lookup_value)
        
        def update(self, d):
            prev_lookup_value = self._lookup()
            super(Row, self).update(d)
            Row._callback(self, prev_lookup_value)
        
        def _lookup(self):
            return Row._lookup_func(self)
    return Row

class LookupTable(Table):
    '''
    Adds a lookup mechanism to the table giving O(1) row lookups
    
    All seach methods (`index`, `count`, `__contains__`) accept either
    a Row object (which is run through the lookup function)
    or the "lookup" value.  The "lookup" values do not need to be unique.
    '''
    
    def __init__(self, headers, lookup):
        '''
        headers : strings or int
            strings -> the column headers names
            int -> the desired number of column headers, filled it with default names
        lookup : function or column or columns
            function -> takes a Row object and returns a unique key
            column -> uses the value in column as the unique key
            columns -> creates a tuple from values in columns as the unique key
        '''
        super(LookupTable, self).__init__(headers)
        if not isinstance(lookup, collections.Callable):
            if isinstance(lookup, basestring):
                lookup = [lookup]
            self._lookup_comp = tuple(lookup)
            lookup = operator.itemgetter(*lookup)
        self.Row = row_upgrade(self.Row, lookup, self._row_update)
        self._index = {} # {lookup value: [indexes]}
    
    def __repr__(self):
        return 'LookupTable<%d Rows>' % len(self._rows)
    
    def __getitem__(self, index):
        if isinstance(index, int): # table[0] -> Row
            return self._rows[index]
        elif isinstance(index, tuple): # table[0,0]; table[0, 'Col2'] -> value
            if len(index) != 2:
                raise TypeError('LookupTable index by tuple must be 2-tuple indicating (row, column)')
            return self._rows[index[0]][index[1]]
        elif isinstance(index, slice): # table[0:10:2] -> new LookupTable
            lookup = getattr(self, '_lookup_comp', self.Row._lookup)
            r = LookupTable(self.headers, lookup)
            r._rows = self._rows[index]
            r._update_indexes()
            return r
        else:
            raise TypeError('LookupTable index must be int, slice, or 2-tuple')
    
    def __setitem__(self, index, rowvalue):
        if isinstance(index, int):
            if not isinstance(rowvalue, self.Row):
                rowvalue = self.Row(rowvalue)
            # Remove old lval
            oldrow = self._rows[index]
            old_lval = oldrow._lookup()
            self._index[old_lval].remove(index)
            if not self._index[old_lval]:
                del self._index[old_lval]
            # Add new lval
            newrow = rowvalue
            new_lval = newrow._lookup()
            if new_lval not in self._index:
                self._index[new_lval] = [index]
            else:
                self._index[new_lval].append(index)
                self._index[new_lval].sort()
            self._rows[index] = newrow
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise TypeError('LookupTable index by tuple must be 2-tuple indicating (row, column)')
            irow, icol = index
            self._rows[irow][icol] = rowvalue
        elif isinstance(index, slice):
            if not isinstance(rowvalue, LookupTable):
                raise TypeError('setitem using slice must be passed a LookupTable')
            if rowvalue.Row.headers != self.Row.headers:
                raise TypeError('setitem using slice must have the same Row for both Tables')
            # Update rows and indexes
            self._rows[index] = rowvalue._rows
            self._update_indexes()
        else:
            raise TypeError('LookupTable index must be int, slice, or 2-tuple')
    
    def __delitem__(self, index):
        if isinstance(index, int): # table[0] -> remove Row
            self.pop(index)
        elif isinstance(index, tuple): # table[0,0]; table[0, 'Col2'] -> cell value to None
            if len(index) != 2:
                raise TypeError('LookupTable index by tuple must be 2-tuple indicating (row, column)')
            irow, icol = index
            del self._rows[irow][icol]
        elif isinstance(index, slice): # table[0:10:2] -> remove Rows
            try:
                for i in range(len(self))[index]:
                    self.pop(i)
            except TypeError: # py 3.1 can't use slice on range object
                for i in list(range(len(self)))[index]:
                    self.pop(i)
        else:
            raise TypeError('LookupTable index must be int, slice, or 2-tuple')
    
    def __eq__(self, other):
        try:
            return ((self.headers == other.headers) and (self._rows == other._rows)
                    and ((self.Row._lookup_func == other.Row._lookup_func) or (self._lookup_comp == other._lookup_comp)))
        except AttributeError:
            return NotImplemented
    
    def _update_indexes(self):
        '''
        Update the lookup indexes using lookup on each Row
        '''
        # Create local variables to speed up loop
        rows = self._rows
        index = self._index
        index.clear()
        for irow, row in enumerate(self):
            lval = row._lookup()
            if lval not in index:
                index[lval] = [irow]
            else:
                index[lval].append(irow)
    
    def _increment_indexes(self, start_row, inc):
        '''
        Update the lookup indexes for start_row and beyond, incrementing by inc
        This should be faster than generating a new index using lookup
        '''
        index = self._index
        try:
            itervalues = index.itervalues() # py 2.x
        except AttributeError:
            itervalues = index.values() # py 3.x
        for irows in itervalues:
            for i, irow in enumerate(irows):
                if irow >= start_row:
                    irows[i] += inc
    
    def _row_update(self, row, prev_lval):
        # Does index need to be updated?
        new_lval = row._lookup()
        if new_lval == prev_lval:
            return
        # Find the matching index(es) using previous lookup value
        # Can be multiple if user added same row object multiple times
        indexes = self._index[prev_lval]
        matching = [ind for ind in indexes if self[ind] is row]
        for match in matching:
            # Remove previous index
            indexes.remove(match)
            if not indexes:
                del self._index[prev_lval]
            # Add new index
            if new_lval not in self._index:
                self._index[new_lval] = [ind]
            else:
                self._index[new_lval].append(ind)
                self._index[new_lval].sort()
    
    def __copy__(self):
        '''Returns a shallow copy of the table'''
        lookup = getattr(self, '_lookup_comp', self.Row._lookup_func)
        r = LookupTable(self.headers, lookup)
        r._rows = self._rows[:]
        # Deepcopy the index to avoid crosstalk if either is sorted or updated
        r._index = copy.deepcopy(self._index)
        return r
    
    def append(self, row):
        '''
        row can be: Row object, dict, or iterable
        '''
        if not isinstance(row, self.Row):
            row = self.Row(row)
        lval = row._lookup()
        if lval not in self._index:
            self._index[lval] = [len(self)]
        else:
            self._index[lval].append(len(self))
        self._rows.append(row)
    
    def insert(self, index, row):
        '''
        row can be: Row object, dict, or iterable
        '''
        if not isinstance(row, self.Row):
            row = self.Row(row)
        lval = row._lookup()
        self._rows.insert(index, row)
        # Increment all rows first, then add new index
        self._increment_indexes(index)
        if lval not in self._index:
            self._index[lval] = [index]
        else:
            self._index[lval].append(index)
            self._index[lval].sort()
    
    def pop(self, index=-1):
        '''
        Remove index row from the table and return it
        '''
        row = self._rows.pop(index)
        lval = row._lookup
        # Remove index and entire entry if no matching lvals
        self._index[lval].remove(index)
        if not self._index[lval]:
            del self._index[lval]
        # Decrement all rows
        self._increment_indexes(index, -1)
        return row
    
    def reverse(self):
        self._rows.reverse()
        # Reverse index by subtracting each index from total length
        index = self._index
        total_len = len(self) - 1
        try:
            itervalues = index.itervalues() # py 2.x
        except AttributeError:
            itervalues = index.values() # py 3.x
        for irows in itervalues:
            for i, irow in enumerate(irows):
                irows[i] = total_len - irow
        return self
    
    def sort(self, key=None, reverse=False):
        '''
        key:
            function called on each Row in the Table
            string or int for sort-by-column
            list of strings or ints for sort-by-multi-column
        
        Returns the Table object to allow operation stringing
            This is needed because you cannot override the default
            behavior of the builtin sorted() function.
        
        Example:
            # Return a table of the top 5 values in column B
            top5 = mytable.sort('B', reverse=True)[:5]
        '''
        super(LookupTable, self).sort(key, reverse)
        self._update_indexes()
        return self
    
    def index(self, row, start=None, stop=None):
        '''
        Returns the index of value in the LookupTable
        
        row can be either a Row object or the lookup value
            of the row
        '''
        if isinstance(row, self.Row):
            row = row._lookup()
        irows = self._index.get(row, [])
        for irow in irows:
            if stop is not None and stop <= irow:
                break
            if start is not None and start < irow:
                continue
            return irow
        raise ValueError('%s is not in the LookupTable' % row)
    
    def count(self, row):
        '''
        Returns the number of times row exists in the LookupTable
        
        row can be either a Row object or the result of
            passing the Row object to the lookup function
        '''
        if isinstance(row, self.Row):
            row = row._lookup()
        return len(self._index.get(row, ()))
    
    def remove(self, row):
        '''
        Removes first occurrence of row in the Table.
        Raises ValueError if the row is not present.
        
        row can be either a Row object or the result of
            passing the Row object to the lookup function
        '''
        self.pop(self.index(row))
    
    def take(self, func_or_indexes):
        '''
        Returns a new LookupTable comprised of rows contained in indexes
        or where func(row) returns True
        '''
        lookup = getattr(self, '_lookup_comp', self.Row._lookup_func)
        r = LookupTable(self.headers, lookup)
        if isinstance(func_or_indexes, _collections.Callable):
            r._rows = list(filter(func_or_indexes, self._rows))
        else:
            r._rows = [self._rows[i] for i in func_or_indexes]
        r._update_indexes()
        return r
    
    def as_dataframe(self, index=None, dtype=None):
        '''
        Return a pandas DataFrame containing the headers and data in the table
        The LookupTable column headers are passed to DataFrame for column headers
        The new array is a shallow copy of the data, not a view
            Changes will not propagate between the two objects
            unless an embedded mutable object is modified
        
        index: pandas Index or array-like
            defaults to lookup keys
        dtype : np.dtype, default None
            Data type to force, otherwise infer
        '''
        try:
            import pandas
        except ImportError:
            raise ImportError('as_dataframe() requires pandas')
        if index is None:
            # Rebuild index as list
            index = [row._lookup() for row in self._rows]
        return pandas.DataFrame([tuple(row) for row in self._rows],
                                columns=self.headers,
                                index=index, dtype=dtype)
    