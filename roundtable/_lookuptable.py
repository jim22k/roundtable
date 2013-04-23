'''
_lookuptable.py

Author: Jim Kitchen
Created: 2012-09-12
'''
import operator as _operator
from ._table import Table

class DuplicateKeyError(Exception):
    '''
    Raised when attempting to add a new row to a LookupTable with a key
    matching an existing row in the table
    
    Every effort is made to ensure that this error is raised before
    a change is made to the LookupTable
    '''
    pass

class LookupTable(Table):
    def __init__(self, headers, lookup):
        '''
        Adds a lookup mechanism to the table giving O(1) row lookups
        Each row in the table must have a unique key associated with it
        Conflicts will raise a DuplicateKeyError
        
        headers : strings or int
            strings -> the column headers names
            int -> the desired number of column headers, filled it with default names
        lookup : function or column or columns
            function -> takes a Row object and returns a unique key
            column -> uses the value in column as the unique key
            columns -> creates a tuple from values in columns as the unique key
        
        LookupTable additional methods
            .lookup(key) -> returns the Row associated with key
            .index(key) -> returns the row index associated with key
            .remove(key) -> removes the row associated with key
        '''
        super(LookupTable, self).__init__(headers)
        if not callable(lookup):
            if isinstance(lookup, basestring):
                lookup = [lookup]
            self._lookup_comp = tuple(lookup)
            lookup = _operator.itemgetter(*lookup)
        self.lookup = lookup
        self._index = {}
    
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
            lookup = getattr(self, '_lookup_comp', self.lookup)
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
            # Check for duplicate keys
            oldrow = self._rows[index]
            oldkey = self.lookup(oldrow)
            newrow = rowvalue
            newkey = self.lookup(newrow)
            # Update index keys if newrow has a different key than oldkey
            if newkey != oldkey:
                self._check_key(newkey)
                del self._index[oldkey]
                self._index[newkey] = index
            self._rows[index] = newrow
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise TypeError('LookupTable index by tuple must be 2-tuple indicating (row, column)')
            irow, icol = index
            # Check for duplicate keys caused by updating item
            oldrow = self._rows[irow]
            oldkey = self.lookup(oldrow)
            newrow = self.Row(oldrow) # create a new row to avoid modifying existing row before the check
            newrow[icol] = rowvalue
            newkey = self.lookup(newrow)
            # Update index keys if updating cell changes key
            if newkey != oldkey:
                self._check_key(newkey)
                del self._index[oldkey]
                self._index[newkey] = irow
            self._rows[irow][icol] = rowvalue
        elif isinstance(index, slice):
            if not isinstance(rowvalue, Table):
                raise TypeError('setitem using slice must be passed a Table')
            if rowvalue.Row is not self.Row:
                raise TypeError('setitem using slice must have the same Row for both Tables')
            # Check for duplicate keys caused by updating items
            # Needs to comprehend cyclical reordering of rows
            oldrows = set(self._rows[index])
            oldkeys = set(map(self.lookup, oldrows))
            newrows = rowvalue
            newkeys = set(map(self.lookup, newrows))
            # Validate new index keys
            if len(newkeys) < len(newrows):
                raise DuplicateKeyError('Duplicate keys exist in newly added rows')
            for newkey in newkeys:
                if newkey in oldkeys:
                    # oldkeys will be removed, so not a real conflict
                    continue
                self._check_key(newkey)
            # Update rows and indexes
            self._rows[index] = newrows._rows
            for oldkey in oldkeys:
                del self._index[oldkey]
            self._update_indexes()
        else:
            raise TypeError('LookupTable index must be int, slice, or 2-tuple')
    
    def __delitem__(self, index):
        if isinstance(index, int): # table[0] -> remove Row
            del self._index[self.lookup(self._rows[index])]
            del self._rows[index]
            self._increment_indexes(index, -1)
        elif isinstance(index, tuple): # table[0,0]; table[0, 'Col2'] -> cell value to None
            if len(index) != 2:
                raise TypeError('LookupTable index by tuple must be 2-tuple indicating (row, column)')
            irow, icol = index
            # Check for duplicate keys caused by removing item
            oldrow = self._rows[irow]
            oldkey = self.lookup(oldrow)
            newrow = self.Row(oldrow)
            del newrow[icol]
            newkey = self.lookup(newrow)
            # Update index keys if removing item changes key
            if newkey != oldkey:
                self._check_key(newkey)
                del self._index[oldkey]
                self._index[newkey] = irow
            del self._rows[irow][icol]
        elif isinstance(index, slice): # table[0:10:2] -> remove Rows
            rows = self._rows[index]
            for row in rows:
                del self._index[self.lookup(row)]
            del self._rows[index]
            self._update_indexes()
        else:
            raise TypeError('LookupTable index must be int, slice, or 2-tuple')
    
    def __eq__(self, other):
        try:
            return ((self.headers == other.headers) and (self._rows == other._rows)
                    and ((self.lookup == other.lookup) or (self._lookup_comp == other._lookup_comp)))
        except AttributeError:
            return NotImplemented
    
    def _update_indexes(self):
        '''
        Update the lookup indexes using lookup on each Row
        '''
        # Create local variables to speed up loop
        rows = self._rows
        index = self._index
        lookup = self.lookup
        for irow in xrange(len(self)):
            index[look_func(rows[irow])] = irow
    
    def _increment_indexes(self, start_row, inc):
        '''
        Update the lookup indexes for start_row and beyond, incrementing by inc
        This should be faster than generating a new index using lookup
        '''
        index = self._index
        for key, irow in index.iteritems():
            if irow >= start_row:
                index[key] += inc
    
    def _check_key(self, key):
        if key in self._index:
            raise DuplicateKeyError('%s is already a key in the LookupTable' % key)
    
    def append(self, row):
        '''
        row can be: Row object, dict, or iterable
        '''
        if not isinstance(row, self.Row):
            row = self.Row(row)
        key = self.lookup(row)
        self._check_key(key)
        self._index[key] = len(self)
        self._rows.append(row)
    
    def extend(self, rowlist):
        '''
        rowlist must be iterable of: Row object, dict, or iterable
        '''
        # Save current length in case of needing to abort for duplicate key
        cur_len = len(self)
        try:
            for row in rowlist:
                self.append(row)
        except DuplicateKeyError:
            del self[cur_len:]
            raise
    
    def insert(self, index, row):
        '''
        row can be: Row object, dict, or iterable
        '''
        if not isinstance(row, self.Row):
            row = self.Row(row)
        key = self.lookup(row)
        self._check_key(key)
        self._rows.insert(index, row)
        # Increment all rows first, then add new index
        self._increment_indexes(index)
        self._index[key] = index
    
    def copy(self):
        '''
        Returns a shallow copy of the table
        This can be used to get slices of the table via .filter()
            ex. mytable.copy().filter(filter_func)
        
        This method is faster than using copy.copy because it avoids
            pickling and unpickling the data
        For a deep copy, use copy.deepcopy
        '''
        lookup = getattr(self, '_lookup_comp', self.lookup)
        r = LookupTable(self.headers, lookup)
        r._rows = self._rows[:]
        r._index = self._index.copy()
        return r
    
    def pop(self, index=-1):
        '''
        Remove index row from the table and return it
        '''
        p = self._rows.pop(index)
        del self._index[self.lookup(p)]
        self._increment_indexes(index, -1)
        return p
    
    def reverse(self):
        self._rows.reverse()
        # Reverse index by subtracting each index from total length
        index = self._index
        total_len = len(self) - 1
        for key, i in self._index.iteritems():
            index[key] = total_len - i
        return self
    
    def sort(self, key=None, reverse=False):
        '''
        key is a function called on each Row in the LookupTable
        '''
        super(LookupTable, self).sort(key, reverse)
        self._update_indexes()
        return self
    
    def sort_by_col(self, col, reverse=False):
        '''
        Sort the rows by column or columns
        col must be a column header string, column index, or list of string/index
        
        This function is syntactic sugar for sort(key=operator.itemgetter(col))
        '''
        super(LookupTable, self).sort(col, reverse)
        self._update_indexes()
        return self
    
    def take(self, func_or_indexes):
        '''
        Returns a new LookupTable comprised of rows contained in indexes
        or where func(row) returns True
        '''
        lookup = getattr(self, '_lookup_comp', self.lookup)
        r = LookupTable(self.headers, lookup)
        if callable(func_or_indexes):
            r._rows = filter(func_or_indexes, self._rows)
        else:
            r._rows = [self._rows[i] for i in func_or_indexes]
        r._update_indexes()
        return r
    
    def lookup(self, key):
        '''
        Returns the row matching key
        '''
        return self._row[self._index[key]]
    
    def index(self, key):
        '''
        Returns the index of the row matching key
        '''
        return self._index[key]
    
    def remove(self, key):
        self.pop(self._index[key])
    
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
            lookup = self.lookup
            index = [lookup(row) for row in self._rows]
        return pandas.DataFrame(self._rows, columns=self.headers,
                                index=index, dtype=dtype)