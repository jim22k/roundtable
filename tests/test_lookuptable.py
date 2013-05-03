from nose.tools import raises
from roundtable import LookupTable
import collections, copy

class TestCreateLookupTable:
    def setUp(self):
        self.headers = ['id', 'x', 'y']
        self.func = lambda row: '%s:(%s,%s)' % tuple(row)

    def test_build_row_with_strings(self):
        t = LookupTable(self.headers, self.func)
        assert t.headers == ('id', 'x', 'y')
        row = t.Row(('a', 1, 2))
        assert row._lookup() == self.func(row)
    
    def test_build_row_with_int(self):
        t = LookupTable(5, 'id')
        assert len(t.headers) == 5
        assert isinstance(t.Row._lookup_func, collections.Callable)
    
    def test_build_func(self):
        t = LookupTable(self.headers, self.func)
        row = t.Row(('a', 1, 2))
        assert row._lookup() == 'a:(1,2)'
    
    def test_build_column(self):
        t = LookupTable(self.headers, 'id')
        row = t.Row(('a', 1, 2))
        assert row._lookup() == 'a'
    
    def test_build_columns(self):
        t = LookupTable(self.headers, ('id', 'y', 'x'))
        row = t.Row(('a', 1, 2))
        assert row._lookup() == ('a', 2, 1)

class TestTableEquivalency:
    '''Test __eq__ and copy'''
    def setUp(self):
        self.headers = ['id', 'x', 'y']
    
    def test_equal_same_function(self):
        func = lambda row: row['x'] + row['y']
        t = LookupTable(self.headers, func)
        t.append((1,2,3))
        t2 = LookupTable(self.headers, func)
        t2.append((1,2,3))
        assert t == t2
        assert t is not t2
    
    def test_equal_same_column(self):
        t = LookupTable(self.headers, 'id')
        t.append((1,2,3))
        t2 = LookupTable(self.headers, 'id')
        t2.append((1,2,3))
        assert t == t2
        assert t is not t2
    
    def test_equal_same_columns(self):
        t = LookupTable(self.headers, ('id', 'y', 'x'))
        t.append((1,2,3))
        t2 = LookupTable(self.headers, ('id', 'y', 'x'))
        t2.append((1,2,3))
        assert t == t2
        assert t is not t2
    
    def test_notequal_diff_headers(self):
        func = lambda row: row['x'] + row['y']
        t = LookupTable(self.headers, func)
        t.append((1,2,3))
        t2 = LookupTable(['notid', 'x', 'y'], func)
        t2.append((1,2,3))
        assert t != t2
    
    def test_notequal_diff_content(self):
        t = LookupTable(self.headers, 'id')
        t.append((1,2,3))
        t2 = LookupTable(self.headers, 'id')
        t2.append((1,2,3))
        t2.append((2,3,4))
        assert t != t2
    
    def test_copy_is_equal_func(self):
        func = lambda row: row['x'] + row['y']
        t = LookupTable(self.headers, func)
        t.append((1,2,3))
        t2 = copy.copy(t)
        assert t == t2
        assert t is not t2
        assert t[0] is t2[0]
    
    def test_copy_is_equal_column(self):
        t = LookupTable(self.headers, 'id')
        t.append((1,2,3))
        t2 = copy.copy(t)
        assert t == t2
        assert t is not t2
        assert t[0] is t2[0]
        t3 = LookupTable(self.headers, 'id')
        t3.append((1,2,3))
        assert t3 == t2
        assert t3[0] is not t2[0]
    
    def test_notequal(self):
        t = LookupTable(self.headers, 'id')
        t.append((1,2,3))
        assert t != (1,2,3)

class TestTableAddRemoveRows:
    '''Test append, extend, insert, pop, remove, del'''
    pass

class TestDuplicateKeyError:
    '''Test DuplicateKeyError is raised when appropriate and original isn't modified'''
    pass

class TestTableAccess:
    '''Test __getitem__'''
    pass

class TestTableUpdate:
    '''Test __setitem__'''
    pass

class TestTableDeletion:
    '''Test __delitem__'''
    pass

class TestTableSorting:
    '''Test sort, sort_by_col, reverse'''
    pass

class TestTableTake:
    '''Test filter by func and indices'''
    pass

class TestTableLookupMethods:
    '''Test lookup(), index()'''
    pass

class TestTableConversion:
    '''Test convert to DataFrame'''
    pass
