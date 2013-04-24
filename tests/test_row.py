from nose.tools import raises
from roundtable._table import RowFactory, Table

def test_RowFactory():
    '''
    Build table and grab created Row
    Row should be in the cache once
    Create a duplicate table, should be two rows in the cache
    Delete duplicate table, should be back to one in the cache
    Delete table, should be removed from cache
    '''
    rand_headers = ('a;lkjsalkeh', 'qu coijslkjheoi', 'hwlnelklz8963io#*#720')
    tbl = Table(rand_headers)
    Row = tbl.Row
    assert isinstance(Row, type)
    assert RowFactory.cache[rand_headers] is Row
    assert RowFactory.count[rand_headers] == 1
    tbl2 = Table(rand_headers)
    assert RowFactory.count[rand_headers] == 2
    del tbl2
    assert RowFactory.count[rand_headers] == 1
    del tbl
    assert rand_headers not in RowFactory.cache
    assert rand_headers not in RowFactory.count

class TestRowCreation:
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.table = Table(self.headers)
        self.Row = self.table.Row
    
    def test_row_headers_become_tuples(self):
        assert self.Row.headers == tuple(self.headers)

    def test_row_instance_headers(self):
        r = self.Row((1,2,3))
        assert r.headers is self.Row.headers

    def test_new_row_from_iterable(self):
        r = self.Row((1,2,3)) # tuple
        r = self.Row([1,2,3]) # list
        r = self.Row(xrange(3)) # xrange
        def gene8r():
            for i in range(3):
                yield i
        r = self.Row(gene8r()) # generator object
        r = self.Row('abc') # string

    def test_new_row_from_dict(self):
        r = self.Row({'x':1, 'y':2, 'z':3})

    def test_new_row_using_keyword(self):
        r = self.Row(dict_or_iterable=(1,2,3))
        r = self.Row(dict_or_iterable={'x':1, 'y':2, 'z':3})

    @raises(TypeError)
    def test_new_row_args_instead_of_iterable(self):
        '''Test against common error of Row(1,2,3) instead of Row((1,2,3))'''
        r = self.Row(1,2,3)

    @raises(TypeError)
    def test_new_row_single_item_only(self):
        '''
        Special case of above for single item only
        User might expect x=1, y=None, z=None
        '''
        r = self.Row(1)
    
    @raises(TypeError)
    def test_new_row_no_col_names(self):
        '''Test against common error of Row(x=1, y=2, z=3)'''
        r = self.Row(x=1, y=2, z=3)
    
    @raises(IndexError)
    def test_new_row_too_many_items(self):
        r = self.Row((1,2,3,4,5))

    @raises(KeyError)
    def test_new_row_extra_dict_items(self):
        r = self.Row({'x':1, 'y':2, 'z':3, 'extra':17})

    def test_new_row_from_iterable_fill(self):
        r = self.Row((1,2))
        assert r[0] == 1
        assert r[1] == 2
        assert r[2] is None

    def test_new_row_from_dict_fill(self):
        r = self.Row({'y':2})
        assert r['x'] is None
        assert r['y'] == 2
        assert r['z'] is None
    
    def test_row_has_no_capitalized_attrs(self):
        '''
        Row attrs that start with a capital letter are reserved for column names
        which are also valid Python variable names.
        Ensure that all Row attrs start with lower case or underscore.
        '''
        for key in self.Row.__dict__:
            assert not key[0].isupper()
    
    def test_creates_attrs_for_capitalized_column_names(self):
        headers = ['123', 'Abc', 'Abc?', '_def', 'aBc_def', 'Abc_', '_mapper']
        t = Table(headers)
        t.append((1,2,3,4,5,6,7))
        r = t[0]
        assert r['123'] == 1
        assert r['Abc'] == 2
        assert r['Abc?'] == 3
        assert r['_def'] == 4
        assert r['aBc_def'] == 5
        assert r['Abc_'] == 6
        assert r['_mapper'] == 7
        assert not hasattr(r, '123') # not valid Python variable name
        assert r.Abc == 2
        assert not hasattr(r, 'Abc?') # not valid Python variable name
        assert not hasattr(r, '_def') # starts with underscore
        assert not hasattr(r, 'aBc_def') # valid, but not capitalized
        assert r.Abc_ == 6
        # _mapper is both a Row attribute and a Row column
        # ensure no conflict exists
        assert hasattr(r, '_mapper')
        assert r._mapper is not r['_mapper']

class TestRowAccess:
    '''Test __getitem__'''
    def setUp(self):
        self.headers = ['x', 'y', 'z', 'CapCol']
        self.table = Table(self.headers)
        self.table.append((1,2,3.5,14))
        self.row = self.table[0]
    
    def test_access_by_index(self):
        assert self.row[0] == 1
        assert self.row[1] == 2
        assert self.row[2] == 3.5
        assert self.row[3] == 14
    
    def test_access_by_reverse_index(self):
        assert self.row[-1] == 14
        assert self.row[-2] == 3.5
        assert self.row[-3] == 2
        assert self.row[-4] == 1
    
    def test_access_by_column_name(self):
        assert self.row['x'] == 1
        assert self.row['y'] == 2
        assert self.row['z'] == 3.5
        assert self.row['CapCol'] == 14
    
    def test_access_by_slice(self):
        assert self.row[0:2] == (1,2)
    
    @raises(IndexError)
    def test_access_index_error(self):
        self.row[7]
    
    @raises(IndexError)
    def test_access_index_error_reverse(self):
        self.row[-7]
    
    @raises(KeyError)
    def test_access_key_error(self):
        self.row['unknown']
    
    @raises(AttributeError)
    def test_access_by_attribute(self):
        '''Common error: DataFrame lets you do this, but simpletable doesn't'''
        self.row.z
    
    def test_access_by_capital_attribute(self):
        '''This is allowed, if names is valid and starts with a capital letter'''
        assert self.row.CapCol == 14
    
class TestRowUpdate:
    '''Test __setitem__ and __delitem__'''
    def setUp(self):
        self.headers = ['x', 'y', 'z', 'CapCol']
        self.table = Table(self.headers)
        self.table.append((1,2,3.5,14))
        self.row = self.table[0]
    
    def test_set_value_by_index(self):
        self.row[1] = 42
        assert tuple(self.row) == (1, 42, 3.5, 14)
    
    def test_set_value_by_reverse_index(self):
        self.row[-1] = 15
        assert tuple(self.row) == (1, 2, 3.5, 15)
    
    def test_set_value_by_column_name(self):
        self.row['x'] = -500
        assert tuple(self.row) == (-500, 2, 3.5, 14)
    
    def test_set_value_by_capcol(self):
        self.row.CapCol = -500
        assert tuple(self.row) == (1, 2, 3.5, -500)
    
    def test_del_value_by_index(self):
        del self.row[0]
        assert tuple(self.row) == (None, 2, 3.5, 14)
    
    def test_del_value_by_reverse_index(self):
        del self.row[-1]
        assert tuple(self.row) == (1, 2, 3.5, None)
    
    def test_del_value_by_column_name(self):
        del self.row['y']
        assert tuple(self.row) == (1, None, 3.5, 14)
    
    def test_del_values_by_slice(self):
        del self.row[0:2]
        print self.row
        assert tuple(self.row) == (None, None, 3.5, 14)
    
    def test_del_value_by_capcol(self):
        del self.row.CapCol
        assert tuple(self.row) == (1, 2, 3.5, None)
    
    @raises(IndexError)
    def test_set_value_bad_index(self):
        self.row[5] = 25
    
    @raises(IndexError)
    def test_set_value_bad_reverse_index(self):
        self.row[-5] = 10
    
    @raises(KeyError)
    def test_set_value_bad_column_name(self):
        self.row['unknown'] = 'N/A'
    
    @raises(IndexError)
    def test_del_value_bad_index(self):
        del self.row[5]
    
    @raises(IndexError)
    def test_del_value_bad_reverse_index(self):
        del self.row[-5]
    
    @raises(KeyError)
    def test_del_value_bad_column_name(self):
        del self.row['unknown']

class TestRowComparison:
    '''Test __lt__'''
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.table = Table(self.headers)
        self.table.append((1,1,1))
        self.row1 = self.table[0]
    
    def test_compare_two_rows(self):
        self.table.append((1,1,2))
        assert self.row1 < self.table[-1]
        assert self.row1 <= self.table[-1]
        assert self.table[-1] > self.row1
        assert self.row1 != self.table[-1]
        self.table.append((1,1,0))
        assert self.row1 > self.table[-1]
        assert self.row1 >= self.table[-1]
        assert self.table[-1] < self.row1
        self.table.append((1,1,1))
        assert self.row1 == self.table[-1]
        assert self.row1 <= self.table[-1]
        assert self.row1 >= self.table[-1]
    
    def test_compare_with_tuple(self):
        assert self.row1 < (1,1,2)
        assert (1,1,2) > self.row1
        assert self.row1 > (1,1,0)
        assert (1,1,0) < self.row1
        assert self.row1 == (1,1,1)
        assert self.row1 != (1,1,2)
        assert self.row1 <= (1,1,1)
        assert self.row1 <= (5,0,0)
        assert self.row1 >= (1,1,1)
        assert self.row1 >= (-5,0,0)
    
    def test_compare_with_nontuple(self):
        assert self.row1 == [1,1,1]
        assert self.row1 != [5]
        assert self.row1 < [1,1,2]
        