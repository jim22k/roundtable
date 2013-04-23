from nose.tools import raises
from roundtable import Table

def test_build_row_with_strings():
    t = Table(['x', 'y', 'z'])
    assert t.headers == ('x', 'y', 'z')
    t = Table('a b c d e'.split())
    assert t.headers == ('a', 'b', 'c', 'd', 'e')

def test_build_row_with_int():
    t = Table(5)
    assert len(t.headers) == 5

class TestTableEquivalency:
    '''Test __eq__ and copy'''
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.t = Table(self.headers)
        self.t.append((1,2,3))
        self.t.append((4,5,6))
        self.t.append((7,8,9))

    def test_table_equal(self):
        t2 = Table(self.headers)
        t2.extend([[1,2,3],[4,5,6],[7,8,9]])
        assert self.t == t2

    def test_table_notequal_same_headers(self):
        t2 = Table(self.headers)
        t2.extend([[1,2,3],[4,5,6]])
        assert self.t != t2

    def test_table_notequal_diff_headers(self):
        t2 = Table(['p', 'd', 'q'])
        t2.extend([[1,2,3],[4,5,6],[7,8,9]])
        assert self.t != t2

    def test_table_notequal_nontable(self):
        t2 = ((1,2,3),(4,5,6),(7,8,9))
        assert self.t != t2

    def test_table_copy(self):
        q = self.t.copy()
        assert self.t is not q
        assert self.t == q
        assert self.t[0] is q[0]
        assert self.t[1] is q[1]
    
    def test_table_copy_copy(self):
        """copy.copy pickles the Table, so the Row objects are '==', but not 'is'"""
        import copy
        q = copy.copy(self.t)
        assert self.t is not q
        assert self.t == q
        assert self.t[0] is not q[0]
        assert self.t[0] == q[0]
        assert self.t[1] is not q[1]
        assert self.t[1] == q[1]

class TestTableAddRemoveRows:
    '''Test append, extend, insert, pop, del'''
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.t = Table(self.headers)
    
    def test_append(self):
        r = self.t.Row((1,2,3))
        self.t.append(r)
        self.t.append((4,5,6))
        self.t.append({'x':7, 'y':8, 'z':9})
        assert len(self.t) == 3
        assert self.t[0] is r
        assert self.t[1] == (4,5,6)
        assert self.t[2] == (7,8,9)
    
    def test_extend(self):
        self.t.append((0,0,0))
        rows = [
            [1,2,3],
            {'x':4, 'y':5, 'z':6},
            [7,8]
        ]
        self.t.extend(rows)
        assert len(self.t) == 4
        assert self.t[0] == (0,0,0)
        assert self.t[1] == (1,2,3)
        assert self.t[2] == (4,5,6)
        assert self.t[3] == (7,8,None)
    
    def test_insert(self):
        self.t.append((1,2,3))
        self.t.append((4,5,6))
        self.t.append((7,8,9))
        self.t.insert(1, (-1,-1,-1))
        assert len(self.t) == 4
        assert self.t[0] == (1,2,3)
        assert self.t[1] == (-1,-1,-1)
        assert self.t[2] == (4,5,6)
        assert self.t[3] == (7,8,9)
    
    def test_insert_same_behavior_as_list_index(self):
        self.t.append((1,2,3))
        self.t.insert(10, (4,5,6))
        self.t.insert(-10, (7,8,9))
        assert len(self.t) == 3
        assert self.t[0] == (7,8,9)
        assert self.t[1] == (1,2,3)
        assert self.t[2] == (4,5,6)
    
    def test_pop(self):
        self.t.append((1,2,3))
        self.t.append((4,5,6))
        assert len(self.t) == 2
        q = self.t.pop()
        assert len(self.t) == 1
        assert q == (4,5,6)
        self.t.append(q)
        y = self.t.pop(0)
        assert len(self.t) == 1
        assert y == (1,2,3)
        assert self.t[0] == q
    
    @raises(IndexError)
    def test_pop_bad_index(self):
        self.t.append((1,2,3))
        q = self.t.pop(10)
    
    def test_del(self):
        self.t.append((1,2,3))
        self.t.append((4,5,6))
        self.t.append((7,8,9))
        assert len(self.t) == 3
        del self.t[1]
        assert len(self.t) == 2
        assert self.t[0] == (1,2,3)
        assert self.t[-1] == (7,8,9)
    
    @raises(IndexError)
    def test_del_bad_index(self):
        self.t.append((1,2,3))
        del self.t[10]

class TestTableAccess:
    '''Test __getitem__'''
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.t = Table(self.headers)
        self.t.append((1,2,3))
        self.t.append((4,5,6))
        self.t.append((7,8,9))
        self.t.append((10,11,12))
        self.t.append((13,14,15))
        self.t.append((16,17,18))
    
    def test_get_row_by_index(self):
        assert self.t[0] == (1,2,3)
    
    def test_get_row_by_reverse_index(self):
        assert self.t[-1] == (16,17,18)
    
    @raises(IndexError)
    def test_get_row_bad_index(self):
        self.t[15]
    
    @raises(IndexError)
    def test_get_row_bad_reverse_index(self):
        self.t[-15]
    
    def test_get_row_slice(self):
        t2 = self.t[1::2]
        assert isinstance(t2, Table)
        assert t2.headers == self.t.headers
        assert len(t2) == 3
        assert t2[0] == (4,5,6)
        assert t2[1] == (10,11,12)
        assert t2[2] == (16,17,18)
    
    def test_get_row_empty_slice(self):
        t2 = self.t[1:1]
        assert isinstance(t2, Table)
        assert t2.headers == self.t.headers
        assert len(t2) == 0
    
    def test_get_cell_by_row_index(self):
        assert self.t[1,1] == 5
    
    def test_get_cell_by_row_reverse_index(self):
        assert self.t[-1,-1] == 18
    
    def test_get_cell_by_row_name(self):
        assert self.t[0, 'z'] == 3
    
    @raises(IndexError)
    def test_get_cell_by_row_bad_index(self):
        self.t[1,15]
    
    @raises(IndexError)
    def test_get_cell_by_row_reverse_bad_index(self):
        self.t[1,-15]
    
    @raises(KeyError)
    def test_get_cell_by_row_bad_name(self):
        self.t[1, 'unknown']
    
    @raises(TypeError)
    def test_get_cell_by_bad_tuple(self):
        self.t[1,2,3]

class TestTableUpdate:
    '''Test __setitem__'''
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.t = Table(self.headers)
        self.t.append((1,2,3))
        self.t.append((4,5,6))
        self.t.append((7,8,9))
        self.t.append((10,11,12))
        self.t.append((13,14,15))
        self.t.append((16,17,18))
    
    def test_set_row_by_index(self):
        row = self.t.Row((0,0,0))
        self.t[0] = row
        self.t[1] = (-1,-1)
        self.t[2] = {'x':10, 'z':10}
        assert self.t[0] == (0,0,0)
        assert self.t[1] == (-1,-1,None)
        assert self.t[2] == (10, None, 10)
    
    def test_set_row_by_reverse_index(self):
        self.t[-2] = (100,100,100)
        assert self.t[4] == (100,100,100)
    
    @raises(IndexError)
    def test_set_row_bad_index(self):
        self.t[15] = (1,1,1)
    
    @raises(IndexError)
    def test_set_row_bad_reverse_index(self):
        self.t[-15] = (1,1,1)
    
    def test_set_row_slice(self):
        t2 = Table(self.headers)
        t2.extend([[-1,-1,-1],[-2,-2,-2],[-3,-3,-3]])
        self.t[1::2] = t2
        assert self.t[1] == (-1,-1,-1)
        assert self.t[3] == (-2,-2,-2)
        assert self.t[5] == (-3,-3,-3)
        assert self.t[1::2] == t2
    
    def test_set_row_empty_slice(self):
        t2 = Table(self.headers)
        t2.extend([[-1,-1,-1],[-2,-2,-2],[-3,-3,-3]])
        self.t[0:0] = t2
        assert len(self.t) == 9
        assert self.t[0] == (-1,-1,-1)
        assert self.t[1] == (-2,-2,-2)
        assert self.t[2] == (-3,-3,-3)
    
    @raises(TypeError)
    def test_set_row_slice_not_table(self):
        self.t[1::2] = [[-1,-1,-1],[-2,-2,-2],[-3,-3,-3]]
    
    @raises(TypeError)
    def test_set_row_slice_wrong_table_rows(self):
        t2 = Table(['p', 'd', 'q'])
        t2.extend([[-1,-1,-1],[-2,-2,-2],[-3,-3,-3]])
        self.t[1::2] = t2
    
    def test_set_cell_by_row_index(self):
        self.t[1,1] = -99
        assert self.t[1] == (4,-99,6)
    
    def test_set_cell_by_row_reverse_index(self):
        self.t[-1,-1] = -99
        assert self.t[5] == (16,17,-99)
    
    def test_set_cell_by_row_name(self):
        self.t[0, 'y'] = -99
        assert self.t[0] == (1,-99,3)
    
    @raises(IndexError)
    def test_set_cell_by_row_bad_index(self):
        self.t[0, 25] = -99
    
    @raises(IndexError)
    def test_set_cell_by_row_bad_reverse_index(self):
        self.t[0, -25] = -99
    
    @raises(KeyError)
    def test_set_cell_by_row_bad_name(self):
        self.t[0, 'unknown'] = -99
    
    @raises(TypeError)
    def test_set_cell_by_bad_tuple(self):
        self.t[0,1,2] = (1,1,1)

class TestTableDeletion:
    '''Test __delitem__'''
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.t = Table(self.headers)
        self.t.append((1,2,3))
        self.t.append((4,5,6))
        self.t.append((7,8,9))
        self.t.append((10,11,12))
        self.t.append((13,14,15))
        self.t.append((16,17,18))
    
    def test_del_row_by_index(self):
        del self.t[0]
        assert len(self.t) == 5
        assert self.t[0] == (4,5,6)
    
    def test_del_row_by_reverse_index(self):
        del self.t[-2]
        assert len(self.t) == 5
        assert self.t[4] == (16,17,18)
    
    @raises(IndexError)
    def test_del_row_bad_index(self):
        del self.t[15]
    
    @raises(IndexError)
    def test_del_row_bad_reverse_index(self):
        del self.t[-15]
    
    def test_del_row_slice(self):
        del self.t[1::2]
        assert len(self.t) == 3
        assert self.t[0] == (1,2,3)
        assert self.t[1] == (7,8,9)
        assert self.t[2] == (13,14,15)
    
    def test_del_row_empty_slice(self):
        del self.t[0:0]
        assert len(self.t) == 6
    
    def test_del_cell_by_row_index(self):
        del self.t[1,1]
        assert self.t[1] == (4,None,6)
    
    def test_del_cell_by_row_reverse_index(self):
        del self.t[-1,-1]
        assert self.t[5] == (16,17,None)
    
    def test_del_cell_by_row_name(self):
        del self.t[0, 'y']
        assert self.t[0] == (1,None,3)
    
    @raises(IndexError)
    def test_del_cell_by_row_bad_index(self):
        del self.t[0, 25]
    
    @raises(IndexError)
    def test_del_cell_by_row_bad_reverse_index(self):
        del self.t[0, -25]
    
    @raises(KeyError)
    def test_del_cell_by_row_bad_name(self):
        del self.t[0, 'unknown']
    
    @raises(TypeError)
    def test_del_cell_by_bad_tuple(self):
        del self.t[0,1,2]

class TestTableSorting:
    '''Test sort, sort_by_col, reverse'''
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.t = Table(self.headers)
        self.t.append((1,2,9))
        self.t.append((1,-5,6))
        self.t.append((-1,8,9))
        self.t.append((10,11,-12))
        self.t.append((-13,11,15))
        self.t.append((16,0,1))
    
    def test_sort_by_single_col(self):
        self.t.sort_by_col('y')
        assert list(self.t.column('y')) == [-5, 0, 2, 8, 11, 11]
        assert self.t[0] == (1,-5,6)
        assert self.t[-2] == (10,11,-12)
        assert self.t[-1] == (-13,11,15)
    
    def test_sort_by_single_col_numeric(self):
        self.t.sort_by_col(1)
        assert list(self.t.column('y')) == [-5, 0, 2, 8, 11, 11]
        assert self.t[0] == (1,-5,6)
        assert self.t[-2] == (10,11,-12)
        assert self.t[-1] == (-13,11,15)
    
    def test_sort_by_multiple_col(self):
        self.t.sort_by_col(['x', 'z'])
        assert list(self.t.column('x')) == [-13, -1, 1, 1, 10, 16]
        assert list(self.t.column('z')) == [15, 9, 6, 9, -12, 1]
    
    def test_sort_by_multiple_col_numeric(self):
        self.t.sort_by_col([0, 2])
        assert list(self.t.column('x')) == [-13, -1, 1, 1, 10, 16]
        assert list(self.t.column('z')) == [15, 9, 6, 9, -12, 1]
    
    def test_sort_by_single_col_reverse(self):
        self.t.sort_by_col('y', reverse=True)
        assert list(self.t.column('y')) == [11, 11, 8, 2, 0, -5]
        # Even though the sort order is reversed, in the case of equal numbers
        # the original row ordering remains
        assert list(self.t.column('z')) == [-12, 15, 9, 9, 1, 6]
    
    def test_sort_by_multiple_col_reverse(self):
        self.t.sort_by_col(['z', 'y', 'x'], reverse=True)
        assert self.t[0] == (-13,11,15)
        assert self.t[1] == (-1,8,9)
        assert self.t[2] == (1,2,9)
        assert self.t[3] == (1,-5,6)
        assert self.t[4] == (16,0,1)
        assert self.t[5] == (10,11,-12)
    
    def test_sort_by_func(self):
        func = lambda row: sum(map(abs, row))
        self.t.sort(func)
        assert self.t[0] == (1,2,9)
        assert self.t[1] == (1,-5,6)
        assert self.t[2] == (16,0,1)
        assert self.t[3] == (-1,8,9)
        assert self.t[4] == (10,11,-12)
        assert self.t[5] == (-13,11,15)
    
    def test_sort_by_func_reverse(self):
        func = lambda row: sum(map(abs, row))
        self.t.sort(key=func, reverse=True)
        assert self.t[0] == (-13,11,15)
        assert self.t[1] == (10,11,-12)
        assert self.t[2] == (-1,8,9)
        assert self.t[3] == (16,0,1)
        # Equal sort items retain original ordering, even with reverse flag
        assert self.t[4] == (1,2,9)
        assert self.t[5] == (1,-5,6)
    
    def test_reverse(self):
        self.t.reverse()
        assert self.t[0] == (16,0,1)
        assert self.t[1] == (-13,11,15)
        assert self.t[2] == (10,11,-12)
        assert self.t[3] == (-1,8,9)
        assert self.t[4] == (1,-5,6)
        assert self.t[5] == (1,2,9)

class TestTableTake:
    '''Test take by func and indices'''
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.t = Table(self.headers)
        self.t.append((1,2,9))
        self.t.append((1,-5,6))
        self.t.append((-1,8,9))
        self.t.append((10,11,-12))
        self.t.append((-13,11,15))
        self.t.append((16,0,1))
    
    def test_take_by_func(self):
        func = lambda row: row['y'] > 3
        t2 = self.t.take(func)
        assert len(self.t) == 6
        assert len(t2) == 3
        assert t2[0] == (-1,8,9)
        assert t2[1] == (10,11,-12)
        assert t2[2] == (-13,11,15)
    
    def test_take_by_indexes(self):
        indexes = [1, 4, 2]
        t2 = self.t.take(indexes)
        assert len(self.t) == 6
        assert len(t2) == 3
        assert t2[0] == (1,-5,6)
        assert t2[1] == (-13,11,15)
        assert t2[2] == (-1,8,9)
    
    def test_take_by_negative_indexes(self):
        indexes = [0, -1, -3]
        t2 = self.t.take(indexes)
        assert len(t2) == 3
        assert t2[0] == (1,2,9)
        assert t2[1] == (16,0,1)
        assert t2[2] == (10,11,-12)
    
    @raises(IndexError)
    def test_take_bad_indexes(self):
        indexes = [0,1,15]
        self.t.take(indexes)
    
    def test_take_by_indexes_generator(self):
        def gene8r():
            for i in (1,4,2):
                yield i
        t2 = self.t.take(gene8r())
        assert len(t2) == 3
        assert t2[0] == (1,-5,6)
        assert t2[1] == (-13,11,15)
        assert t2[2] == (-1,8,9)
    
    def test_take_duplicate_indexes(self):
        indexes = [3,1,1,1]
        t2 = self.t.take(indexes)
        assert len(t2) == 4
        assert t2[0] == (10,11,-12)
        assert t2[0] is self.t[3]
        assert t2[1] is self.t[1]
        assert t2[1] == (1,-5,6)
        assert t2[2] == t2[3] == t2[1]
        assert t2[2] is t2[3] is t2[1]
    
    def test_take_equivalency_indexes(self):
        indexes = range(6)
        t2 = self.t.take(indexes)
        assert self.t == t2
        assert self.t is not t2
    
    def test_take_equivalency_func(self):
        func = lambda row: True
        t2 = self.t.take(func)
        assert self.t == t2
        assert self.t is not t2

class TestTableColumnMethod:
    '''Test column()'''
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.t = Table(self.headers)
        self.t.append((1,2,3))
        self.t.append((4,5,6))
        self.t.append((7,8,9))
    
    def test_column_by_index(self):
        assert list(self.t.column(0)) == [1,4,7]
        assert list(self.t.column(1)) == [2,5,8]
        assert list(self.t.column(2)) == [3,6,9]
    
    def test_column_by_reverse_index(self):
        assert list(self.t.column(-1)) == [3,6,9]
        assert list(self.t.column(-2)) == [2,5,8]
        assert list(self.t.column(-3)) == [1,4,7]
    
    def test_column_by_name(self):
        assert list(self.t.column('x')) == [1,4,7]
        assert list(self.t.column('y')) == [2,5,8]
        assert list(self.t.column('z')) == [3,6,9]
    
    @raises(IndexError)
    def test_column_by_bad_index(self):
        self.t.column(15)
    
    @raises(IndexError)
    def test_column_by_bad_reverse_index(self):
        self.t.column(-15)
    
    @raises(KeyError)
    def test_column_by_bad_name(self):
        self.t.column('unknown')

class TestTableConversion:
    '''Test convert to ndarray and DataFrame'''
    def setUp(self):
        self.headers = ['x', 'y', 'z']
        self.t = Table(self.headers)
        self.t.append((1,2,3))
        self.t.append((4,5,6))
        self.t.append((7,8,9))
    
    def test_ndarray(self):
        import numpy as np
        ta = self.t.as_array()
        assert isinstance(ta, np.ndarray)
        a = np.array([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
        assert a.size == ta.size
        assert a.shape == ta.shape
        assert np.all(a == ta)
        assert a is not ta
    
    def test_dataframe(self):
        import pandas as pd
        tp = self.t.as_dataframe()
        assert isinstance(tp, pd.DataFrame)
        df = pd.DataFrame({'x': [1,4,7],
                           'y': [2,5,8],
                           'z': [3,6,9]})
        assert len(tp) == len(df)
        assert (tp.columns == df.columns).all()
        assert (tp == df).all().all()
        assert tp is not df
        