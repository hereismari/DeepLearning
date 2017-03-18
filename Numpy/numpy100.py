'''
    100 practical numpy exercises in order to RULE IT!
    https://github.com/rougier/numpy-100
'''

def print_delimiter(func):
    upper = '-----------------------------\n'
    down = '\n'
    def func_wrapper():
        print upper
        func()
        print down
    return func_wrapper()

import numpy as np

@print_delimiter
def _2():
    print '2.Print the numpy version and the configuration'
    print
    print 'version:', np.__version__
    print 'configs:', np.show_config()

@print_delimiter
def _3():
    print '3.Create a null vector of size 10'
    print
    null_vector = np.zeros(10)
    print null_vector

@print_delimiter
def _4():
    print '4. How to find the memory size of any array'
    print
    null_vector = np.zeros(10)
    print 'null vector has size of %d bytes' % (null_vector.size * null_vector.itemsize)

@print_delimiter
def _5():
    print '5. How to get the documentation of the numpy add function from the command line?'
    print
    np.info(np.add)

@print_delimiter
def _6():
    print '6. Create a null vector of size 10 but the fifth value which is 1'
    print
    v = np.zeros(10)
    v[4] = 1
    print v

@print_delimiter
def _7():
    print '7. Create a vector with values ranging from 10 to 49'
    print
    v = np.array(range(10, 50))
    print v

@print_delimiter
def _8():
    print '8. Reverse a vector (first element becomes last)'
    print
    v = np.array(range(10, 50))
    v = v[::-1]
    print v

@print_delimiter
def _9():
    print '9. Create a 3x3 matrix with values ranging from 0 to 8'
    print
    v = np.arange(9).reshape(3, 3)
    print v

@print_delimiter
def _10():
    print '10. Find indices of non-zero elements from [1,2,0,0,4,0]'
    print
    v = np.array([1, 2, 0, 0, 4, 0])
    print v.nonzero()

@print_delimiter
def _11():
    print '11. Create a 3x3 identity matrix'
    print
    v = np.identity(3)
    print v

@print_delimiter
def _12():
    print '12. Create a 3x3x3 array with random values'
    print '13. Create a 10x10 array with random values and find the minimum and maximum values'
    print
    v = np.random.random((3, 3, 3))
    print v
    print np.min(v), np.max(v)

@print_delimiter
def _20():
    print '20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?'
    print
    print(np.unravel_index(100,(6,7,8)))

@print_delimiter
def _21():
    print '21. Create a checkerboard 8x8 matrix using the tile function'
    print
    print np.tile( np.array([[0,1],[1,0]]), (4,4))
