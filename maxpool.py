import numpy as np
import math
from icecream import ic
from skimage.measure import block_reduce

def ceil(x):
    return int(math.ceil(x))

# IMPORTANT ASSUMPTION:
# A matrix of MAX_BLOCK_SIZE * MAX_BLOCK_SIZE is large enough to hold
# the largest convolution kernel
# Our code is coded with this assumption: otherwise, additional measures
# need to be taken to generate the correct result
MAX_BLOCK_SIZE = 4
CHECK = False

class MaxPool2D:
    def __init__(self, I, kernel_size):
        assert kernel_size[0] < MAX_BLOCK_SIZE, "Kernel height must be less than MAX_BLOCK_SIZE"
        assert kernel_size[1] < MAX_BLOCK_SIZE, "Kernel width must be less than MAX_BLOCK_SIZE"
        self.I = I
        self.kernel_size = kernel_size
        self._shape = (ceil(I.shape[0]/kernel_size[0]), ceil(I.shape[1]/kernel_size[0]))
        ic(self._shape)
        self._blocked_shape = ceil(self._shape[0]/MAX_BLOCK_SIZE), ceil(self._shape[1]/MAX_BLOCK_SIZE)
        ic(self._blocked_shape)
        self.computed = set()
        self.recompute_count = 0
        self.partially_computed = set()
        self.partial_recompute_count = 0


    @property
    def shape(self):
        return self._shape

    @property
    def blocked_shape(self):
        return self._blocked_shape

    def __matmul__(self, h):
        return MaxPool2D(self, h)

    def get_partial_block(self, i, j, x, y):
        """Returns block (i, j) ensuring that block[:x+1, :y+1] is valid"""
        assert 0 <= i < self._blocked_shape[0]
        assert 0 <= j < self._blocked_shape[1]
        def compute_size(i, j):
            nrows = MAX_BLOCK_SIZE if i < self.blocked_shape[0] - 1 else self._shape[0] % MAX_BLOCK_SIZE
            ncolumns = MAX_BLOCK_SIZE if i < self.blocked_shape[1] - 1 else self._shape[1] % MAX_BLOCK_SIZE
            nrows = MAX_BLOCK_SIZE if nrows == 0 else nrows
            ncolumns = MAX_BLOCK_SIZE if ncolumns == 0 else ncolumns
            return (nrows, ncolumns)

        if (i, j) in self.partially_computed:
            self.partial_recompute_count += 1

        I, ksize = self.I, self.kernel_size
        self.partially_computed.add((i, j))

        nrows, ncolumns = compute_size(i, j)
        accumulator = np.zeros((nrows, ncolumns))
        cx = 0
        max_di = min(ksize[0], I.blocked_shape[0] - i*ksize[0])
        max_dj = min(ksize[1], I.blocked_shape[1] - j*ksize[1])
        prepend_rows = [None] * max_dj
        for di in range(max_di):
            print(f"[Row {di}]")
            cy = 0
            prepend_column = None
            next_prepend_rows = []
            for dj in range(max_dj):
                print(f"[Column {dj}]")
                b = I.get_partial_block(i*ksize[0] + di, j*ksize[1] + dj, x=x, y=y)
                ic(b)

                n_rows = b.shape[0]
                if prepend_rows[dj] is not None:
                    ic(prepend_rows[dj])
                    b = np.concatenate((prepend_rows[dj], b), axis=0)

                max_x = ksize[0] * (b.shape[0] // ksize[0])
                next_prepend_rows.append(b[max_x:, :])
                # Keep only "good" portion
                b = b[:max_x+1, :]

                if prepend_column is not None:
                    ic(prepend_column)
                    b = np.concatenate((prepend_column, b), axis=-1)


                bp = b
                ic(bp)
                c = block_reduce(b, ksize, np.max)
                dx = b.shape[0] // ksize[0]
                dy = b.shape[1] // ksize[1]
                max_y = ksize[1] * (b.shape[1] // ksize[1])
                accumulator[cx:cx+dx, cy:cy+dy] = c[:dx, :dy]
                prepend_column = b[:n_rows, max_y:]
                ic(n_rows)
                cy += dy

            # Take care of leftover prepend column
            if prepend_column is not None and prepend_column.shape[1] != 0:
                ic(prepend_column)
                c = block_reduce(prepend_column, ksize, np.max)
                dxx = c.shape[0]
                accumulator[cx:cx+dxx, -1] = c.squeeze(-1)

            cx += dx
            prepend_rows = next_prepend_rows

        cy = 0
        prepend_column = None
        print("[Finalizing]")
        for b in prepend_rows:
            if b is not None and b.shape[0] != 0:
                ic(b)
                if prepend_column is not None:
                    ic(prepend_column)
                    b = np.concatenate((prepend_column, b), axis=-1)

                bp = b
                ic(bp)
                max_y = ksize[1] * (b.shape[1] // ksize[1])
                c = block_reduce(b, ksize, np.max)
                dy = b.shape[1] // ksize[1]
                ic(c)
                ic(dy)
                accumulator[-1, cy:cy+dy] = c[:, :dy].squeeze(0)
                prepend_column = b[:, max_y:]
                ic(accumulator[-1, :])
                cy += dy

        if prepend_column is not None and prepend_column.shape[1] != 0:
            accumulator[-1, -1] = np.max(prepend_column)


        return accumulator


    def get_block(self, i, j):
        """Get the (i, j)-th block. We use a constant amount of intermediate memory and a lot of recomputation"""
        return self.get_partial_block(i, j, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE)
    
class Matrix:
    def __init__(self, A):
        self.A = A
        self.id = 0
        self.do_print = False

    def __getitem__(self, *idx):
        return self.A.__getitem__(*idx)

    def __matmul__(self, h):
        return MaxPool2D(self, h)

    def get_partial_block(self, i, j, x=None, y=None):
        """Get the (i, j)-th block, ensuring that block[:x+1, :y+1] is valid.. The 'real' implementation would load this from disc into memory."""
        n, m = self.A.shape
        return self.A[i*MAX_BLOCK_SIZE:(i+1)*MAX_BLOCK_SIZE,j*MAX_BLOCK_SIZE:(j+1)*MAX_BLOCK_SIZE]

    def get_block(self, i, j):
        """Get the (i, j)-th block. The 'real' implementation would load this from disc into memory. """
        n, m = self.A.shape
        return self.A[i*MAX_BLOCK_SIZE:(i+1)*MAX_BLOCK_SIZE,j*MAX_BLOCK_SIZE:(j+1)*MAX_BLOCK_SIZE]

    @property
    def shape(self):
        """Returns the size of the matrix in terms of (n_row_elements, n_column_elements)"""
        return self.A.shape

    # Defines
    @property
    def blocked_shape(self):
        """Returns the size of the 'blocked' matrix"""
        return (ceil(self.A.shape[0]/MAX_BLOCK_SIZE), ceil(self.A.shape[1]/MAX_BLOCK_SIZE))

def main():
    print("[Initializing]")
    # This is our "input" to the neural network
    np.set_printoptions(edgeitems=30, linewidth=100000)
    KSIZE = (3, 3)
    SIZE = 8
    I = Matrix(np.arange(SIZE**2).reshape(SIZE, SIZE))
    cmp = I @ KSIZE
    ic(I.A)
    tmp = Matrix(block_reduce(I.A, KSIZE, np.max))
    ic(tmp.A)


    print("[Testing]")
    for i in range(cmp.blocked_shape[0]):
        for j in range(cmp.blocked_shape[1]):
            # Compare the final result of the convolution
            # with the true result
            # Now is when computation is actually done
            computed_block = cmp.get_block(i, j)
            ic(computed_block)
            true_block = tmp.get_block(i, j)
            ic(true_block)
            assert np.allclose(computed_block, true_block), f"(mp) block {(i, j)} does not match"




if __name__ == '__main__':
    main()