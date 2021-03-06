import numpy as np
import math
from scipy.signal import convolve2d
from icecream import ic

def ceil(x):
    return int(math.ceil(x))

# IMPORTANT ASSUMPTION:
# A matrix of MAX_BLOCK_SIZE * MAX_BLOCK_SIZE is large enough to hold
# the largest convolution kernel
# Our code is coded with this assumption: otherwise, additional measures
# need to be taken to generate the correct result
MAX_BLOCK_SIZE = 9
CHECK = False

class Convolve2D:
    def __init__(self, I, h):
        assert h.shape[0] < MAX_BLOCK_SIZE, "Kernel height must be less than MAX_BLOCK_SIZE"
        assert h.shape[1] < MAX_BLOCK_SIZE, "Kernel width must be less than MAX_BLOCK_SIZE"
        self.I = I
        self.h = h
        self._shape = (I.shape[0] - h.shape[0] + 1, I.shape[1] - h.shape[1] + 1)
        self._blocked_shape = ceil(self._shape[0]/MAX_BLOCK_SIZE), ceil(self._shape[1]/MAX_BLOCK_SIZE)
        self.id = I.id + 1
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
        return Convolve2D(self, h)

    def get_partial_block(self, i, j, x, y):
        """Returns block (i, j) ensuring that block[:x+1, :y+1] is valid"""
        def compute_size(i, j):
            nrows = (self.I.shape[0] - self.h.shape[0] + 1) % MAX_BLOCK_SIZE
            nrows = nrows if nrows != 0 else MAX_BLOCK_SIZE
            ncolumns = (self.I.shape[1] - self.h.shape[1] + 1) % MAX_BLOCK_SIZE
            ncolumns = ncolumns if ncolumns != 0 else MAX_BLOCK_SIZE
            nrows = nrows if i == self.blocked_shape[0] - 1 else MAX_BLOCK_SIZE
            ncolumns = ncolumns if j == self.blocked_shape[1] - 1 else MAX_BLOCK_SIZE
            return (nrows, ncolumns)

        if (i, j) in self.partially_computed:
            self.partial_recompute_count += 1

        I, h = self.I, self.h
        self.partially_computed.add((i, j))

        nrows, ncolumns = compute_size(i, j)
        accumulator = np.zeros((nrows, ncolumns))
        b = I.get_partial_block(i, j, x=x, y=y)
        c = convolve2d(b, h, mode='full')
        vl, vh = c[h.shape[0] - 1:c.shape[0] - (h.shape[0] - 1), h.shape[1] - 1:c.shape[1] - (h.shape[1] - 1)].shape

        accumulator += c[h.shape[0] - 1:h.shape[0] - 1 + nrows, h.shape[1] - 1:h.shape[1] - 1 + ncolumns]
        xp = x + h.shape[0] - MAX_BLOCK_SIZE
        yp = y + h.shape[1] - MAX_BLOCK_SIZE


        # Check if the row dependencies are satisfied
        # We use the assumption that the block_length > kernel_length and block_length > kernel_length
        if x + h.shape[0] >= MAX_BLOCK_SIZE and i + 1 < I.blocked_shape[0]:
            b = I.get_partial_block(i + 1, j, x=xp, y=y)
            c = convolve2d(b, h, mode='full', fillvalue=0)
            l, w = min(h.shape[0], b.shape[0] + 1), min(h.shape[1], b.shape[1] + 1)
            accumulator[vl:, :] += c[:l - 1, w-1:w - 1 + ncolumns]

        if y + h.shape[1] >= MAX_BLOCK_SIZE and j + 1 < I.blocked_shape[1]:
            b = I.get_partial_block(i, j + 1, x=x, y=yp)
            c = convolve2d(b, h, mode='full', fillvalue=0)
            l, w = min(h.shape[0], b.shape[0] + 1), min(h.shape[1], b.shape[1] + 1)

            accumulator[:, vh:] += c[l- 1:l - 1 + nrows, :w - 1]

        if x + h.shape[0] >= MAX_BLOCK_SIZE \
                and y + h.shape[1] >= MAX_BLOCK_SIZE \
                and i + 1 < I.blocked_shape[0] \
                and j + 1 < I.blocked_shape[1]:
            b = I.get_partial_block(i + 1, j + 1, x=xp, y=yp)
            c = convolve2d(b, h, mode='full', fillvalue=0)
            l, w = min(h.shape[0], b.shape[0] + 1), min(h.shape[1], b.shape[1] + 1)

            accumulator[vl:, vh:] += c[:l - 1, :w- 1]

        return accumulator


    def get_block(self, i, j):
        """Get the (i, j)-th block. We use a constant amount of intermediate memory and a lot of recomputation"""
        def compute_size(i, j):
            nrows = (self.I.shape[0] - self.h.shape[0] + 1) % MAX_BLOCK_SIZE
            nrows = nrows if nrows != 0 else MAX_BLOCK_SIZE
            ncolumns = (self.I.shape[1] - self.h.shape[1] + 1) % MAX_BLOCK_SIZE
            ncolumns = ncolumns if ncolumns != 0 else MAX_BLOCK_SIZE
            nrows = nrows if i == self.blocked_shape[0] - 1 else MAX_BLOCK_SIZE
            ncolumns = ncolumns if j == self.blocked_shape[1] - 1 else MAX_BLOCK_SIZE
            return (nrows, ncolumns)

        if (i, j) in self.computed:
            self.recompute_count += 1

        I, h = self.I, self.h
        self.computed.add((i, j))

        nrows, ncolumns = compute_size(i, j)
        accumulator = np.zeros((nrows, ncolumns))
        # TODO: possibly also make this a partial block?
        b = I.get_block(i, j)
        c = convolve2d(b, h, mode='full')
        vl, vh = c[h.shape[0] - 1:c.shape[0] - (h.shape[0] - 1), h.shape[1] - 1:c.shape[1] - (h.shape[1] - 1)].shape

        accumulator += c[h.shape[0] - 1:h.shape[0] - 1 + nrows, h.shape[1] - 1:h.shape[1] - 1 + ncolumns]

        if i + 1 < I.blocked_shape[0]:
            b = I.get_partial_block(i + 1, j, x=h.shape[0] - 1, y=MAX_BLOCK_SIZE - 1)
            c = convolve2d(b, h, mode='full', fillvalue=0)
            l, w = min(h.shape[0], b.shape[0] + 1), min(h.shape[1], b.shape[1] + 1)

            accumulator[vl:, :] += c[:l - 1, w-1:w - 1 + ncolumns]

        if j + 1 < I.blocked_shape[1]:
            b = I.get_partial_block(i, j + 1, x=MAX_BLOCK_SIZE - 1, y=h.shape[1] - 1)
            c = convolve2d(b, h, mode='full', fillvalue=0)
            l, w = min(h.shape[0], b.shape[0] + 1), min(h.shape[1], b.shape[1] + 1)

            accumulator[:, vh:] += c[l- 1:l - 1 + nrows, :w - 1]

        if i + 1 < I.blocked_shape[0] and j + 1 < I.blocked_shape[1]:
            b = I.get_partial_block(i + 1, j + 1, x=h.shape[0] - 1, y=h.shape[1] - 1)
            c = convolve2d(b, h, mode='full', fillvalue=0)
            l, w = min(h.shape[0], b.shape[0] + 1), min(h.shape[1], b.shape[1] + 1)

            accumulator[vl:, vh:] += c[:l - 1, :w- 1]

        return accumulator
    
class Matrix:
    def __init__(self, A):
        self.A = A
        self.id = 0
        self.do_print = False

    def __getitem__(self, *idx):
        return self.A.__getitem__(*idx)

    def __matmul__(self, h):
        return Convolve2D(self, h)

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
    I = Matrix(np.random.rand(1032, 532))
    #I = Matrix(np.random.rand(484, 643))
    #I = Matrix(np.random.rand(30, 30)) # id=0
    # These are the values of the convolution filters
    # We are applying four convolution "layers" but we
    # do not have space to store the intermediate results
    h1 = np.random.rand(4, 5) # id=1 when debugging
    h2 = np.random.rand(3, 4) # id=2 when debugging
    h3 = np.random.rand(6, 6) # id=3 when debugging
    h4 = np.random.rand(8, 3) # id=4
    b1 = I @ h1 # Result after first convolution layer (nothing is actually computed)
    b2 = b1 @ h2 # Result after second convolution layer (nothing is actually computed)
    b3 = b2 @ h3 # Result after third convolution layer (nothing is actually computed)
    b4 = b3 @ h4 # Result after fourth convolution layer (nothing is actually computed)

    c1 = Matrix(convolve2d(I.A, h1, mode='valid'))
    c2 = Matrix(convolve2d(c1.A, h2, mode='valid'))
    c3 = Matrix(convolve2d(c2.A, h3, mode='valid'))
    c4 = Matrix(convolve2d(c3.A, h4, mode='valid'))


    print("[Testing]")
    for i in range(b4.blocked_shape[0]):
        for j in range(b4.blocked_shape[1]):
            # Compare the final result of the convolution
            # with the true result
            # Now is when computation is actually done
            computed_block = b4.get_block(i, j)
            true_block = c4.get_block(i, j)
            assert np.allclose(computed_block, true_block), f"(b4) block {(i, j)} does not match"


    print(f"Total recomputation done (b1): {b1.recompute_count}")
    print(f"Partial recomputation done (b1): {b1.partial_recompute_count}")
    print(f"Total recomputation done (b2): {b2.recompute_count}")
    print(f"Partial recomputation done (b2): {b2.partial_recompute_count}")
    print(f"Total recomputation done (b3): {b3.recompute_count}")
    print(f"Partial recomputation done (b3): {b3.partial_recompute_count}")
    print(f"Total recomputation done (b4): {b4.recompute_count}")
    print(f"Partial recomputation done (b4): {b4.partial_recompute_count}")


if __name__ == '__main__':
    main()
