"""

nn_layers_pt.py

PyTorch version of nn_layers

"""

import torch
import torch.nn.functional as F
import numbers
import numpy as np
import math

"""

function view_as_windows

"""

def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.
    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).
    Parameters
    ----------
    arr_in : Pytorch tensor
        N-d Pytorch tensor.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.
    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.
    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle
    Examples
    --------
    >>> import torch
    >>> A = torch.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])
    >>> A = torch.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])
    >>> A = torch.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not torch.is_tensor(arr_in):
        raise TypeError("`arr_in` must be a pytorch tensor")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = torch.tensor(arr_in.shape)
    window_shape = torch.tensor(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    # window_strides = torch.tensor(arr_in.stride())
    window_strides = arr_in.stride()

    indexing_strides = arr_in[slices].stride()

    win_indices_shape = torch.div(arr_shape - window_shape
                          , torch.tensor(step), rounding_mode = 'floor') + 1
    
    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = torch.as_strided(arr_in, size=new_shape, stride=strides)
    return arr_out

#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, f_height, f_width, input_size, in_ch_size, out_ch_size):
        
        # Xavier init
        self.W = torch.normal(0, 1 / math.sqrt((in_ch_size * f_height * f_width / 2)),
                                  size=(out_ch_size, in_ch_size, f_height, f_width))
        self.b = 0.01 + torch.zeros(size=(1, out_ch_size, 1, 1))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.v_W = torch.zeros_like(self.W)
        self.v_b = torch.zeros_like(self.b)
        
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W.clone().detach(), self.b.clone().detach()

    def set_weights(self, W, b):
        self.W = W.clone().detach()
        self.b = b.clone().detach()

    def forward(self, x):
        ###################################
        # Q4. Implement your layer here
        ###################################
        batch_size, in_ch_size, in_width, in_height = x.shape
        out_ch_size, in_ch_size, f_height, f_width = self.W.shape
        
        out_width = (in_width-f_width) + 1  #stride = 1
        out_height = (in_height-f_height) + 1   #stride = 1

        out = torch.empty((batch_size,out_ch_size,out_width,out_height))

        for i in range(out_ch_size):
            filt = self.W[i,:,:,:].reshape((-1,1))
            y = view_as_windows(x,(batch_size,in_ch_size,f_width,f_height)).reshape((out_width,out_height,batch_size,-1))
            result = torch.matmul(y,filt).squeeze(dim=-1).transpose(0,2).transpose(1,2)
            out[:,i,:,:] = result + self.b[0,i,0,0]
        
        return out
        
    
    def step(self, lr, friction):
        with torch.no_grad():
            self.v_W = friction*self.v_W + (1-friction)*self.W.grad
            self.v_b = friction*self.v_b + (1-friction)*self.b.grad
            self.W -= lr*self.v_W
            self.b -= lr*self.v_b
            
            self.W.grad.zero_()
            self.b.grad.zero_()

# max pooling
class nn_max_pooling_layer:
    def __init__(self, pool_size, stride):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        ###################################
        # Q5. Implement your layer here
        ###################################
        batch_size,in_ch_size, in_width, in_height = x.shape
        pool_width = self.pool_size
        pool_height = self.pool_size
        stride = self.stride
        out_width = int((in_width-pool_width)/stride) +1
        out_height = int((in_height-pool_height)/stride) +1
        out = torch.empty((batch_size,in_ch_size,out_width,out_height))

        for i in range(batch_size):
            for j in range(in_ch_size):
                tmp = view_as_windows(x[i,j],(pool_width,pool_height),step=stride).reshape(out_width,out_height,-1).max(dim=-1).values
                out[i,j,:,:] = tmp
        
        return out

# relu activation
class nn_activation_layer:

    # linear layer. creates matrix W and bias b
    # W is in by out, and b is out by 1
    def __init__(self):
        pass

    def forward(self, x):
        return x.clamp(min=0)

# fully connected (linear) layer
class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        
        # Xavier/He init
        self.W = torch.normal(0, std/math.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+torch.zeros((output_size))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.v_W = torch.zeros_like(self.W)
        self.v_b = torch.zeros_like(self.b)

    ## Q1
    def forward(self,x):
        # compute forward pass of given parameter
        # output size is batch x output_size x 1 x 1
        # input size is batch x input_size x filt_size x filt_size
        output_size = self.W.shape[0]
        batch_size = x.shape[0]
        Wx = torch.mm(x.reshape((batch_size, -1)),(self.W.reshape(output_size, -1)).T)
        out = Wx+self.b
        return out

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W.clone().detach(), self.b.clone().detach()

    def set_weights(self, W, b):
        self.W = W.clone().detach()
        self.b = b.clone().detach()
    
    def step(self, lr, friction):
        with torch.no_grad():
            self.v_W = friction*self.v_W + (1-friction)*self.W.grad
            self.v_b = friction*self.v_b + (1-friction)*self.b.grad
            self.W -= lr*self.v_W
            self.b -= lr*self.v_b
            self.W.grad.zero_()
            self.b.grad.zero_()


# softmax layer
class nn_softmax_layer:
    def __init__(self):
        pass

    def forward(self, x):
        s = x - torch.unsqueeze(torch.amax(x, axis=1), -1)
        return (torch.exp(s) / torch.unsqueeze(torch.sum(torch.exp(s), axis=1), -1)).reshape((x.shape[0],x.shape[1]))


# cross entropy layer
class nn_cross_entropy_layer:
    def __init__(self):
        self.eps=1e-15

    def forward(self, x, y):
        # first get softmax
        batch_size = x.shape[0]
        num_class = x.shape[1]
        
        onehot = np.zeros((batch_size, num_class))
        onehot[range(batch_size), (np.array(y)).reshape(-1, )] = 1
        onehot = torch.as_tensor(onehot)

        # avoid numerial instability
        x[x<self.eps]=self.eps
        x=x/torch.unsqueeze(torch.sum(x,axis=1), -1)

        return sum(-torch.sum(torch.log(x.reshape(batch_size, -1)) * onehot, axis=0)) / batch_size
