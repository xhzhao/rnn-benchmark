import  tensorflow as tf
import  numpy as np
import  sys
import time

dry_run = 50
num_iter = 100
count = 20     # number of iterations
#cuda = False   # whether GPU is used or not
train = True  # True: test training performance; False: test forward performance only
daily = True

#if 'cuda' in sys.argv:
#    cuda = True
if 'train' in sys.argv:
    train = True
if 'daily' in sys.argv:
    daily = True

if daily:
    sizes = [[64,50,500,500],
         [128,25,4096,4096]
        ]
    print("daily test")
else:
    sizes = [[64,15,500,500],
         [64,20,500,500],
         [64,25,500,500],
         [64,30,500,500],
         [64,35,500,500],
         [64,40,500,500],
         [64,45,500,500],
         [64,50,500,500],
         [16,25,512,512],
         [32,25,512,512],
         [64,25,512,512],
         [128,25,512,512],
         [16,25,1024,1024],
         [32,25,1024,1024],
         [64,25,1024,1024],
         [128,25,1024,1024],
         [16,25,2048,2048],
         [32,25,2048,2048],
         [64,25,2048,2048],
         [128,25,2048,2048],
         [16,25,4096,4096],
         [32,25,4096,4096],
         [64,25,4096,4096],
         [128,25,4096,4096]
        ]


for idx in range(len(sizes)):
    size = sizes[idx]
    N = size[0]    # batch size
    T = size[1]    # sentence length
    D = size[2]    # embedding size
    H = size[3]    # hidden size

    X = np.random.randn(N,T,D)
    target = np.random.randn(N,T,H)

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=H, state_is_tuple=True)

    init_state = cell.zero_state(batch_size=N, dtype=tf.float64)

    rnn_name = "rnn" + str(idx)

    with tf.variable_scope(rnn_name, reuse=None):
        outputs, last_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float64,
            initial_state=init_state,
            inputs=X,
        )

    loss = tf.nn.l2_loss(outputs - target)

    optim = tf.train.GradientDescentOptimizer(0.01)
    train_op = optim.minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for j in range(dry_run + num_iter):
        if j == dry_run:
            start = time.time()

        if train:
            sess.run(train_op)
        else:
            sess.run(outputs)

    dura = (time.time() - start) / num_iter  # time of ONE iteration
    gflops = T * 4 * (N * H * D * 2 + N * H * H * 2) / 1e9
    GFLOPS = gflops / dura  # giga floating-point operations per second
    SPS = N / dura  # number of processed sentences per second
    print("size = %s, duration = %.4f, gflops = %.4f, GFLOPS = %.4f, SPS = %.4f" % (size, dura, gflops, GFLOPS, SPS))
