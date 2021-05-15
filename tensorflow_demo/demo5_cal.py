import tensorflow as tf

"""
    变量与常量
"""

# w = tf.Variable(3)
w = tf.constant(3)

x = w + 3
y = x + 5
z = x * 3

# 方式一
with tf.Session() as sess:
    # w 值作为变量需要初始化
    # w.initializer.run()
    print(sess.run(y))
    # 这里为了去计算 z，又重新计算了x和w,除了Variable值，tf是不会缓存其他比如contant等的值的
    # 一个Variable的生命周期是当它的initializer运行的时候开始，到会话session.close的时候结束
    print(sess.run(z))

print('====' * 10)

# 方式二：推荐
# 如果我们想要有效的计算y和z，并且又不重复计算w和x两次，我们必须要求Tensorflow 算 y 和 z 在一个图里
with tf.Session() as sess:
    y_res, z_res = sess.run([y, z])
    print(y_res)
    print(z_res)
