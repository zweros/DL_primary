import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2

# 创建一个计算图上下文环境
# 配置里面是把具体运行过程在哪里执行给打印出来
with tf.Session() as session:
    # 碰到 session.run() 就会立即去调用计算
    # session.run(x.initializer)
    # session.run(y.initializer)
    x.initializer.run()  # 等价于 tf.get_default_session().run(x.initializer)
    y.initializer.run()
    res = f.eval()  # 等价于  res = tf.get_default_session().run(f)
    print('res = ', res)
