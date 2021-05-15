import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2

# 可以不分别对每个变量去进行初始化
# 并不立即初始化， 在 run 运行的时候才初始化
init = tf.global_variables_initializer()

# 创建一个计算图上下文环境
# 配置里面是把具体运行过程在哪里执行给打印出来
with tf.Session() as session:
    # 运行
    init.run()
    # 获取计算结果
    res = f.eval()
    print('res = ', res)
