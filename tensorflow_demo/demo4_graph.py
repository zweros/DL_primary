import tensorflow as tf

"""
   计算图
"""

#  创建任何节点默认都会加入默认图中
x = tf.Variable(3, name='x')
print(x.graph is tf.get_default_graph())

graph = tf.Graph()
x3 = tf.Variable(4, name='x3')

# 大多数情况下上面运行的很好，有时候或许想要管理多个独立的图
# 可以创建一个新的图并且临时使用with块是的它成为默认的图
with graph.as_default():
    x2 = tf.Variable(5, name='x2')

x4 = tf.Variable(6, name='x4')

print(x2.graph is graph)  # True
print(x2.graph is tf.get_default_graph())  # False

print(x3.graph is tf.get_default_graph())  # True
print(x4.graph is tf.get_default_graph())  # True
