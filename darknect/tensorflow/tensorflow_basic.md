
# 1.矩阵操作
## 1.1矩阵生成

     这部分主要将如何生成矩阵，包括全０矩阵，全１矩阵，随机数矩阵，常数矩阵等
### 1.  tf.ones（全1） | tf.zeros（全0）

      tf.ones(shape,type=tf.float32,name=None)
      tf.zeros([2, 3], int32)
      用法类似，都是产生尺寸为shape的张量(tensor)


      sess = tf.InteractiveSession()
      x = tf.ones([2, 3], int32)
      print(sess.run(x))
      #[[1 1 1],
      # [1 1 1]]



### 2. tf.ones_like （全1）| tf.zeros_like （全0）  按模型生成

      tf.ones_like(tensor,dype=None,name=None)
      tf.zeros_like(tensor,dype=None,name=None)
      新建一个与给定的tensor类型大小一致的tensor，其所有元素为1和0

      tensor=[[1, 2, 3], [4, 5, 6]] 
      x = tf.ones_like(tensor) 
      print(sess.run(x))
      #[[1 1 1],
      # [1 1 1]]



### 3. tf.fill  填充

      tf.fill(shape,value,name=None)
      创建一个形状大小为shape的tensor，其初始值为value

      print(sess.run(tf.fill([2,3],2)))
      #[[2 2 2],
      # [2 2 2]]



### 4. tf.constant

      tf.constant(value,dtype=None,shape=None,name=’Const’)
      创建一个常量tensor，按照给出value来赋值，可以用shape来指定其形状。value可以是一个数，也可以是一个list。
      如果是一个数，那么这个常亮中所有值的按该数来赋值。
      如果是list,那么len(value)一定要小于等于shape展开后的长度。赋值时，先将value中的值逐个存入。不够的部分，则全部存入value的最后一个值。

      a = tf.constant(2,shape=[2])
      b = tf.constant(2,shape=[2,2])
      c = tf.constant([1,2,3],shape=[6])
      d = tf.constant([1,2,3],shape=[3,2])

      sess = tf.InteractiveSession()
      print(sess.run(a))
      #[2 2]
      print(sess.run(b))
      #[[2 2]
      # [2 2]]
      print(sess.run(c))
      #[1 2 3 3 3 3]
      print(sess.run(d))
      #[[1 2]
      # [3 3]
      # [3 3]]
