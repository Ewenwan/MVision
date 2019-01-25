# Keras 基于Python的深度学习库

[ Keras中文文档](https://keras-cn.readthedocs.io/en/latest/)

[ Keras教程](https://github.com/cdlwhm1217096231/keras_tutorials)   
    
    Keras是一个高层神经网络API，Keras由纯Python编写而成并基Tensorflow、Theano以及CNTK后端
    
[快速开始序贯（Sequential）模型](https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/)

[快速开始函数式（Functional）模型](https://keras-cn.readthedocs.io/en/latest/getting_started/functional_API/)
    
    
# 安装
    pip3 install tensorflow-gpu==1.8.0
    # Python语言用于数字图像处理 
    # scikit-image 是基于scipy的一款图像处理包，它将图片作为numpy数组进行处理，正好与matlab一样
    pip3 install scikit-image
    # Keras:基于Python的深度学习库
    # Keras是一个高层神经网络API，Keras由纯Python编写而成并基Tensorflow、Theano以及CNTK后端
    pip3 install keras
# 快速开始：30s上手Keras

    Keras的核心数据结构是“模型”，模型是一种组织网络层的方式。
    Keras中主要的模型是Sequential模型，
    Sequential是一系列网络层按顺序构成的栈。
    你也可以查看函数式模型来学习建立更复杂的模型

    Sequential模型如下

        from keras.models import Sequential

        model = Sequential()
        
    将一些网络层通过.add()堆叠起来，就构成了一个模型：
        from keras.layers import Dense, Activation

        model.add(Dense(units=64, input_dim=100))
        model.add(Activation("relu"))
        model.add(Dense(units=10))
        model.add(Activation("softmax"))

    完成模型的搭建后，我们需要使用.compile()方法来编译模型：
         model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    编译模型时必须指明损失函数和优化器，如果你需要的话，也可以自己定制损失函数。
    Keras的一个核心理念就是简明易用，同时保证用户对Keras的绝对控制力度，
    用户可以根据自己的需要定制自己的模型、网络层，甚至修改源代码。
   
        from keras.optimizers import SGD
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

    完成模型编译后，我们在训练数据上按batch进行一定次数的迭代来训练网络
        model.fit(x_train, y_train, epochs=5, batch_size=32)
    当然，我们也可以手动将一个个batch的数据送入网络中训练，这时候需要使用：    
        model.train_on_batch(x_batch, y_batch)
    随后，我们可以使用一行代码对我们的模型进行评估，看看模型的指标是否满足我们的要求：
        loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    或者，我们可以使用我们的模型，对新的数据进行预测：
        classes = model.predict(x_test, batch_size=128)
        
        
  
  
