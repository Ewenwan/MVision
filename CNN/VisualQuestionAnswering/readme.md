# Visual Question Answering(VQA)视觉问答

[【自然语言处理】--视觉问答（Visual Question Answering，VQA）从初始到应用](https://blog.csdn.net/LHWorldBlog/article/details/81124981)

视觉问答（Visual Question Answering，VQA），是一种涉及计算机视觉和自然语言处理的学习任务。这一任务的定义如下： A VQA system takes as input an image and a free-form, open-ended, natural-language question about the image and produces a natural-language answer as the output[1]。 翻译为中文：一个VQA系统以一张图片和一个关于这张图片形式自由、开放式的自然语言问题作为输入，以生成一条自然语言答案作为输出。简单来说，VQA就是给定的图片进行问答。

VQA系统需要将图片和问题作为输入，结合这两部分信息，产生一条人类语言作为输出。针对一张特定的图片，如果想要机器以自然语言来回答关于该图片的某一个特定问题，我们需要让机器对图片的内容、问题的含义和意图以及相关的常识有一定的理解。VQA涉及到多方面的AI技术（图1）：细粒度识别（这位女士是白种人吗？）、 物体识别（图中有几个香蕉？）、行为识别（这位女士在哭吗？）和对问题所包含文本的理解（NLP）。综上所述，VQA是一项涉及了计算机视觉（CV）和自然语言处理（NLP）两大领域的学习任务。它的主要目标就是让计算机根据输入的图片和问题输出一个符合自然语言规则且内容合理的答案。

