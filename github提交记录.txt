cd 目标文件夹
echo "# MVsion" >> README.md  # 添加文件
git init #初始化
git config --global user.email "you@example.com"  # 邮箱
git config --global user.name "Your Name"         # 用户名
git add README.md            # 添加文件
git commit -m "first commit" # 提交 
git remote add origin https://github.com/Ewenwan/MVision.git #增加一个远程服务器端版本库
git push -u origin master # 将本地文件提交到Github。

提交所有文件:
git clone https://github.com/Ewenwan/PyML.git #复制本地
将其他文件全部 放入 PyML 文件夹 
git add .
git commit -m "add all files"
git push -u origin master # 将本地文件提交到Github


git status  查看 文件修改情况
git add 对应文件
git commit -m "add files"
git push -u origin master # 将本地文件提交到Github
