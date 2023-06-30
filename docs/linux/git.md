# Git

> Reference: [Learn Git Branching](https://learngitbranching.js.org/)
## 基本操作
  
```bash
# 查看仓库
git remote -v

# 更改远程url
git remote set-url origin git@github.com:csl122/anything.git
```
## add and commit
    
```bash
git add <file>: 将指定的文件添加到暂存区。例如，git add readme.txt
git add . : 将当前目录下的所有文件添加到暂存区，包括新建的和修改过的文件
git add -u : 将已修改和已删除的文件添加到暂存区，不包括新建的文件
git add -A : 将所有的改动文件添加到暂存区，包括新建的、修改过的和已删除的文件
git add --interactive : 类似于--patch选项，但是以交互式方式显示文件的改动，可以选择性地添加到暂存区

git commit -m "commit message": 提交暂存区的改动并添加提交描述信息
git commit -a: 将所有改动过的文件添加到暂存区，并提交到本地仓库。需要注意的是，使用该命令时，新添加的文件不会被提交。
git commit --amend: 修改最近一次的提交。如果你发现刚刚的提交有错误或遗漏了一些改动，可以使用该命令进行修改。

```
## branch
    
```bash
git branch <branch_name>: 创建分支，创建一个名为 <branch_name> 的新分支。
git checkout <branch_name>: 切换分支，切换到 <branch_name> 分支。
git switch <branch_name>: 切换分支，切换到 <branch_name> 分支。
git checkout -b <branch_name>: 创建并切换到分支，创建一个名为 <branch_name> 的新分支，并切换到该分支。
git branch: 查看所有分支，显示当前仓库中所有的分支。
git branch -f <branch_name> <commit_id>: 强制移动分支的指向位置，将 <branch_name> 分支移动到指定的 <commit_id> 处。
git merge <branch_name>: 合并分支，将 <branch_name> 分支合并到当前分支。
git branch -d <branch_name>: 删除分支，删除名为 <branch_name> 的分支。
git branch -D <branch_name>: 强制删除分支，强制删除名为 <branch_name> 的分支。
git branch -m <new_branch_name>: 重命名分支，将当前分支重命名为 <new_branch_name>。
git branch -r: 查看远程分支，显示与当前仓库关联的远程分支。
git branch -a: 查看所有分支，显示当前仓库中的所有分支，包括本地和远程分支。
git push origin <branch_name>: 将本地分支推送到远程，将当前分支推送到名为 origin 的远程仓库。
git checkout -b <local_branch_name> origin/<remote_branch_name>: 获取远程分支到本地，创建一个名为 <local_branch_name> 的新本地分支，并将名为 <remote_branch_name> 的远程分支与之关联。
```

## merge
    
```bash
# currently in main branch
git merge bugFix : 在main的最新节点下面创建一个新的节点有两个branch的最新内容
```
## rebase
    
```bash
git rebase <branch>: 将当前分支以 <branch> 为基础进行变基。这将把当前分支上的提交移动到 <branch> 的最新位置，并更新当前分支的提交历史。
git rebase -i <commit>: 交互式变基，允许用户编辑提交历史。使用此命令时，git会打开一个文本编辑器，显示当前分支的提交历史，并允许用户重新排序、删除或合并提交

```
## HEAD
    

如果想看 HEAD 指向，可以通过 cat .git/HEAD 查看
如果 HEAD 指向的是一个引用，还可以用 git symbolic-ref HEAD 查看它的指向。
当我们checkout一个分支名的时候其实是: 
HEAD -> main -> C1
HEAD 指向 main， main 指向 C1
而当我们checkout一个节点的时候: git checkout C1
HEAD -> C1
    HEAD 指向 C1, main也指向 C1

```bash
git checkout <commit_hash>: 将HEAD指向对应的commit hash节点
```
## 相对引用 ~ 和 ^
    
```bash
1. ^ 符号：向上移动一个提交
    - commit^：表示给定提交的父提交。例如，HEAD^表示当前提交的父提交。
    - commit^n：表示给定提交的第n个父提交。例如，HEAD^2表示当前提交的第二个父提交（对于合并提交）。

2. ~ 符号：向上移动多个提交
    - commitn：表示给定提交的n代父提交。例如，HEAD~3表示当前提交的第三代父提交。

这些符号可以与各种git命令和引用一起使用，例如分支名、标签名、commit哈希或引用操作符（例如HEAD）。

以下是一些示例：

3. HEAD^: 表示当前提交的父提交。
4. HEAD^^：表示当前提交的父提交的父提交。
5. HEAD~3：表示当前提交的第三代父提交。
6. branch_name^：表示分支名称对应提交的父提交。
7. tag_name~2：表示标签名称对应提交的第二代父提交。
如果是merge后的结果有两个parent, 可以通过^来给定哪一个parent
8. commit_hash^2：表示给定提交哈希的第二个父提交。
```
## 撤销变更
    
```bash
git reset 一般用来回退本地的修改, 将当前分支指针指向之前, 之后的更改会以未加入暂存区的形式显示
git revert 常用来回退origin上的修改, 新建一个commit, 该commit中是之前做过的事情逆向变化

git reset --soft [commit]：保留之前的提交，并将之后的提交放入暂存区，我们可以重新修改并提交。
git reset --mixed [commit]：保留之前的提交，并将之后的提交放入工作区，我们可以重新修改，并将修改后的内容重新提交。
git reset --hard [commit]：将之前的提交和之后的提交全部丢弃，回到指定的提交状态，我们会丢失之前的修改。

git revert HEAD: HEAD原本所指向的C2后出现了新提交记录 C2’ 引入了更改, 这些更改刚好是用来撤销 C2 这个提交的。也就是说 C2’ 的状态与 C1 是相同的。
```
## cherry-pick
    
```bash
git cherry-pick 允许你从一个分支（或多个分支）中选择一个或多个提交，并将它们应用于当前分支。
git cherry-pick <commit>
git cherry-pick <commit1> <commit3> <commit2> 按照132的顺序接到当前branch后面
git cherry-pick <start-commit>..<end-commit> 合并一个范围的提交
注意, 即便给的是branch name, 拼接到后面的也是指向的那一个节点的commit而已
```
## 交互式rebase
    
```bash
git rebase -i <commit>: 交互式变基，允许用户编辑提交历史。使用此命令时，git会打开一个文本编辑器，显示当前分支的提交历史，并允许用户重新排序、删除或合并提交
git rebase -i <base> <branch>
git rebase -i HEAD~4
可以自己调整HEAD之前的这些东西, omit调, 或者调整顺序, 都可以
```
## 提交技巧
    
```bash
如果我们在fix bug之前有很多print的东西, 但是我们最后不想要这些print的commit, 
我们可以只获取bugFix 这个节点的commit来放到main里面, 可以用两种方式
git rebase -i main bugFix : 将中间无用节点删去, based on main, 与bugFix一起生成一条新路

git checkout main
git cherry-pick bugFix : 仅将bugFix放到main后面

```
## tag and describe
    
```bash
git tag -a v1.0 -m "Version 1.0"
git push origin v1.0 : 推送标签至远程仓库
git tag -d v1.0 : 这会删除名为v1.0的标签

git describe <ref>
<ref> 可以是任何能被 Git 识别成提交记录的引用，如果你没有指定的话，Git 会使用你目前所在的位置（HEAD）。
<tag>_<numCommits>_g<hash>
```
## fetch
    
```bash
fetch会将远程没有下载到本地的commit下载到本地, 并将远程的oigin/main指针指向远程main的位置
1. 从远程仓库下载本地仓库中缺失的提交记录
2. 更新远程分支指针(如 o/main)

git fetch origin foo : 仅下载远程仓库中foo分支中的最新提交记录，并更新了 o/foo
git fetch origin source:dest 将远程的source branch 拉到本地的dest branch上, 如果source为空, 本地的dest会被删除
git fetch origin :bugFix : 在本地新建一个bugFix分支
```
## pull
    
```bash
git pull
== git fetch + git merge
git pull --rebase
== git fetch + git rebase origin/main
```
## push
    
```bash
git push <remote> <place>
git push origin main : 切到本地仓库中的“main”分支，获取所有的提交，再到远程仓库“origin”中找到“main”分支，将远程仓库中没有的提交记录都添加上去，搞定之后告诉我

git push origin foo~:main : 可以将 foo^ 解析为一个位置，上传所有未被包含到远程仓库里 main 分支中的提交记录。
git push origin main:newBranch : 可以推送一个新的branch
git push origin :side : 删除远程中的side分支
```
## pull request

 - 如果远程不允许直接push, 而需要pull request来更新main分支的话
 - 新建一个feature分支, 将本地的修改放到这个branch中, 然后推送到远程
 - 然后reset本地的main和远程的一致, 以防止之后pull什么的出现麻烦

## remote tracking

- 直接了当地讲，`main` 和 `o/main` 的关联关系就是由分支的“remote tracking”属性决定的。`main` 被设定为跟踪 `o/main` —— 这意味着为 `main` 分支指定了推送的目的地以及拉取后合并的目标。
- 当你克隆仓库的时候, Git 就自动帮你把这个属性设置好了。
- 当你克隆时, Git 会为远程仓库中的每个分支在本地仓库中创建一个远程分支（比如 `o/main`）。然后再创建一个跟踪远程仓库中活动分支的本地分支，默认情况下这个本地分支会被命名为 `main`。

```bash
git checkout -b totallyNotMain o/main : 可以创建一个名为 totallyNotMain 的分支，它跟踪远程分支 o/main。
git branch -u o/main foo : 这样 foo 就会跟踪 o/main 了
git branch -u o/main: 如果当前就在 foo 分支上, 还可以省略 foo
```
