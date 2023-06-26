# Natural Language Processing

tags:: IC, Course, ML, Uni-S10
alias:: NLP, 自然语言处理
Ratio: 7:3
Time: 星期一 16:00 - 18:00, 星期四 11:00 - 13:00

- ### 容易忘记的东西 & 新东西
	- BERT的embedding用的是Token Embeddings 是随网络一起训练的, 用的是nn.Embedding. 并没有用其他预训练的embedding 例如word2vec. torch.nn包下的Embedding，作为训练的一层，随模型训练得到适合的词向量。
	- Transformer中的很多层, 很多head, 会有各自的侧重点, 例如会每个词attend to 前一个, 后一个, 自己, 或者句号等等. 很神奇的是SEP的注意力会被放很多
	  collapsed:: true
		- ![image.png](../assets/image_1683417669330_0.png)
		- ![image.png](../assets/image_1683417686522_0.png)
		- Words in a sentence tend to attend to the special token [SEP] using BERT because [SEP] is used to indicate the end of one sentence and the beginning of another. By attending to [SEP], BERT can better understand the context and relationship between the two sentences. Additionally, the [SEP] token helps BERT to differentiate between different segments of the input text, which is important for tasks such as sentence classification and question answering.
	- FlashAttention
		- 要点在于分解qkv, 每一个向量都分成很多片来分别计算, 这样子可以放到更快速的SRAM中, 而不是在HBM里面处理, 比如说q的前两个, k的前两个, v的前两个, 先一起进行一个att操作, 最后合并所有
		- ![image.png](../assets/image_1686432619825_0.png)
		- 在传统算法中，一种方式是将Mask和SoftMax部分融合，以減少访存次数。然而，FlashAttention则更加激进，它将从输入Q,区，V到输出 。的整个过程进行融合，以避免 S，卫矩阵的存储开销，实现端到端的延迟缩减。然而，由于输入的长度八通常很长，无法完全将完整的 Q,K,V,0及中间计算结果存储在SRAM中。因此，需要依赖HBM进行访存操作，与原始计算延迟相比没有太大差异，甚至会变慢（没具体测）。
	- sequence (句子) -> `inputs = tokenizer(sequence)` ->`encoded_sequence = inputs["input_ids"]` -> `[101, 138, 18696, 155, 1942, 3190` (词典中的indices) -> `tokenizer.decode` -> sequence
		- tokenizer.tokenizer(sequence) 会输出断开的词汇列表`['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']`
		- input_ids 是encoder 模型的输入
		- [Glossary](https://huggingface.co/docs/transformers/glossary#input-ids)
		- ```python
		  # %%
		  from PIL import Image
		  import requests
		  
		  from transformers import CLIPTokenizer, CLIPTextModel
		  
		  version = "openai/clip-vit-large-patch14"
		  
		  transformer = CLIPTextModel.from_pretrained(version).eval()
		  tokenizer = CLIPTokenizer.from_pretrained(version)
		  
		  # %%
		  prompts = ["a photo of a cat", "a photo of a dog"]
		  
		  batch_encoding = tokenizer(prompts, truncation=True, max_length=77, return_length=True,
		                                          return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
		  
		  tokens = batch_encoding["input_ids"].to('cpu')
		  out = transformer(input_ids=tokens).last_hidden_state
		  
		  # %%
		  print(out.shape)
		  torch.Size([2, 77, 768])
		  
		  print(batch_encoding.keys())
		  dict_keys(['input_ids', 'length', 'attention_mask'])
		  
		  print(tokenizer.decode(batch_encoding["input_ids"][0]))
		  <|startoftext|>a photo of a cat <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>
		  
		  print(tokenizer.tokenize(prompts[0]))
		  ['a</w>', 'photo</w>', 'of</w>', 'a</w>', 'cat</w>']
		  
		  print(transformer(input_ids=tokens).keys())
		  odict_keys(['last_hidden_state', 'pooler_output'])
		  ```
- ## Notes
	- 术语表
	  collapsed:: true
		- [[LM]]: Language model
		- [[NLI]]: Natural Language Inference, 给出两句话之间的关系
		- [[TF-IDF]]: Term Frequency-Inverse Document Frequency
		- [[FFNN]]: Feed Forward Neural Network
		- [[RNN]] : Recurrent Neural Network
		- [[LSTM]]: Long Short-Term Memory
		- [[BPTT]]: back-propagate through time
		- [[BiRNN]]: BiDirectional RNN
		- [[Auto-regressive]]: 自回归
		- [[MT]]: Machine Translation
		- [[Attention]]: 注意力机制
		  collapsed:: true
			- a dynamic weighted average
			- In the immediate context, it allows us to dynamically look at individual tokens in
			  the input and decide how much weighting a token should have with respect to the current timestep of decoding.
			- Types of attention
			  collapsed:: true
				- additive/MLP
				- Multiplicative
				- Self-attention
		- [[MLP Attention]]: 加性attention
		  collapsed:: true
			- {{embed ((63e1a7c2-05bb-4762-b60f-e7ca15370cd7))}}
		- [[Self-attention]]: 自注意力
		  collapsed:: true
			- ![image.png](../assets/image_1675865377160_0.png)
			- 给进一个序列, 序列中有若干词语, 词语们已经被表示成了embeddings, 为了得到每个词语在语境中的embedding, 我们进行self-attention, 每个词语本身的embedding用可学习的w转换成kqv, 用q 去query其他词语的k来获得score, 用这个softmax以后的score来合成自己的新value, 作为它的具有语境意义的新表示.
		- [[Multi-head Self-attention]]: 多头自注意力
		  collapsed:: true
			- 在这个基础上用了多个attention head, 不同的weight来重新建立qkv, 以期望获得注意力机制组合使用查询、键和值的不同 子空间表示(representation subspaces)可 能是有益的。
			- 希望模型可以基于相同的注意力机制学习到不同的行 为，然后将不同的行为作为知识组合起来，捕获序列内各种范围的依赖关系(例如，短距离依赖和⻓距离依 赖关系)。
			- ![image.png](../assets/image_1675866657287_0.png)
		- [[BERT]]: Bidirectional Encoder Representations from Transformers
		  collapsed:: true
			- BERT是一个由Transformer而来的双向编码器, 仅包含编码过程, 目的是仿照cv领域的feature extractor, 做到能够给下游任务复用, 因此更像是一个文本理解器, 运用的也是多头自注意力, 其实就是transformer的块block.
			- 不同之处在于, BERT使用了segment embedding片段嵌入和可学习的positional embedding, BERT的输入需要预先处理过, 可以是一句话, 也可以是两句话, 第一句话segment 会标记成0000, 第二句话会标记成1111. 输入的最前面会prepend 上<cls> token, 用来后续输出分类任务用, 每句话的结尾会append上<sep> token 用来表示句子之间的分割. 进入网络的真正输入是embedded tokens + embedded segments + embedded position.
			- segment embedding: 用0标记第一句话的token, 1标记第二句话的token; 例如[00001111]
			- BERT对与词语遮蔽预测(给定mask位来对vocab预测概率) 和下一句对不对预测进行了预训练, 如果我们要进行fine tuning 工作的话只需要把BERT的最后层拿掉, 只拿第一个class token的学习结果
			- 例: vovab = 10000, num_hiddens = 768; 输入序列长度为8, 一共两个samples batch size = 2
			- BERT会take (2, 8)的tokens, 对他进行一系列embedding, 输入到transformer blocks里, 输出(2, 8, 768)
		- [[MLM]]: Masked Language Modeling
		- [[BPE]]: Byte Pair Encoding, 一种用于把文本中词语分成subparts来进行stemming 和lemmatisation的手段
		- [[ELMo]]: Embeddings from Language Models, 两个从前到后和从后到前的recurrent language model
		- [[LLM]]: Large Language Model
		- [[Zero-shot]]: 给出instruction without examples
		- [[Few-shot]]: 给出instruction with few examples
		- [[CoT]]: chain-of-thought
		- [[RLHF]]: Reinforcement learning from human feedback
		- [[POS]]: Part of Speech, 词性的意思, 每个元素的tag, label, 有很多种标准
		- INTJ (interjection): psst, ouch, hello, ow, 感叹词
		- PROPN (proper noun): UK, Jack, London, 专有名词
		- AUX (auxiliary verb): can, shall, must, 助动词
		- CCONJ (coordinating conjunction): and, or, 连接词
		- DET: (determiner): a, the, this, which, my, an, 限定词
		- [[NER]]: Named Entity Recognition, 又称作**专名识别**、**命名实体**，是指识别文本(中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等，以及时间、数量、货币、比例数值等文字。
		- [[HMM]]: Hidden Markov Model, 隐马尔可夫模型, Hidden Markov Chains: states are not given, but hidden
		  collapsed:: true
		  ○ Words are observed
		  ○ POS are hidden
			- 有隐含的state来决定概率的, 隐含的states可以从observation中推断出来
		- [[Viterbi]]: **维特比算法**（英语：Viterbi algorithm）是一种[动态规划](https://zh.wikipedia.org/wiki/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92)[算法](https://zh.wikipedia.org/wiki/%E7%AE%97%E6%B3%95)。它用于寻找[最有可能产生观测事件序列](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E5%90%8E%E9%AA%8C%E6%A6%82%E7%8E%87)的**维特比路径**——隐含状态序列，特别是在马尔可夫信息源上下文和隐马尔可夫模型中。
		- [[CNF]]: Chomsky Normal Form, X -> YZ, X -> w
		- [[CKY]] algorithm: bottom-up的动态规划算法, 用于句子成分分析, 列表合成, 再合成
		- [[PCFG]]: Probabilistic/stochastic phrase structure grammar is a
		  context-free grammar PCFG = (T, N, S, R, q), where:
		  ● T is set of terminals
		  ● N is set of nonterminals
		  ● S is the start symbol (non-terminal)
		  ● R is set of rules X→ ,where X is a nonterminal and is a sequence of terminals & nonterminals
		  ● q = P(R) gives the probability of each rule
		-
	- Week2 intro & word representations, word2vec, skip-gram, CBOW
	  collapsed:: true
		- Lecture 1 Introduction
		  collapsed:: true
			- Natural language processing
			  collapsed:: true
				- NLP is the processing of natural language by computers for a task
				- Natural language is a language that has developed naturally in use
				- Ultimate goal: systems to understand NL (in written language), execute requested tasks & produce NL
				- Key words: Machine Learning, Linguistics, Cognitive Science, Computer Science
			- Subtasks of language understanding
			  collapsed:: true
				- Lexicon
				  collapsed:: true
					- 词汇
					- morphological analysis 形态学分析
					  collapsed:: true
						- What is a ‘word’ and what is it composed of?
						  collapsed:: true
							- segmentation (分割词汇), Word normalisation (将不同形式的词语normalise, 例如大小写, 有没有点, 连不连起来, 英式美式),
							- Lemmatisation (单复数, ing等到base form), Stemming (reduce 到 root, 例如connect, connction),
							- Part-of-speech tagging (识别词语的类别, 例如名词, 代词)
							- Morphological analysis (recognise/generate word variants): Number, gender, tense, 前缀, 后缀
				- Syntax
				  collapsed:: true
					- 句法, 语法, 句子的结构, how words are put together
					- ![image.png](../assets/image_1673971995738_0.png)
					- ![image.png](../assets/image_1673972049160_0.png)
				- Semantics
				  collapsed:: true
					- 语义, 词语句子的意思, Meaning of words in context
					- Word Sense Disambiguation: given a word and its context, assign the correct meaning given a set of candidates, 正确理解一个词在一个语境中的含义
					- Understanding who did what to whom, when, where, how and why? 理解谁对谁做了什么事
					  collapsed:: true
						- Semantic role labelling: assign semantic roles to words in sentence; fail a sentence if that is not possible 对语义角色进行标记
				- Discourse
				  collapsed:: true
					- 论述, 句子与句子间的联系, How do sentences relate to each other?
					- References & relationships within and across sentences. 前后句子之间的联系, 代词的指代
				- Pragmatics
				  collapsed:: true
					- 语用论, 语言的用意, What is the intent of the text? How to react to it?
					- ![image.png](../assets/image_1673972264925_0.png)
					- 例如上图问的不是yes or no, 而是有其他的隐含意思, 指示对方做些什么的
				- 曾经这些是被单独分别解决的, 现在我们用一个端到端模型来处理所有
			- History
			  collapsed:: true
				- 19050: Foundational work: automata theory, information theory
				- 1960-1970: Rule-based models
				- 1980-1990: the empirical revolution
				- 2000’s: better statistical and machine learning methods
				- 2014+: the neural revolution
				- 2018+: the era of pre-trained language models
			- Why ML
			  collapsed:: true
				- Creation and maintenance of linguistic rules often infeasible or impractical
				- Learning functions from data instead creating rules based on intuition
				- Optimizing weights and balances automatically instead of tuning
				  them manually
				- Data/examples can be abundant for certain applications and languages, e.g. human translations, movie reviews, ...
			- Why DL
			  collapsed:: true
				- ![image.png](../assets/image_1673973364076_0.png)
			- Applications
			  collapsed:: true
				- Spam filtering
				- Translation
				- Grammar correction
				- text prediction
				- Stable diffusion
				- ChatGPT
		- Lecture 1.2 Word Representations
		  collapsed:: true
			- Words as vectors, word embeddings, skip-gram, Byte pair coding
			- Similar words have similar vectors, 相似含义的词语, 有相近的vector
			- 1-hot vectors: 独热编码, 170k个单词, 170k个bit, 相似的词语和完全不同的词语有相同的distance
			- WordNet: map words to broader concepts
			  collapsed:: true
				- “cat”, “kitten” →”feline mammal”
				  “London”, “Paris”, “Tallinn” → “national capital” “yellow”, “green” → “colour”
				- 但是, rely on manual curate, miss out rare and new meanings, vectors are still 互相垂直距离相同, Disambiguating word meanings is difficult
			- Distributed vectors, distributed representations: Each element represents a property and these are shared between the words.
			  collapsed:: true
				- ![IMG_1109.jpeg](../assets/IMG_1109_1674126990982_0.jpeg)
				- ![image.png](../assets/image_1674127094866_0.png)
				- cos是similarity, 越大越好, 用于高纬; L2 norm是距离, 越小越好
				- 但是我们不想知道这些label, 希望他们可以自动生成
			- Distributional hypothesis
			  collapsed:: true
				- Words which are similar in meaning occur in similar contexts. 相似的词语会有相近的陪伴词们
				- Count-based vectors
				  collapsed:: true
					- count how often a word occurs together with specific other words (within a context window of a particular size). 通过一个与附近其他词语同时出现的词语的次数, 例如 magazine 和newspaper和 read经常一起出现
					- TF-IDF:
					  collapsed:: true
						- down-weighting 那些经常出现在所有地方的词语
						- ![image.png](../assets/image_1674127543538_0.png)
					- 非常sparse, very large 170k
					- 所以我们最好希望能有个固定大小的parameters, 让神经网络去学, 300-1k的长度
			- Word2vec
			- Continuous Bag-of-Words (CBOW)
			- Skip-gram
			  collapsed:: true
				- Predict the context words based on the target word wt, 或者根据wt, 预测其周围的词语
				- target x 的1-hot vector给到网络, 用weight W得到他的embedding, 再用W'得到整个词典长度的vector y, 用softmax找到可能的词汇, 用的是categorical cross-entropy
				- 整个模型只有两个W, W'
				- ![image.png](../assets/image_1674128283586_0.png)
				- 但是, 我们需要对整个词典进行softmax 的计算, 这一点downside,
				- 优化
				  collapsed:: true
					- 不用1-hot, 直接获取一行embedding
					- negative sampling
					  collapsed:: true
						- 让可能的词语可能性最大化, 不应该的词语最小化
						- ![image.png](../assets/image_1674128885888_0.png)
						- 小的数据集, 需要5-20个负面词语, 大的只需要2-5个, 大大减少了参数量
			- Analogy Recovery, 相似度恢复
			  collapsed:: true
				- 例如queen = king - man + woman
			- Multilingual word embeddings
			  collapsed:: true
				- 不同语言词语放到一起
			- Multimodal word embeddings
			  collapsed:: true
				- 图和词语
			-
			-
		- Tutorial Preprocessing and embedding
		  collapsed:: true
			- Q: What is a **token** and why do we need to **tokenize**?  
			  A: A **token** is a string of contiguous characters between two spaces, or between a space and punctuation marks. For segmentation.
			- **Capitalization** inducing **sparsity** in the dataset.
			  A: Process of normalization (removing capitalization, etc.)
			- 几个**Pre-processing methods**, 虽然他们不在neural-based NLP anymore, 但extensively employed in rule-based and statistical NLP.
			  collapsed:: true
				- ### Stop Word removal
				  
				  Stop words are generally the most common words in the language which who's meaning in a sequenece is ambiguous. Some examples of stop words are: The, a, an, that.
				- ### Punctuation removal
				  collapsed:: true
				  
				  Old school NLP techniques (and some modern day ones) struggle to understand the semantics of punctuation. Thus, they were also removed.
					- ```python
					  import re # regex
					  re_punctuation_string = '[\s,/.\']'
					  tokenized_sentence_PR = re.split(re_punctuation_string, sentence)
					  tokenized_sentence_PR = list(filter(None, tokenized_sentence_PR)) # return none empty strings
					  ```
				- ### Stemming
				  collapsed:: true
				  
				  In the case of stemming, we want to normalise all words to their stem (or root). The stem is the part of the word to which affixes (suffixes or prefixes) are assigned. Stemming a word may result in the word not actually being a word. For example, some stemming algorithms may stem [trouble, troubling, troubled] as "troubl". 把词语简化到其词根
					- ```python
					  from nltk.stem import PorterStemmer
					  porter.stem(word)
					  ```
				- ### Lemmatization
				  collapsed:: true
				  
				  Lemmatization attempts to properly reduce unnormalized tokens to a word that belongs in the language. The root word is called a **lemma**, and is the canonical form of a set of words. For example, [runs, running, ran] are all forms of the word "run. 把词语简化到其基础形态, the lemmatizer requires parts of speech (POS) context about the word it is currently parsing.
					- ```python
					  from nltk.stem import WordNetLemmatizer
					  wordnet_lemmatizer = WordNetLemmatizer()
					  wordnet_lemmatizer.lemmatize(word, pos="v")
					  ```
				- Note that other NLP tools, such as [SpaCy](https://spacy.io/) or [Stanza](https://stanfordnlp.github.io/stanza/) are popular alternatives which provide higher levels of abstractions than NLTK.
			- #### Implementing a Word2Vec algorithm from scratch
			  collapsed:: true
				- collapsed:: true
				  1. 创建vocabulary
					- ```
					  vocabulary = []
					  vocabulary.append(token)
					  ```
				- collapsed:: true
				  2. 创建word2idx 和 idx2word 两个字典
					- ```python
					  word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
					  idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
					  ```
				- collapsed:: true
				  3. 创建 look-up table, 词语到1hot
					- ```
					  def look_up_table(word_idx):
					     x = torch.zeros(vocabulary_size).float()
					     x[word_idx] = 1.0
					     return x
					  ```
				- collapsed:: true
				  4. Extracting contexts and the focus word
					- 对句子[0, 2, 3, 6, 7] (数字是字典里的位置)里的每个词, 找到对应的window size范围内的周围的词, 写成[这个词index, 周围词index]
				- collapsed:: true
				  5. 用w1 来映射到embedding, w2映射到词汇表每个词的相似度概率
					- y 是周围词index, 会被网络强化, 相当于哪一类
					- 最后就是输入进某个词语, 可以给出词汇表中每个词与其的相似程度
				- 6. skipgram 中用了negative sampling, 来减小非相关词的可能性
			- **预训练模型**, 如glove, 会有6billion个tokens, 40k个词汇, 有多个dimension版本, 如50, 300等, 已经事先给每个词语做好了word representation
	- Week3 classification, 朴素贝叶斯, 罗辑回归, CNN, debiasing
	  collapsed:: true
		- Lecture 2 Classification
		  collapsed:: true
			- 什么是classification:
			  collapsed:: true
				- $y_{predicted} = argmax_yP(y|x)$
				- 预测的label是让given x这个数据集时, 发生概率最高的那个labely, 上面的式子也叫做后验概率
				- NLP tasks
				  collapsed:: true
					- ![image.png](../assets/image_1674573303934_0.png)
			- Naive Bayes Classifier [[朴素贝叶斯]]
			  collapsed:: true
				- 例如数据集中有几句话, 我们只对某三个词语进行统计并且作为feature, 每句话都有一个label, good or bad. 这个数据集中一共五句话, 三个good, 两个bad, 因此先验p(y)就是3/5和2/5对于good类和bad类. 我们也可以算出, given是good类, 出现某个词语的概率, 这个叫做likelihood, 方法是good类中总共的词语数量作为分母, 分子是这个词语的出现次数. 通常情况下会smoothing
				- ![image.png](../assets/image_1674577384800_0.png)
				- 因为我们假设各个词语独立分布, 因此总的似然p(x|y)可以用连加log后各自的值来表现
				- ![image.png](../assets/image_1674577639475_0.png)
			- 这些方法有问题的, 因为直接用词频, 实际上并没有考虑到限定词的语义相反问题
			  collapsed:: true
				- Conditional independence assumption
				- Context not taken into account
				- New words (not seen at training) cannot be used
			- Input representation
			  collapsed:: true
				- 可以使用bag of words来表示一句话, 例如统计一句话中的词语们的出现频率来作为一个vector
			- Discriminative vs Generative algorithms #card #ML
			  collapsed:: true
				- 判别式模型通过生成判别boundary来判别新的数据的类别. 判别式模型是对条件概率建模，学习不同类别之间的最优边界，无法反映训练数据本身的特性，能力有限，其只能告诉我们分类的类别. 决策边界, SVM, NN, 回归, 决策树
				- 生成式模型通过生成一个符合数据的分布来进行概率判断, 在哪个类别中的概率更高. 一般会对每一个类建立一个模型，有多少个类别，就建立多少个模型。比如说类别标签有｛猫，狗，猪｝，那首先根据猫的特征学习出一个猫的模型，再根据狗的特征学习出狗的模型，之后分别计算新样本X跟三个类别的联合概率 P（y|x），然后根据贝叶斯公式：分别计算 P（y|x），选择三类中最大的 P（y|x）作为样本的分类。贝叶斯网络, 朴素贝叶斯, KNN
				- 两者的特点
				  collapsed:: true
					- **判别式模型特点：**
					- 判别式模型直接学习决策函数Y=f(X)，或者条件概率P（Y|X），不能反映训练数据本身的特性，但它寻找不同类别之间的最优分裂面，反映的是异类数据之间的差异，直接面对预测往往学习准确度更高。具体来说有以下特点：
					- 对条件概率建模，学习不同类别之间的最优边界。捕捉不同类别特征的差异信息，不学习本身分布信息，无法反应数据本身特性。学习成本较低，需要的计算资源较少。需要的样本数可以较少，少样本也能很好学习。预测时拥有较好性能。无法转换成生成式。**生成式模型的特点：**
					- 生成式模型学习的是联合概率密度分布P（X,Y），可以从统计的角度表示分布的情况，能够反映同类数据本身的相似度，它不关心到底划分不同类的边界在哪里。生成式模型的学习收敛速度更快，当样本容量增加时，学习到的模型可以更快的收敛到真实模型，当存在隐变量时，依旧可以用生成式模型，此时判别式方法就不行了。具体来说，有以下特点：
					- 对联合概率建模，学习所有分类数据的分布。学习到的数据本身信息更多，能反应数据本身特性。学习成本较高，需要更多的计算资源。需要的样本数更多，样本较少时学习效果较差。推断时性能较差。一定条件下能转换成判别式。总之，判别式模型和生成式模型都是使后验概率最大化，判别式是直接对后验概率建模，而生成式模型通过贝叶斯定理这一“桥梁”使问题转化为求联合概率。
			- Logistic regression
			  collapsed:: true
				- ![image.png](../assets/image_1674579729074_0.png)
				- 以0.5为界, 选择类别1 还是0
				- 也可以对每个类进行一次逻辑回归, 得到的值用softmax
			- 逻辑回归对比朴素贝叶斯
			  collapsed:: true
				- ![image.png](../assets/image_1674580057701_0.png)
			- Learnt feature representations - Neural networks
			  collapsed:: true
				- Inputs (very basic model), one-hot representation
				- Inputs (better model), Automatically learnt dense feature representations or Pre-trained dense representations
				- ![image.png](../assets/image_1674580810466_0.png){:height 296, :width 629}
				- Advantages: Automatically learned features, Flexibility to fit highly complex relationships in data
			- Document representation
			  collapsed:: true
				- 1. 可以用句子里面的所有词的每个dimension的均值average作为这个document这个dimension的值, work well, 但不够好
				  2. 不可以固定document words长度来形成一个表, 这样子一方面大小固定, 另一方面会有固定的word position信息
				     3.
			- Recurrent Neural Networks
			  collapsed:: true
				- ![image.png](../assets/image_1674581258504_0.png)
				- ![image.png](../assets/image_1674581303728_0.png)
				- W是hidden to hidden, H是input to hidden
				- Vanishing gradient problem
				  collapsed:: true
					- The model is less able to learn from earlier inputs
					- 遥远过去的gradient会消失或者爆炸, 因为tanh和sigmoid的导数在0-1和0-0.25之间
			- CNN
			  collapsed:: true
				- CNNs are composed of a series of convolution layers, pooling layers and fully connected layers
				- Convolution to detect
				- pooling to reduce dimensionality
				- FC train weights of learned representation for a specific task
				- ![image.png](../assets/image_1674581878894_0.png)
			- RNN vs CNN
			  collapsed:: true
				- CNNs can perform well if the task involves key phrase recognition, CNN主要识别关键phrase
				- RNNs perform better when you need to understand longer range dependencies, 是别的事更长远的dependency
		- Lecture 2.2 Model [[de-biasing]]
		  collapsed:: true
			- preventing a model learning from shallow heuristics, 阻止模型学习一些比较浅层过于显著的知识, 希望NLP模型可以更加好的泛化
			- 几种可能的策略
			  collapsed:: true
				- Augment with more data so that it ‘balances’ the bias, 对数据集增强, 使用更多的数据
				- Filter your data, 对数据集进行过滤操作, 减少bias属性
				- Make your model predictions different from predictions based on the bias (robust & shallow models), 使得我们的模型的预测不同于那些基于bias的预测, 也就是说撇开使用bias的预测
				- Prevent a classifier finding the bias in your model representations, 阻止分类器找到存在于模型中的bias
			- Bias指的是什么?
			  collapsed:: true
				- 可以是比如long sentence, 句子长度, 某些词语, 就是任何你不想让模型学习的一些浅层的没用的东西, 任何帮助我们泛化模型的东西, 这个是依赖于经验的. 这个bias不同于性别的bias, 性别这个时候可能更适合直接被hide, 而不是debias
			- Notation: 即将用到的两个标注
			  collapsed:: true
				- The probabilities of each class given our bias **(bi)** p(c|bias), 给定bias, 是这个class的概率, 我们称之为 Bias probability
				- probabilities from our model **(pi)**
			- 我们把上面几种可能的策略结合起来, 得到下面的这个策略 [[Product of Experts]] POE
			- ![image.png](../assets/image_1674755201283_0.png)
			- ![image.png](../assets/image_1674755252417_0.png)
			- 这个思路从高层来看, 就是希望我们的模型去学习除了bias以外的任何东西, 实现了去bias的目标, 也就是学习everything around the bias, 会少考虑bias的sample, feature
			- 从底层来看, 就是加入了来自于bias的惩罚项, 约束模型从bias中学东西
			- 那么我们如何找到这个bias probability bi呢
			  collapsed:: true
				- Target a specific known ‘biased’ feature
				- **creating a ‘biased’ model**
			- Creating a ‘biased’ model
			  collapsed:: true
				- 1. Use a model not powerful enough for the task (e.g. BoW model, or TinyBERT)
				     使用弱小的模型来预测, 找到一些simple且浅层的patterns
				  2. Use incomplete information (e.g. only the hypothesis in NLI)
				     使用不完全的信息
				  3. Train a classifier on a very small number of observations
				     用很小的观测集来进行训练
			- 除了上述的直接加bias到交叉熵里面的方法以外, 还有一种weight the loss of examples based on performance of our biased model
			  collapsed:: true
				- ![image.png](../assets/image_1674755794018_0.png)
				- bi值越大 这个1-bi 就越小, 把这个乘上loss也就越小; 意思就是说 bi如果越自信, loss就越小, 就越不用学习
			- ![image.png](../assets/image_1675123870257_0.png)
			- 正确的label是contradiction, p^是重新softmax后结合了b和p的概率结果, 会作为loss 的依据, 可以看到bias项被拉高了, 与真实label的差距变大了, loss也相应增加了, 模型会更加考虑惩罚这个label背后的逻辑, 例如简单的看两个句子之间overlapping的词汇数这种简单粗暴易出错的逻辑feature, 从而实现对bias的抵制, 真正模型学习中也就会减少对该特征的考虑, 更多考虑到unbiased的内容, 增加根据真正有用的feature预测到对的label的概率. 在真正的预测过程中, 我们只会使用中间的这个p. debiasing会有regularisation的作用, 因此类似于本训练数据集分布的测试集的结果会稍微比通常的训练结果差, 但是会对泛化的数据集有更好的效果
			- ![image.png](../assets/image_1675161709462_0.png)
		- Tutorial Text classification: Sentiment analysis using [[FFNN]]
		  collapsed:: true
			- 怎么训练一个Feed Forward Neural Network 来对句子的sentiment进行分类
			- 句子和label长这个样子
			- ```python
			  train = ['i like his paper !',
			           'what a well-written essay !',
			           'i do not agree with the criticism on this paper',
			           'well done ! it was an enjoyable reading',
			           'it was very good . send me a copy please .',
			           'the argumentation in the paper is very weak',
			           'poor effort !',
			           'the methodology could have been more detailed',
			           'i am not impressed',
			           'could have done better .',
			           ]
			  
			  train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
			  ```
			- 第一步先要把句子分成一个个token, 并且保证每个句子里的token数一致, 也就是都是最长的那句话的length, 采用0 padding
			- 然后对每个词语都进行字典归纳, 每个词语都有一个字典的index
			- 然后就把这些句子, 用这些词语在字典里的index表示出来了, 最后的效果就是10句话, list里首先有十个元素, 每个元素又是一个list, 这些list都是等长的词语在词典中的index们
			- ```python
			  tensor([[ 1,  2,  3,  4,  5,  0,  0,  0,  0,  0,  0],
			          [ 6,  7,  8,  9,  5,  0,  0,  0,  0,  0,  0],
			          [ 1, 10, 11, 12, 13, 14, 15, 16, 17,  4,  0],
			          [18, 19,  5, 20, 21, 22, 23, 24,  0,  0,  0],
			          [20, 21, 25, 26, 27, 28, 29,  7, 30, 31, 27],
			          [14, 32, 33, 14,  4, 34, 25, 35,  0,  0,  0],
			          [36, 37,  5,  0,  0,  0,  0,  0,  0,  0,  0],
			          [14, 38, 39, 40, 41, 42, 43,  0,  0,  0,  0],
			          [ 1, 44, 11, 45,  0,  0,  0,  0,  0,  0,  0],
			          [39, 40, 19, 46, 27,  0,  0,  0,  0,  0,  0]])
			  ```
			- 要放到神经网络里, 我们还需要给这些句子进行embedding, embedding函数会给句子中的每个词语一个embedding, （几句话，每句话的长度， 每个词语的embedding长度(batch size, max_sent_len, embedding dim)
			- 然后要把句子里的这么多词语的embedding整合成一个作为这个句子的表示, 我们用avg来进行这个操作, embedding 的每一个feature都是句中所有词语该feat的均值
			- 网络结构很简单, 可以随意设计, 几个fc relu 就好了,
	- Week4 Evaluation, language modeling, N-gram, RNN, LSTM
	  collapsed:: true
		- Lecture 2.3 Evaluation
		  collapsed:: true
			- Accuracy
			  collapsed:: true
				- ![image.png](../assets/image_1675162139398_0.png)
			- F1
			  collapsed:: true
				- ![image.png](../assets/image_1675162170636_0.png)
				- macro averaging: average of each class F1 scores, 每个类的F1单独求出以后, 直接求均值
				- micro averaging: TPs, TNs, FNs, FPs are summed across each class, F1的基本元素都各自加起来以后算一个F1
				  collapsed:: true
					- ![image.png](../assets/image_1675162434073_0.png)
					- 底下这堆, 其实就是dataset 的大小, 也其实就是accuracy了
		- Lecture 3 [[Language Modeling]]
		  collapsed:: true
			- What is language modeling? and why?
			  collapsed:: true
				- assigning probabilities to sequences of words. 给词序列分配可能性, 例如对下一个词语的预测和对挖空词的预测 (next word, masked word)
				  Language modelling is the task of training a computer to predict the probability of a sequence of words. It involves teaching the computer to understand and generate human language by using statistical and machine learning techniques. Essentially, it’s about teaching a computer to write and speak like a human.
				- 语言建模的意义在于我们需要generating language responses, rather than choosing a specific class, 生成语言反馈, 而非仅仅只是选择一个特定的类别
				- 应用如下
				  collapsed:: true
					- ● Word completion (on phones)
					  ● Machine translation
					  ● Summarization
					  ● Copilot coding assistants
					  ● Chatbots
					  ● And more....
			- N-gram models
			  collapsed:: true
				- Counting the likelihood of the next word
				  collapsed:: true
					- We aim to compute P(w|h) where: w is the word (or symbol) and h is the history
					- $P(w_n|w_{1}^{n-1})$ 利用w1 - wn-1 的信息去预测wn (full context)
					- ![image.png](../assets/image_1675163094346_0.png)
				- N-gram models approximate history by just the few last words, 这里的N就是指一共多少词
				  collapsed:: true
					- ![image.png](../assets/image_1675163205665_0.png)
					- $P(w_n|w_{1}^{n-1}) \approx P(w_n|w_{n-N+1}^{n-1})$
					- MLE as relative frequencies, 用在数据集中出现的频率作为似然
					- 语料库越大, 效果越好, 通常情况下Trigram 足够了
					- ![image.png](../assets/image_1675163481721_0.png)
				- Evaluating language models: Perplexity 困惑度
				  collapsed:: true
					- Estimate joint probability of an entire word sequence by multiplying together a number of conditional probabilities 通过joint probability 来计算整个word sequence的出现概率
					  collapsed:: true
						- ![image.png](../assets/image_1675164588330_0.png)
						- 注意这里condition一直都是从一到上一个
					- Bi-gram language model: probabilities
					  collapsed:: true
						- ![image.png](../assets/image_1675165013664_0.png)
						- 这里用的是一个历史记录作为condition
					- ![image.png](../assets/image_1675165068107_0.png)
					- Issue: longer output, lower likelihood, 长的输出会只有很少的似然值
					  Solution: perplexity: inverse probability of a text, normalized by the # of words
					- ![image.png](../assets/image_1675165595396_0.png)
					- 即prob的-1/n次方, It’s a measure of the surprise in a LM when seeing new text; 衡量了一个语言模型在看到新的text的时候的惊喜程度, 惊喜程度越低越好, 但是Perplexity is specific to the test-set
					- FAQ
					  collapsed:: true
						- If we are finding the perplexity of a single word, what is the best possible score? 1, perplexity最小最小就是1
						- If our model uniformly picks words across a vocabulary of size |V|, what is the perplexity of a single word? |V|, prob是1/|V|, 做inverse就是|v|
						- ![image.png](../assets/image_1675166098898_0.png)
				- Cross Entropy
				  collapsed:: true
					- ![image.png](../assets/image_1675165743814_0.png)
					- 对于我们的语言模型而言, T就是给定的training set, 语料库, q是一串词的出现概率
					- Perplexity就是以某个值为底的H对数
					  collapsed:: true
						- ![image.png](../assets/image_1675165964107_0.png)
						  collapsed:: true
							-
			- GPT3 tangent
			  collapsed:: true
				- 我们人工给定一些例子, 会影响gpt3的生成
				- Extrinsic vs Intrinsic evaluation
				  collapsed:: true
					- ● If the goal of your language model is to support with another task
					  ○ The best choice of language model is the one that improves downstream task performance the most (extrinsic evaluation)外部评估就是在下游任务的表现情况
					  ● Perplexity is less useful in this case (intrinsic evaluation) 内部评估就是在内部的表现情况
				- Evaluating GPT-3
				  collapsed:: true
					- ![image.png](../assets/image_1675166247007_0.png)
					- ![image.png](../assets/image_1675166256493_0.png)
			- Sparsity in N-gram models
			  collapsed:: true
				- 稀疏性怎么解决呢, 因为语言模型的语料库中会有很多不怎么出现的词语, 比如女王走之后就出现了对国王的称呼, 而在过去的几十年内都没有这个, 会被标识为<UNK>
				- Techniques to mitigate sparsity:
				  collapsed:: true
					- Add-1 smoothing
					  collapsed:: true
						- ![image.png](../assets/image_1675166362362_0.png)
						- ![image.png](../assets/image_1675166394788_0.png)
						- 对于那些稀疏的词语, 大量的其他的0变成1会导致原本高概率的东西大大降低概率
						- ● Easy to implement
						  ● But takes too much probability mass from more likely occurrences
						  ● Assigns too much probability to unseen events
						  ● Could try +k smoothing with a smaller value of k
					- Back-off
					  collapsed:: true
						- We could back-off and see how many occurrences there are of ‘royal highness’ 减少词, 减弱specification
						- ![image.png](../assets/image_1675166514023_0.png)
					- Interpolation
					  collapsed:: true
						- ![image.png](../assets/image_1675166531466_0.png)
		- Lecture 3.1 Neural language models, [[RNN]] & [[LSTM]]
		  collapsed:: true
			- Neural language models
			  collapsed:: true
				- Improvements of Neural-based language models
				  collapsed:: true
					- Avoids n-gram sparsity issue 避免了n-gram稀疏问题, 那些不怎么出现的低概率词语
					- Contextual word representations i.e. embeddings #Question
				- 4-gram Feed-forward LM (FFLM)
				  collapsed:: true
					- ![image.png](../assets/image_1675689441008_0.png)
					- 将三个上文词语embedding concat起来作为context输入以后激活得到输出层进行softmax得到词典中所有词的概率分布, 来决定下一个词是哪一个
					- 相对于普通3gram提升很大, 但是不敌RNN
				- RNN
				  collapsed:: true
					- ![image.png](../assets/image_1675691349757_0.png)
					- Many to one: sentiment classification
					- Many to many: Machine translation
					- many to many: video classification on frame level
					- ![image.png](../assets/image_1675691432247_0.png)
					- 其中使用的function和parameters都是一样的
					- ![image.png](../assets/image_1675691477170_0.png)
					- ![image.png](../assets/image_1675691526325_0.png)
					- In theory RNN retains information from the infinite past.
					  – All past hidden state has influence on the future state.
					- In practice RNN has little response to the early states.
					  – Little memory over what seen before.
					  – The hidden outputs blowup or shrink to zeros.
					  – The “memory” also depends on activation functions.
					  – ReLU and Sigmoid do not work well. Tanh is OK but still not “memorize” for too long.
					- Loss and back propagation through time (BPTT)
					  collapsed:: true
						- ![image.png](../assets/image_1675691956352_0.png)
						-
				- Vanilla RNNs for language modelling in classification
				  collapsed:: true
					- ![image.png](../assets/image_1675689685681_0.png)
					- 上图是用于分类工作的RNN, 每个时间点给入一个词语, 历史信息会通过f与当前输入结合处理给到下一个环节, 最后输出分类
					- 由于词语量的增加会导致每一层的激活函数一直累积起来, 导致导数会向0靠拢, 出现vanishing gradients problem
				- Vanilla RNNs: Many-to-many
				  collapsed:: true
					- Every input has a label: Language modelling -> predicting the next word
					- The LM loss is predicted from the cross-entropy losses for each predicted word:
					- ![image.png](../assets/image_1675689932905_0.png)
					- 下一个输入用当前输出的结果来feed into nn, 来计算cross entropy作为loss, 这么做可以让RNN模型学习根据历史信息和当前输入来获得下一个词语的概率, 从而不断预测下一个词语
					- [[Teacher Forcing]]
					  collapsed:: true
						- ![image.png](../assets/image_1675690408483_0.png)
						- 没有teacher forcing下一个输入将会是预测的chased, 但是teacher forcing会强制给定正确的输入 sat来guide nn学习到正确的概率分布, 加速学习过程
						- ![image.png](../assets/image_1675732079125_0.png)
						- During training, we use teacher forcing with a *ratio*. I.e. we well teacher force
						  ratio% of time. The ratio can be 100% (i.e. full teacher forcing), 50%, or you can even anneal it during training.
						- ![image.png](../assets/image_1675732174448_0.png)
						- Teacher forcing creates a phenomenon called [[Exposure Bias]].
						- Exposure Bias is when the data distributions the model is conditioned on vary
						  collapsed:: true
						  between training and inference
							- In this case, if we set teacher forcing to 100%, then the model is never
							  conditioned on its own predictions
							- So during inference, when the model uses the previously generated
							  words as context, it is different than what it has seen in training
						- It’s unclear how much of an issue exposure bias actually is. There are multiple
						  papers on either side of the argument.
					- Weight tying, reducing the no. parameters
					  collapsed:: true
						- 为了节约参数, 可以用相同的embedding weights, 一开始我们把one-hot变成了embedding, 最后的预测的embedding也用这个weights 来输出词语的概率
				- Bi-directional RNNs
				  collapsed:: true
					- 如何实现语法纠正? 如何利用前后文所有信息来做分类任务? 双向的RNN可以
					- ![image.png](../assets/image_1675690762021_0.png)
					- For classification: 拼接起正向和反向的输出来帮助分类
					  collapsed:: true
						- ![image.png](../assets/image_1675690789941_0.png)
					- Multi-layered RNNs: 多层RNN, 得到更多深层信息
					  collapsed:: true
						- ![image.png](../assets/image_1675690862974_0.png)
					- Bidirectional multi-layered RNNs: 多层加双向
					  collapsed:: true
						- ![image.png](../assets/image_1675690892555_0.png)
				- LSTM: Long Short Term Memory
				  collapsed:: true
					- 将memory分成了long term memory和current working memory
					- ![image.png](../assets/image_1675691590143_0.png)
					- RNN:
					  collapsed:: true
						- • Recurrent neurons receive past recurrent outputs and current input as inputs.
						  • Processed through a tanh() activation function
						  • Current recurrent output passed to next higher layer and next
						  time step.
					- ![image.png](../assets/image_1675691650495_0.png)
					  collapsed:: true
						- Constant Error Carousel
						  • Key of LSTM: a remembered cell state
						  • Ct is the linear history carried by the constant error carousel.
						  • Carries information through and only effected by a gate
						  • Addition of history (gated).
					- Gate
					  collapsed:: true
						- ![image.png](../assets/image_1675691751785_0.png)
					- Forget gate
					  collapsed:: true
						- ![image.png](../assets/image_1675691769983_0.png)
						- • The first gate determines whether to carry over the history or forget it
						  • Called “forget” gate.
						  • Actually, determine how much history to carry over.
						  • The memory C and hidden state h are distinguished.
					- Input Gate
					  collapsed:: true
						- ![image.png](../assets/image_1675691787804_0.png)
						- The second gate has two parts
						  • A tanh unit determines if there is something new or
						  interesting in the input.
						  • A gate decides if it is worth remembering.
					- Memory Cell Update
					  collapsed:: true
						- ![image.png](../assets/image_1675691804193_0.png)
						- Add the output of input gate to the current memory cell
						  • After the forget gate.
						  • ⊕: Element-wise addition.
						  • Perform the forgetting and the state update
					- Output and Output Gate
					  collapsed:: true
						- ![image.png](../assets/image_1675691818113_0.png)
						- The output of the memory cell
						  • Similar to input gate.
						  • A tanh unit over the memory to output in range [-1, 1].
						  • A sigmoid unit [0,1] decide the filtering.
						  • Note the memory is carried through without tanh.
					- the “Peephole” Connection
					  collapsed:: true
						- ![image.png](../assets/image_1675691862408_0.png)
						- Let the memory cell directly influence the gates!
					- ![image.png](../assets/image_1675691879433_0.png)
					- application and 优势
					  collapsed:: true
						- • Does not suffer from Vanishing Gradient problem.
						  collapsed:: true
							- 1. Additive formula means we **don’t have repeated multiplication** of the same matrix (we have a derivative that’s more ‘well behaved’)
							  2. The forget gate means that our model can learn **when to let the gradients vanish, and when to preserve them**. This gate can take different values at different time steps.
						- • Very powerful, especially in deeper networks.
						  • Very useful when you have a lot of data.
					- ![image.png](../assets/image_1675692122428_0.png)
					-
	- Week5 Statistical Machine Translation, Neural MT, Encoder-decoder, BiRNN, Attention, Metrics, Inference, Data Augmentation
	  collapsed:: true
		- Lecture 4 Machine Translation
			- Statistical machine translation 根据传统统计学得到的机器翻译
			  collapsed:: true
				- pipeline of several sub-models
				  collapsed:: true
					- alignment model: 用于对齐文本 is responsible for extracting the phrase pairs
					- translation model p(s|t): 用于翻译到对应的语言文本 is essentially a lookup table. We won’t dive into it, but you can imagine that statistics over large pairs of parallel texts can help identify parallel phrases
					- language model p(t): 用于检测对应语言文本是否符合科学, contains the probability of target language phrases, can be trained on monolingual data and gives us probability of a phrase occurring in that language.
				- **Parallel corpus** means we have aligned sentences: i.e. an english sentence and its corresponding french sentence
				- we want to model p(t|s), target given source
				  collapsed:: true
					- ![image.png](../assets/image_1675703577995_0.png)
				- translation is some combination of language model and translate model, 有语言模型还有翻译即变换的模型
				- language model p(t): can be trained on monolingual data and gives us probability of a phrase occurring in that language.
				- translation model p(s|t): How likely is the source phrase going to occur given a candidate translation t. We will have multiple candidate translations (example on next slide) 给定一个备选的目标语言句子, 源语言句子出现的可能性有多大
				- ![image.png](../assets/image_1675703720492_0.png)
				- language 和 translation组合的原因在于, 得到translation的结果以后, 放到语言模型中, 评估这句话在这个语言之中科不科学, 出现的概率高不高, 来验证这句话质量好不好 Language model p(t) then says: for each of the candidate targets, how likely is that phrase to occur in the target language. 下图中红色部分就是验证target language中这句话出现的概率
				  collapsed:: true
					- ![image.png](../assets/image_1675729336588_0.png)
				- downsides:
				  collapsed:: true
					- sentence alignment Long sentences may be broken up, short sentences may be merged. There are even some languages that use writing systems without clear indication of a sentence end (for example, Thai).
					- word alignment问题, 并不一定一一对应, 并且还会有一些没有对应意思的词语
					- Statistical anomalies: Real-world training sets may override translations of, say, proper nouns. 例如训练集里面出现了太多 took train to paris, 翻译起来就算原文是train to berlin 也会变成paris
					- Idioms: Only in specific contexts do we want idioms to be translated. For
					  example, using in some bilingual corpus (in the domain of Parliment), "hear" may almost invariably be translated to "Bravo!" since in Parliament "Hear, Hear!" becomes "Bravo!”. 我们不希望习语被翻译, 希望保留原本
			- Neural Machine Translation
				- Encoder  Decoder
				  collapsed:: true
					- solve sequence-to-sequence tasks
					- encoder represents the source as a **latent encoding**, 编码器用于将源语言编码成某种潜空间编码, 用来表示输入的语言序列, 用于给decoder使用
					- decoder generates a **target** from the latent encoding, 可以把decoder看作是language model, 用于将latent encoding转换为语言
					- 三种方式来实现编解码器
					  collapsed:: true
						- RNNs
						- Attention
						- Transformers
					- 通常encoder最后产生的hidden state 表示整个序列, 双向的网络可能会有两个h concat起来, 表示整个序列的latent encoding
				- BiDirectional RNN ([[BiRNN]]) 双向循环神经网络
				  collapsed:: true
					- ![image.png](../assets/image_1675731217398_0.png)
					- 双向的结构适合于encoding 翻译源, 因为他并没有在预测, 而是在编码, 需要得到整个句子的意思, RNN的init都是随机或者0初始化的作为hidden history, 这种结构可以将开始的信息与将来的信息结合起来, 获得整个语境的信息, 实现方式是每个时间节点的输出都变成了两个反向的RNN的h的concat结果 [h_i ; h’_0]
					- ![image.png](../assets/image_1675731674697_0.png)
					- ![image.png](../assets/image_1675731700252_0.png)
					- 通过双向循环神经网络的双向输出hidden state 可以得到对于源sequence 的encoding, c, 可以是单单hi, 也可以是两个输出的concat, 也可以是取一些平均; 这个c会被用作input放到decder结构中, 用来预测生成目标语言的词语序列
					- Decoder可以就是一个我们训练的typical language model, 比如一个RNN, 只不过它最开始的输入的hidden state并不是0, 而是encoding
					- 注意因为我们这里使用了concat的方法, 原本的embedding 的 dim是d, 现在我们就用2d了
					- 我们会用到 [[Teacher Forcing]] 来在训练阶段减少错误累积, 帮助训练, 但是inference的过程是 [[Auto-regression]], 自回归的, 这个时间点的预测会成为下一个时间点的输入
					- ![image.png](../assets/image_1675732414097_0.png)
					- Problems
					  collapsed:: true
						- All sequences are being represented by a d-dimensional vector. For longer sequence lengths, the ability to retain all the source information in this vector diminishes. 不管多长的sequence 都是这样子一个长度的embedding vector, 太长的sequence 根本就没有source information了
						- 对于decoder 来说就是相当于, 一开始给的context vector c没了 diminish了
						- vanishing gradients and the lack of ability to ‘remember’ things from earlier parts of the sequence
					- 为了解决上述问题, 我们就要提供所有的hidden state h们, 这样子的话就不会只依据于一个最终的输出c (由最终的两个h concat)了,  (引出[[attention]]注意力机制) during decoding, the current decoding timestep could look at all the source words, and find out which source words are most important to its current stage of decoding. 在解码阶段, 我们可以看到所有的源词语, 还能够找到哪几个是对于这个阶段最为重要的, 那就解决了信息缺失的问题, 而找到重要的词语, 或者说weight词语的hidden state 的过程, 就被叫做attention, 因为我们在pay attention to some important words than the other words
					-
				- [[Attention]] 注意力机制
				  id:: 63e1a744-5b0f-40ab-9f5e-4e53346baf10
					- a dynamic weighted average
					  id:: 63e1a74e-6dda-46ab-a2ca-9cdcf5de628b
					- In the immediate context, it allows us to dynamically look at individual tokens in
					  id:: 63e1a77c-f1f4-434c-9dc0-a2934287263a
					  the input and decide how much weighting a token should have with respect to the current timestep of decoding.
					- Types of attention
					  id:: 63e1a798-6d0f-4b1c-9250-690953da3ffb
						- additive/MLP
						  id:: 63e1a7ab-48b4-4455-a781-87bc715ebbc0
						- Multiplicative
						  id:: 63e1a7b7-efb3-4ec7-a7fb-67050aacb2eb
						- Self-attention
						  id:: 63e1a7be-1405-440e-ab5e-ddfd71719d57
					- [[MLP Attention]]
					  id:: 63e1a7c2-05bb-4762-b60f-e7ca15370cd7
						- $c_t\ =\ \Sigma_{i=1}^{I}\alpha_i h_i$
						  collapsed:: true
							- c_t is the context vector for the t’th decoding timestep
							- We then loop over all the hidden states i, and weight it by a scalar value
							  alpha
							- So if alpha is 0, then the i’th hidden state is 0
							- If alpha is 1, then we retain the full information of that hidden state - We then sum together our weighted hidden states to obtain a
							  contextualised representation for the t’th decoding step
							- Now the question is... how do we obtain alpha?
						- ![image.png](../assets/image_1675733243112_0.png)
						  collapsed:: true
							- 1. What we’re trying to do is decode the y_t word
							  2. And we have access to bidirectional encodings of each source word
							  3. Think of the attention module as a black box for a second. We’ll look at how it works in the next slide
							  4. So, before we decode y_t, we’re going to feed **all our encoder hidden states** AND the **decoder hidden state (s_t-1)** to our attention module
							  5. The module will output a **context vector, c_t**.
							  6. **c_t is a dynamic and contextualised representation**. It uses the decoder hidden state information to try and figure out which source words are most important when for decoding y_t
							  7. We send c_t to the t’th RNN step, alongside the previously generated token y_t-1.
							  8. One final change that the methodology introduced (not strictly related to attention itself), is that the output projection layer now also takes in c_t and the word embedding for y_t-1 (alongside s_t) to predict the t’th word.
						- ![IMG_DD2AAD24FA0B-1.jpeg](../assets/IMG_DD2AAD24FA0B-1_1675733471341_0.jpeg)
						  collapsed:: true
							- 因为这个是additive attention, energy score 是由加法得到的, 由我们需要attention的两个s和h与各自learnable weights W和U向乘后相加再激活得到的
							- 我们先看a, 即energy 函数 alignment function, 用来获得attention 分数的函数, 这个函数输入了我们正在decode的state s (1 x d) 和我们encoder中的所有h concat起来的一个(i x 2d)向量, 这里i 是词语数, 也就是sequence 长度, 2d是因为双向RNN的两个结果相接.
							- a函数括号内部, s进行了repeating操作, 为了和U对齐, 好一个个attention, 因为我们用了矩阵加速, 所以要考虑到这个对齐, W和U都是可学习参数; 括号外的v作用是把tanh输出的(i x d)向量转换成(i x 1), 也就是变成i个scalar 作为每个hidden state h_i的注意力分数,
							- 算出每个h的注意力分数以后, softmax就能得到weight
							- 最后用weight 来进行加权平均就得到了我们需要的context c
							- 1. Tell them: alpha is a scalar value. h_i is 2d
							  2. Alpha represents how important the i’th source word is to the current decoding step.
							  3. To obtain this, we need to calculate the energy scores for each word (e_i).
							  4. Energy scores are calculated by using an alignment function: a.
							  5. Once we have energy scores, we concatenate them together. Then we apply a softmax. The softmax is the normalized relevance of each source word with respect to the current decoding step.
							  6. Then we perform the alpha*h_i multiplication: This is a keypoint of attention. It applies a “mask” to each of our hidden states. Low energy values tend to 0 which means that we do not need that word’s information to decode the next word
							  7. Think of the alignment function as a 2 layered MLP. The first layer combines our decoder hidden state with all our encoder hidden states. The second layer (v) uses this contextualised representation to predict energy scores: i.e. unnormalized “importance” of each of the source words.
							-
						- 细节实现和维度信息
						  collapsed:: true
							- ![image.png](../assets/image_1675959759448_0.png)
							- 左边的橙色vectors是BiDRNN的每个时刻(即每个输入词语)的hidden state, 由于是双向的, 所以拼接起来后是2D, 但是我们的decoder想要一个D的, 所以会通过一个projection 投射到D, 作为Decoder的hidden 输入
							- ![image.png](../assets/image_1675960019882_0.png)
							- decoder rnn网络中每个时间节点的hidden state s都会与所有的encoder 的hidden state们进行attention, 举例来说就是s0和每个hi算加性注意力分数, 根据这个分数a, 来决定每个hi要取多少, 加权得到的就是context vector ct, 每个注意力分数都可以被解释为这个词语对于现在的decoding step 的重要性
							- ![image.png](../assets/image_1675960257567_0.png)
							- 维度信息如下图所示, 注意这里计算的是对于某一个h的注意力分数, 所以后面都是D x 1
							- ![image.png](../assets/image_1675960384717_0.png)
							- ![image.png](../assets/image_1675961588066_0.png)
							- ![image.png](../assets/image_1675961600210_0.png)
							- 使用了attention的RNN decoder 就会有三个input, c, s 和i, 分别是context from attention, hidden state from previous output, input
						-
		- Lecture 4.1 Evaluation metrics of MT/NLG systems
		  collapsed:: true
			- NLG Evluation
			  collapsed:: true
				- Human evaluation:  好, 但是不现实, 很贵
				- Automatic evaluation
				  collapsed:: true
					- 通常基于统计计数 n-grams的; 也就是n个词组成的对, 有多少次出现在reference 中
					  collapsed:: true
						- BLEU, Chr-F, TER, METEOR, ROUGE
						- **BertScore** is non n-gram count based, model based
			- BLEU: reports a modified precision metric for each level of n-gram 没关注recall
			  collapsed:: true
				- MP (Modified Precision)分数来源于每个unique n-gram在references 中出现的最大次数的总和, 例如the 在r1出现三次, r2出现一次, 那就只算3次, 然后其他的n-gram词语也得这么算
				- ![image.png](../assets/image_1675963919749_0.png)
				- ![image.png](../assets/image_1675963941678_0.png)
				- 例如上图, 每个词语是一个1-gram, r1中已经都出现过了1次, r2中的就不算了
				- ![image.png](../assets/image_1675963978049_0.png)
				- ![image.png](../assets/image_1675964008502_0.png)
				- Definition of BLEU:
				  collapsed:: true
					- Is a precision based metric over the product of n-gram matches
					- The matches are scaled by the brevity penalty which penalises shorter
					  translations.
					- There are a couple of interpretations about why we use a BP. They’re
					  mostly about encouraging the Hyps to be of a similar length to a reference (see BP equation). Feel free to research more about it in your own time. An intuitive reason is for its existence is to account for the lack of recall term in the metric.
				- Practically, there are some differing definitions and implementations of BLEU. When you want to report this score, it is good practise to use a standardized library (e.g. SacreBLEU)
			- Chr-F: Character n-gram Fß score
			  collapsed:: true
				- Balances character precision and character recall
			- TER: Translation Error Rate
			  collapsed:: true
				- minimum number of edits required to change a hypothesis into one of the references
			- ROUGE-n: Measures the F-score of n-gram split references and system outputs
			  collapsed:: true
				- ROUGE balances both precision and recall via the F-Score.
				- Though originally a translation metric, ROUGE is more common in
				  captioning/summarization literature than translation. Obviously it can be used for
				  translation though.
				- ROUGE-n measures the F-score of n-gram split references and system outputs
				- ![image.png](../assets/image_1675964411735_0.png)
			- METEOR: Unigram precision and recall with R weighted 9* higher than P; Modern way
			  collapsed:: true
				- ![image.png](../assets/image_1675964490240_0.png)
			- Shortcomings for N-gram methods
			  collapsed:: true
				- ![image.png](../assets/image_1675964576575_0.png)
			- BertScore: computes pairwise cosine similarity for each token in the candidate with each token in the reference sentence
			  collapsed:: true
				- 不再是n-gram了
				- Biggest drawback is now not the n-gram matching. Rather, the scores can vary if evaluated against different BERT models
			- Inference: 不知道真实label怎么办
			  collapsed:: true
				- Greedy decoding
				  collapsed:: true
					- ![image.png](../assets/image_1675964842183_0.png)
					- Chosen word might be best at current timestep. But as we decode the rest of the sentence, it might be worse than we thought. If we were able to see what the future candidates might be, we might be able to predict a better word for the current time step
				- Beam search
				  collapsed:: true
					- ![image.png](../assets/image_1675964858062_0.png)
					- In practise, k is normally between 5-10. For the example we’re about to work through, k=2
					- Note that we will end up with k hypothesis at the end of decoding. Decoding finishes when we hit an EOS token
					- Example run through:
					- t=0: arrived and the are the 2 most likely words
					- t=1: for each of these words, decode the next k. So we have [start
					  arrived the, start arrived witch, start the green, start the witch]
					- Then we prune all but the top-k: [start the green, start the witch]
					- t=2: Repeat. Now we have [start the green mage, start the green witch,
					  start the witch arrived, start the witch who]
					- After pruning: [start the green witch, start the green who]
				- Temperature sampling
				  collapsed:: true
					- ![image.png](../assets/image_1675964874846_0.png)
					- Temperature sampling lets us inject non-determinism into our decoder.
					  collapsed:: true
						- Perhaps not ideal for translation, but can be useful for language
						  modelling
					- Gif is the post softmax value of each of the 10 classes.
					- Higher temperature leads to smoother softmax operation.
					- Thus more diverse (but sometimes less coherent) outputs.
			- Data Augmentation
			  collapsed:: true
				- Backtranslation
				- Synonym replacement
				- ![image.png](../assets/image_1675964955494_0.png)
			- Batching, Padding and Sequence Length
			  collapsed:: true
				- Group similar length sentences together in a batch, 把长度相似的放进一个batch里面, 这样子padding的就比较少, 也比较类似, more efficient
				- Train model on simpler/smaller sequence lengths first: 先训练那些长度短的句子
				  collapsed:: true
					- Models have been shown to better model more complicated sequences when they’ve been exposed to sequences in gradually increasing complexity; 因为实验证明要先短后长
		- Tutorial
		  collapsed:: true
			- TODO 上课的时候让我们实现了一下encoder decoder, 就是一些weight的学习, 要匹配好sequence的维度, 放入nn的rnn里面
			- TODO 看了一眼RNN的实现视频:
			  collapsed:: true
				- 输入维度会是以时间为导向 (T, N, D) 也就是处理的时候就和RNN图里一样, 是一个个时间节点顺序处理的, 对每个时间节点T, 取N个sample, 对D进行运算得到N个h, 放到下一个时间点作为输入之一, 继续运算
				- 库里的rnn还会给到几层的参数, 所以会有一个几层在第一维度
			- encoder RNN 的一些代码细节
			  collapsed:: true
				- ```python
				  # Encoder, embedding 把one-hot编码成我们要的embedding dim
				  self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
				  # 用了自带的rnn, 给定了词语的embedding dim和hidden state的dim
				  self.forwards_rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
				  self.backwards_rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
				  self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
				  
				  # shape(inputs) = [B, I] (e.g. B=3, I=6) 3 sentences of length 6
				  
				  # > embed the inputs
				  embedded = self.embedding(inputs)
				  # shape(embedded) = [B, I, E] (e.g. B=3, I=6, E=5) 3 sentences of length 6, each word represented by a 5-dimensional vector
				  
				  # > create a tensor of zeros to initialize the hidden state of the RNN (N=1, B, D), where N is the number of layers
				  zeros = torch.zeros(1, inputs.shape[0], self.forwards_rnn.hidden_size)
				  
				  # > run the forwards RNN, 给如embedded 词语序列和初始的hidden state
				  _, final_forwards = self.forwards_rnn(embedded, zeros)
				  # shape(final_forwards) = [1, B, D], 1 is the number of layers
				  ```
	- Week6 Transformers
	  collapsed:: true
		- Transformers Architecture
			- ![image.png](../assets/image_1676399140084_0.png)
			- ![image.png](../assets/image_1676630755354_0.png)
			-
		- Transformers
		  collapsed:: true
			- Encoder
			  collapsed:: true
				- ![image.png](../assets/image_1676399925210_0.png)
			- Decoder
			  collapsed:: true
				- ![image.png](../assets/image_1676399949811_0.png)
			- Encoder-Decoder
			  collapsed:: true
				- ![image.png](../assets/image_1676399972110_0.png)
			- Vision Transformers
			  collapsed:: true
				- ![image.png](../assets/image_1676400041891_0.png)
			- Multimodal Transformers
			  collapsed:: true
				- ![image.png](../assets/image_1676400066836_0.png)
			- ![image.png](../assets/image_1676400132802_0.png)
		- Encoder
		  collapsed:: true
			- ![image.png](../assets/image_1676400207590_0.png)
			- Each encoder layer consists of 2 sublayers. The first sublayer contains a multi-
			  head self-attention module, and the second contains a position-wise feedforward
			  network.
			- the output of the “Residual & Norm” after the Position-wise
			  feedforward will form the input to the Multi-head self-attention in the next encoder layer
			- ![image.png](../assets/image_1676400298242_0.png)
			  collapsed:: true
				- The input will be an encoding of each of our words. So here, we have 3 words represented with 4 dimensions. More generally, you would have S words in the input sequence, each represented with D dimensionality.
				- The MHA module processes the input and outputs another set of S x D encodings.
				- These encodings get sent to the Residual & Norm, which outputs another set of S x D encodings.
		- Self-attention (scaled dot product attention)
		  collapsed:: true
			- ![image.png](../assets/image_1676400412857_0.png)
			- 每个词都和其他词attention, 找到对应attention分数, 根据这个分数来weight每个词语对应的V的值, 求加权平均得到这个词最终的value
			- Query: 表示查询, self-attention中表示要查询与我相关的东西
			- Key: 表示键值, 与query匹配查看两者的相似度, 指向的是value
			- Value: 表示值, 可以看作是代表这个东西的潜在feature 表达, 用于表示这个东西
			- ![image.png](../assets/image_1676400758682_0.png)
			- ![image.png](../assets/image_1676400770259_0.png)
			- 一个self-attention head就类似于这么三个W, 用来把词语的embedding变换成QKV的表示, 这个变换是学习来的
			- ![image.png](../assets/image_1676400849575_0.png)
			- ![image.png](../assets/image_1676400868199_0.png)
			- ![image.png](../assets/image_1676400898399_0.png)
			- 上图就演示了一个词做self-attention 的过程, sigma是softmax完了以后的注意力权重, 这个权重乘上每个词对应的v, 会得到S 词语个数个weighted v, 加起来就是他的新v, 对应于做self-attention的这个词语照顾了上下文意义的新向量表示value
			- ![image.png](../assets/image_1676400999159_0.png)
			- 这里的d_h是KQV的hidden dimension, 这里对乘法注意力分数除以了一个根号下的d_h的目的是让softmax得到更平滑的weight, 因为如果值太大的话, 经过exp会差别很大, 运算下来norm到概率就会被那个稍微大一点的值dominate, 从而变成了类似于one-hot的表示, 而缩放的话可以让他们的分布更平滑, 例如(0.2, 0.3, 0.5) 而非(0.01, 0.01, 0.98); 所以要看情况需要怎么样的形式
			- The intuition is that we divide through by sqrt(dq)​ in order to control how large the dot product between queries (Q) and keys (K) can get.
			- Why do we do this? The idea is when we come to softmax (or apply any non linear function) having a down-scaled version of our attention weights allows the distribution to become smoother and less "peaky". This is useful as it means distribution is spread more smoothly across the words.
			- But why the square root? The square root is just a popular choice in the literature as it has been shown to work quite well. That's not to say other factors wouldn't work too but taking the square root is just the most popular choice.
			- ![image.png](../assets/image_1676401112239_0.png){:height 317, :width 685}
			- 最后的解雇也是S x d_h
			- sigma是row-wise的softmax, 对每一行进行
		- Multi-head attention
		  collapsed:: true
			- Why? Intuitively multiple attention heads allows for attending to parts of the sequence differently (e.g. some heads are responsible for longer-term dependencies, others for shorter-term dependencies)
			- ![image.png](../assets/image_1676401188200_0.png)
			- ![IMG_1145.jpeg](../assets/IMG_1145_1676591221675_0.jpeg)
			- ![IMG_1146.jpeg](../assets/IMG_1146_1676591572788_0.jpeg)
			-
		- [[Layer Normalisation]] #card
		  id:: 64419f8e-794a-4de5-a628-4345a46dfaa4
			- ![image.png](../assets/image_1676401257459_0.png)
			- The purpose of gamma and beta is to allow the neural network to learn the optimal scale and shift for each feature after normalization.
			- By applying these learned parameters to the normalized feature vectors, the neural network can adjust the range and mean of each feature to better fit the task at hand.
			- ![image.png](../assets/image_1676401283689_0.png)
		- Residual
		  collapsed:: true
			- Help mitigate the vanishing gradient problem (tiny weight changes)
			- ![image.png](../assets/image_1676401941844_0.png)
			- By adding the previous layer's output directly to the current layer's output, a residual connection allows the current layer to focus on learning the difference between the two outputs, rather than learning an entirely new transformation.
		- Position Wise Feedforward Network
		  collapsed:: true
			- Is an MLP
			- Position-wise 的意思是同样的weight会用于每一个句子中的词语, 不会用不一样的
			- ![image.png](../assets/image_1676402381986_0.png)
			- 用来增加一些non-linearity
		- Positional Encodings
			- Transformers are position invariant by default, 对于不同词语的顺序没有感知, not inherently sequential, 需要positional encoding 来inject position information into embeddings
			- 理解上就是, a, b, c, d, e做attention的时候, 不管位置怎么变, 怎么排列, attention分数都是分别算得, 最后的结果都是weighted avg, 并没有位置上的影响存在
			- 我们的做法就是在词语的embedding基础上加上对于它位置信息的encoding
			- ![image.png](../assets/image_1676402611091_0.png)
			- Sinusoids
				- ![image.png](../assets/image_1676402689475_0.png)
				- PE is a function of 2 inputs: position of a word, and the model dimensionality
				- Notice what is happening with these arguments. The 2i and 2i+1 imply that we’re
				  collapsed:: true
				  going to be looping over the indexes in range of d.
					- E.g. Consider we **had a vector of 512 dimensions**. We would be looping
					  over it. At an **even index** (e.g. 0), we’d apply the **sin variant** of PE. At an
					  **odd index** (e.g. 1), we would apply the **cosine** variant.
				- Now observe that the only difference between the functions is whether we use a
				  sin or a cosine
				- Apart from that, the functions divide the position argument by 10000^(2i/d)
				  collapsed:: true
					- Note that when we’re **early on in looping over d**, **i** will be **small**. The resulting **exponent would be therefore be small**. This makes the denominator **small**. Thus the **overall value** of pos/10000^(2i/d) would be **larger** than when we’re later in our loop over d
					- You can play around with the implications of how this affects the output sin and cosine in your own time (just thought it was worth mentioning)
				- ![image.png](../assets/image_1676402852986_0.png)
				  collapsed:: true
					- In implementation, we have a PE matrix in **[maxT, d]**
					- maxT is the maximum sequence length we ever want to support (e.g. 5000)
					- d is our ‘model dimensionality’
				- ![image.png](../assets/image_1676402877285_0.png)
				  collapsed:: true
					- Consider position 0 (i.e. the first word in the sequence)
					  collapsed:: true
						- Positional encoding function means no manipulation is done to the
						  vector
					- Consider position 1
					  collapsed:: true
						- Positional encoding function shows that indexes up to 10 have values close to 1 added to them
					- Consider position 22
					  collapsed:: true
						- Positional encoding function shows that the first few indexes are affected
						  heavily by positional encodings. Some indexes have values close to -1 added to them, followed shortly by a value +1 added to them
				- ![image.png](../assets/image_1676629585364_0.png)
				- i越大的时候, 2i/d越大, 对应上面曲线的x也愈大, 会让y值越接近于0, positional encoding式子就变成了sin, cos(0), 也就是图中的白线和黑线, 对应1 和0.
				- 而在i小的时候, y值是平滑下降的, 对应cos sin(ax)中的a是越来越小的 (0->1), sin和cos的频率是越变越大的
			- Learned (in BERT)
		- Test time (inference)
		  collapsed:: true
			- ![image.png](../assets/image_1676630876697_0.png)
			- we perform auto-regressive generation
			- 1. Our source sentence gets encoded via our encoder
			- 2. We feed an SOS token to our decoder. It then predicts the first word (i.e. I)
			- 3. We append the prediction to our SOS token, and then use this to predict the
			  next word (i.e. am)
			- 4. And so forth. At some point our decoder will predict an EOS token (not shown
			  here), and we’ll know that this is the end of the generation.
			- 我们进行的是一个iterative 的 prediction 过程, 每次的预测, 会作为下一个decoder的输入之一, 帮助他继续预测, 直到EOS 或者长度达标了
		- Train time
		  collapsed:: true
			- ![image.png](../assets/image_1676631018727_0.png)
			- However, during training, we won’t get the model to generate auto-regressively.
			- We’re actually going to to feed the whole target sequence as input to the
			  decoder and decode the sentence in one go 使用完整的句子, 一次性把所有东西全给预测完
			- This means we can run the decoder stack once. 只需要运行一次啦
			- As opposed to running it for as many tokens as we have in the target
			  sequence (which is what we do when performing inference)
			- If we feed in the whole sequence to the decoder, we need a way to tell the decoder not to look at future tokens when computing attention for one of the tokens. 但是我们给进去的每一个词, 需要只知道它自己和前面的东西, 虽然给的是完整的序列可以一次性算完(因为teacher enforcing, 每一个decoder block内部直接可以并行, 因为我们做的是attention, 并没有内部时序关系), 但是对每个词预测计算loss的时候还是需要mask掉当前词语之后的所有词, 因为要仿照测试时并不知道后续东西的模式
			  collapsed:: true
				- E.g. when trying to compute the attention-based representation for “am”, we should not be allowed to look at “a” or “student”, since these tokens are in the future
				- We enforce this unidirectionality with masking
		- Decoder
		  collapsed:: true
			- Difference between Encoder and decoder
			  collapsed:: true
				- different in train and test time
				  collapsed:: true
					- masked multi-head self-attention
					- 100% teacher enforcing
					- auto-regression during test (until a max length, or EOS)
				- cross attention
			- Architecture
			  collapsed:: true
				- ![image.png](../assets/image_1676630630902_0.png)
				- ![image.png](../assets/image_1676630815107_0.png)
			- Masked Multi-head Self-attention
			  collapsed:: true
				- 由于attention是对整个序列而言的, 因此可以被看作是bi-directional的; recall之前用RNN做的encoder-decoder, decoder部分是用的单向的RNN, 我们在这里解码器也应当让每个时间节点的词语只能看到现在和历史, 而不能偷看到未来
				- Target sequence在training阶段是知道的, 长度为T, 那么我么所使用的mask也应当是T长度的, 对于每个词T, 看向整个sequence T, 因此第二个维度也是T. 即T x T
				- ![image.png](../assets/image_1676636867972_0.png)
				- upper triangular 设为-inf, indicating future tokens not be accessed
			- Cross Attention
			  collapsed:: true
				- 就像之前的RNN based 的ende一样, 需要根据encoder的信息, 来生成后续的信息. 每decode 一个token, 我们都得要知道我们要去看哪几个encoded words, 这个就需要用cross attention来实现. Q取的是decoder里面的Q, 但是K 和V 使用的是encoder最后一层输出的K V. As we decode a token, we look at all encoded tokens to find out which are more important for the current decoding step
				- Last Encoder layer中的K V 会被所有decoder layers使用
				- ![image.png](../assets/image_1676637461865_0.png)
			-
		- Other tricks
		  collapsed:: true
			- ![image.png](../assets/image_1676637668897_0.png)
			- Weight tying: Embedding matrix and output matrix are shared
			  collapsed:: true
				- word 到embedding的matrix和model dim到output word的matrix 是shared
			- Learning rate annealing
		- Questions
		  collapsed:: true
			- What are the differences between training and testing in a transformer
			- How does masked self-attention work?
			- What are teacher forcing ratio do we use in Transformers?
			- What is cross attention doing?
			- What should the shape of the cross attention matrix be?
			- What does a transformer class do every training loop?
			  collapsed:: true
				- Create a source and target mask
				- run the encoder
				- run the decoder
				- output logits for token prediction
	- Week7 Tokenise, contextual word repre, pertaining models, BERT
	  collapsed:: true
		- Module 6.1: Pre-training models
			- Byte Pair Encoding: Method of tokenisation
				- Intuition:
				  collapsed:: true
					- 如果我们只处理完整的常用词, 我们会难以处理那些词语的变形, 例如前缀后缀, 合成词, 以及一些错误拼写问题
				- Solution: Subword units
				  collapsed:: true
					- break long and complex words down into smaller parts, so that each part occurs frequently enough in the training data to learn a good embedding for it.
					- 每个词语都由几个部分组成, 比如一些常见的词根词缀变形等等, 把他们分开以后, 可以让语言模型理解不同部分的含义
					- ![image.png](../assets/image_1676995557154_0.png)
				- Method: Byte Pair Encoding
				  collapsed:: true
					- BPE is successfully used by GPT, GPT-2, GPT-3, RoBERTa, BART, and DeBERTa.
					  collapsed:: true
						- [Byte-Pair Encoding tokenization - Hugging Face Course](https://huggingface.co/course/chapter6/5)
					- 我们当然不会手动做这件事情, 用BPE这个算法来进行词语subpart 分解
					- Training
						- ![image.png](../assets/image_1676995801797_0.png)
						- ![image.png](../assets/image_1676995809247_0.png)
						- ![image.png](../assets/image_1676995820990_0.png)
						- ![image.png](../assets/image_1676995829341_0.png)
					- 上述例子中用到了underscore _, 用处是
					  collapsed:: true
						- 1. It distinguishes the suffixes.
						  2. It tells us where to insert spaces when putting words back together.
					- 上面是training algorithm, 如果遇到新的词语, 就可以根据训练得到的组合方法, 按merge dict的顺序, 来对字母们进行组合
					- Inference
					  collapsed:: true
						- ![image.png](../assets/image_1676995998686_0.png)
						- ![image.png](../assets/image_1676996013833_0.png)
						- ![image.png](../assets/image_1676996020137_0.png)
						- 最差的情况就是把一个词分解成单个的characters
					- dealing with Unicode characters, we may encounter characters that we haven’t seen during training. Unicode时候会有没有遇见过的, 所以我们把字母character扩展到了字节byte based, vocabulary就可以有2^8 256大小, 也就可以出现其他未见过的characters了
				- Method: Wordpieces
				  collapsed:: true
					- Used by BERT and other models based on BERT (e.g. DistilBERT).
					  collapsed:: true
						- [WordPiece tokenization - Hugging Face Course](https://huggingface.co/course/chapter6/6)
					- using corpus statistics to decide how to split words into subwords.使用的也是语料库数据来决定如何细分词语
					- ![image.png](../assets/image_1676996473763_0.png)
			- Contextual word representations
				- 一个词语可能有多个意思, 在不同的语境有不同的语境意义, 如何根据语境学习到每个词语合适的embedding?
				- RNN: We can train a recurrent neural network to predict the next word in the sequence, based on the previous words in the context.
				  collapsed:: true
					- Internally it will learn to encode any given context into a vector.
					- RNN 序列化的一个个处理输入的词语, 后面的词语可以看到前面的东西
				- ELMo: Embeddings from Language Models
					- ![image.png](../assets/image_1676996902217_0.png)
					- 从前到后和从后到前的信息 get
					- Contextual word embeddings
						- When we need a vector for a word, combine the representations from both directions. 用双向的context 信息作为这个词语的vector embedding
						- ![image.png](../assets/image_1676996992755_0.png)
						- 例如上面两个方向的RNN, 最后得到的就是brown的整句话考虑进去的embedding
						- ![image.png](../assets/image_1676997057480_0.png)
					- ELMo could be integrated into almost all neural NLP tasks with a simple concatenation to the embedding layer.
						- ELMo几乎可以用在任何NLP任务中用来加强word embedding的context意义
						- ![image.png](../assets/image_1676997097549_0.png)
			- Encoder-decoder models
			  collapsed:: true
				- ![image.png](../assets/image_1676997184203_0.png)
			- Pre-training encoders: BERT and relatives
				- BERT: Bidirectional Encoder Representations from Transformers
				- Takes in a sequence of tokens, gives as output a vector for each token.
				- ![image.png](../assets/image_1676997232255_0.png)
				- BERT是个encoder, 拿进来一堆词语的token, 生成他们的embedding vector
				- ![image.png](../assets/image_1676999008877_0.png)
				- Self-attention
				  collapsed:: true
					- The new representation of each word is calculated based on all the other words.
				- Multi-head self-attention
				  collapsed:: true
					- Each head learns to focus on different type of information.
				- 上图的x1 x2的embedding由几个部分组成
					- ![image.png](../assets/image_1676999076051_0.png)
					- position embeddings 告诉transformer词语之间的顺序, segment embeddings告诉transformer这个词语属于哪个句子
					- Token Embeddings 是随网络一起训练的, 用的是nn.Embedding. 并没有用其他预训练的embedding 例如word2vec
					- torch.nn包下的Embedding，作为训练的一层，随模型训练得到适合的词向量。
				- Masked Language Modeling
				  collapsed:: true
					- 我们的目标是让模型能够理解这些词语在句子中的意思, 从而可以给出他们的contextual vector利用于更多下游任务中, 其中一种方法就是mask掉输入中的一些词语, 让模型猜测他们应该是什么, 以这种方式鼓励模型学习general-purpose language understanding.
					- ![image.png](../assets/image_1676999242118_0.png)
					- ![image.png](../assets/image_1676999262898_0.png)
					- 所谓的MLM其实是一个于BERT之上的神经网络, 通常是由MLP组成, in是embedding dim, out是vocabulary大小, BERT会对有MASK的一句话(6个词, 包含2个mask) encode得到6个embedding, 这两个mask的embedding会给到MLM网络, 用来map到词汇表中, 通过鼓励map到真正的词, 来鼓励BERT学习到真正的语意信息
				- Next sentence prediction
				  collapsed:: true
					- BERT的第二个训练目标, 判断两句话是不是按顺序出现在一个源文本的, predict的是一个label 但是这个被证明没啥用
				-
			- Putting pre-trained models to work
			  collapsed:: true
				- BERT-like models give us a representation vector for every input token. BERT会给输入文本每个词一个representation, 我们可以利用这些vectors来表示tokens或者句子
				- Option 1:
				  Freeze BERT, use it to calculate informative representation vectors. Train another ML model that uses these vectors as input.
				- Option 2 (more common these days):
				  Put a minimal neural architecture on top of BERT (e.g. a single output layer) Train the whole thing end-to-end (called fine-tuning).
				- Sentence classification
				  collapsed:: true
					- ![image.png](../assets/image_1676999448022_0.png)
				- Token labeling
				  collapsed:: true
					- ![image.png](../assets/image_1676999459091_0.png)
				- Sentence pair classification
				  collapsed:: true
					- ![image.png](../assets/image_1676999471079_0.png)
				- Question answering
				  collapsed:: true
					- ![image.png](../assets/image_1676999484387_0.png)
			- Text classification with BERT in practice
			  collapsed:: true
				- ```python
				  checkpoint = 'bert-base-cased' # 选择所需要的模型名称, 用于在huggingface中索引
				  
				  # 自动载入用于二分类的对应checkpoint的模型
				  model = AutoModelForSequencialClassification.from_pretrined(checkpoint, num_labels=2)
				  model.to(device)
				  
				  # 加载数据
				  datasets = load_dataset('glue', 'sst2')
				  
				  # tokenisation, 对应model的tokeniser
				  tokeniser = AutoTokenizer.from_pretrained(checkpoint)
				  tokenised = tokeniser('this is an example sentence', truncation=True)
				  
				  optimiser = AdamW(model.parameters(), lr=lr)
				  
				  num_epochs = 10
				  
				  for epoch in range(num_epochs):
				  	for batch in train_loader:
				      	outputs = model(batch)
				        	loss = outputs.loss
				          loss.backward()
				          optimiser.step()
				          optimiser.zero_grad()
				  
				  ```
			- 一些例子 看ppt
			- Parameter-efficient fine-tuning
			  collapsed:: true
				- Models fine-tuned for one task are usually better at that particular task, compared to models trained to do many different tasks.
				- let’s keep most of the parameters the same (frozen) and fine-tune only some of them to be task-specific.
				- Prompt tuning
				  collapsed:: true
					- Include additional task-specific “tokens” in the input, then fine-tune only their embeddings for that particular task, while keeping the rest of the model frozen.
					- ![image.png](../assets/image_1677000017701_0.png)
				- Prefix tuning
				  collapsed:: true
					- ![image.png](../assets/image_1677000033246_0.png)
				- Control Prefixes
				  collapsed:: true
					- Training different prefixes for each property/attribute you want the output to have. For example, the domain or desired length of the text.
				- Adapters
				  collapsed:: true
					- Inserting specific trainable modules into different points of the transformer, while keeping the rest of the model frozen.
					- ![image.png](../assets/image_1677000075018_0.png)
				- BitFit
				  collapsed:: true
					- ![image.png](../assets/image_1677000092202_0.png)
				- Low-rank adaptation
				  collapsed:: true
					- ![image.png](../assets/image_1677000109136_0.png)
			- The keys to good models of language
			  collapsed:: true
				- 1. Transfer learning (model pre-training)
				  2. Very large models
				  3. Loads of data
				  4. Fast computation
				  5. A difficult learning task
		- Module 6.2: pre-training encoder-decoder, decoder models, advanced prompting, human feedback
		  collapsed:: true
			- Pre-training encoder-decoder models
			  collapsed:: true
				- Encoder-decoder:
				  Input is processed using an encoder, then output is generated using a decoder.
				  Particularly popular for **machine translation**.
				- can also be used for **classification**, by constructing a new output layer and connecting it to the last hidden state from the decoder.
				- Ideas of pre-training encoder-decoder models
				  collapsed:: true
					- Can’t really do Masked Language Modelling any more, there isn’t a direct correspondence between input tokens and output tokens. 这里不能再直接把decoder对应位置输出给到MLM了, 这里和BERT的情况不一样, BERT是给的一个对应mask位置的embedding, 但是en-de给的是一个新生成的东西, 生成模型, 不是一对一的, 比如我也可以生成不一样长度的, 也可以叫他只生成mask掉的东西, 因此会有区别. 所以用的新的方式是给decoder supervision, 告诉他应该生成什么, 例如把原文给他, 作为label, 或者更难一点, 期望他仅仅生成mask的词语
					- Prefix language modeling
					  collapsed:: true
						- ![image.png](../assets/image_1677426536236_0.png)
					- Sentence permutation deshuffling
					  collapsed:: true
						- ![image.png](../assets/image_1677426546229_0.png)
					- BERT-style token masking
					  collapsed:: true
						- ![image.png](../assets/image_1677426557746_0.png)
				- Replace corrupted spans
				  collapsed:: true
					- We can corrupt the original sentence in various different ways, then optimise the model to reconstruct the original sentences. Good for generation tasks. 打乱原句子, 让ende来恢复原句子
					- ![image.png](../assets/image_1677426812637_0.png)
				- Instructional training
				  collapsed:: true
					- Can be pre-trained trained on different supervised tasks, by stating in the input what task the model should perform. 告诉模型你要干什么, 模型做我说的事情
					- T5 (Text-To-Text Transfer Transformer): trained using the span corruption unsupervised objective, along with a number of different supervised
					  tasks.
					- Existing datasets can be converted into more conversational-sounding instructions using templates. 可以将原有的数据集进行一些更改, 把他们转变成自然语言的形式
					  collapsed:: true
						- ![image.png](../assets/image_1677427011948_0.png)
					- 训练模式: train with natural language instructions as inputs, and annotated answers as target outputs. 使用自然语言指示作为输入, 标记过的答案作为目标输出
			- Pre-training decoder models
			  collapsed:: true
				- Decoders: 语言模型, Are able to access context on the left. Language models, good for generating text. 前面输出的结果作为后面的输入, 后面的只能看到前面的东西
				- 训练模式: We can train on unlabeled text, optimizing $p_\theta (w_t|w_{1:t-1})$, Great for tasks where the output has the same vocabulary as the pre-training data. 适用于输出词汇和与训练词汇相似的任务: dialogue systems, summarization, simplification, etc.
				- Learning methods: 一些模型学习回答问题的办法
				  collapsed:: true
					- 1. Fine-tuning: Supervised training for particular input-output pairs.
					  Or we can put a new layer on top and fine-tune the model for a desired task.
					- 2. Zero-shot: Give the model a natural language description of the task, have it generate the answer as a continuation. 给一个自然语言描述的任务, 不给example, 直接让模型输出想要的答案
					  3. One-shot: In addition to the description of the task, give one example of solving the task. No gradient updates are performed. 会给到一个example, 不再进行梯度更新
					  4. Few-shot: In addition to the task description, give a few examples of the task as input. No gradient updates are performed. 给到更多的example, 同样没有梯度更新
				- Fine-tuning decoder models
				  collapsed:: true
					- Once pre-trained, we can fine-tune these models as classifiers, by putting a new output layer onto the last hidden layer.
					  The new layer should be randomly initialised and then optimized during training.
					  We can backpropagate gradients into the whole network.
					- pre-trained model可以在最后的隐藏层后面加一个新的输出层, 用来根据decoder信息来完成特定的任务, GPT就是先是用生成来与训练, 最后finetune成了辨别器
					- ![image.png](../assets/image_1677427716997_0.png)
					- 可以在sentence中token到一些特殊的token例如<CLASS>, 用来给分类器分类学习
				-
			- Advanced prompting
			  collapsed:: true
				- Chain-of-thought
				  collapsed:: true
					- 在举例中给到reasoning, 可以encourage model 在回答类似问题的时候做类似的reasoning, 而不是只生成一个答案
					- 甚至可以模拟code, 给出适当的code, 并且真的可以被执行得到答案
					- Zero-shot chain-of-thought
					  collapsed:: true
						- 可以不说是the answer is, 而是lets think step by step, 也可以提示模型给出chain of thought
				- Retrieval-based language models
				  collapsed:: true
					- ![image.png](../assets/image_1677433577373_0.png)
					- 增加了从数据库中获取信息的能力
				- Limitations of instruction fine-tuning
				  collapsed:: true
					- Language models are being trained with instruction fine-tuning, using manually created ground truth data that follows instructions. This data is expensive to collect. 指令和答案的训练集很昂贵, 因为是人工的
					- ● Problem 1: tasks like open-ended creative generation have no right answer.
					      Write me a story about a dog and her pet grasshopper. 开放性问题
					  ● Problem 2: language modeling penalizes all token-level mistakes equally, but some errors are worse than others.有的错误更严重, 但是惩罚起来是一样的
					-
			- Learning from human feedback
			  collapsed:: true
				- Reinforcement learning from human feedback
				  collapsed:: true
					- 强化学习的概念呢就是说有一个环境给到的评价, reward, 比如说一个给文本创建概括的任务, 人可以给模型的输出打分来告诉模型这个好不好
					- ![image.png](../assets/image_1677433720871_0.png)
					- 问题依然在于人力成本太高了, 解决方案可以是用一个LM来预测人类可能打的分
					- 另一个问题是人类的判断是noisy且有失焦准的, 解决方案是不去给一个分, 而是去判断两个例子哪个更好, 这样子能减少noise
	- Week8 POS, Constituency parsing, Dependency parsing
	  collapsed:: true
		- Module 7.1 POS Tagging
			- Tagset
				- ● ADJ (adjective): old, beautiful, smarter, clean...
				  ● ADV (adverb): slowly, there, quite, gently, ...
				  ● INTJ (interjection): psst, ouch, hello, ow
				  ● NOUN (noun): person, cat, mat, baby, desk, play
				  ● PROPN (proper noun): UK, Jack, London
				  ● VERB (verb): enter, clean, play
				  ● PUNCT (punctuation): . , ()
				  ● SYM(symbol):$,%,§,©, +,−,×,÷,=,<,>,:),♥
				  ● X (other): ? (code switching)
				- ● PREP (preposition): in, on, to, with
				  ● AUX (auxiliary verb): can, shall, must
				  ● CCONJ (coordinating conjunction): and, or
				  ● DET (determiner): a, the, this, which, my, an
				  ● NUM (numeral): one, 2, 300, one hundred
				  ● PART (particle): off, up, ‘s, not
				  ● PRON (pronouns): he, myself, yourself, nobody
				  ● SCONJ (subordinating conjunction): that, if, while
			- Why need PoS Tagging
				- 对于NER而言, 可以先识别出noun, 再进行命名实体识别, 即识别出一些专有名词
				- 一些预处理可以在POS tagging的基础上完成, 例如情感分析的预处理可以提取出形容词
				- 也可以用于syntactic and semantic parsing, 句法和语意分析
				- For many applications such as spam detection, sentiment analysis at scale (e.g. millions of emails going through a server) you can focus on nouns,verbs and adjectives removing redundant tokens. 提炼出垃圾邮件中的名次动词和形容词, 来增加效率
			- POS tagging libraries
				- Spacy,NLTK
			- baseline method
				- Assign each word its most frequent POS tag
				- unknown words the tag NOUN
				- 90% acc
			- Facing difficulties
				- Ambiguity
				- Unknown words
			- Probabilistic POS tagging
				- Given a sequence of words W = w1, w2, w3, ..., wn
				- Estimate the sequence of POS tags T= t1, t2, t3, ...., tn
				- Compute $P(T|W)$
				- Generative approach (Bayes):
					- ![image.png](../assets/image_1677601073265_0.png)
					- ![image.png](../assets/image_1677601093083_0.png)
					- 基于 HMM, Hidden Markov Model,
					- 一个是transition probability, tag作为state, state之间有transition, transition有概率, 这就是P(T), 由前一个tag 跟下一个tag的可能性
					- 另一个是Emission probability, 某个tag可能的词中, 是这个词的可能性, 相当于tag发射到词. 注意我们这里求的是词序列对应某个tag序列的可能性, 因此这里知道的只能是这个tag有多大可能对应这个词
					- ![image.png](../assets/image_1677601316881_0.png)
					- 因此也就需要两个表, transition 以及emission
					- Example
						- ![image.png](../assets/image_1677601378296_0.png)
						- ![image.png](../assets/image_1677601393939_0.png)
						- ![image.png](../assets/image_1677601414623_0.png)
						- 注意这个emission表是每种tag对应于某个词的可能性, 横着看, 用来算given t, 某个词出现的概率
						- 接下来要真正计算每个词可能的词性了, 也就是
						- ![image.png](../assets/image_1677601605566_0.png)
						- given John 是pronoun的可能性就是, pronoun 本身在start symbol后出现的概率p(t) 乘上是pronoun的情况下John出现的概率p(w|t). 也要酸楚其他的词性的可能性.
						- 因为这个Viterbi 算法的最优前提特点, 我们只需要以前一个词的最优词性为前提即可, 不需要找到所有的可能最大化整句话的prob, 因此wants只需要考虑John是propn的情况作为前一个时间点即可
						- ![image.png](../assets/image_1677601910508_0.png)
						- 如果出现了下个词基于上词概率全部为零, 说明有可能上个词的词性选错了, 这个时候就可以进行重新选择上个词性的操作, 例如
						- ![image.png](../assets/image_1677602026363_0.png)
						- 不再认为race为noun, 而是verb了
					- HMM tagger
						- ![image.png](../assets/image_1677602209463_0.png)
						- ![image.png](../assets/image_1677602237900_0.png)
						- 有很强的前提假设, 未来只基于current state, observation 只基于当前state
						- HMM的其中一个作用就是: **Decoding/inference**: task of determining the hidden state sequence corresponding to the sequence of observations
							- ![image.png](../assets/image_1677602422177_0.png)
					- Vertibi algorithm
						- Dynamic programming
						- ![image.png](../assets/image_1677602486535_0.png)
						- ![image.png](../assets/image_1677602494628_0.png)
						- ![image.png](../assets/image_1677602508863_0.png)
						- ![image.png](../assets/image_1677602606724_0.png)
						- ![image.png](../assets/image_1677602622254_0.png)
						- This gives us the chain of states that generates the observations with the highest probability 惠特比算法会把所有可能性不为0的path都考虑一下
						- Viterbi’s running time is O(SN2), where S is the length of the input and N is the number of states in the model
					- Beam search as alternative decoding algorithm
						- 因为有的tagset太大了, 不适合全部算一遍, 这个时候可以用beam search, 仅仅选取k=2比如说个top most promising paths, 只考虑prob top2的那两个
					- MEMM for POS tagging
						- maximum entropy classifier (MEMM)
						- sequence version of logistic regression classifier, a discriminative model to directly estimate posterior
						- ![image.png](../assets/image_1677602985784_0.png)
						- ![image.png](../assets/image_1677603070831_0.png)
						- 不仅考虑到了上一个时间点的tag信息, 还考虑到了这个时间点的word信息
						- ![image.png](../assets/image_1677603262189_0.png)
						- ![image.png](../assets/image_1677603229915_0.png)
						- ![image.png](../assets/image_1677603241507_0.png)
					- approaches to POS tagging
						- ![image.png](../assets/image_1677603287610_0.png)
		- Module 7.2 Constituency parsing 选区分析, 句法成分分析, 把句子分成一块块, 生成句法结构
			- Goal: Generate the structure for the sentence 生成句子的结构
			- Applications
				- Grammar checking
				- Semantic analysis
					- Question answering
					- Named entity recognition
				- Machine translation
			- Challenges
				- One label per group of words (any length in principle)
				- Structural plus POS ambiguity:
					- 不仅会有每个词词性的模糊, 还有结构上的模糊不清
				- ![image.png](../assets/image_1677605022706_0.png)
			- Constituency parsing 句子成分分析
				- A constituent is a sequence of words that behaves as a unit, generally a phrase, 句子成分, 单元, 短语
				- ![image.png](../assets/image_1677605287630_0.png)
			- The CKY algorithm
				- Grammar needs to be in Chomsky Normal Form (CNF), into two smaller sequences, 变成两部分, 而不是CFG那样子的可以是一堆
				- Rules are of the formX→YZ or X→w
				- ![image.png](../assets/image_1677605434680_0.png)
				- 这种binarisation makes CKY 非常的高效: O(n3|G|): n is the length of the parsed string; |G| is the size of the CNF grammar G
				- ![image.png](../assets/image_1677605540201_0.png)
				- ![image.png](../assets/image_1677605557977_0.png)
				- 每个表中元素代表着这一行中覆盖的他以及他左边的所有词的组合的结构名称, 可以由其左边已有的结构和下面的结构结合, 需要符合两个子结构加起来的字数等于这个结构的总字数, 滴入第二行的VP是3个字, (eats a fish), 需要两个字结构分别是1和2个字, 那就可以是eats和a fish的组合, 分别对应于第二行的第一个(V, VP)和第三行的第二个的a fish(NP), 组合成了VP
				- 右上角的元素会是S, 表示这是一个句子
				- CKY for parsing (CKY 找到所有可能性)
					- 以上的例子recognise了每个结构的成分名称, 如果要用来分析, 要考虑到不同的组合可能性, 并且关联到之前的元素
					- ![image.png](../assets/image_1677605809660_0.png)
					- ![image.png](../assets/image_1677605889793_0.png)
					- 但是可以从上面看出来, 这种方法, 会把所有的情况全部都罗列出来, 每种句子可能的组成方式都会有, 这个情况就相当棘手, 因为如果句子一长, 语法一复杂, 就会有太多类似的树, 我们需要基于统计来分析, 找到最有可能的句法分析(们)
			- Statistical parsing - learning (通过概率找到最可能的那个)
				- Treebanks: “Learn” probabilistic grammars from labelled data e.g. Penn Treebank
				- What does it mean to “learn” a grammar?
				- 学习grammar的意思就是, 根据数据库训练集对每种rule进行一个probability的assign
				- ![image.png](../assets/image_1677606931068_0.png)
				- Probabilistic/stochastic phrase structure grammar (PCFG)
					- a context-free grammar PCFG = (T, N, S, R, q)
					- ● T is set of terminals
					  ● N is set of nonterminals
					  ● S is the start symbol (non-terminal)
					  ● R is set of rules X→ , where X is a nonterminal and is a sequence of terminals & nonterminals
					  ● q = P(R) gives the probability of each rule
					  ![image.png](../assets/image_1677616282899_0.png)
					- 例如NP -> 其他的 所有可能性和为1
					  collapsed:: true
						- ![image.png](../assets/image_1677616351708_0.png)
					- 上图可见 VP有两种分法, 分别为0.7 和0.3的, 就可以写出两颗树来
					- ![image.png](../assets/image_1677616431570_0.png)
					- 而这一整颗树, 也就是这句话的句法分析结构是这样子的可能性, 就是把每个分的操作的可能性给乘起来The probability of each of the trees is obtained by multiplying the probabilities of each of the rules used in the derivation
					- ![image.png](../assets/image_1677616503088_0.png)
					- ![image.png](../assets/image_1677616516289_0.png)
					- 因为是有一个VP的不同分支, 所以在句法分析的意义上这句话出现的概率就是两种句法出现的概率的和
					- 但是 以上的naive的方法又有无法scale的问题: enumerating all options
				- The CKY algorithm for PCFG
					- Application
					  collapsed:: true
						- ○ Recognition: does this sentence belong to the language? 每个语言有自己的
						  ○ Parsing: give me a possible derivation 给出可能的句法分析
						  ○ Disambiguation: give me the best derivation 去除歧义
					- Same dynamic programming algorithm, but now to find most likely parse tree
					- ![image.png](../assets/image_1677618324146_0.png)
					- $\pi[i,j,X]$= **maximum** probability of a constituent with non-terminal X 
					   spanning words i...j  inclusive; i,j指的是一个子序列
					- 例如$\pi[2,5,NP]$意思就是the parse tree with the highest probability for words 2-5, whose head is NP
					- 目的就是通过CKY的方式找到最可能的那个parsing P(t) for the whole sentence
					  collapsed:: true
						- ![image.png](../assets/image_1678193720984_0.png)
					- ![image.png](../assets/image_1677618871752_0.png)
					- ![image.png](../assets/image_1677619043392_0.png)
					- 从上面这两个图可以看出, 递归的应用, 使得这个问题变成了不断寻找split以后的分支的可能性, 要首先考虑分裂以后两个分支可能的句法组合, 例如在计算(3,8)为VP的可能性的时候, 考虑VP可能分成两种组合, 然后再在两种组合内寻找分裂点, 逐渐递归到base case. 概率其实就是层层相乘的结果, 会递归回来. 然后取可能性最大的那个句法组合, 以及该句法组合的split点作为这个(3,8)为VP的可能性
					- 这个方法对每种可能都进行了概率运算, 而不是原始的CKY算法那样子把所有的可能性都找出来, 列出来. 这个CKY for PCFG 会把最优解给出来
					- ![image.png](../assets/image_1677619065356_0.png)
					- Example
						- ![image.png](../assets/image_1678194833743_0.png)
						- 上面的例子, S是最后我们要求的整句话的可能性. 也就是$\pi (1,8,S)$的最大值
						- 可以被分解为, S的split的可能性乘最大可能性的分裂点两边的可能性, 也就是分裂点为2...7使得两边可能性乘起来最大的那个分裂点
						- 由于S只有一种分法, 所以是$1 \times \pi (1,?,NP) \times \pi (1,?,VP)$
						- 上面的问号, 需要一个个试, 找到能让这个最大的那个
				- Evaluating parsers
					- ● Parseval metrics: evaluate structure
					  collapsed:: true
						- ○ How much of constituents in the hypothesis parse tree
						  look like the constituents in a gold-reference parse tree
						  ○ A constituent in hyp parse is labelled “correct” if there is a
						  constituent in the ref parse with the same yield and LHS symbol
						      ■ Only rules from non-terminal to non-terminal
						  ○ Metrics are more fine-grained than full tree metrics, more
						  robust to localised differences in hyp and ref parse trees
					- ![image.png](../assets/image_1678195807396_0.png)
					- 左边四个是我们预测到的有的句法结构, 右边是真实的, 只要我们预测的在右边有, 就precision有了
					- ![image.png](../assets/image_1678196979817_0.png)
					- 这里为了简便, 只考虑了non terminal和非一元的(就是至少有两个分叉的那些句法结构) 左边就只有一个NP, 右边有6个, 所以recall只有1/6了
				- Issues with PCFG
					- Poor independence assumption
						- CFG是context free的, 每个词语的dependency只有他们自己, 不存在相互关联, 他们只依赖于他们自己的POS tag
						- 在英语中，主语功能的NPs更有可能被派生为主语（91%），而宾语功能的NPs更有可能被派生为非主语（66%）。
						- ![image.png](../assets/image_1678197823597_0.png)
						- In this case, S → NP^S VP^S could indicate that the
						  first NP is the subject
						- ![image.png](../assets/image_1678197857276_0.png)
						- 给句法结构注释上了上一层的信息, 告知其为subject的可能性, 因为上一层是S的话, NP更有可能是subject, 因此下一层会更可能是PRP; 而右侧的VP之下的NP则更可能为object, 因此细分
					- Lack of lexical conditioning
						- 缺少了词汇条件, 没有考虑到特俗词汇的句法事实, 例如固定的搭配, 因为有些词汇更容易组合在一起
						- ![image.png](../assets/image_1678203619987_0.png)
						- The affinity between ‘dump’ and ‘into’ is greater than the affinity between ‘masks’ and ‘into’. Conversely , the affinity btw. ‘kilos’ and ‘of’ is greater than btw. ‘catch’ and ‘of’
				- Probabilistic Lexicalised CFG
					- Add annotations specifying the head of each rule, 指明每条规则的head
					- ![image.png](../assets/image_1678203851054_0.png)
					- ![image.png](../assets/image_1678204175144_0.png)
					- head是自下而上传递的, Head是核心语言概念，是每条规则的核心子成分, 可以使用rule来分辨他们, 能够大大提升准确率, 例如
						- ![image.png](../assets/image_1678204398838_0.png)
			-
		- Module 7.3 Dependency parsing
			- 示例
				- ![image.png](../assets/image_1678205292366_0.png)
			- Connect words in sentence to indicate dependencies between them - much older linguistic theory. Build around notion of having heads and dependents.
			- ○ Head (governor), also called argument: origin
			  ○ Dependent, also called modifier: destiny
			- 示例
				- ![image.png](../assets/image_1678206176371_0.png)
				- ![image.png](../assets/image_1678206211503_0.png)
				- 可以见到, prefer被认为是这句话的root, 所有其他的词都是由这个head出发的, 是dependent. 因此prefer是root. flight也是一个巨大的head, linked to很多的dependency
			- Main advantages in dependency parsing
				- ○ Ability to deal with languages that are morphologically
				  rich and have a relatively free word order. E.g. Czech location adverbs may occur before or after object:
				  I caught the fish here vs I caught here the fish
				  不再受到语序的影响
				- ○ Would have to represent two rules for each possible place of the adverb for constituency
				  如果是constituent parsing的话必须为选区副词的每个可能的位置表示两条规则
				- ○ Dependency approach: only one link; abstracts away from word order
				  但是dependency 的话就只需要一个link就可以了
				- ○ Head-dependent relations provide approximation to semantic relationship between predicates and arguments
				- ○ Can be directly used to solve problems such as co-reference resolution, question answering, etc.
				  可以被直接用于 co-reference resolution, question answering
					- For example, if the dependency parser identifies that "John" is the subject of the sentence "He went to the store," then it can be inferred that "he" refers to John.
					- This graph can then be used to identify the relevant information in a text corpus that can be used to answer the question.
			- Dependency formalisms - general case
				- A dependency structure is a directed graph G = (V, A)
				- ○ Has a single ROOT node that has no incoming arcs
				  ○ Each vertex has exactly one incoming arc (except for ROOT)
				  ○ There’s a unique path from ROOT to each vertex
				  ○ There are no cycles A→B,B→A
				- 特性
					- 形成了一棵树
					- 每个词只有一个指向它的head
					- connected
					- 只有一个root
			- Dependency parsing - two approaches
				- Shift-reduce (transition-based)
					- ○ Predict from left-to-right
					  ○ Fast (linear), but slightly less accurate
					  ○ MaltParser
					- Greedy choice of attachment for each word in order, guided by ML classifier, Linear time parsing!
					- ![image.png](../assets/image_1678209000379_0.png)
					- ![image.png](../assets/image_1678209706185_0.png)
					- ![image.png](../assets/image_1678209723527_0.png)
					- ![image.png](../assets/image_1678210184350_0.png)
					- right arc就是把buffer里面的两个词中的右边那个取出来作为determinant, 没取出来的那个是head, 上面的例子里面ate就是head指向了fish
					- Each action is predicted by a discriminative classifier over each move, 使用了ML来预测, feature是stack 顶部的词语和POS, buffer 顶部的词和他的POS
					- 改进方案:
						- Replace binary features by embeddings, Concatenate these embeddings
				- Spanning tree (graph-based, constraint satisfaction)
					- ○ Calculate full tree at once
					  ○ Slightly more accurate, slower
					  ○ MSTParser
			- Dependency parsing - evaluation
				- Accuracy, precision or recall
					- Count identical dependencies: span (non-typed parsing) or span and type of dependency (typed parsing)
					- ![image.png](../assets/image_1678211015392_0.png)
			- Neural parsing - simple approach
				- Parsing as translation
					- Linearise grammar from treebank: convert tree to bracketed representation, all in one line
					- Pair sentences and their linearised trees
					- 把dependency结构变成线性的一行话, 然后pair起来, 直接让模型学
					- ![image.png](../assets/image_1678211163530_0.png)
				- Graph-basedmethods
	- Week9 Revision
	  collapsed:: true
		- Exam topics
		  collapsed:: true
			- 2122
			  collapsed:: true
				- 1. Smoothing, CKY, HMM & Viterbi, character-level language model & token-level (老ppt里)
				- 2. many-to-one or many-to-many RNN, metrics, machine translation, BPE, Transformers and (attention-based) RNNs
				- 3. self-attention, BERT, MLM, how NER is performed, Transformer parameters
			- 2021
			  collapsed:: true
				- 1. word2vec, skip-gram, BPE, n-gram language model, CKY, parse trees
				- 2.  word sense disambiguation, pre-processing techniques, CNN for hate detection
				- 3. FFLM, number of learnable parameters, GRU, NMT, summarise hidden states of RNN, BLEU, loss of NMT
				- 4.   neural models based on the autoregressive, autoencoding, comparison, HMM, POS, Viterbi, how Transformers can learn long sequences
			- 1920
			  collapsed:: true
				- 1. skip-gram using softmax and negative sampling,  n-gram language model, CKY
				- 2. NN for sentiment classification, CNN for sentiment analysis,
				- 3. markov assumption, BPTT, cross-entropy, attention
				- 4. word2vec skip-gram model (original formulation), n-gram language model, recurrent neural net language model, and transformer-based embeddings, CNN in text classification, HMM, Viterbi, transformer-based and attentive recurrent neural machine translation.
			- 1819
			  collapsed:: true
				- 1. negative sampling and skip-gram and loss function, bigram language model for a corpus, pre-processing techniques, perplexity, mitigate issues with zero-count, HMM POS table
				- 2. CNN for sentiment analysis, classifier
				- 3. (RNN) language model, initialise an RNN with pre-trained word embeddings, self-attention, role of attention in NMT, loss function in RNN, metrics to evaluate RNN LM
			- 总结一下:
			  collapsed:: true
				- 计算: CKY, HMM & Viterbi, n-gram, parameters计算
				- 难点: CNN的细节, filer size, padding, 如何应用的; 各个算法使用的loss func, metrics, 基础的pre-processing的方法们, disambiguate
		- Ed
		  collapsed:: true
			- Layer Norm serves 2 purposes in the Transformer:
			  collapsed:: true
				- 1) Rescaling inputs to improve generalization
				- 2) Inducing non-linearities into the model
			-
		- ---
		- 全局的理解
			- Classifier
			  collapsed:: true
				- Bag of words
				  collapsed:: true
					- 提炼出训练集中有多少词汇, V维向量, 每句话用V维向量每个维度的词汇出现的次数作为其vector来进行后续任务. 会出现很不准确的情况, 其假设是所有词语都是单独的feature, 但是实际上词语间有强关联, 例如'not good'两个feature间highly correlated, 此时P (x1, x2|y) is not equal to P (x1|y) × P (x2|y), 此时会极大影响模型判断.
				- Binary Naive Bayes
				  collapsed:: true
					- 一句话中(某个正例)出现了和没出现某个feature, 而不是这个feature的个数. 所以训练的时候就是p('good'|+)计算的是出现了正例的句子数除以正例句子数; 如果add one smoothing的话, 分子加一, 分母加上feature种类总数
			- Language Model
			  collapsed:: true
				- language model可以认为是decoder, 用于生成文本, 也可以用于预测某个文本的存在可能性
				- CE用来计算loss, PPL用来评估语言模型, 1最好
				- n-gram仅考虑了统计数据, 长dependency没办法考虑到
				- RNN 可以用于从前往后生成文本
				- Bi-directional RNNs 可以用于理解前后语义的情况下生成文本, 可以用于语法纠正
				- LSTM 用cell state和gates选择性保留gradient
				- 4-gram FF LM 在concat了三个embedding以后进行了一个tanh激活, 再投到V进行softmax, 目的是提取到有用的希望被关注到的feature
				- Beam search, greedy decoding, Temperature sampling: 保留k个, 保留1个, smooth以后的softmax的概率用来sample
			- Encoder
			  collapsed:: true
				- BERT只有encoder, 因为目的是encode texts to latent representation, 来表示某个词带有context的含义或者整句话的含义, 服务于下游目标
				- 注重编码提取上下文特征用于下游, 或预测中间词: classification, MLM, NER, Q&A; BERT, RoBERTa
			- Decoder
			  collapsed:: true
				- GPT只有decoder, 是个language model, 用来生成文本, 而不是encode 文本. 训练过程中就是会看前面已经有的信息, 然后生成下一个字符, 它也有理解语境的能力, 只是直接用来生成了. 他没有source and target的概念
				- 注重直接用已有输入和模型生成预测后面的词: LM, text gen, dialogue systems, summarization, simplification, classification(最后一个输出), Q&A, NLI; GPT
			- Encoder-Decoder
			  collapsed:: true
				- Transformer 两个都有, 适用于seq2seq的任务, 比如说 machine translation, encoder and decoder can have separate vocabularies and focus on different languages.
				- seq2seq场景, 需要对source进行特殊理解的场景: translation, summarisation, Q&A; Transformer, BART
				- 也可以用于classification, 在decoder的最后输出后面接分类层
				- 当我们增加model dim的时候self-attention和PWFF的parameter的增长都是quadratic的, 因为Wqkv是D->D/h, PWFF是D->ff(4D)->D
				- the cross attention matrix shape of [T x S], T is my target sequence length and S is my input sequence length
				  collapsed:: true
					- Q in T x D; K in S x D; V in S x D
					  QK^T = [T x D] x [D x S] = T x S <---- This is the cross attention matrix that is being asked about
					  (QK^T)V = [T x S] x [S x D] = T x D <---- This is the output of the cross attention formula
			- CNN
			  collapsed:: true
				- perform well if the task involves key phrase recognition
				- Text classification, NER, Semantic role labelling
			- RNN
			  collapsed:: true
				- d2l里面输入的是V大小作为input, 输出的是hidden state大小的vector
				- 正常训练中, 我们可以通过事先训练好的word embedding 来作为输入
				- 可以作为 LM, 可以作为Decoder, 双向的RNN适合作为Encoder
				- Sentiment analysis, text generation
			- M7
			  collapsed:: true
				- POS: For many applications such as spam detection, sentiment analysis at scale (e.g. millions of emails going through a server) you can focus on nouns,verbs and adjectives removing redundant tokens. Another common application is Named Entity Recognition where we use POS tagging to only recognise nouns assuming named entities (people, organisations, places etc) are nouns in English.
		- 基础
		  collapsed:: true
			- ![image.png](../assets/image_1678285711404_0.png)
			- ![image.png](../assets/image_1678285737690_0.png)
			- For regression (predicting a score):
			  collapsed:: true
				- Output layer of size 1
				- Linear activation for the output layer, so the value is not restricted
				- Use Mean Squared Error (MSE) as the loss
			- For binary classification:
			  collapsed:: true
				- Output layer of size 1
				- Use sigmoid to predict between two classes (0 and 1)
				- Use binary cross-entropy as the loss
			- For multi-class classification (predicting one class out of many):
			  collapsed:: true
				- With k classes, have output layer of size k
				- Use softmax activation to get a probability distribution
				- Use categorical c
			- For multi-label classification (possibly predicting many classes):
			  collapsed:: true
				- With k classes, have output of size k
				- Use sigmoid
				  activation, making each output is an independent binary classifier
		- Word representations
			- 难点
				- Intuition: Represent words as vectors, so that similar ones have similar vectors
				- options:
					- [[One-hot]] (1-of-V): V长度的vector, 每个词占用一个bit的1, 其他都是0
					  collapsed:: true
						- super sparse, all are orthogonal and equally distant
					- [[WordNet]]: map words to broader concepts, 猫->猫科哺乳
					  collapsed:: true
						- rely on manually curated database
						- miss rare and new meanings
						- ambiguity mouse有两个含义
					- Distributed vectors: 一堆属性的程度来形容一个词语, furry, danger的打分可以区分猫和老虎
					  collapsed:: true
						- elements representing properties shared btw words,
						- 使用cosine来描述相似度
					- Count-based vectors: 用其周边词的出现频率来描述这个词
					  collapsed:: true
						- vector还是很长, 且sparse, 很多都是0
						- 要解决那些到处都出现的词的问题, 他们没有额外信息
						- 使用TF-IDF 来weight 这些context词 Term Frequency-Inverse Document Frequency. d是target word, w是context word, D是ducument sentences
						- TF: w和d一起出现的次数/所有context word和d一起出现次数和
						- IDF: log(|D|/有w的句子个数)
						- weight = TF*IDF
					- [[Word Embeddings]]
						- embedding the words into a real-valued low-dimensional space; short vectors and dense
					- [[Word2vec]] as word embeddings
						- usually lowercase as pre-processing (tokenise)
						- Continuous Bag-of-Words ([[CBOW]])
						- [[Skip-gram]]
				- Continuous Bag-of-Words ([[CBOW]])
				  collapsed:: true
					- Predict the target word wt based on the surrounding context words
					- ![image.png](../assets/image_1678374526858_0.png)
					- 有一个embedding layer和一个output layer, context words们经过embedding layer得到他们的embeddings然后加起来然后fed into output layer生成V大小的向量给softmax, 给定target的one hot就可以和softmax的结果计算**categorical [[cross-entropy]]**了
				- [[Skip-gram]]
				  collapsed:: true
					- Predict the context words based on the target word wt
					- ![image.png](../assets/image_1678374580463_0.png)
					- 也有两个weight matrices, 两个都可以用作embedding. 与CBOW不同的是, skip-gram输入了target, 要预测context. 根据context窗口的大小, 会有不同个数的context 词. 例如如果有四个context词的话, 就会与这个target形成四个training samples, 更新网络四次. 同样也是对输出的V进行softmax, 然后与正确的context计算**categorical cross-entropy**. 实际训练的loss是同时一个batch有好几句话, 每句话都选择同一个位置的把这四个context的log likelihood加起来, 希望他变大
					- Intuition 是optimise target的embedding和context经过w‘的embedding的相似度
					- Downside 是每次算一个context的概率都需要对整个V进行softmax, 例如300dim的embedding和1w的V, 因为有两个weights, 就需要做5M的计算
					- 解决方法:
					  collapsed:: true
						- lookup, 直接index到需要的embedding, 而不是做矩阵乘法
						- negative sampling: logistic regression for the real context word and k other noise words 即最后的输出结果找出那1+k个词的位置进行sigmoid, 仅对他们几个进行bp, 从而只更新这几个词的embedding. noise词的选择可以是random的也可以是根据频率来. 大模型需要的negative 少一些, 小模型多一些
						- 假如1个context配5个negative, 每次只需要更新6个 embedding, 在output layer只需要更新6*300个parameters, 而不是3M了, 当然embedding layer还是一样的
				- 仍然遗留的问题
				  collapsed:: true
					- 罕见词和未见过的词
					- morphological similarity 单复数(因为我们用了一些pre-processing
					- 词语的语境意义无法分辨
					- Contextualised word embeddings 可以解决, 这被现代的语言模型所使用
				- Byte Pair Encoding([[BPE]])
				  collapsed:: true
					- Instead of manually specifying rules for lemmatisation or stemming, let’s learn from data which character sequences occur together frequently
					- BPE首先会得到base vocabulary 即所有单个的字符, 然后通过在一轮轮训练中得到merge最常见的符号pair们的rule来得到单词们的共同的root之类的元素作为新的词汇, 直到得到想要的词汇量为止, 因而可以处理一些不认识的单词, 因为最差的情况就是把一个词分解成了一个个字符.
					- byte-level (可以处理所有的字符)的 BPE被广泛运用在了[[GPT]]还有[[BERT]]模型中作为tokeniser, 用于把输入的词语分解到char以后merge成词汇表中的的词汇, 例如 lowest_被分解成一个个字母后会apply训练得到的merge rule merge成low和est_ 两个词语 (_用来标注这是词语的结尾, 可以用来重新组合), 这样子就可以被网络进行处理了 (embedding啊什么的)
				- [[Wordpieces]]
				  collapsed:: true
					- 与BPE相似, 但是除了每个词的开头char以外都会在前面##, 选择mergerule不一样, merge成最长的subword
				- [[BPE]]的优势
				  collapsed:: true
					- The advantage of BPE in NLP is that it can effectively handle out-of-vocabulary words and rare words by breaking them down into smaller subword units that are more likely to be in the vocabulary. This can improve the performance of NLP models that rely on word embeddings, such as neural machine translation, sentiment analysis, and named entity recognition.
			- 疑点
				- skip-gram的loss和训练方式到底是怎么样的
				  collapsed:: true
					- 暂时觉得就是
		- Model architectures
		  collapsed:: true
			- 难点
			- 疑点
		- ---
		- Classification
		  collapsed:: true
			- 难点
			  collapsed:: true
				- common NLP classification tasks
				  collapsed:: true
					- 1. Hate speech detection
					  2. Sentiment analysis
					  3. Fact verification
					  4. Spam detection
					  5. Error detection
					  6. Natural Language Inference (NLI)
				- [[BoW]] (Bag of words) 来表示一句话
				  collapsed:: true
					- 用某些词语的词频来表示一句话, 每个词语可以看作是一个feature, 可以只选一些有意义的词语, 例如说 good bad
					- Suppose we have a collection of three documents:
					- Document 1: "The cat in the hat."
					  Document 2: "The dog chased the cat."
					  Document 3: "The cat ran away from the dog."
					- To apply the BoW model to this collection, we first create a vocabulary of all the unique words in the documents:
					- Vocabulary: the, cat, in, hat, dog, chased, ran, away, from
					- Next, we represent each document as a vector of word counts, where the element at index i represents the count of the ith word in the vocabulary in that document. For example, the BoW representation of Document 1 is:
					- Document 1 BoW vector: [2, 1, 1, 1, 0, 0, 0, 0, 0]
					- This means that Document 1 contains 2 occurrences of the word "the," 1 occurrence of the words "cat," "in," and "hat," and 0 occurrences of all the other words in the vocabulary.
					- Similarly, the BoW vectors for Document 2 and Document 3 are:
					- Document 2 BoW vector: [1, 1, 0, 0, 1, 1, 0, 0, 0]
					  Document 3 BoW vector: [1, 1, 0, 0, 1, 0, 1, 1, 1]
					- We can now use these BoW vectors to perform various tasks, such as document classification or information retrieval, by comparing the similarity between vectors using measures such as cosine similarity or Euclidean distance.
				- [[Naive Bayes Classifier]]
				  collapsed:: true
					- ![image.png](../assets/image_1678392422957_0.png)
					- ![image.png](../assets/image_1678392699558_0.png)
					- 找到最符合数据x的label y, 也等于给定y出现x的概率 乘上y本身这个label出现的概率, 其中P(x|y)有个independence假设, 可以是所有feature的conditional prob的乘积
					- ![image.png](../assets/image_1678393282708_0.png)
					- 先验p(y)=3/5 和 2/5 分别是正和负; P(good | +）= 2/4 四个划分为正例的feature词里有两个是good
					- Add-one smoothing: 解决probability 为零的问题
					  collapsed:: true
						- ![image.png](../assets/image_1678394007674_0.png)
						- P(good | +）= 2+1 / 4+3 = 3/7; 3是有三种不同的词
					- 假设测试例中这三个词都出现了一次, 最后为正的概率就是3/5 * 三个词condition+的概率积
					- improvement
					  collapsed:: true
						- 1. 句子中重复出现某个特征词, 也不作重复处理, 当作是一个 (Binary Naive Bayes, 即便有两个movie, 也只乘一遍它的概率)
						- 2. append ‘NOT_’ after any logical negation until next punct. 使得否定的含义更清晰, 不然朴素贝叶斯只根据词语本身来判断, 无法看到前面的否定含义, 也无法得到context和dependency
					- 问题: context, new words
				- [[Logistic Regression]]
				  collapsed:: true
					- 学习给每个词语的weight (How important an input feature is to the classification decision) 和bias, 经过logistic function(sigmoid)来进行分类, 通过BCE loss来得到loss, 通过gd更新参数
					- 如果是多个类的话, 可以用fc map到多个类, 相当于给每个类一条weight, 最后得到的结果进行softmax 这个叫做multinomial logistic regression
				- 两种baseline分类方法的重要结论
				  collapsed:: true
					- LR: better dealing with correlated features, larger datasets
					- NB: faster, good on small datasets
					- 帮助我们知道哪些feature是有用的, 如何和class联系的, 帮助我们理解dataset, 可以与大模型比较来理解这个任务的nature
				- 为什么我们需要更好的理解我们的数据呢?
				  collapsed:: true
					- To select and engineer features
					- To identify patterns and relationships. This can help us to understand the underlying structure of our data and make more informed decisions about how to model and analyze it
					- Preprocessing optimization: By understanding the data, we can identify common preprocessing steps and wisely adjust them
					- Identification of data quality issues
				- Neural Networks ([[NN]]s)
				  collapsed:: true
					- 2个非常非常naive的表达句子的方式
					  collapsed:: true
						- 每个词都有比如说3维的表达了, 句子的表示就用所有词的每一个维度的avg来表示, 形成一个三维的句表达, 表现还可以
						- 还有一种固定下来句长, 用所有的词向量的concat表达, 这样子首先长度fix了, 其次词语位置固定了, 非常不好
				- Recurrent neural networks ([[RNN]]S)
				  collapsed:: true
					- Usually the last hidden state is the input to the output layer
					- $h_{t+1} = f(h_t,x_t) = tanh(Wh_t + Ux_t)$, $y_t = W_{hy}h_t + B_y$考过的公式
					- $W \in \mathbb{R}^{H\times H}, U\in \mathbb{R}^{E\times H}$
					- ![image.png](../assets/image_1678399171952_0.png)
					- Vanishing gradient problem as tanh's derivatives are between 0 and 1
				- [[CNN]]s
				  collapsed:: true
					- Filter: width是embedding dims, height是window大小, 即一次考虑几个词, 通常2-5(bigram to 5-gram)
					- stride: 每次移动filter的多少
					- ![image.png](../assets/image_1678400671710_0.png)
					- maxpooling就是同样用window对这得到的结果进行汇聚, 另外有多少个filters就有多少这样的东东
				- RNNs vs CNNs
				  collapsed:: true
					- RNNs perform better when you need to understand longer range dependencies, RNNs are typically suited for tasks that require understanding the temporal dependencies between the input features. well-suited for tasks involving variable-length sequences.
					  collapsed:: true
						- Language modeling: predicting the probability distribution of the next word in a sequence given the previous words
						- Speech recognition: recognizing spoken words or phrases from an audio signal
						- Machine translation: translating a sentence from one language to another
						- Named entity recognition: identifying and classifying named entities in text
						- Sentiment analysis: classifying the sentiment of a text as positive, negative, or neutral
						- Text generation: generating new text based on a given input or a specific style or topic
					- CNNs can perform well if the task involves key phrase recognition
					  collapsed:: true
					  CNNs are typically suited for tasks that involve extracting local features from the input, such as identifying important n-grams or combinations of words that are indicative of a certain category or relationship. They are good at processing fixed-length inputs, making them well-suited for tasks that involve text classification or semantic role labeling.
						- Text classification: classifying text into one or more predefined categories
						- Semantic role labeling: identifying the semantic relationships between words in a sentence
						- Question answering: answering natural language questions based on a given context or passage
						- Named entity recognition: identifying and classifying named entities in text
						- Relation extraction: identifying the relationships between entities in text
						- 举例: Determining if an article is Fake News based on the headline
						  collapsed:: true
							- CNNs are well-suited for this type of classification task because they can capture local features or patterns within the text that are important for determining the category. In the case of a headline, local features could include specific words or combinations of words that are indicative of fake news, such as "shocking new evidence" or "breaking news". These features can be detected by the convolutional layers of the CNN, which scan the input text with a sliding window to extract local features.
				- [[De-biasing]]
				  collapsed:: true
					- preventing a model learning from shallow heuristics
					- 有些feature很浅很明显带有偏见很容易被错误利用, 我们不希望这些浅层的信息被model抓到而不去探索深层次的东西, 比如说sentence length, 一切阻止我们generalise model的东西
					- Possible strategies
					  collapsed:: true
						- Augment with more data to balance bias
						- filter data
						- make model predict different from predictions based on the bias
						- prevent model finding the bias
					- bi is the prob of each class given the bias; pi is the prob from our model
					- [[Product of Experts]]
					  collapsed:: true
						- 训练的时候, 我们让我们的模型去学习bias之外的东西, 通过增加bias的类的prob, 可以鼓励自己的模型learn everything around that bias
						- $\hat{p}_i = softmax(log(p_i)+log(b_i))$
						- 这个bias的prob可以由比如: 根据某个bias的feature来得到, 用一个简单的模型学习, 仅用一个sample中的部分数据, 减少数据集中sample数量
					- Weight the loss of examples based on the performance of our biased model
					  collapsed:: true
						- 用 1-bi给正确类的预测prob 来weight loss; 如果bias对这个结果非常自信, 1-1=0会大大减少由这种简单的sample带来的权重更新; 这会鼓励模型更多的去看那些bias model 觉得苦难的, 发现不了的深层次的feature. give more weight to examples that are more difficult for the biased model to predict accurately, with the aim of reducing the impact of any biases present in the original model.
					- 适用场景: out-of-distribution test set, generalised for 更多分布外数据
					- 不适合: geder bias, 直接hide gender info就好
				- Micro averaged F1 = acc
			- 疑点
		-
		- [[Language Modeling]]
		  collapsed:: true
			- 难点
			  collapsed:: true
				- 什么是[[LM]]
				  collapsed:: true
					- Language modeling involves assigning probabilities to sequences of words.
					- Predicting the next word in a sequence of words
					- Predicting a masked word in a sentence
					- 简而言之就是预测和**生成**词, 其本身也能够计算一个句子存在的可能性
				- [[N-gram]]
				  collapsed:: true
					- 是最基础的LM, 通过训练集中的统计数据, 来进行生成, 选择最有可能的(或者按概率)的下一个词语, 可以用CE, perplexity来衡量其好坏, 即在测试集中适应程度. 由于n的限制, 没法找到长距离的dependency. (NN LM们可以解决)
					- P(w|h), w在given 前面的词语们h, 即history的概率, n-gram就是只考虑w和h一共n个词, 这个概率可以由统计得到, 即abc一起出现的数量除以ab一起出现的数量
					- $P(w_n | w_1^{n-1}) \approx P(w_n | w_{n-N+1}^{n-1})$ (assume they are likely)
				- 整句话的prob可以写作所有n-gram的概率乘积, 在log space可以表达成加法
				  collapsed:: true
					- ![image.png](../assets/image_1678410089751_0.png)
				- [[Perplexity]] 困惑度
				  collapsed:: true
					- It’s the inverse probability of a text, normalized by the # of words, measure of the surprise in a LM when seeing new text 惊喜程度越低, 越熟悉. 测试集中越低越好. 是个intrinsic evaluation
					- 最小为1, 如果完全随机的话最大就是|V|
					- ![image.png](../assets/image_1678410220134_0.png)
					  id:: 640a81b3-9d4c-4701-a2bd-942addba3574
				- Cross Entropy
				  collapsed:: true
					- 问题在于, 我们没办法知道真实的应该的prob, 下面这个就是一句话的CE, normalised by N, 即词语数量
					- ![image.png](../assets/image_1678410456421_0.png)
					- 真实处理过程中 是有个window size的, context只会取一部分, 还可以决定有没有strided sliding window
					  collapsed:: true
						- ![ppl_sliding.gif](../assets/ppl_sliding_1678875647324_0.gif)
				- Converting Cross Entropy Loss to Perplexity
				  collapsed:: true
					- $Perplexity(M) = e^H$
				- 虽然LM是生成, 但是也可以帮助分类, 例如说事先概括文本, 纠正错误, 甚至直接让他进行分类. GPT之类的模型考虑的可不是单纯概率, 还有语境意义等等. 也就可以用让他们perform classification, 做选择题, NLI之类的任务来评估
				- 如何解决sparsity的问题
				  collapsed:: true
					- Use Neural-based language models
					- Add-1 Smoothing
					  collapsed:: true
						- 对于那些sparse statistics的词, steal probability mass from more frequently words; 可以获得更好的泛化性能, 但是会从高prob处偷走很多, 给unseen的太多了
						- ![image.png](../assets/image_1678411568510_0.png)
					- Back-off
					  collapsed:: true
						- back-off and see how many occurrences there are of 退一步看少一个词的该组合, 如果再没有, 再少看一个词, 直到只剩下一个自己
						- ![image.png](../assets/image_1678412078151_0.png)
					- Interpolation
					  collapsed:: true
						- 插值法, 给n-gram变成 几个比n等和小的n-gram的加权组合
						- ![image.png](../assets/image_1678412156766_0.png)
			- 疑点
		- ---
		- Neural language models
		  collapsed:: true
			- 难点
			  collapsed:: true
				- 优势和进步
				  collapsed:: true
					- 避免了n-gram的sparsity问题
					- 能够contextual word representations
				- Feed-forward LM ([[FFLM]])
				  collapsed:: true
					- ![image.png](../assets/image_1678452804833_0.png)
					- 4-gram的话就是3个context words经过embedding, concat起来作为context用tanh激活后用一个FC来进行分类预测第四个词. 效果比smoothed 3-gram LM好, 但是还得看RNN
				- [[RNN]]s for language modeling
				  collapsed:: true
					- $h_{t+1} = f(h_t,x_t) = tanh(Wh_t + Ux_t)$,  $y_t = W_{hy}h_t + B_y$
					- $W \in \mathbb{R}^{H\times H}, U\in \mathbb{R}^{E\times H}$
					- ht是最后一个hidden state, 可以作为整句话的context, 输出给一个输出层, 用于下游任务
					- Vanishing gradient
					  collapsed:: true
						- Activation functions: sigmoid and tanh cause the gradients to become very small as they are backpropagated through time
						- Time step size: too large, gradient becomes unstable
						- Initialization of the weights
						- 可以用 gradient clipping 帮助把gradient限制在一个固定区间内, 而不改变方向
					- Many-to-many
					  collapsed:: true
						- 每个时间t都会给出一个output预测下一个词, 与下个时间的正确词进行CE
					- Teacher forcing
					  collapsed:: true
						- 在训练的时候不用上一个时刻的预测词作为下一个时刻的输入, 而是使用真实的标签作为指导输入, 让训练更加稳定
					- Weight tying 如何减少weights
					  collapsed:: true
						- 把hidden state看作是embedding, 用同一个embedding matrix来map回V; 因此使用的embedding dim和hidden state dim相同, 下图中的U 为 H x H, E为H x V
						- ![image.png](../assets/image_1678457210367_0.png)
				- [[Bi-directional RNN]]s
				  collapsed:: true
					- 如何用于language modeling: The hidden states from both directions are then concatenated to obtain a final hidden state, which is used to make the prediction. 用双向信息来预测下一个词.
					- 应用场景: 由于获得了双向信息, 可以detect grammatical errors, 也可以将两个方向的最终hidden state拼接起来, 作为整句话的hidden state (长度为2H)
					- 优势: capture both past and future context of a sequence, can better handle noisy or missing information in the input sequence
				- [[Multi-layered RNN]]s
				  collapsed:: true
					- each layer can learn to represent a different level of abstraction in the data. lower layers learn local dependencies, higher levels learn long-term dependencies
				- [[LSTM]]
				  collapsed:: true
					- ![IMG_1168.jpeg](../assets/IMG_1168_1678459989163_0.jpeg)
					- ![image.png](../assets/image_1678460035395_0.png)
					- How do LSTMs help with Vanishing gradients
					  collapsed:: true
						- The gradients through the cell states are hard to vanish
						  Additive formula means no repeated multiplication
						  forget gate allows model to learn when to preserve gradients
				- [[GRU]]s
				  collapsed:: true
					- GRU is quicker to compute and has fewer parameters. no cell state, only history
			- 疑点
		- Machine Translation
		  collapsed:: true
			- 难点
			  collapsed:: true
				- Statistical Machine Translation
				  collapsed:: true
					- 有多个sub-models
					- 需要两门语言的corpus, 对应的句子对们
					- p(t|s) = p(s|t)p(t)
				- Pipeline of Statistical Machine Translation
				  collapsed:: true
					- Alignment model
					  collapsed:: true
						- 寻找目标语言库中, 对应的phrase pairs
					- Translation model
					  collapsed:: true
						- 对对应的phrase pair进行概率评估, 每个pair给出可能性, 找的是p(s|t) 给定一个可能的target, source是这样这可能性
					- Language model
					  collapsed:: true
						- 对给出来的句子, 进行目标语言内存在可能性评估 也就是 p(t)
				- Downsides of statistical MT
				  collapsed:: true
					- Sentence alignment: 多种翻译可能, 也可能被夹断
					- Word alignment: 不一定有对应的词语
					- Statistical anomalies: 统计上的过多搭配, 可能主导翻译
					- 习语, 不存在的词 ==
				- Neural Machine Translation
				  collapsed:: true
					- en-de used in seq2seq tasks (summarisation, Q&A)
					- X->encoder->latent encoding->decoder->Y;
					- encoder 是个编码器, decoder就是一个语言模型
					- BiRNN 通常是个不错的选择, 可以考虑到历史和未来, 利用上可以用hidden states for each word or the concat of two last HS ([h_i; h’_0])
					- BiRNN表示整句话也可以利用[h_i; h’_0] 或者avg of 所有词位, 输出给decoder RNN 作为hidden state(不是0了), 然后给一个[SOS]作为初始input就可以开始auto-regressive了, 训练的时候同样可以采用teacher forcing (按照比例来比如50%)
					- Teacher forcing
					  collapsed:: true
						- Using incorrect predictions as input can cause accumulation of errors, can be hard to optimise, so teacher forcing
						- 但是有[[Exposure Bias]] 问题, 模型只会依赖于训练时的正确引导input, 而不会根据自己的预测来进行了
					- ![image.png](../assets/image_1678465689209_0.png)
					- BiRNN问题和缺点
					  collapsed:: true
						- d维无法很好保存历史信息, decoder中会加入到生成的词语的信息, 也会让encoder来的context c消失, 依然有vanishing gradient problem
						- 这也引出了attention, 不再只依靠c, 而是能够直接看到输入的所有信息
					- Attention (additive MLP)
					  collapsed:: true
						- 对每一个decoder step 都会用上一步的state与所有的encoder hidden state 进行attention, 得到每个hidden state的权重, 加权求和得到一个这一步特有的context c
						- 权重是由这个encoder的ht与decoder中的s的相关性定义的, 相关性由两个权重matrix与他们分别相乘的和的tanh得到, v把d维缩放到1, 变成一个energy score. 下面的a会得到i个值, softmax完了以后就是每个hi的weight了
						- ![image.png](../assets/image_1678469326609_0.png)
						- Decoder 会用到st-1, ct, yt-1, 即hidden state, context, input来计算新的st, 计算y^t时, 会用到st和ct
					- BLEU: 最受欢迎的evaluation metric, modified precision (MP)
					  collapsed:: true
						- MP = total unique overlap/total MT ngrams
						- total unique overlap是提炼出ngram以后, 每个gram在两个reference中找match, 哪个match多这个ngram的count就算哪个数
						- ![image.png](../assets/image_1678470730225_0.png)
						- BLEU则是1-n个gram的MP的乘积的Brevity penalty结果, 惩罚短句子, 希望获得差不多长度的句子
						- ![image.png](../assets/image_1678470970250_0.png)
					- 其他的翻译任务metrics:
					  collapsed:: true
						- Chr-F is an F beta -score based metric over character n-grams. This metric balances character precision and character recall
						- TER (number of minimum edits) is performed at the word level, and the “edits” can be a: Shift, Insertion, Substitution and Deletion.
						- ROUGE-L: F-Score of the longest common subsequence (LCS), /ref就是recall, hyp就是precision
						- METEOR: 适用于概括和标题, 有多少chunks是match的, 考虑了形态变化
						- BERT score: 不再使用ngram, 使用BERT-based contextual  embedding, 比较他们的embedding 的cosine similarity, 每个词都有一个emb, 两两计算, 用最好的匹配. 缺点在于依赖于bert, 不同的model给出不同结果
					- Inference
					  collapsed:: true
						- 由于不再有label, 需要auto-regressive, 就得要选择给下一个input的词是哪一个
						- Greedy decoding: 直接只用最高概率的, 只保留一条线
						- Beam search: 保留k条线, 取乘起来总概率最高的那k条线
						- Temperature sampling: divide logits by T, A high temperature >1 value increases the randomness of the sampling process by making the less probable words more likely to be selected. Thus more diverse.  controlling the trade-off between creativity and coherence in generated text from NLP models, such as language models. 用除以T来smooth softmax, 得到想要的flat的概率分布, 用这个新的flat的概率分布sample词, 而不是取概率最高的那几个. 这里是采样!不是取最大!
					- Data Augmentation
					  collapsed:: true
						- Backtranslation
						- Synonym replacement
						- Group similar length sentences together in the batch 
						  Train your model on simpler/smaller sequence lengths first
			- 疑点
		- Transformers
		  collapsed:: true
			- 难点
			  collapsed:: true
				- Encoder, decoder and EN-DE
				  collapsed:: true
					- Encoder: 注重编码提取上下文特征用于下游, 或预测中间词: classification, MLM, NER; BERT, RoBERTa
					- Decoder: 注重直接用已有输入和模型生成预测后面的词: LM, text gen; GPT
					- EN-DE: seq2seq场景, 需要对source进行特殊理解的场景: translation, summarisation, Q&A; Transformer, BART
				- 结构:
				  collapsed:: true
					- ![image.png](../assets/image_1678488097639_0.png)
				- Self-attention (scaled dot product attention)
				  collapsed:: true
					- 每个词都和其他词attention, 找到对应attention分数, 根据这个分数来weight每个词语对应的V的值, 求加权平均得到这个词最终的value
					- QKV三个新东西, 由三个W project词的embedding得到, 从embedding dim D(也通常是model dim, n个heads concat起来的总长度) map到 单个head的dim d_h
					- 每对Q和K的点积结果为标量, 表示相似度, 一共有SxS个, 用根号d_h normalise后的结果是注意力分数, 对每行做softmax以后就是weight, 维度不变. 句中词的V的加权平均就是做SA的词的value,  最后是S x d_h = Z, S个词的values
					- ![image.png](../assets/image_1678489267356_0.png)
				- Multi-head attention
				  collapsed:: true
					- Intuitively multiple attention heads allows for attending to parts of the sequence differently (e.g. some heads are responsible for longer-term dependencies, others for shorter-term dependencies)
					- 输入进来每个词的embedding是D, h个heads, h组QKV projection matrices把他们变成了d_h, head dim = d_h = model dim(D) / h, 用来做SA. 每个词的h个头的d_h会concat起来重新得到D作为这个词的encoding, 经过一个FC还是D, 保证了D不变, 可以在addnorm以后进行再一次的MHA了
				- Layer normalisation (NORM)
				  collapsed:: true
					- 对每句话的每个词的整条特征向量进行norm, 即对每个词的D维向量进行norm
					- BN的话则是对所有句子中相同词语位置的词语的相同维度位置的数字们进行norm
					- Why gamma and beta have d dims?
					  collapsed:: true
						- they allow for different scaling and shifting of each feature dimension. In NLP tasks, different feature dimensions can have different magnitudes and ranges, which can make it difficult to normalize them effectively with a single scalar factor.
				- Residual Connection (ADD)
				  collapsed:: true
					- allows the current layer to focus on learning the difference between the two outputs, rather than learning an entirely new transformation
				- Position Wise Feedforward Network
				  collapsed:: true
					- MLP, 同样的weight会用于每一个句子中的词语
				- Positional Encodings
				  collapsed:: true
					- Transformers are position invariant by default, 对于不同词语的顺序没有感知, not inherently sequential, 需要positional encoding 来inject position information into embeddings, 我们直接在词embedding上加PE. dim轮流用sin cos, 值的大小受pos影响
					- Intuitively, front words are 010101; back words effected with diff patterns
					- ![image.png](../assets/image_1678490794617_0.png)
				- Decoder
				  collapsed:: true
					- Masked multi-head self-attention
					  collapsed:: true
						- 预测时, encoder encode了source, 作为decoder的参考, 我们给decoder一个SOS, 每次预测的新词, 和前面的词拼起来作为新的输入来预测新的词 auto-regressive 直到达到限制或是EOS
						- 训练时限制attention不看到未来的词, 我们feed是整个sequence, 但每一个时间点都只能看到现在和过去的, transformers的训练是百分百的teacher forcing. mask set values in the upper triangular to be -inf, 来让softmax不关注
					- cross attention
					  collapsed:: true
						- 为了让decoder每一步都需要知道要关注哪些encoded tokens, perform attention to all encoded tokens in the last layer of the encoder, using Q from current decoder layer, and K, V from encoder. 即在搞清楚target sequence里面自己的关系以后, 再和source 探索一下关系, 获得新的values 这里的cross attention matrix T x S.
					-
			- 疑点
		- ---
		- Pre-training models
		  collapsed:: true
			- 难点
			  collapsed:: true
				- 全局理解
				  collapsed:: true
					- fine-tune是针对某个specific 的task的, fine-tune 整个模型而非冻结在大训练集的情况下表现更好
					- Contextual word embedding 需要把整句话给到模型来得到一个词的embedding
					- zero/one/few shot learning 是以prompt的方式引导模型, 不更新参数
				- Tokenisation and word embedding
				  collapsed:: true
					- Byte Pair Encoding & Wordpieces
					  collapsed:: true
						- 1.2中全部有提到
					- Contextual word representations
					  collapsed:: true
						- take the context into account when constructing word representations
						- ELMo: Embeddings from Language Models
						  collapsed:: true
							- 使用两个从左到右 从右到左得到的hidden state concat起来; 三层LSTM的三个输出
							-
				- pre-training encoder models
				  collapsed:: true
					- Masked Language Modeling
					  collapsed:: true
						- 用[MASK] 符号代表来遮蔽掉部分词汇, 让BERT生成整句话的encodings, 取出对应mask位置的encoding 用mlp进行V中分类, 与真实label求loss BP
						- 80% [MASK], 10% random, 10% original
						- Random: 只有mask会让模型只学习mask的部分, 不学习那些不mask的不正确的部分
						- Original: 防止random的不正确的样例让模型认为没有mask的就是不正确的, 让模型也能处理那些正确的词语
					- Another approach: predict next sentence (label yes or no) 没用
					- 使用方法:
					  collapsed:: true
						- BERT会在tokenise的时候加一个CLS 专门用来表示类别, SEP来分开句子来处理多句, 可以在每个token后接out layer, QA也可以实现by label in candidate answer span
					- Some ideas: mask a span, distil model with similar performance, sparse attention
					- Parameter-efficient fine-tuning 如何有效训练并得到最广泛的应用
					  collapsed:: true
						- Prompt tuning:  include task specific prompt and fine-tune only their embeddings for that particular task, froze remaining
						- Prefix tuning: include these trainable task-specific “tokens” into all layers of the transformer.
						- Adapters: insert specific trainable modules and freeze rest
						- BitFit: Keep most of the model parameters frozen, fine-tune only the biases.
					- Keys to good performance
					  collapsed:: true
						- model pre-training, large models, Loads of data, Fast computation, A difficult learning task
				- Pre-training encoder-decoder models
				  collapsed:: true
					- 无法使用MLM, 因为output的是一整个sequence, 而不是一对一的输出, 没办法对应位置去分类.
					- 3种训练方式:
					  collapsed:: true
						- Prefix language modeling
						- Sentence permutation deshuffling
						- BERT-style token masking
					- Corrupt original sentences in ways and optimise the model to recover them. 比如span mask掉, 预测出span是什么
					- Instructional training: train with natural language instructions as inputs, and annotated answers as target outputs.
				- Pre-training decoder models
				  collapsed:: true
					- train on unlabeled text, optimising p(xn|h)
					- Fine-tuning: Supervised training for particular input-output pairs.
					- No gradient update
					  collapsed:: true
						- Zero-shot: 给出自然语言指令, 是LLM的优势
						- One-shot: 给出指令并给出一个例子
						- Few-shot: 给出指令并给出很多例子, 给例子能够指导其回答需要的答案
					- Chain-of-thought
					  collapsed:: true
						- Show examples of reasoning and do reasoning (lets think step by step 作为prompt来引出COT)
					- Retrieval-based language models: 从数据库中获取factual knowledge
					- Limitations of instruction fine-tuning
					  collapsed:: true
						- data is expensive as manually created
						- creative generation have no right answer
						- penalise token-level mistakes equally, but some are worse
					- [[RLHF]]: Reinforcement learning from human feedback
					  collapsed:: true
						- train our language model to maximize this a human reward, using reinforcement learning
						- to reduce expensive direct human participants, model human preferences as a separate (NLP) problem, ask for pairwise comparisons
						- InstructGPT and ChatGPT 就用了这种方式, 非常流行 且效果很好
			- 疑点
		- ---
		- Structured Prediction
		  collapsed:: true
			- 难点
			- 疑点
		- Tagging
		  collapsed:: true
			- 难点
			- 疑点
		- Parsing
		  collapsed:: true
			- 难点
			- 疑点
- ## Coursework
  collapsed:: true
	- Deadline: 3.7
	  SCHEDULED: <2023-03-07 Tue>
	- 3.4
		- 1. 直接测一个uncased and cased
		  2. 给最佳组合找到最佳的learning rate和batch size (使用train val, 再在test上面得到结果
		  3. 测试learning rate scheduler
		  4. 测试不同的数据增强方法的区别
		  5. 测试bagging 和ensemble
		  6. length of the input sequence
		  7. To what extent does model performance depend on the data categories? E.g. Observations for homeless vs poor-families
	- Task:
		- 实现一个用于判断文本有没有屈尊优越高人一等(patronising or condescending) 的含义的二分类模型
			- transformer-based model
			- F1 score: over 0.48 on dev set; over 0.49 on test set
			- dev set evaluation will be a public test on LabTS
			- test set 的label is private, 打分的时候才能看
		- Report, 用于打分, pdf
			- answer the questions in the Marking Scheme section
	- Submission:
		- PDF of your report
		- SHA1 key for your GitLab repository
			- – Dev set predictions as dev.txt
			  – Test set predictions as test.txt
			- also contain the code, but not be marked
	- Data and evaluation:
		- use the dontpatronizeme_pcl.tsv
		- practice split中分好了train和dev, dev我们作为test用
	- Marking scheme
		- 1) Data analysis of the training data (15 marks): a written description of the training data
		- 2) Modelling (40 marks): implementation of a transformer model
		- 3) Analysis (15 marks):Analysis questions to be answered
		- 4) Written report (30 marks): awarded for the quality of your written report
	- Ideas:
		- data augmentation
		- 平衡数据
		- prompting
		- {0,1}   = No PCL
		- {2,3,4} = PCL
	- Transformer:
	- Links:
		- [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) #[[Hugging Face]]
		- [LUKE](https://huggingface.co/docs/transformers/model_doc/luke)#[[Hugging Face]]
		- [DeBERTa-v2](https://huggingface.co/docs/transformers/model_doc/deberta-v2)#[[Hugging Face]]
		- [An Algorithm for Routing Vectors in Sequences | Papers With Code](https://paperswithcode.com/paper/an-algorithm-for-routing-vectors-in-sequences)
		- [ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)#[[Hugging Face]]
		- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=roberta)#[[Hugging Face]]
		- [Processing the data - Hugging Face Course](https://huggingface.co/course/chapter3/2?fw=pt)#[[Hugging Face]]
		- [Hugging Face 的 Transformers 库快速入门（一）：开箱即用的 pipelines - 小昇的博客](https://xiaosheng.run/2021/12/08/transformers-note-1.html)
		- [Transformer 模型通俗小传 - 小昇的博客](https://xiaosheng.run/2022/04/04/transformers-biography.html)
		- [Hugging Face 的 Transformers 库快速入门（四）：微调预训练模型 - 小昇的博客](https://xiaosheng.run/2021/12/17/transformers-note-4.html)
		-
- ## Info
  collapsed:: true
	- 7:3
	- 星期一 16:00 - 18:00, 星期四 11:00 - 13:00
	- 3 hours lectures + 1 hour tut/lab
	- CW: 1.30 - 3.7
- ## Syllabus
  collapsed:: true
	- ![image.png](../assets/image_1673967503450_0.png)
		-
- ## Links
	- [Scientia](https://scientia.doc.ic.ac.uk/2223/modules/70016/materials)
	- [Ed — Digital Learning Platform](https://edstem.org/us/courses/29425/discussion/)
	- [textbook](https://web.stanford.edu/~jurafsky/slp3/) #书
	- [COMP70016: Natural Language Processing | Department of Computing | Imperial College London](https://nlp.pages.doc.ic.ac.uk/spring2023/)
	- [ImperialNLP/NLPLabs-2023 · GitHub](https://github.com/ImperialNLP/NLPLabs-2023)
	- [Tensors — PyTorch Tutorials 1.13.1+cu117 documentation](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) #tutorial
	- [NLP-progress](https://github.com/sebastianruder/NLP-progress) #GitHub #科技资源 #Academic
	- [Browse the State-of-the-Art in Machine Learning | Papers With Code](https://paperswithcode.com/sota) #科技资源 #Academic
	- [SpaCy](https://spacy.io/) #NLP #Academic #科技资源
	- [Stanza](https://stanfordnlp.github.io/stanza/) #NLP #GitHub #Academic #科技资源
	- huggingface
	- [exBERT](https://huggingface.co/exbert/?model=bert-base-cased&modelKind=bidirectional&sentence=The%20girl%20ran%20to%20a%20local%20pub%20to%20escape%20the%20din%20of%20her%20city.&layer=0&heads=..0,1,2,3,4,5,6,7,8,9,10,11&threshold=0.7&tokenInd=null&tokenSide=null&maskInds=..&hideClsSep=true)
	- [GitHub - bhoov/exbert: A Visual Analysis Tool to Explore Learned Representations in Transformers Models](https://github.com/bhoov/exbert)
