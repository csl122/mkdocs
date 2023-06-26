# Cryptography Engineering

public:: true
tags:: IC, Security, Course, Uni-S10
alias:: CE
Ratio: 8:2
Time: 星期一 9:00 - 11:00

-
- ## Notes
  
  Template:: Module Notes
	- 术语表
	  collapsed:: true
		- [[Shift cipher]] (THE CAESAR CIPHER)
		  collapsed:: true
			- 统一往后移动0-25位
		- [[Substitution cipher]] (MONOALPHABETIC SUBSTITUTION)
		  collapsed:: true
			- 26个字母打乱替换对应关系
		- [[Vigenere cipher]] (POLYALPHABETIC SUBSTITUTION)
		  collapsed:: true
			- key set是各种原字母表的permutation, 用多个key来轮流进行替换加密, 相同字母可以被替换成不同的密文
		- {{embed ((63c2c66c-9ab2-45a2-a347-e981592ee698))}}
		- **Kerckhoffs’s principle** is that a cryptographic system must be secure even if everything is known about the system with the exception of the secret key.
		- **DES**: Data Encryption Standard
		  collapsed:: true
			- 64-bit data blocks, 56-bit keys (8 bits for redundancy)
		- **AES**: Advanced Encryption Standard
		  collapsed:: true
			- 128-bit data blocks, key size of 128, 192, 256 bits
		- **ECB** Electronic Code Book 电子密码本模式
		  collapsed:: true
			- 相同明文加密到相同密文
		- **CBC** Cipher Block Chaining 密码分组链接模式
		  collapsed:: true
			- 前一个密文与当前明文XOR后的结果加密
		- **CFB** Cipher FeedBack 密文反馈模式
		  collapsed:: true
			- 前一个密文加密结果与明文XOR
		- **OFB** Output FeedBack 输出反馈模式
		  collapsed:: true
			- 初始向量不断加密形成密钥序列与对应明文XOR
		- **CTR** counTeR 计数器模式
		  collapsed:: true
			- 初始向量不断累加后加密, 与对应明文XOR
		- **Collision resistance** ：有人找到两条散列相同的消息（任意两条消息）有多难 .
		- **Preimage Resistance** ：给定哈希值，找到另一个哈希相同的消息有多难？也称为单向散列函数 .
		- **Second preimage resistance** ：给定一条消息，找到另一条散列相同的消息 .
		- {{embed ((63ce8f73-25c8-48cc-817a-d17069e19ad3))}}
		- **Linear Cryptanalysis (线性分析)**: approximate all non-linear components with linear functions; for example the S-boxes of DES; then do a probabilistic analysis about the key. 将明 文和密文的一些对应比特进行XOR并计算其结果为零的概率如果密文具备足够的随机性，则任选一些明文和密文的对应比特进行 XOR 结果为零的概率应该为1/2。 如果能够找到大幅偏离1/2的部分，则可以借此获得一些与密钥有关的信息。 使用线性分析法，对千 DES 只需要 2^ 47组明文和 密文就能够完成破解，相比需要尝试 2^56个密钥的暴力破解来说，所需的计算址得到了大幅减少。
		- **Integrity** refers to that data can only be modified by authorized parties or that unauthorized modifications of data are detectable
		- **Availability** means that we can access data when we need it or will receive it across a network within the expectations set out in service-level agreements
		- **Nonce**: 临时造的数字, 只用一次, 但是生成方式并不一定是random的
		- **Semantic Security**: a **semantically secure** [cryptosystem](https://en.wikipedia.org/wiki/Cryptosystem) is one where only negligible information about the [plaintext](https://en.wikipedia.org/wiki/Plaintext) can be feasibly extracted from the [ciphertext](https://en.wikipedia.org/wiki/Ciphertext).
		- **Avalanche effect** desired: small changes in the input should result in
		  unpredictable changes in the output.
		- **Horton’s Principle** states that it is wise to sign what is meant and not what is being said. but not in blinding protocols for anonymity solutions.
		- **Abelian group**: Commutative group, x+y = y+x
		- **QKE**: quantum-key-exchange
		- **qubit**: quantum bit
		- **BB84**: Quantum key exchange protocol, Invented by Charles Bennett and Gilles Brassard in 1884
		- information-theoretic security: attacker with infinite computing power
		  cannot break the scheme
		- computational security: attacker with specified computation resources
		  (e.g. probabilistic polynomial time) cannot break the scheme
	- 算法表
	  collapsed:: true
		- DES: Data Encryption Standard, Symmetri ccryptosystem
		- AES: Advanced Encryption Standard, Symmetri ccryptosystem
		- SHA: Secure Hash Algorithm
		- RSA: public-key cryptosystem that is widely used for secure data transmission. It is also one of the oldest. The acronym "RSA" comes from the surnames of Ron Rivest, Adi Shamir and Leonard Adleman
		- MAC: Message Authentication Code
		- HMAC: Hash Message Authentication Code
		- ECC: Elliptic Curve Cryptography
		- DH: Diffie-Hellman key exchange
		- ECDH: Elliptic Curve Diffie-Hellman key exchange
		- ECDHE: Elliptic Curve Diffie-Hellman Ephemeral
		- ECDHE-RSA: Elliptic Curve Diffie-Hellman Ephemeral key exchange, signed by RSA
		- ECDHE-RSA-AES128-GCM-SHA256: using AES128 加密, GCM mode, hash采用SHA256
		- DSA: Digital Signature Algorithm
		- ECDSA: Elliptic Curve Digital Signature Algorithm
	- Week1 Mathematics
	  collapsed:: true
		- ((63c55d15-f4b2-474b-870d-f5cf1db87fbd)) Addition modulo n
		  collapsed:: true
			- $a +_n b = (a +b)(mod\ n)$
			- $inv_n: Z_n -> Z_n$
			- $inv_n(a) = −a(mod\ n) = n - a(mod\ n)$
			- 因为是mod n操作, 所以定义域和值域都得是n
			- inv的结果和原值进行了该group的operation以后会是一个固定的element e
			- 例如: $12 +_13 inv_13 (12) = (12 + 13-12(mod n)) (mod 13) = 12 + 1 (mod 13) = 0$
			- 分数的modulo:
			  collapsed:: true
				- $5/2 mod 7 = 5 * 2^{-1}\ mod\ 7 = 20 mod 7 = 6$
				- 2n = 5 mod 7; n=6
		- ((63c56058-2cf2-46a3-979f-7c9225796653))
		  collapsed:: true
			- ((63c56080-b507-4555-9937-a1678ad09ddd))
			- 数学群包含了一个集合, 一个固定的元素(与inv中和后的固定值), operation, 以及一个inverse函数.
			- Commutative (Abelian) group是交换群, 或者阿贝尔群, 需要满足交换律和结合律, 例如加法和乘法是关于0 和 1的交换群
		- ((63c56519-6dd1-4516-baf4-57776e6fcc61))
		  collapsed:: true
			- ((63c56536-1a43-47a1-b55e-5cdb50728d3f))
			- 群同态, 通过一个mapping函数, 从G到G‘, 这个mapping不管是对某两个数的操作结果进行mapping还是先map两个数到新的域再进行新的操作, 都能得到一样的结果
			- ![image.png](../assets/image_1673881139811_0.png)
			- ((63c56645-dd64-4330-bc3d-22562f08e41a))
			- 例如modulo操作, 两个数加起来mod n, 和两个数分别mod n再加起来的结果mod n, 是一样的
			- ![image.png](../assets/image_1673881372010_0.png)
			- 有一个同模的概念, 用a = b mod N 表示a mod n = b mod n
	- Week2 Ch 7 & 9 Historical ciphers, Perfect secrecy
	  collapsed:: true
		- Slides
			- Aims of [[Cryptography]]
			  collapsed:: true
				- CIA #术语卡
				  id:: 63c2c66c-9ab2-45a2-a347-e981592ee698
				  collapsed:: true
					- [[Confidentiality]]: 机密性, 保证信息只被需要知道的人获取
					  id:: 63c2c6c0-f7b0-4706-8ce0-b7b3dafd7d25
					  collapsed:: true
						- The prevention of unauthorised users reading
						  sensitive (private, secret) information
					- [[Integrity]]: 完整性, 既指消息传递中保证完整不变, 也指保证发送者的身份
					  id:: 63c2c6d2-3b91-4091-a6fa-6b7ecb513f11
					  collapsed:: true
						- The prevention of unauthorised modification of data, and
						  the assurance that data remains unmodified
					- [[Availability]]: 可访问性, 防止未授权的信息拒绝访问,即便系统被攻击, 依旧有办法保证最低限度的function, 能够保证信息被获取
					  id:: 63c2c6d9-3f3e-48bb-996e-6d56e7ed0cc8
					  collapsed:: true
						- The property of being accessible and useable upon demand by
						  an authorised entity
			- Symmetric Cryptosystems Definition Basic Examples
			  collapsed:: true
				- [[Symmetric Encryption]]
				  collapsed:: true
					- *m*: plaintext messages*
					  k*: key*
					  c*: ciphertext
					- *e*: encryption function to encrypt plaintext m using key k
					  *d*: decryption function to decrypt ciphertext c using key k
					- Kerckhoffs’s principle is that a cryptographic system must be secure even if everything is known about the system with the exception of the secret key.
					- 根据科考夫原则, d and e是公开的, c也是可以被公开的, m的机密性取决于k, 只有k和m本身是不应该被获取的
				- Examples #[[Symmetric Encryption]]
				  collapsed:: true
					- [[Shift cipher]] (THE CAESAR CIPHER)
					  collapsed:: true
						- An early substitution cipher, we replace each letter of plaintext with a shifted letter further down the cyclic alphabet, 例如key=4(E), 就是把字母顺移4位, a->e, b->f, ...
						- Vulnerable to frequency analysis
						  collapsed:: true
							- The frequency of occurrences of each character are very consistent across the same language
							- The longer the ciphertext, the easier this becomes
					- [[Substitution cipher]] (MONOALPHABETIC SUBSTITUTION)
					  collapsed:: true
						- 每个字母换一个字母替代
						- a key is used to alter the cipher alphabet, the key is a permutation of original alphabet; key space is 26! 26个字母的排列
						- vulnerable to frequency analysis
					- [[Vigenere cipher]] (POLYALPHABETIC SUBSTITUTION)
					  collapsed:: true
						- key set是各种原字母表的permutation, 用多个key来轮流进行替换加密, 相同字母可以被替换成不同的密文
						- thus a key k is a tuple (k1,k2,...,kp) of p > 1 many permutations of set {a,b,...,z}, 真正用到的key是keyset中选择了p种permutation, 然后依次apply到每个字母上
						- for example, we have e(k1,k2)(hello) = shljv for permutations k1 and k2 satisfying k1(h) = s, k2(e) = h, k1(l) = l, k2(l) = j, and k1(o) = v; 这个例子中的key用了两位, 所以hello中的两个l被不同的key加密, 获得了不同的ciphertext, Spreads out the occurrences of characters making frequency analysis hard
						- in general, this cipher uses key k1 on letters m1,mp+1,m2p+1,... and keykj onlettersmqp+j forallqwith1≤qp+j≤k
						- 但是: weak to Kasiski examination, Repeated phrases in the ciphertext give away clues as to the length of the running key, 两个相同phrase中间的距离可以帮助获取key的长度, 然后就可以做frequency analysis了
					- [[Questions]]
					  collapsed:: true
						- Is the Vigenere cipher more secure than the simple substitution cipher?
						  collapsed:: true
							- 在破解难度上确实比simple substitution高了, 因为有running keys, 但是仍然并不安全
						- Is the Vigenere cipher secure?
						  collapsed:: true
							- 并不安全哈, 通过重复短语分析出key的长度以后就可以通过frequency analysis来找到对应的原字母了
						- in what sense are the shift cipher, substitution cipher, and Vigenere cipher symmetric encryption?
						  collapsed:: true
							- 相同的密钥可以用来加密, 或者做一些小的调整, 例如反向, inverse 就可以用来解密
							  collapsed:: true
								-
			- Three notions of security
			  collapsed:: true
				- id:: 63c2dec5-ed59-4fcc-82a9-5c03367b6440
				  collapsed:: true
				  1. Information-theoretic security: also known as [[unconditional security]] or [[perfect security]]; a cipher cannot be broken even with **infinite computing power** 
				     This holds forever
				  2. Computational Security: cipher cannot be broken within specified computing power, e.g. where an attacker can compute only within probabilistic polynomial time 在有限的算力和时间内无法被破解
				     如后量子计算时代, integer factorisation 可能都是很简单的
				  3. Provable security: show security via a problem reduction; “Cipher can be broken implies that a known and believed to be hard problem can be solved.” cipher能被破解意味着一个很难的问题能被解决
				     [[Question]] : Why is the term “provable” in ‘provable security’ potentially misleading?
					- 是不是因为这里解释的是, 能被破解的proof, 而不是不能被破解的proof, 因此这里的provable指的不是prove这个是secure的, 而是由subproblem来prove其不安全
			- ((63c325bd-e4e5-4a57-8a8f-090b16c3cd0c))
			  collapsed:: true
				- ((63c2e3e7-f36a-48c0-9737-3c6e3934711f))
				- 密文是c的可能性是 对于每个可能的key (我们用的key恰好是这个key的可能性乘上明文刚好是c用这个key解密得到的值的概率乘积) 的和
				- Example:
				  collapsed:: true
					- ((63c2e3ca-be33-4b36-b00e-9504aa90dec4))
				- Conditional probability
				  collapsed:: true
					- p(C = c | P = m) = probability that ciphertext is c given that plaintext is m; 明文m的加密是c的可能性, 即能把明文m加密成c的密钥出现的可能性和
					- ((63c2e6d4-6bf1-4b2a-9641-fa5f7067bd51))
					- 但是attacker想知道的是, 得到了c这个密文, 这个密文所对应的明文的概率
					  collapsed:: true
						- ((63c2e768-e485-4eef-8b34-486bbd5c67c1))
						- 这里有个Assumption: for all c in C, we have p(C = c) > 0. 不然上面的分母就变成0了. [[Question]]: why does this assumption not lose generality?
						  collapsed:: true
							- 可能是因为我们不会去考虑那些不可能出现的ciphertext吧, 不然就没有意义了呀, 不会去用的东西, 考虑他干嘛?
						- 例子
						  ((63c2e789-74dc-49cc-b79d-daf68ee736a7))
						- 这样就可以推测出最有可能的对应的明文了, for perfect secrecy, we want to prevent inferences about most likely plaintext
						  collapsed:: true
							-
			- ((63c325aa-7555-445e-aef9-33856d8f46d0))
				- ((63c30bcc-f3ae-4111-b9cc-f9860a6aa3df))
				- 所谓完美的保密, 就是密文是某个明文的概率和这个明文本身出现的概率是相同的, 我们没法从密文中获取到任何新的关于明文的信息, no information gain in this; 与之同理的是, 明文是某个密文的概率和这个密文出现的概率是相通的.
				- 另外一个条件则是, $| K | ≥ | C | ≥ | P |$. 数量, key要比密文多, 密文要比明文多, 这样才能保证不会重复
				  collapsed:: true
					- 但这也意味着我们需要至少明文数量的key来实现完美的加密系统, 因此完美加密是有限实用的 (limited practical).
				- ((63c3259f-8d13-4f83-8ef3-5b3bf4576cf9))
				  collapsed:: true
					- ((63c30f59-5e1d-495b-8ac5-ac6097d3cd22))
					  图中所示的加密系统是完美安全的充分必要条件是: 每个key的使用概率都是1/k, 即均衡的; 只有一个key能让m加密成c, 即key对于mc的mapping是唯一的
					- Proof: 由perfect secrecy -> S1 & S2
					  collapsed:: true
						- [S2](((63c32566-b678-443c-97ad-385061d9b5dd))): PS 意味着 $p(C =c|P =m)=p(C =c)>0$
						  由于p(C =c)>0, 因此一定会有k in K allowing $e_k(m) = c$
						  还需要证明k的唯一性, 即一对一的关系, 一个k对应一个c
						  那我们可以证明Claim: $|{e_k(m) | k ∈ K}| = |C|$
						  这个可以由双向的inclusion证明, 首先很明显对m加密生成的字符肯定在C的集合里面; 另一方面, 任何密文c, 由于其存在可能性大于零, 则一定可以由某个key对m加密生成, 因此c也包涵在${e_k(m) | k ∈ K}$集合之中, 双向包含则意味着这是一个injective, 一对一的关系, 也就是唯一性保证了
						  ((63c328c2-656c-4bee-a640-f16cb3fe6d85))
						- [S1](((63c32588-23e3-4d86-a633-f306a25ce7b3))): 用贝叶斯理论重写了一下完美加密的等式, 由于我们说只有一个唯一的key可以把明文m加密成c, 因此$p(C = c | P = mi)$与这个key的出现概率是一样的, 因此可以重写成$p(K = ki)$, 于是因为等式左右要相等, 所以key的出现概率和c的概率是要相等的. 这意味着$p(K = ki)$都是相等的, 也就是1/K了
						  ((63c31a57-638c-4345-8705-4111928fd787))
				- Examples of perfectly secure cryptosystem #[[Perfect Secrecy]]
				  collapsed:: true
					- ((63c3253e-8557-4fc0-a51d-16c608ca1831))
					  collapsed:: true
						- 明文有多长, key就要有多长, key从26个字母里面independently, uniformly, randomly 选择明文长度个, 第几个字母就意味着要移几位(0开始), 组合数呢有$26^n$
					- ((63c325e5-9101-4ade-ae90-3f799a444447)) (Vernam cipher)
					  collapsed:: true
						- ((63c32640-db4a-447f-bf7a-05a04aa849dc))
						- 异或, 不同的时候结果才是1; 密钥可以直接用来加解密
						- n长度的明文, 会有$2^n$种表现, key也会有同样数量, 因此n长度的key的选取可能性是$1/2^n$
				- ((63c3331a-987a-4cbf-b68d-ffb282e1f18f))
					- 1. Key has to be as long as the message.
					  2. Key has to be truly random.
					  3. Key can be used at most once.
					- 如果重复使用的话, Eve可以通过自己generate一些文字让Alice加密, 来获取key, 并且用这个key解密之后A与B的密文
		- Notes
			- ((63c338fb-bec0-4868-84e3-a5fe94ccbeef)) #术语卡
			  collapsed:: true
				- A cryptosystem is given by a set $P$ of plaintexts,
				  a set $C$ of cipertexts, a set $K$ of keys, an encryption function $e : K × P → C$,
				  and a decryption function $d : K × C → P$ such that for all $k$ in $K$ and for all m
				  in $P$ we have that $m = dk(ek(m))$.
			- ((63c33962-3eec-4549-847a-ac5c022ed703)) #术语卡
			  collapsed:: true
				- A cryptosystem is symmetric if it has an efficient and deterministic method for converting keys used for encryption into the corresponding keys used for decryption.
			- ((63c3398d-8bbd-497f-8c00-cb758ad53cf3)) #术语卡
			  collapsed:: true
				- In cryptography, Kerckhoff’s Principle
				  stipulates that the security of a cryptosystem should not rest on the secrecy
				  of its algorithms.
				- cryptographic system must be secure even if everything is known about the system with the exception of the secret key.
			- ((63c339d5-a1fc-4b07-932b-aef369bdc69e)) #术语卡
			  collapsed:: true
				- Let $A$ be a non-empty set. Then Perm($K$) is
				  the set of permutations of $A$, which are functions $f : A → A$ that are 1-1 and
				  onto.
			- ((63c339fe-1497-44e7-9d7c-36656504fb63)) #术语卡
			  collapsed:: true
				- ((63c2dec5-ed59-4fcc-82a9-5c03367b6440))
		- Exercises
			- ((63c32bd0-6c3a-4952-9b76-a136def3c3fb))
			  collapsed:: true
				-
			- ((63c32cf9-97f6-4fef-a8c5-4b696186f42f))
			- ((63c32d08-63ae-4744-8c63-48dac2bad567))
			- ((63c32d12-47d2-4af7-9dda-ca8babb845fd))
			- ((63c32d1b-950c-42b9-a450-70e47c744632))
			- ((63c32d28-d8af-48ed-b949-411be4180f48))
			- ((63c32d30-2fe8-4c9d-99c3-a8b8083f0f2a))
			  collapsed:: true
				-
			- ((63c32d3a-af96-466a-9f8e-6f8c7d48ad89))
			  collapsed:: true
				- ((63c55656-8b79-49fd-97b4-9bfe703cdd23))
				- 为什么C要至少比P大
				  collapsed:: true
					- 因为CP是1-1的关系, 每个p必须要有c对应, 所以C的大小一定至少是等于P的大小的, C也可以更大, 但是会redundant, 所以我们assume p(C=c)>0
				- 为什么K要至少比C大
				  collapsed:: true
					- 因为对于某一个明文m, 对于所有可能的密文, 都必须要有一个key 能够把m加密成这些密文; 而相同的key只能加密明文m到同一个密文, 这意味着key的数量至少要和C的大小一样; 举个例子, shift cipher的进阶perfect版本需要每位字母都拥有一个独立随机的key来进行加密, 各个字母互不相干, 才能让破坏者无法从密文中推理到任何明文的信息
		- 宏观理解
			- CIA三个特性, 其中integrity 还会指保证发送者的身份
			- 对称加密的核心在于解密用的密钥可以用简单的方式从加密的密钥中推出
			- 加密系统由KCP 和 加密函数e解密函数d组成
			- 对称加密有shift (Caesar), substitution (monoalphabetic, 所有明文用一个26字母permutation), Vigenere(polyalphabetic, 轮流running key用一组permutations substitute, Kasiski examination破解), 后面后有俩perfect的 对称加密
			- P(C=c)的概率是对于每个可能的key (我们用的key恰好是这个key的可能性乘上明文刚好是c用这个key解密得到的值的概率乘积) 的和; 某个明文是某个密文的概率是建立这俩联系的所有key的概率和 (完美秘密需要只有一个key)
			- Perfect secrecy是对于所有明文和密文, p(P=m|C=c) = p(P=m), 或者p(C=c|P=m) = p(C=c) 且是大于零的, 因为不出现的c没有意义, 也意味着K>=C>=P. Shannon 定义下三者一样大且k的概率为1/K且每一组m和c都一一对应只有一个唯一的k时是perfectly secure的. 因为p(C=c)>0就一定有k从m映射到c; m加密得到的一定在C中, c一定存在意味着一定有m到c, 所以c的解密也一定在P中, 所以是双向包含的injective关系. 所以一定是唯一的key在map. 1/K的概率则由贝叶斯分解p(P=m|C=c)和key的唯一性可以推出p(K=ki)=p(C=c), 各个key概率相同
			- 剩下两个对称加密: modified shift cipher, 每个明文都由一个随机的shift量来决定, 可能性有26^n. one-time pad, 每个bit都和随机的1或者0 进行XOR, 需要每个位置的key都随机, key只能用一次, key长度和原文长度一致, 有2^n种可能, 重复使用可以chosen text attack获得密钥
			- perfect secrecy does not depend on the probability distribution p(P=p)
	- Week3 Ch 13 & 14 Block ciphers, DES, Hash functions
	  collapsed:: true
		- Slides
			- ((63cdc1a3-cb13-49e2-b4e6-c0dc7a39dbc8))
				- 明文被分成了等长的一个个blocks
				  collapsed:: true
					- $m = m_1m_2 ...m_k$
				- Block ciphers assume cryptosystem (ek(·),dk(·)) for data of block length, 块是基本加密单位
				- 我们上周讲到的加密系统都是使用1bit长度的block, 例如onetime-pad就是每一位都会用一个sample的密钥进行加密
				- Modes of operation: Encryption of mi may depend on encryption of $m_j$ where j < i. 加密操作模式, 这节课讲到的加密系统, 可能会有与前后明文块相关的加密, 例如第二块的加密和第一块有关
				- DES: Data Encryption Standard
				  collapsed:: true
					- 64-bit data blocks, 56-bit keys (8 bits for redundancy)
				- AES: Advanced Encryption Standard
				  collapsed:: true
					- 128-bit data blocks, key size of 128, 192, 256 bits
				- N.B. DES and AES都是iterated block ciphers
				- ((63cd968e-01d3-4f82-8eed-04337dfea944)) 13.2 P244
				  collapsed:: true
					- Block ciphers的iteration被叫做Rounds, 每个round i, 加密用的key k会determine一个round key $k_i$ (子密钥subkey), 从k计算出$k_i$的过程叫做key schedule,
					  collapsed:: true
						- rounds数量越多, 越安全
						- key的长度越长, 越安全
						- ![image.png](../assets/image_1674422860421_0.png)
					- Round function: 我们需要round function来决定如何根据右侧和子密钥生成加密后的比特序列
					  collapsed:: true
						- Feistel function: 被用在DES中, DES采用了16轮循环的Feistel网络, 下图是应用了Feistel function F的一轮的操作
						  collapsed:: true
							- ((63cd97d8-7709-4cd7-9cab-2d56e7ad11f6))
							- 64-bit的输入被首先分成了两半, 左右., 我们不对左进行额外操作, 我们把右侧作为输入和子密钥一起放入轮函数F中, 其得到的结果与左侧原文进行XOR, 结果就是加密后的左侧, 右侧在一轮中不进行加密, 直接下放
							- ![image.png](../assets/image_1674418865950_0.png)
							- 由于右侧在一轮中不进行加密, 因此轮与轮间, 我们需要对调左右, 这也是为啥上图L-R的原因, 因为对调了. 并且对调操作只在两轮之间进行, 最后一轮不需要
							- Feistel网络的一个好处在于, 由于一边的数据是不进行操作的, 我们将下一轮加密输出结果用相同的子密钥重新运算, 是完美还原为原文的, 也就是说Feistel的解密是需要相反顺序apply子密钥即可, 这也就不需要轮函数可以逆向计算出输入的值(解密的时候用的是相同的右侧和子密钥). 因此轮函数可以设计的任意复杂. 同时可以保证相同结构就能实现加密和解密.
				- ((63cd9b44-fcba-4afe-90e3-1fc44cf3fcb2)) 13.2 P244
				  collapsed:: true
					- 1. Number of rounds: trade-off btw security and performance
					  2. Key schedule: key到subkey的映射, 要可行, 可反
					  3. Feistel function F: needs to contain non-linear behavior to ensure enough information-theoretic resiliency. 非线形来保证足够的对信息论的抵抗
					- **Advantage**: Only have to implement or build an encryption device, no need for a decryption device, only need to manage D/E state. 只需要一个device来执行加密解密
				- ((63cd9c09-b3ca-4c6c-87f6-e4ae34c6632b)) 13.4 P247; 13.5 P247
				  collapsed:: true
					- DES Key schedule: we refer to Nigel Smart’s book for details on that; the point to remember is that this **uses permutations** and **cyclic shifts** based on the key and round number.
				- ((63cd9ca8-cd76-4b9b-98d5-2e9aac2d2e90)) 13.3 AES
				  collapsed:: true
					- DES 和 AES seem immune against:
					  collapsed:: true
						- differential cryptanalysis (差分分析): a chosen-plaintext attack, 修改某个明文, 查看密文是怎么变的
						- linear cryptanalysis (线性分析): approximate all non-linear components with linear functions; for example the S-boxes of DES; then do a probabilistic analysis about the key. 将明 文和密文的一些对应比特进行XOR并计算其结果为零的概率如果密文具备足够的随机性，则任选一些明文和密文的对应比特进行 XOR 结果为零的概率应该为1/2。 如果能够找到大幅偏离1/2的部分，则可以借此获得一些与密钥有关的信息。 使用线性分析法，对千 DES 只需要 2^ 47组明文和 密文就能够完成破解，相比需要尝试 2^56个密钥的暴力破解来说，所需的计算址得到了大幅减少。
				- ((63cd9d30-8e69-4f26-876e-6ed81fd31839))
					- 如 果需要加密任意长度的明文，就需要对分组密码进行迭代，而分组密码的迭代方法就称为分组 密码的“模式” 。其实就是迭代过程中的对于明文作出的一些变化, 来让相同的明文块加密后没有可见pattern
					- ECB Electronic Code Book 电子密码本模式
					  collapsed:: true
						- ![image.png](../assets/image_1674423155155_0.png)
						- Real problem with this mode: If we encrypt m more than once with the same key, these ciphertexts are equal. Thus, this mode gives us deterministic encryption which attackers can exploit. Bad!
						- ECB 模式是所有模式中最简单的一种 。 ECB 模式中，明文分组与密文分组 是一一对应的关 系，因此，如果明文中存在多个相同的明文分组，则这些明文分组最终都将被转换为相同的密 文分组 。 这样一来，只要观察一下密文，就可以知道明文中存在怎样的重复组合，并可以以此 为线索来破译密码，因此 ECB模式是存在一定风险的。
						- 假如存在主动攻击者 Mallory, 他能够改变密文分组的顺序 。 当接收者对密文进行解密时， 由于密文分组的顺序被改变了，因此相应的明文分组的顺序也会被改变 。 也就是说， **攻击者 Mallory无需破译密码就能够操纵明文**。例如对调收款人和付款人block, 就可以交换转账信息
					- CBC Cipher Block Chaining 密码分组链接模式
					  collapsed:: true
						- CBC 模式 是 将前 一 个密文分组与当前明文分组的内容混合起来进行加密的.这样就可以避免 ECB 模式的弱点 。CB 模式只进行了加密，而 CBC 模式则在加密之前进行了 一次 XOR。
						- 每个block的明文会首先与上一个block的密文进行异或操作, 再被该block的密钥进行加密, 第一个block和*initialisation vector IV*操作, IV需要完全随机.
						- 同样的, 解密的时候需要先用这个block的key解密, 然后再与上一个blcok的密文进行异或操作才是真正的原文
						- IV 和 key k是需要的共识
						- 常用于TLS/SSL中 推荐使用
						- ![image.png](../assets/image_1674425521575_0.png)
						- ![image.png](../assets/image_1674423514162_0.png)
					- CFB Cipher FeedBack 密文反馈模式
					  collapsed:: true
						- ![image.png](../assets/image_1674425919250_0.png)
					- OFB Output FeedBack 输出反馈模式 不可以平行
					  collapsed:: true
						- 需要*initialisation vector IV*, 由IV来生成一串key stream 
						  $Y=Y_1Y_2Y_3...Y_q$, 
						  $Y_0 = IV$
						  $Y_i = e_k(Y_{i−1})$
						  $c_i =m_i⊕Y_i$ (如果m长度为1的话那就是个stream cipher了, 就不需要padding了)
						- 加密过程与onetime-pad就很相似, 只不过OTP用的是每个letter都随机生成key, 而这里的的Yi都是由Y_i-1用key生成的
						- 并没有perfect secret, 因为并不是完全随机分布的
						- ![image.png](../assets/image_1674426112304_0.png)
					- CTR counTeR 计数器模式 可以平行
						- 需要*initialisation vector IV* as a counter: IV ≃ binary encoding of counter value. 序列中每次往IV里面加一
						- $c_i = m_i ⊕ e_k(IV + i)$
						- Advantages:
						  1. This is a recent US NIST Standard.
						  2. We may compute all $c_i$ in parallel, which is not possible in OFB, CFB, and CBC modes. 可以平行快速计算
						  3. If $m_i = m_j$ , then $c_i != c_j$ in general. So this improves over ECB. 相同的明文块不会导向相同的密文了
						- Requirement: For l-bit data blocks, the plaintext message m must not be
						  longer than 2^l bits. #Question
						- 推荐
						- ![image.png](../assets/image_1674426321766_0.png){:height 394, :width 592}
					- ![image.png](../assets/image_1674426402658_0.png){:height 542, :width 623}
				- AES (Rijndael)
				  collapsed:: true
					- ![image.png](../assets/image_1674424676268_0.png)
					- 128-bit data blocks, key size of 128, 192, 256 bits
					- 512位已经到10^154的数量级了, 再多就没啥必要了
			- Symmetric ciphers take away
			  collapsed:: true
				- 对称密码中, 加密的密钥和解密的密钥是相等的
				- 计算能力再高, 一次性密码本也没办法破译
				- 长度为56位的密钥平均破解次数为2^56的一半2^55
			- ((63cdc18a-256a-444d-b9bb-5f5dd79bb5f1))
			  collapsed:: true
				- 为了保证信息的完整性和不被篡改, 即Integrity, 我们需要方法来验证信息
				- ((63cdc2a5-7373-4ed8-b7e1-0620b6993b2b)) h: capture the integrity of m as a hash h(m). 单向散列函数, 消息的指纹
				  collapsed:: true
					- Input x is arbitrarily long. Output y is of fixed length n.
					  从任意长度的比特序列到固定长度比特序列的mapping
					- Security Properties 区别于哈希表的安全属性
					  collapsed:: true
						- **Preimage resistance**: Given y in the image of h, it should be compu- tationally infeasible to compute an x with h(x) = y. 
						  给定一个y, 很难找到一个可以生成y的x, 至少应该是O(2^n)
						- **Collision resistance**: It should be computationally infeasible to find
						  two inputs x and x′ (in particular, x ̸= x′) such that h(x) = h(x′).
						  难以找到了两个x使得这个hash函数能生成同样的y, 至少是$O(2^{n/2})$, 注意这里指的是找到有两个人同一天生日, 而不是找到特定某一天生日的两个人, 因此要更容易一点, 所以是n/2. 也因此, 我们如果需要保证2的80次的运算成本, 至少需要2的160次的总量空间, 也就是160位
						- **Second preimage resistance**: 给定一个x, 找到能生成同样y的x'
				- Message authentication codes (MACs): MACk(m) as a key-dependent
				  hash 消息认证码, 消息被正确传送了吗
				- N.B. Cannot say who created hash values. If this is an issue in an application, MACs are an alternative since a MAC is produced by someone who knows the used key.
		- Notes
			- ((63ce77e4-7243-4b92-9505-e550b80fad78))
				- Block ciphers trade off this tension between the degree of security and the
				  practical utility of a cryptosystem. 块加密在安全程度和实用性之间找到了某个平衡
				- 明文块的长度为128bits, 密文块的长度也会是128bits, 但是密钥的长度可以是任意的, 取决于加密算法如何应用
			- ((63ce7876-5dc2-42d8-8ab6-37f4f3583e8f))
			  collapsed:: true
				- ((63ce790e-7e63-492a-bce7-65b3440c1c0d))
				- DES is a instance of the above feistel cipher, 还有key scheduling, Feistel funcion, pre-processing prior以及post-processing.
				  collapsed:: true
					- 第一步首先是pre-processing, 对明文按照一定规则进行permutation, 称之为Initial Permutation IP
					  $L_0R_0 = IP(m)$
					- 算法如下, 每一轮结束后, 会对左右两边交换, 给到下一轮, 由于右边R是原封不动地传递下来的, 因此下一轮的左边输入L_i等于R_i-1, 而下一轮的右边输入, 则要对上一轮的左边输入进行加密操作后才能得到
					- ((63ce7a45-de3b-490c-9f14-14b4c83ac8bf))
					- 这里的post-processing 包括了$L_16R_16 - R_16L_16$的swap以及apply inverse的initial permutation; 必须swap才能让L1‘ 由R2替代, R1’由L2替代, 因为上述算法的过程是第一步就需要交换, 解密过程如下 Exercise 16
					- ((63ce7d65-9b1b-48fb-aeb1-94c5b19eb3f7))
					- we can now reason that L′2 = R1 and R′2 = L1. Similarly, we get that L′3 = R0 and R′3 = L0.
					- Feistel function F: Does it contain enough non-linear behavior to ensure enough information-theoretic resiliency? #Question
					- Differential Cryptanalysis: a **chosen-plaintext attack** #术语卡
					  id:: 63ce8f73-25c8-48cc-817a-d17069e19ad3
					  collapsed:: true
						- 事先选择一定数量明文, 让被攻击的加密算法加密, 获得密文, 通过这个过程获得加密算法的一些信息
						  id:: 63ce99c0-c9dc-4243-be5c-10c053910d72
						- id:: 63ce8fcc-9af0-40f7-9e61-191827bad0d6
						  1. choosing m and m′ with particular differences
						  2. computing c = ek(m) and c′ = ek(m′)
						  3. analyzing the “difference” c ⊕ c′
						  4. repeating the first three steps as often as needed
					- Linear Cryptanalysis: #术语卡
					  collapsed:: true
						- this approximates all non-linear components with linear functions; for example the S-boxes of DES; then it does a probabilistic analysis of this approximated cipher to infer information about the key.
						  collapsed:: true
							-
			- Hash Functions
			- Preimage resistance
			  collapsed:: true
				- A hash function h is preimage resistant if, given an arbitrary y in the image of h, it should be computationally infeasible to compute an x with h(x) = y. 给定一个哈希值, 找到一个能对应上的x
			- Collision resistance
			  collapsed:: true
				- A hash function h is collision-resistant if it is computationally infeasible to find two inputs x 6 = x′ such that h(x) = h(x′). 找到两个具有相同哈希值的x
			- second-preimage resistane
			  collapsed:: true
				- A hash function h is second-preimage resistant if, for an arbitrary input x, it is computationally infeasible to find an input x′ with x != x′ such that h(x) = h(x′). 给定一个x, 找到一个具有相同哈希值的x‘
		- Exercises 16-23
		  collapsed:: true
			- ((63ce9874-47a7-4c4a-986f-d580161a3b7e))
			- ((63ce983b-fbce-4b4f-83f1-52134a26bb84))
			  collapsed:: true
				- ((63ce98c5-badc-4bc2-9b3c-12ee0111bd21))
			- ((63ce98d6-b321-4e4f-820a-9a29343541eb))
			  collapsed:: true
				- 1. ECB模式下, 某个ciphertext block的一个bit改变了, 只会影响到这一个block的明文, 但是这个区块中50%的内容需要被改变
				  2. CBC模式下, avalanche effect会导致这个区块和之后的一个区块都被影响
				     3.ECB模式下, 插入了一个bit, 最后删除一个, 会导致这个block和后面所有block都被影响
				- 4. CBC模式下同样如此
			- ((63ce9a5f-86b1-420d-b7b6-6521d3c9fcdd))
			  collapsed:: true
				- 如果OFB的key stream是完全random的, 那其实就是a instance of onetime-pad
				- 但是OFB只有第一个key是random的, 后面都是depend于这个key. 因此并不是一次性密码本
			- ((63ce9adc-1602-4899-8153-2af7c432d6af))
			  collapsed:: true
				- 1. CTR 可以parallel的原因: 一旦IV known了, 我们也能够知道IV + i, 对于所有的i, 而如果我们知道m, 那就直到所有的m blocks, 就可以evaluate同时所有. 但是如果m是一dcv个个block提供的, 就得一个个来了
				  2. CBC, OFB, CFB无法parallel, 因为CBC 依赖于上一个密文, OFB依赖于上一个Y key, CFB依赖于上一个密文
				  3. OFB的correctness for decryption: 
				     ((63ce9c52-a5a9-4160-9356-025f1b5591e1))
				- 4. 加入明文block长度为 a bits, 给到一个IV, CTR模式最高支持的block数量是
				     $2^a - 1 - IV$
			- Hash Functions
			- ((63cea12d-5fc3-4136-ad82-a4df334fa476))
			  collapsed:: true
				- 要想要找到两个人的生日是相同的, 23个人, 就有超过50%的概率了, 这是因为我们算1减去所有人生日都是不同的概率, 也就是1-(365*364*363*...*365-N+1)/365^N, 50%的概率基本上就是365开根号, 而70个人甚至99.9%有俩人同一天生日, 所以collision resistance的compute time是O(2^n/2), 即开了个根号
			- ((63cea416-5f7d-4461-8f52-5ea3c3e11ec7))
			  collapsed:: true
				- 例如 cannot performed in probabilistic polynominal time
			- ((63cea45f-1475-4483-bf5b-7ffb8e884992))
			  collapsed:: true
				- 第二原象指的是给定一个x, 找到一个具有相同y的x'
				- ((63cea4b7-b0df-42a8-aba4-33c75e4bd5ba))
		- 全局理解
			- Block ciphers 将明文分成了等长的块, 传统的加密方式也可以理解为块的大小为一个字母, 或者一个bit. 但是这里的Block cipher会使用同样一个key对每个块进行加密, 并且每个块的大小会更大来提升效率, 例如DES的64, AES的128. 为了保证加密算法的non-deterministic, 各种mode会被使用来创建一些前后块的关联, 或是一次性的信息来让一样的信息块也能被加密成不一样的密文
			- DES对每个64位的块用56位的key进行加密, 加密时分左右部分, 有轮数, 每一轮的字密钥都由key生成, 可复现. 这个结构式Feistel Cipher. 优势是 加解密用的同一个结构, 方便快速.
			- AES更为复杂, 256bit的key就能实现不错的加密效果
			- 密文的长度和明文一样长
			- differential and linear cryptanalysis: chosen-text attack来分析明文密文对应关系; 用linear function 估计non-linear (轮函数DES)
			- ECB: 块与密文deterministic 对应, 改变分组顺序很容易
			- CBC: 推荐, 加密(上密 XOR m), 本密文的改变, 会影响自己的解密和下一个的解密; 解密的时候由于知道所有c, 因此可以并行运算, 并解密任意分组; 需要IV和k
			- OFB: 对IV进行迭代加密生成每个块的XOR对象, XOR对象流可以单独生成, 这个对象流与one-time pad的key很相似, 但是并不是完全随机的. 由于按位异或, 只有对应bit会出错
			- CFB: 加密(上密) XOR m, 对上个block的密文加密与m XOR
			- CTR: 推荐 加密(计数器) XOR m随机IV作为计数器的基, 第几个就往IV加几, 因此可以完全平行运算. 块的数量被IV大小限制, 2^l - 1 - IV个块最多. 由于按位异或, 只有对应bit会出错
	- Week4 Hash, MAC, RSA, DH
	  collapsed:: true
		- 24-30, 53-54
		- 数论
		  collapsed:: true
			- Euler Totient
			  collapsed:: true
				- 小于n的, 与n互质的数的个数
				- (p−1)·(q−1)
			- 原根 primitive root
			  collapsed:: true
				- ![image.png](../assets/image_1675071157228_0.png)
				- ![image.png](../assets/image_1675071313383_0.png)
			- 欧拉定理
			  collapsed:: true
				- ![image.png](../assets/image_1675071391290_0.png)
			- FACTORING problem: given N, known to be the product of two
			  primes, find these primes:
			  given an RSA modulus N, find primes p and q with N = p · q
			- Discrete Logarithm Problem:“For appropriate choices of prime p and element g, it is hard to compute a knowing ga mod p.”
		- Slides
		  collapsed:: true
			- 回顾一下hash的三个resistence, 按照难度从低到高, **second >= preimage**
			  collapsed:: true
				- Collision resistance: 找到有两个x, 拥有相同的hash
				- Preimage resistance: 给定y, 找x
				- second preimage resistance: 给定x, 找一个有相同y的x‘
			- Hash function design
			  collapsed:: true
				- ((63d6e861-dca7-433a-973a-135b7947b0a1))
				  collapsed:: true
					- 假设Oracle可以帮我们给y找到一个随机的x, 那么我们可以利用这个oracle破坏collision resistance
					- 随机选择一个x, 并计算出它的y
					- 把y给到oracle, 得到一个x'
					- 如果x‘和x不一样就找到了collision, 没有的话回到第一步
				- ((63d6e8e2-4172-4787-803a-d431d54599cd))
				  collapsed:: true
					- 要证明碰撞抵抗, 就得证明相同的h, 对应的是相同的x
					- ![IMG_1114.jpeg](../assets/IMG_1114_1675029122792_0.jpeg)
					  collapsed:: true
						-
				- ((63d6ea95-e4e9-4d8b-ba59-3bd5627ccdd8))
				  collapsed:: true
					- ![image.png](../assets/image_1675029196073_0.png)
					- 用有限域的压缩函数f, 来构建一个可抵抗碰撞的hash函数
				- ((63d6eb26-d4ce-4fab-9904-9126d5f922b6))
				  collapsed:: true
					- Avalanche effect desired: small changes in the input should result in
					  unpredictable changes in the output. 需要改变一点, 全局都能大变化
					- MD4, MD5, SHA-0, SHA-2, SHA-3
				- ((63d6ebca-906e-413a-bb46-b71eb2b8d74f))
				  collapsed:: true
					- Davies-Meyer hash
					- ((63d6ebf6-011d-4bc5-9ef0-ff7c5eb075d0))
					- ((63d6ec04-60a3-4ca0-a2d1-44b4a252cdda))
					- 利用迭代的方式, 上一块的结果, 会用来放进下一块里面, 压缩函数是用block cipher加密, 密钥用的是数据x
				- ((63d6f248-a7eb-4e35-a00d-4b329924f072))
				  collapsed:: true
					- ((63d6f9d1-1e9c-4b6c-b528-811eae39c428))
					- the x is used now as “data” and not as key and where we make use of a key
					  derivation function (KDF) g
			- ((63d6fa18-b74a-47c8-b807-7ea467970fc2))
			  collapsed:: true
				- ((63d6fa27-680a-43b6-9000-e425c7bc8ecb))
				- 由于h已知, 中间人可以很轻易的用自己的消息, 修改. 因此我们需要这个hash key-dependent, 来保证只有他们两个人可以
				- ((63d6fa91-0bd5-4403-9533-a61378385a1f))
				  collapsed:: true
					- MAC codes give us integrity, not confidentiality. An idea for both confi-
					  dentiality and integrity:
					- 给到密文和密文的mac, 保证的是密文的integrity
					- 只有拥有key的人才能创建MAC, 才能验证MAC的integrity; 当然key也需要key management, 如何存储, 转移, 交换密钥是一个大问题
				- ((63d7075b-574c-4050-9845-b6321fa91415))
				  collapsed:: true
					- $H_{i+1} = f (H_i || m_i)$
					- 通过迭代关系, 轻松的不需要key, 只需要利用前一个的密文, 就可以得到加了一些内容的新MAC
					- ((63d7089c-f2ea-42b0-9a5b-42f4a1bae336))
				- ((63d70944-02c3-4929-82ff-193360d5a4e1))
				  collapsed:: true
					- ((63d70951-968b-42ee-a3fe-0294689c2977))
			- ((63d70a01-38fc-486a-8e2d-266810c416b0))
			  collapsed:: true
				- 为什么想发明公钥密码?
				  collapsed:: true
					- 对称密钥需要被安全地传输分享
					- 公钥密码不需要, 公共密码是公开的, 但是需要注意公钥的认证问题: 怎么来判断这个公钥就是你或者某网站的公钥呢？
				- [[RSA]] **Factoring problem**
				  collapsed:: true
					- primes > 1024 bits, key > 2048 bits
					- ![image.png](../assets/image_1676218392289_0.png)
					- ![image.png](../assets/image_1675071853484_0.png)
					- ![image.png](../assets/image_1675071892837_0.png)
					- ![image.png](../assets/image_1675071906431_0.png)
					- Problem
					  collapsed:: true
						- 相同的明文意味着相同的密文, 会出现ECB那种直接一块块加密的模式一样的问题, 可以通过用随机数blind m
			- ((63d793c6-39df-434f-a2a4-d27e9efea6f2))
			  collapsed:: true
				- Diffie-Hellman Key Exchange protocol: **Discrete Logarithm Problem**
				  collapsed:: true
					- ![image.png](../assets/image_1675072497630_0.png)
					- 这里的g, 生成元应当为p的一个原根, 生成元的个数为phi(p-1) 也就是g的0-p-1次方能够mod p得到1-p-1所有数字, 这样子才足够随机, 因为不会有两个a撞车得到一样的g^a mod p
				- Discussion
				  collapsed:: true
					- works in any mathematical group in which operation f (a) = ga is hard
					  to “invert”
					- both agents generate parts of the shared key, more trustworthy
					- satisfies *forward secrecy*: if any longterm keys of Alice or Bob get com-
					  promised, it won’t compromise secrecy of shared key K: 一方的key泄露不影响共享key
					- elliptic curves可以用更少的bits 生成安全的密钥
					- key 通常需要被trim, 例如使用hash func
				- Man in the middle attack
				  collapsed:: true
					- ![image.png](../assets/image_1675074224997_0.png)
					  collapsed:: true
						-
		- Notes
			- second-preimage resistance 是比collision resistance 要强的, 因此能够break second preimage resistance 就可以很有效的去打破碰撞抵抗了
			- 怎样证明collision resistance呢? 给定两个不同的x, 他们生成的h是相同的, 如果根据计算结果, 两个x应当是相同的, 那就是collision resistance了
			- Hash Function Construction
			  collapsed:: true
				- ((63d7c620-3001-4b31-ae92-89d57dd14ffa))
				  collapsed:: true
					- ((63d7c64f-0cb0-4ce5-9060-194700bd0723))
					- 以上的算法并不能够抵御碰撞, 例如8->4位的映射, 如果mi位数不够的话需要padding, 例如m1=010, padding后就是m2=0100, 两个数完全不一样, 但是由于padding的存在, 会让两个数字的hash值一致
					- 但是可以通过在最后加一个block记录信息的长度来避免这个问题, 例如 0011 和0100, 一个3, 一个4, 就不一样啦
			- ((63d7cb8d-c849-43c0-bf03-ef14d31408c2))
			  collapsed:: true
				- 我们需要加密的hash来保证不知道密钥的人无法修改信息, 也只有知道密钥的人才能够验证
				- ((63d7cbd9-42ad-4e70-901a-d051aa0f51e5))
				- MAC值就是所谓的[[integrity tag]]
				- Security properties
				  collapsed:: true
					- 拥有MAC值的黑客无法找到一条与原文不同的m‘但是拥有相同的MAC值.
					- 唯一的问题就是key management
					- 普通的MAC, 直接对明文操作的话是不保证confidentiality的 , 如果要保密的话, 那就得对加密的密文进行MAC操作
					- ![image.png](../assets/image_1675087001308_0.png)
					  collapsed:: true
						- 两部分组成, 密文和密文的MAC, 两个密钥要不同, 因为加密机制可能就是不同的, 并且我们要limit the use of a key for a specific purpose, 例如这里就是C和I两个purposes, 这样就需要双重破解了
						- 上述的做法都是针对密文而言的, 有个潜在问题就是这个**Horton’s principle** “signs what is meant (the m1), not what is being said (the ek1 (m))”. 因此可以改进为
						  MAC直接对明文m进行操作
					- **Horton’s Principle** states that it is wise to sign what is meant and not what is being said. but not in blinding protocols for anonymity solutions.
					- This is a good principle. However, there are special use contexts were it
					  is beneficial to sign what is being said, for example in blinding protocols for
					  anonymity solutions. 由中间人的时候就应该sign e了
				- MAC Construction
				  collapsed:: true
					- ((63d7cede-df34-42da-9b1b-810d988a4b7c))
					- MAC post-processing:
					  collapsed:: true
						- CBC-MAC 用了block cipher in CBC mode
						- ![image.png](../assets/image_1675088182413_0.png)
						- m_q 需要padding, 方式有0 padding, 或者再加上一个长度block
						- 第一个方法致命缺陷在于无法知道哪些内容真的是原文
						- 第二个方法则可以通过长度得知, 但是会对明文有长度限制
			- ((63d7d3c0-6cec-48b1-af88-d5b5ae1969e2)) **Discrete Logarithm problem**
			  collapsed:: true
				- ![image.png](../assets/image_1675088976250_0.png)
				- ((63d7d507-de34-461c-adbe-3c637d2bc608))
				  collapsed:: true
					- ((63d7d515-a6dc-4521-bcf1-237e9ec8c965))
				- ((63d7d5f2-2e16-4c7d-9d9b-bc042d8e6cd1))
				  collapsed:: true
					- ((63d7d613-d235-444e-865b-38a51bcd9271))
					- ((63d7d636-e4ef-48d1-86a1-7f7cd97b4730))
					-
					-
			- Menti
				- ![Xnip2023-01-30_10-30-53.png](../assets/Xnip2023-01-30_10-30-53_1675166639231_0.png)
				- 1,2,5是对的, 任何需要身份认证, 确认身份责任的地方需要用到MAC
				- ![Xnip2023-01-30_09-07-05.png](../assets/Xnip2023-01-30_09-07-05_1675166653832_0.png)
					- 1
		- 全局理解
		  collapsed:: true
			- Collision resistance does not imply preimage resistance, 因为你可以构建0开头后面数字都是不一样的直接用原文表示和一个1开头后面是collision resistance的hash fun, 但是只要是0开头就能知道原文
			- 三种resistance 没有孰强孰弱, 生日悖论对于计算量的减少是根号2的
			- DM 和MMO hash 都用到了block cipher, 但是加密的密钥选用上有区别, 一个是用data x, 一个是用key derivation function
			- MAC 让有key的人才能验证消息的MAC对不对, 也是可以用block cipher 来实现, 因为都是很长的信息流. MAC的作用是在未经认证的通道传输用于验证身份
			- RSA 就是mod pq的的一个加密手段, 利用了任何有限群里的元素的phi(N)方都是1这个拉格朗日定理带来的特性, 因为d·e=1+s(p-1)(q-1)
	- Week5 Elliptic Curves, Finite Fields
	  collapsed:: true
		- 63, 41-42, 55-58, 60
		- Slides
			- Elliptic Curves for Public Key Cryptography
			  collapsed:: true
				- $y^2 = x^3 + ax + b$
				- **Elliptic Curve Discrete Logarithm Problem**: Given two points P and Q on an elliptic curve such that Q is a multiple of P , it is a hard problem to find some k such that Q = k ⋆ P .
				- ![image.png](../assets/image_1675640183372_0.png)
				- 简单来说, 椭圆曲线相对于之前直接使用g的a次方mod N 难以知道a是多少来说, 利用了椭圆曲线函数, 在椭圆曲线上面区一个基点, 通过不断做切线交点对x对称操作, 以及和基点连线取交点的x对称操作得到新的k * P 点, 这个点会作为密钥交换的公共部分, 别人知道这个是很难得知这个k到底是几, 也就是到底做了几次P + P的操作, 但是对方在前面乘个j, 和自己乘上kP得到的都是jk*P, 达成了目的
				- 但是, 在密码学中, 这个应用的是有限域中的椭圆曲线离散log问题, 函数的输出被限定在了一个prime大小的有限域中, 要求的不是等式左右两边直接算出来的值相等, 而是mod p的值相等, 椭圆上的点也就被表示成了符合下面这个mod 公式的点集合, 画在图中 将是离散的, 沿x轴对称的
				- ((63e03e4a-7ec3-4918-bbe5-c8b68e324caa))
				- k * P 就是进行了k次 P + P的操作, 加法是group定义的; given points P and Q on curve, it is hard to determine whether there is some k with Q = k ⋆ P, and to compute such k; 难以知道进行了几次P+P的操作得到Q点
			- ECDSA Signature Generation
			  collapsed:: true
				- 生成过程:
				- Let G be a generator of prime order q > 2^160 in E′′(K). The private key for agent A is a random 256-bit number k in N. The corresponding public key for agent A is K = k ⋆ G. The security requirement is that nobody should be able to learn the private key from public information – including the public key K. m message to be signed
				- points P on elliptic curve can be mapped to integer interval [0, q − 1]:
				  collapsed:: true
					- f (P ) = “x-coordinate of point P ” mod q
				- ((63e0443b-49de-423b-a49c-9446e2f308a1))
				- 首先用了两遍hash, 再选择一个一次性的随机数字乘上基点G得到经过u次变换后的点P, 再对这个点P进行0-q-1的map工作得到处理完成的r, s由hash, 私钥k, r, u操作mod q得到 (r,s)就是signature
				- 验证过程
				- ((63e045b1-a22f-41c2-a350-24f16af72cfe))
				- 生成同样的hash, 验证f(u*G) = f(a*G + b*K)
				- To hedge against one of the **2 hash functions** being broken
				- 重要安全提示: 不能使用相同的ephemeral key u对不同消息进行签名, 因为attacker可以通过这个计算出私钥k
				  collapsed:: true
					-
			- EC math
			  collapsed:: true
				- [[Fields]]
				  collapsed:: true
					- ((63e046b7-2572-4e89-b870-3e4f6d2c003a))
					- 域限定了输入的取值范围, 乘法是加法的叠加,
					- ((63e04744-ad99-470e-b64b-4d27551b83bf))
					- Fp是质数群, 包括了0-p-1, 有限的值域, 通过mod操作来重新映射
				- [[Finite fields]]((63e049d6-e6dc-424f-9d7b-8ff955d9862a))
				  collapsed:: true
					- Let (K, +, ∗, 0, 1) be a finite field.
					- the **[[order]]** of K is the size of set K, K的set大小就是K的阶 order
					  collapsed:: true
						- the **[[characteristic]]** of K is the smallest n > 1 such that the nfold sum
						  1 + · · · + 1 equals 0, 特征是几个1加起来能mod N等于0; 只有prime power order 的域才有characteristic, 2的3次size的域, characteristic还是2, 但其中乘的操作会在多项式上进行以达到闭环目的, 元素都是多项式, 他们的degree 也就是几次会小于等于2, 会有一个Ideal, 作为类似于之前(mod 7)的一个模数, 只不过这里是一个多项式, 可以是p(x)=1+x+x^3, 其他多项式相乘mod这个p(x)得到的结果会在这些元素之内, 形成一个环, 且是一个大小为8的域, characteristic为2, 因为大家的系数a都是从(0,1)中取的. 为什么有八个: a+bx+cx^2, abc从0、1中取一共有八种组合
					- Z7 就是0-6的域, order 和characteristic 都是7
					- Z*7就是一个域的乘法可交换群, order为6, 因为需要元素都和7互质且0不可
					- 域的大小size或是order必须要prime power的, 比如size为2^3=8的域中的8个元素就是(1, x, x^2)
					- 域中的加法inverse就是加几可以mod N到0, 乘法inverse则为乘几可以mod N为1
					- Facts about finite fields k
					  collapsed:: true
						- • there is some prime p and n ≥ 1 such that the order of K is p^n
						  • two finite fields with the same order are isomorphic as fields
						  • fields of characteristic 2 are exactly those of order 2n for some n ≥ 1
					- ![image.png](../assets/image_1675646464608_0.png)
					- ![image.png](../assets/image_1675646685585_0.png)
					- 这里对有限域定义了群操作, 第一个e是0, 写错了, 我没改掉, 加法是成立的, 不管怎么样都能找到inverse
					- 但是乘法必须要在下面这个subgroup中实现, 因为不一定可以乘起来mod出1来
				- ((63e049cf-5808-4f0f-81eb-c8e79f49712b))
				  collapsed:: true
					- ((63e049c8-877c-457c-96a9-bc2e113d42a0))
					- 投影平面就是一个多维坐标集合, 坐标取值范围是K
					- ((63e04a1b-66fc-4821-a1d0-834659d0c867))
					- 三等符号的含义是两个坐标可以通过一个scalar 缩放成一样
				- ((63e04a6f-abd3-4238-a4a0-a0fb8a0ed789))
				  collapsed:: true
					- 齐次椭圆曲线方程
					- ((63e04aa2-5b26-401f-902b-b171da7dc1e5))
					- E(K)是在K域中的点集合, mod了K的大小
				- ((63e04b37-7f76-428a-bf10-f426023f6790))
				  collapsed:: true
					- 将Z看作是1, 进行一个缩放得到相通但简单的表达
					- ((63e04b75-3feb-42fc-8e85-f2f08b9dbd5a))
					- 由于没有z了, 要表达无限远O = (0, 1, 0), 就需要特地拿出来一个
				- ((63e04bbb-e1b1-409b-b257-649a87c40f6a))
				  collapsed:: true
					- ((63e04bde-a129-4926-bc2c-7dbab4120043))
					- 3值表达更为高效, 但是2值表达更容易理解
				- ((63e04eff-7b6c-40cd-bed1-2b96e4965460))
				  collapsed:: true
					- 当finite field的characteristic为2 或者3的时候, 曲线的形式就不能简化为short Weierstrass form了. 并且即便式子不同, 但是所表达的曲线上的点在field中会存在重复, 也就是不同的coefficient, 相同的点集. 因此不能够使用2或者3
					- ((63e04eeb-9ffb-4fe4-b892-f51738c73a4f))
				- Chord-tangent process, Group law of EC
				  collapsed:: true
					- ![image.png](../assets/image_1675644838909_0.png)
				- ((63e04fac-dad3-4e78-89f1-a09111fe481a))
				  collapsed:: true
					- 如何计算P3, 根据P1 P2得到P1 + P2
					- ((63e04fdb-7ef9-4e3c-b022-e92f86295598))
		- Notes
		- 宏观理解
			- 简单来说, 椭圆曲线相对于之前直接使用g的a次方mod N 难以知道a是多少来说, 利用了椭圆曲线函数, 在椭圆曲线上面区一个基点, 通过不断做切线交点对x对称操作, 以及和基点连线取交点的x对称操作得到新的k * P 点, 这个点会作为密钥交换的公共部分, 别人知道这个是很难得知这个k到底是几, 也就是到底做了几次P + P的操作, 但是对方在前面乘个j, 和自己乘上kP得到的都是jk*P, 达成了目的
			- E(有Z)中计算比较高效(因为没有inverse), E'没了Z要另外包括O, 他俩isomorphic. E''是特征非2,3的域上的椭圆曲线, 可以写作最简形式
			- ![image.png](../assets/image_1678739128569_0.png)
			- EC被用在了数字签名, ECDSA
				- G是EC的基, K是对G做了k(私钥)次运算后的公钥, h是信息的hash, u是一次性密钥对G做u次运算生成了一个r, s作为签名, 汇总了h, k*r和u的逆. (r, s)是签名. 验证签名时计算v是否等于r.
			- RSADSA RSA的数字签名
				- 用私钥对hash后的m加密, 发送(m, s), 验证者对比公钥解密后的hash是否与自己hash的m相同
				- preimage resistance: can compute a preimage m of the decrypted s, but is an existential forgery since the attacker cannot control the content
				- collision resistance: sender can be a potential attacker, can repudiate the m with m', which has the same hash
				- second-preimage resistance: attacker can find another m'
	- Week6 ECs Weierstrass form, EC efficiency, secure sharing, Reed-Solomon Codes, Shamir Secret Sharing
	  collapsed:: true
		- 78-82
		- Slides
		  collapsed:: true
			- Elliptic Curves
			  collapsed:: true
				- 在一个小的有限域上的EC的形象表达 in short Weierstrass form
				  collapsed:: true
					- ((63e05059-ba36-4521-866e-3d73d5405565))
					- ((63e05060-8886-4038-813a-c7e496835e73))
					- 可以看出点的个数是有限的, 而某个点的连加最终都会走到infinity
				- ((63e050b6-54de-4b39-b50f-93e85858fd71))
				  collapsed:: true
					- 用数学的方法evaluate 这个curve的安全性
					- ((63e90c42-f520-4692-88e5-fd47fbcbc922))
					- 某个finite field的order的trace of frobenius可以被定义出来, 与order 本身和描绘出的曲线点集大小有关
					- 当n=1的时候, p^n是prime, t=1, 这个时候就不安全
					- n>1的时候也有类似的情况, 要尽量避免
					- 我们需要|E(K)| 能够被一个大的prime整除
					  collapsed:: true
						-
				- ((63e050a8-7194-4732-8eb3-6e5a1114ea63))
				  collapsed:: true
					- 1. may choose values of curve coefficients, may choose finite field: this
					     gives us a very large number of possible groups for Group Law 椭圆曲线参数, 域的大小, 有很多可以选择的参数来构建不同的group law
					- 2. finding E and K with strong cryptographic properties is relatively easy 容易找到高强度的椭圆曲线group law
					- 3. curves E(K) with strong cryptographic properties much more secure
					     than, say, a 7 → ga mod p in terms of the bit-size of p 比特数量要求减少了, 但是保持了相近的安全程度
					- 4. thus we may decrease size of p when using Elliptic Curves, without
					     losing security when compared to operating in multiplicative group
					     ({1, 2, . . . , p − 1}, λxλy : x · y mod p, 1) with suitable generator g 可以使用小一点的p了
					- ![image.png](../assets/image_1676218346640_0.png)
				- ((63e90fba-8101-4937-909a-f9412fad2218))
				  collapsed:: true
					- $K^2$ 是affine form, 只有(X, Y), 计算中会有除法的存在, 而除法在group内做的事multiplicative inverse, 即寻找和他乘起来mod N为1的值, 计算上面很昂贵
					- 解决方案是使用projective coordinates (X, Y, Z) instead of (X, Y ); One good method is to transform (0, 1, 0) back to (0, 1), and (X, Y, Z) back to (X/Z^2, Y /Z^3) for Z  != 0. 只有在最后一步有除法了, 中间就没有了
				- ((63e9148e-9454-42e6-a067-79591a0ac1bc))
				- ((63e9168e-e6c6-4eef-9b60-29bd95f7d866))
				  collapsed:: true
					- ((63e9170f-c137-414b-b639-6f5a7d5d886c))
					- 可以通过快速幂来计算mod, 增加速度
				- ((63e9176f-d18d-4b13-b7fe-ae8f9acac9b7))
				  collapsed:: true
					- ((63e91786-a0ba-45ca-b611-5543084bcd8e))
			- Secrete Sharing
			  collapsed:: true
				- ((63e917d5-5755-4757-a228-7ef3e29c461b))
				  collapsed:: true
					- Error-correcting codes used in CD/DVD, BluRay, WiMAX, and in space missions such as the famous interstellar mission Voyager.
					- 错误修正代码,
					- Mathematical setting:
					- ((63e91909-58ad-4783-89a0-93bf866994a5))
					- 这里的P是一个多项式集合, 集合了各种可能的多项式, 多项式的系数f_i是从finite field中取的, 因为有一共t+1个fi, 因此P的大小也就是$q^{t+1}$.
					- X则是各个parties, 不同的sharing 者, 他们各自拥有自己的密钥, 即如果自己是1, 那么自己将拥有把x=1代入这个式子后的值作为我的share 密钥. 而这个多项式的系数就是集齐几个parties才能够得到的公共宝库
				- ((63e91c25-7cad-4cf3-a536-20dec8f02d76))
				  collapsed:: true
					- Evaluate code word f in P at all points in X = {x1, x2, . . . , xn}:
					- $C = {(f (x_1), f (x_2), . . . , f (x_n)) | f \in P}$
					- 每个party会拥有code words中的一个
					- C虽然有n个元素, 每个元素的长度是$log_2 q$, 但是只需要t+1个元素就够了, 因为我们设计的时候, 就是t个元素, 多余的就是redundency for error correction t < n
				- ((63e931e4-b775-46ee-8746-d9f7ecadb9e5))
				  collapsed:: true
					- q = 101
					  t = 2
					  n = 7
					  X = {1, 2, 3, 4, 5, 6, 7}
					- q是一个prime number, 用来mod的一个field order大小, t是多项式的order, 决定了最少需要t+1个parties来解, n是所有的多项式x代入结果冗余, x是q的子集, 包括了代表parties的几个x值, 用来代入多项式获得f(x)
					- ((63e93448-4b47-4d0b-8043-eb15eae36845))
				- ((63e9345c-a186-46bb-b069-544ed2d08e4b))
				  collapsed:: true
					- if t < n and no transmission error occurred in communication of c, we can fully recover f from c.
					- ((63e93491-7d56-48e0-9a5a-86fafb16f0df))
					- 根据Fundamental Theorem of Algebra, 解唯一, t+1个点时
				- ((63e934c1-7214-493d-8fc2-3113c1c9c042)) Lagrange interpoltion
				  collapsed:: true
					- ((63e93627-3035-4110-b59b-7b533f3ef1a6))
				- ((63e93631-f0e2-4604-bdf4-4389c1a3fcb4))
				  collapsed:: true
					- (t + 1)-out-of-n secret sharing scheme
					- Security property: t or less than t parties, when colluding, should not
					  learn anything about the secret, which is an element s in field F_q.
					- ((63e936ed-8db1-4061-b729-c6ecd1752eff))
					- 现在的s是函数的常数项, 正常解的话, 就是得到参数以后求f(0)
					- 如果用Lagrange的话, 就可以直接算出来
					- ((63e9379c-1c26-4e02-a0af-699454595569))
					- 这里的recombinant vector只与X有关, 可以重复使用
					- Example
					  collapsed:: true
						- ((63e938e5-8529-44bb-8e5c-d8addbc9ce26))
						- 每个party根据他的id, 会得到f(id)的值, 即为一个share, 因为我们的函数2次方, 所以有3个shares就可以了
						- ((63e93955-07f6-41f4-bea2-bd69d4ec48ec))
					-
					-
				- ((63e93972-4a16-47b4-b075-46cbd9961bdf))
				  collapsed:: true
					- 只要展示show this for the case when | Y | = t的情况没办法还原就够了
					- 由于只有t个, 所以只能找出来一个g, order为t-1, 极端情况下能得到一个g(0)刚好等于s, 但是即便得到了, 他们也不知道这个就是s
					- ((63e93a49-0fde-4d7d-a4b8-0e4cd4c736a7))
					- 可能性太多了, 根本无法确定是不是真的s
		- Notes
			- ![image.png](../assets/image_1676280995042_0.png)
			- ECDSA 椭圆曲线数字签名算法
			- 利用的是部分好用的椭圆曲线, 并非所有曲线都是好使的
			- 由于椭圆曲线的性质, 相同安全性需要的key的size要小于RSA
			- 由于ephemeral的存在, 是non-deterministic的
			- private key是一个long term key, 使用一次性的ephemeral keys来让签名结果不具确定性, 即每次都会不一样
			- ![image.png](../assets/image_1676288202436_0.png)
			- ((63ea32eb-8b78-43f7-b40a-6eff5b6d8cf8))
			- 上面的Figure就给到了由$p(x) = 1 + x+x^3$给到的quotient ring, 是一个finite field, size是2^3 = 8, 即有8个元素, 且每个匀速都有能变成1的multiplicative inverse
			- quotient ring 是除了p(x)以及他的高阶延伸的所有coefficient 组合的多项式的集合, 如表中所示, 因此第一个选项是正确的
			- 第二个选项可以见表中的x(x+1)得到了x^2+x也是field内的元素, 因此是错误的
			- 第三个选项, The ring of polynomials over the rational numbers is a field 应该是对的
			  collapsed:: true
				- Given the polynomial ring F_K[x] for a field K, rational functions are quotients p(x)/q(x) - as formal expressions - where p(x) and q(x) are in F_K[x] and q(x) is not 0. It turns out that this set of rational functions is then a field where 0, 1,  and * is what you might expect.
				- In particular, this is true for the field K = Q of rational numbers. The ring F_K[x] is not a field only because non-unit elements p(x) that are not 0 do not have a multiplicative inverse. This is a problem that rational functions solve; and note that the ring F_K[x] embeds into this field where p(x) is mapped to p(x)/1.
				- The ring F_K[x] is infinite when K is infinite, as K embeds into F_K[x] where field elements are identified with their constant polynomial. For K = Q, the field of rational functions is infinite for the same reason.
			- 第四个选项, AES中也运用到了finite fields和多项式
			- 第五个选项, fields的大小得是prime power
			- ![image.png](../assets/image_1676288206889_0.png)
			- Shamir‘s secret sharing 是安全的, 信息学上的安全的
			- Recombinant vector是由party集 X决定的
			- 可以用来管理key, 例如多少个人满足才能获取密钥
			- 加法是同构的, f+g (n) = f(n) + g(n), 但是乘法是不满足的
		- 宏观理解
			- Secret share 主要用的是Shamir scheme, 多项式次数+1个group的值可以解出这个多项式的系数, 通常0次的常数项作为secret;
			- ![image.png](../assets/image_1678725456879_0.png)
			- 通过Lagrange interpolation 来简化计算, 用recombinant vector来计算得出0次项的值
			- ![image.png](../assets/image_1678725042033_0.png)
			- 例如3个parties, 就需要计算三遍这个值, 然后各自的值乘上各自的secret以后的和就是s0了. 这个算出来的其实是g(0), 但由于我们的party数比次数高, 所以g(0)=f(0); 如果个数不够, determine的g的degree就不够, 0点的可能性就在域内都有可能.
			- If parties 1 and 3 pool together their shares, they can determine a polynomial g(x) of degree 1 but the polynomial f(x) has degree 2 and so for each value z in Z31 there is a polynomial h(x) of degree 2 with h(0) = z, and where h(1) = g(1) and h(3) = g(3). Therefore, parties 1 and 3 cannot learn any information about the secret s from combining their shares.
			- Reed-Solomon codes generalize this to the case in which some of the shares may be erased s (i.e., no longer available) or may be corrupted e. e<t< (n−s)/ 3. n是total number of shares, t是多项式的degree. 当error数大于或等于degree的时候就recover不了了.
	- Week7 Complex numbers, Quantum secret exchange
	  collapsed:: true
		- 83-91
		- Slides
		  collapsed:: true
			- ((63f25aa5-c3ca-4b2d-839e-ddf50d2ce7c9))
			  collapsed:: true
				- ![image.png](../assets/image_1676827324394_0.png)
				- a is real part, b·i is the imaginary part, where $i^2 = -1$
			- ((63f25b02-ae81-4061-be87-12e3940b933b))
			  collapsed:: true
				- ((63f25b12-deba-4172-a08b-260f10065848))
				- 实数部分和虚数部分分别相加即可
			- ((63f25b36-0755-4bac-b07c-43be217dbaf5))
			  collapsed:: true
				- 分配律 Distributivity Law 依然成立
				- ((63f25b69-fc4c-4191-b27a-c89864e6efa5))
			- ((63f25b80-ba64-458f-8860-ef5130cabeb4))
			  collapsed:: true
				- (C, +, 0) is a commutative group that + and · satisfies the distributivity laws of a field.
				- 1 is the unit for · in C, 任何非0的c都有multiplicative inverse
				- ![image.png](../assets/image_1676827603841_0.png)
				- ((63f25bdd-1477-493b-8b61-3279415dbe78))
				  collapsed:: true
					- The conjugate of c = a + bi negates the imaginary part c^bar = a - bi
					  id:: 63f25bdf-2006-4b29-b611-f54794677b34
					- ((63f25c7e-975d-4a1b-a9ae-21d9b5d97752))
					- Norm是根号下的系数平方和, inverse可以用 conjugate 简写
			- ((63f25cc3-4e8b-4531-bb78-a38e5ba9e02b))
			  collapsed:: true
				- 虚数也可以构成vector, 例如二维的虚数vetor
				- ![image.png](../assets/image_1676827881290_0.png)
				- ((63f25cf7-1839-417f-8eca-6ea22fff42f9))
				- ((63f25d09-6b84-4b32-a02f-442866c6e692))
				  collapsed:: true
					- Vector相加, 就是对应维度项的相加
					- ![image.png](../assets/image_1676828086524_0.png)
					- scalar vector相乘, 就是每个维度都乘上这个scalar
					- ![image.png](../assets/image_1676828099159_0.png)
				- ((63f25ddf-8c2d-47e1-ba9a-6ba87d61fa50))
				  collapsed:: true
					- vector space operations are component-wise:
					- ![image.png](../assets/image_1676828160118_0.png)
				- ((63f25e16-6d52-479f-9721-3e856893699f))
				  collapsed:: true
					- 2x2 complex matrices 可以作为transformation matrix 变换quantum bits
					- ![image.png](../assets/image_1676828228746_0.png)
				- ((63f25e57-15f7-44d7-a73f-38887795990b))
				  collapsed:: true
					- 两个2x2的矩阵相乘, 和普通矩阵乘法是一样的
					- ((63f25e7f-c7c4-4c62-924b-edc58fe5acc2))
					  collapsed:: true
						-
			- ((63f25ebf-166e-43b4-a116-31fe7e029d03)) (two bases)
			  collapsed:: true
				- A “qubit” q is an element of C^2 with norm 1
				- qubit 需要norm为1, 长度为1, 一个qubit由两个complex number组成, 每个complex number有两个系数, 一共四个系数, 他们的各自的平方的和要等于1
				- Since qubits are elements of C2 with norm 1, they correspond to the points on Bloch’s sphere, a 2-dimensional sphere. Therefore, there are uncountable many qubits! This is in striking contrast to the fact that there are only 2 classical bits: 0 and 1. qubits的数量和可能性是无穷的
				- ![image.png](../assets/image_1676828394359_0.png)
				- qubits可以被写成ket notation form, 两个常用的qubits是 [[ket right]] and [[ket up]]
				  collapsed:: true
					- ((63f25f54-2eb8-43ef-a6b8-ed907a670aa5))
					- 由这两个qubits组成的set ‘+’, 即plus basis, 是C^ 的一个basis
					- ((63f25fa1-2ebb-4576-8725-d255a84453ed))
				- Another basis we will use for quantum-key exchange uses “ket 45 degrees” and “ket −45 degrees” and is × = {| ↗〉, | ↖〉} where
				  collapsed:: true
					- ((63f25fed-a49e-415d-85e9-05ab98a1f668))
				- States of 2-dimensional quantum systems are elements of C2. 即二维量子系统中的states 是C2的元素们都是states
				- In quantum physics, we cannot “read” those exact states, but we can measure them in a given basis. The measurement collapses the measured state to a basis vector.
				- 在量子力学系统中, 我们没有办法对这些state进行read操作而不破坏他们的原本状态, 我们只可以measure, 测量他们的状态, 会截获并得到一个结果, 这个结果依据我们measure使用的basis会有不同, 采用发送者相同的basis会得到相同的state, 但是如果使用的是45度差的则会有50%的概率得到当前使用的basis中的两个qubits中的任意一个. 这种测量后才得到结果并且结果依照basis而定的情况就叫做measurement collapse
			- ((63f263bf-cde5-48aa-ba2d-18f99382f770))
			  collapsed:: true
				- measurement collapse
				  collapsed:: true
					- ((63f2920d-4712-495e-a6df-2bdf7531c6fb))
				- qubit的系数会对measurement最后的basis结果有概率影响
				  collapsed:: true
					- ((63f265af-fecc-4a89-aa67-24a23270e0d1))
			- ((63f29291-eb16-465d-9ac6-e9cfd6b518c5))
			  collapsed:: true
				- **Creating a source of uncertainty**: We have two bases of the complex vector space of qubits, + and × and a qubit does not reveal in which basis it was prepared.
				  collapsed:: true
					- 两个basis带来的不确定的source, 不知道用的是哪个base, 第一重uncertainty
				- **No information gain**: Measuring any vector from one of those bases in the other basis has an equal probability of observing any of the vectors of that other basis.
				  collapsed:: true
					- For example, measuring | ↖〉 in the + basis gives us no information
					  gain as to which vector | →〉 or | ↑〉 of + we might observe.
					- 不正确的basis会导致结果也不确定, 50 50, 没有info gain
				- **Probabilistic certainty**: Measuring a basis vector in the same basis is guaranteed to observe that same basis vector with probability 1. For example, measuring | ↖〉 in basis × observes | ↖〉 with probability 1.
				  collapsed:: true
					- 但是同样的basis会给到一样的结果
			- ((63f29542-729f-49d3-8203-598a96e714d6))
			  collapsed:: true
				- Players: Alice and Bob share a quantum channel and an authenticated public channel. Eve wants to intercept or manipulate the key in transit.
				- Aim: Alice and Bob want to securely exchange a bitstring of length m so that any tampering or analysis of Eve will be detected.
				- ((63f29ab8-5335-4b53-95b2-d1678cdb4359))
				- ((63f29ac9-6180-40ec-bf10-ae54690d921d))
				- Steps:
				  collapsed:: true
					- collapsed:: true
					  1. Alice generates n = 8 · m bits of randomness x1y1x2y2 . . . x4my4m.
						- For sake of illustration, let us say that m = 2 and that Alice generates
						  1011100101001010 and so x = 11100011 and y = 01011000.
						- 根据上面的encode scheme, 来生成qubits, xy生成一个, 所以有4m个
					- 2.From this randomness of 8 · m bits x1y1x2y2 . . . x4my4m, Alice prepares 4m
					  collapsed:: true
					  qubits | x1〉, | x2〉, . . . , | x〉4m.
						- ![image.png](../assets/image_1676844234142_0.png)
					- collapsed:: true
					  3. Alice sends the qubits | x1〉, | x2〉, . . . , | x4m〉 in that order to Bob over the
					     quantum channel. 从量子通道, 发送给bob这些qubits
						- | ↑〉, | ↖〉, | ↑〉, | ↗〉, | ↗〉, | →〉, | ↑〉, | ↑〉
					- 4.Bob will generate a classical bitstring z = z1z2 . . . z4m of bitlength 4m.
					  collapsed:: true
					  Bob will then measure each of the 4m qubits | x′_j 〉he receives as follows:
						- ![image.png](../assets/image_1676844385963_0.png)
						- ![image.png](../assets/image_1676844401324_0.png)
						- 遇到basis匹配错误的情况, bob就会得到一个在measure basis上的随机qubit
					- collapsed:: true
					  5. Alice and Bob share their bitstrings y = y1y3 . . . y4m and z = z1z2 . . . z4m
					     on the authenticated, public, and classical channel.
						- Step 5.1: Alice will delete in x1x2 . . . x4m all xj for which yj != zj .
						- Step 5.2: Bob will delete in x′1x′2 . . . x′4m all x′j for which yj != zj .
						- ((63f29ea9-825b-4884-bba4-cbfc434d2657))
						- 这个情况下, 除非有通讯错误, 或是恶意操弄, 这里的5 bits 对于双方都是一样的
					- collapsed:: true
					  6. ((63f29f2e-d134-46c2-8f5a-2429ed7a860a))
						- 要验证两个人得到的5 bits是不是一样的, 首先B生成一个k/2大小的测试集index集 I, I会被公开, 也就是第五步里最后去掉不一致的bits后的长度的一半大小的测试集T, K_B就是x‘中去掉T’中有的元素以后的bits, 公开T
						- Alice用公开的信息I, 创建T测试集, 使用相同方法得到K_A, 公开T
						- 他们两个人再都看一下两者的T有多少不一致的bits
						- 少的话就没问题, 多的话, 就有问题了, 这里的key只是likely一致, 如果使用过程中发现不一致, 还需要重新进行这个过程
				- ((63f2a45f-9374-4dbc-ba2c-7c0e314e0f2d))
				  collapsed:: true
					- 传统的通道中, Eve可以保存数据, 中间篡改, 能够读但是不修改数据, 但是量子通道中, 因为数据不可读, 只能测, 而一旦测量就会有坍塌, 无法得知源数据真正的样子, 因此Eve只能拦截, 剪切, 而无法拷贝
					- However, in quantum computations:
					  • Cut, paste, and analyze: Eva can transfer a qubit (a “cut and paste” in computer science) but this transfer must collapse or destroy the original bit (“copy and paste” is impossible). This follows from the no-cloning theorem of quantum physics.
					  • Reading quantum states: This can only be done through measurements, which will change the quantum state to one of the basis vectors with respect to which the measurement is made. Therefore, it is physically impossible to read a quantum state without modifying it.
				- ((63f2a5dc-4cbc-4599-a95d-1d03cc032999))
				  collapsed:: true
					- ((63f2a5e5-0ea8-4658-8a76-8244d708f684))
					- ((63f2a5fb-94b6-4d2f-b703-a63dfc5f0d9b))
					  collapsed:: true
						-
		- Notes
			- ![image.png](../assets/image_1676883974844_0.png)
				- 1. NO, 短暂即逝的, 没有长期的storage, quantum states 会被别的粒子影响
				  2. NO, 得要用2 complex numbers, 4 real numbers 可以用来表示两个complex numbers, 四个数字squared norm等于1
				  3. YES, 可以有多种qubits的表现形式, 比如energy state, low 是0, high是1, 之类的; quantum tech的一个问题就是找到一种合适的表现形式来表达 This system may be realised in different ways: as a single photon whose state is either in vertical or horizontal polarization, as an electron that has either spin up or spin down, as an atom that is either in its ground energy state or at an excited energy state, and so forth.
				  4. YES, 可以通过编码进行表示经典比特
				  5. NO, 不能被read, 只能被measure, measure 以后会改变成one of vectors in the eigenspace,
			- ![image.png](../assets/image_1676939509993_0.png)
				- 1. protocol最后会检测相同的个数, 错误少的话认为没啥问题
				  2. 是的, 需要authenticated 公共通道
				  3. 没有被广泛使用
				  4. 可以用不同的basis
				  5. 对的, 不random的话就出问题了
			- 关于Eve作为中间人attack的问题, 貌似量子通道内, Eve能做的只有拦截, 没办法做到重新发送, 所以没办法假装是双方
		- 宏观理解
			- 关于qubits
			  collapsed:: true
				- qubit 没有longterm storage, 只能被观测, 无法被储存, 阅读而不改变其state
				- qubit的表示用了两个complex number, 作为一个二维复数空间的单位向量, 其模必须为1, 也就是其复数表示的四个系数的平方和必须为1, 当然前面可以有个scalar. 因此所有qubits会形成一个Bloch‘s sphere, 一个二维的球, 由无数个qubits组成, 这与传统的2位有很大区别
				- 拥有多种physical manifestations, 例如polaristion, spin up down, energy state
				- 可以以某种人为的定义表示经典的0和1
				  collapsed:: true
					- 两个linearly independent qubits可以组成一个basis, 我们用到了两组basis, 分别是+ = {右, 上}和x = {右上, 右下}, 用几何平面的类似的向量表示
			- 关于measurements
			  collapsed:: true
				- measure一个qubit需要定义一个basis来测量, 任何量子会collapse to这个basis的两个qubits, 量子坍塌的概率是实打实的随机. 具体坍塌到哪一个qubit, 由被测量的这个qubit写作以这个basis为底以后的两个单位向量的模方决定|ψ⟩ = α · |→⟩ + β · |↑⟩, 则其概率为a^2 和 b^2
			- BB84 protocol
			  collapsed:: true
				- 用quantum channel建立一个shared bitstring K, 同时可以发现通讯错误和恶意修改
				- 需要一个authenticated public channel
				- 0 = 右+, 右上x; 1 = 上+, 右下x
				- m长度的key, 需要8m长度的比特串来运行. 分为4m长的x表示bit, 和4m长的y表示basis, 组合得到4m个qubits
				- Bob也生成4m长的z表示他的basis来测量Alice发过来的4m个qubits, 不一样的basis会导致结果随机, 得到4m长的bitstring x
				- A和B在公共频道share了z和y, 比较了两者不同, 把不同的位置的 x删掉, 两个人的bitstring就一样了, 由于测不准概率为1/2, 删掉后的期望长度就是2m
				- 由于他俩不知道他们的x是否一样了, 所以还有一步. B随机从x里面选出1/2size个数的bits, 将(i, xi)们发送到公共频道, 验证, 如果两个人的这个一样或者差别很小, 就验证通过
				- 验证过程又有一半的丢了, 所以最后得到的share key size 为8m/2/2/2 = m
				- 注: 极小概率还是不同但会被发现, 重新run; bit位数可能不够, hash一下
			- Eve
			  collapsed:: true
				- 无法克隆, 无法不测量的情况下阅读
				- 量子通道是单向通道, 比如一根光纤, 只能中间观测一下, 任由qubit远去
				-
		- Questions
			- What is an authenticated public channel? Why is it authenticated?
			- Is it still secure If Eve acts as a man-in-the-middle who pretends to be Bob and measures all the qubits from Alice and sends her own message to Bob? In other words, Alice may consider Eve as Bob, and Bob may consider Eve as Alice.
	- Week8 Commitments and Oblivious Transfer P121
	  collapsed:: true
		- 65-67
		- Slides
			- 为什么需要commitments?
			  collapsed:: true
				- Security protocols are wise to assume that their participants may not be
				  honest.
			- Aim
			  collapsed:: true
				- Design protocols in which parties can only cheat by also exposing
				  that they cheated. 设计一种protocol, 如果某一方cheat了, 是能够发现它cheat了的
			- ((63fa7a53-6afb-4834-a45f-62d728239093))
			  collapsed:: true
				- 远程进行石头剪子布游戏, 先出的一方人会吃亏, 因为后面那个人可以选择击败的选项, 但是如果先出的一方先给出一个commitment来表示他出了某一个, 但是另一方并不能知道这个是什么(concealing), 另一方出完了以后, 第一方再把真正的选项给出来, 第二方可以进行算法运算得到commitment, 如果和一开始一样(binding), 即验证成功
				- Root of the problem
				  collapsed:: true
					- asynchronous nature of play. 异步造成了这个问题, 如果大家同时, 就不会有这个问题了
				- ((63fa7aeb-abfa-43e6-a83f-e97d1a56a266))
				  collapsed:: true
					- hA defined as H(RA || paper)
					- H是hash fucntion
					- Ra是随机数, 用来阻止B从hA反向推出原值什么 (concealing)
					- hash function的难以找到second preimage特性用来bind原值和commitment (binding)
			- ((63fa7b9a-ead1-408f-bd50-e812e8df92cb))
			  collapsed:: true
				- Concealing Property of Commitment Scheme: (Hiding Property) protect sender
				  collapsed:: true
					- Bob won’t know which value A has committed to, since Bob does
					  not know RA after step 1. B 无法提前知道commitment背后的真实值
					- 接受者 没法提前知道
				- Binding Property of Commitment Scheme: protect receiver
				  collapsed:: true
					- A 无法逃避其所commit的值, 没办法找到新的值, 得到一样的commiment. 例如在上面的石头剪刀布的游戏中, A需要找到一个random R'A 使得hash完的结果一样.
					- 发送者 没法抵赖
			- ((63fa7c24-be54-48d8-8d67-08b67dac89e8))
			  collapsed:: true
				- information-theoretic security: attacker with infinite computing power
				  cannot break the scheme
				- computational security: attacker with specified computation resources
				  (e.g. probabilistic polynomial time) cannot break the scheme
			- ((63fa7dad-e182-40dc-89f3-d5acb506ab78))
			  collapsed:: true
				- A commitment scheme is a public algorithm C which takes a value x and
				  collapsed:: true
				  some randomness r as input to produce a commitment
					- $c = C(x,r)$
				- ((63fa8010-d349-4cf9-9d6a-bcfc6d2ca9d1)) 能不能找到另一对x r得到相同的commitment值
				  collapsed:: true
					- ((63fa8031-57b4-4ff5-983c-9b3b882d842e))
				- ((63fa805b-eeaa-4497-b9f3-293292c951d5)) 能不能通过commitment 得到原来的值
				  collapsed:: true
					- ((63fa80f3-38e4-4604-b888-387a8eb834a6))
					- attacker 就是adversary, 发送了两个消息, 收到了commitment, 要猜到是哪个
				- Lemma 20.3: There is no commitment scheme that is both information-theoretically binding and information-theoretically concealing.
				  collapsed:: true
					- 无法做到完美的又保密, 又绑定
					- 如果绝对的绑定的话, 每个commitment值只对应一组r和x, 无限计算能力就可以找到这组
					- 如果绝对的保密的话, 每个commitment应该有无限种可能, 这个时候也就不绑定了
				- Lemma 20.4: For a hash function H, the commitment scheme is “at best” computationally binding and computational concealing.
				  collapsed:: true
					- 用hash function的话, 因为值域是有限的, 所以binding只能是很难找到第二个原像.
					- concealing 找的是某一个存在的原像, 也是只有计算上的安全, 但是如果有很多collision 存在的话, 那就是information- theoretically secure了, 要求是这个R从一个很大很大的集合中取, 就会有很多collision, 无法确定哪个是真的
				- ((63fa840a-b0c5-4f75-820c-331441de87a9))
				  collapsed:: true
					- ((63fa8414-e62c-4d02-bd6b-ced5a2cf8869))
					- G是一个可交换群, 是由prime q作为size用一些东东generate的, 比如说选定一个原根g, g的123456次方 模q 可以获得1 - q-1之间的数字, 也可以用椭圆曲线来做这件事情, 差不多就是换了个运算方法而已
					- g的x次方得到的值基本上就是q里面随机取一个数的那种随机, 把这个作为h公开, 别人也很难知道到底是哪个x
				- ((63fa8945-2d12-48e5-a528-7858e0cb64cc))
				  collapsed:: true
					- 如何找到一个safe prime: Let p and q be prime with p = 2 · q + 1. Then p is called a “safe prime”.
					- ((63fa89b5-e936-48fb-85f9-627b2e9cac25))
					- F*p是一个order为p-1的乘法交换群, 因为去掉了0
					- 如何找到Generator g
					  collapsed:: true
						- ((63fa8a04-3d25-4b6a-85bc-4f1c60601971))
						  collapsed:: true
							- 由random r找一个不等于1的g, g和r是绑定的; 这里的g由安全质数p推得, 将可以用来$g^x\ mod\ p$ 来生成一个大小为q的subgroup G, x为0~q-1的情况下得到的值应该都不一样, 会遍历G中的所有元素.
							- 举个例子, sefe prime p=5, q=2, g由上面的算法可以找到4
							- 4^0 mod 5=1, 4^1 mod 5=4, 4^2 mod 5=1, ...
							- 这个G的大小就是2=q, 也就是order为q, 包含1和4两个元素
							- 也就是后面的承诺算法中, 任何用g生成的数字mod p以后得到的结果都会是在G之中的, 且q有多大, 用来生成这个结果的power就有多难找
							  collapsed:: true
								-
					- 如何找到一个公开密钥 h
					  collapsed:: true
						- ((63fa8b1a-9dbe-4eba-81ee-64832d1fcc3e))
						- h除了不能等于1以外, 还不能等于g
						- h由于种种原因, 在这种情况下生成的话, 一定可以被g生成
						- 在承诺系统中, h作为公共数据, 是需要被随机生成的, 不能有人知道h到底是g的几次方
						- 并且h非常随机, 可以是p以内的任何数, 无法知道这个z到底是哪个
						- 参考下面的公式, 如果有人能从h得知g的几次方(z)等于h, 他就可以给c1, 也就是$g^a$进行一个to power of z, 求得这个值的乘法逆, 用这个乘法逆乘上c2, x就被还原了
					- ((63fa8b99-ac21-4e85-a3a8-2c47a0169155))
					  collapsed:: true
						- $C(x, a) = E_a(x) = (g^a, x · h^a)$
						- 承诺的值$(x, a)$
						- 基于ElGamal的承诺系统, 上面的式子中, x是message, 是被承诺的信息, a是一个random number, 用EG来加密的话就是会生成一组密钥对,
						- c1 ($g^x$) 是我作为发送者发送的一次性公共密钥, 除了我以外没人知道a是多少, 一次性的来保证每次加密的结果都不同
						- c2 ($x · h^a$) 是我作为发送者, 将信息embedding进去的部分, 用来将这部分加密混乱的密钥是接受者提供的公共密钥的a次方, 如果对方的私有密钥是b的话, h可以表示为$h = g^b$, 也就是说$h^a = g^{ab}$
						- 接受者只要把自己的私钥和c1结合取modulo 逆乘上上面的c2就能把迷惑项消除, 从而得到x
						- ![IMG_1158.jpeg](../assets/IMG_1158_1677366379302_0.jpeg)
						- 以上是ElGamal作为加密或者签名的时候的用法, 在承诺的时候, h, 连同g, p, q都会是公开的数字, 唯一大家不知道的是随机生成的a, 和承诺的消息x, 当它们两个放出来的时候, 就可以验证承诺值对不对了
						- ((63fa91bc-94b3-4bcf-b2c4-d2a95b8c0205))
						- 这里要注意的是x和a的取值, 他们的取值, 其实是根据实际情况定的. 在安全考虑下一定是会抛去一些不安全的小数字.x = 0是不允许的。这将产生g^x = g^0 = 1，其中1是组的identity元素，这完全破坏了安全性。请注意，x = 1也是不安全的，因为这将产生g^1 = g，这是公开的。a和q的取值都必须在q以内, 不然就重复了, 不再唯一了
					- ((63fa91ff-4755-4d11-840e-cec439c72d34))
					  collapsed:: true
						- Lemma 20.6: The commitment scheme Ea(x) is information-theoretically
						  binding and computationally concealing.
						- 使用ElGamal的承诺系统因为用的是加密操作, 能够保证数据一定能够还原, 因此是必定binding的, 必须是一对一的, 得到commitment是一定能够计算出a和x的(只要有无限的算力), 这也导致了这个系统只能是computationally concealing.
						- Proof: Ea(x) “is” ElGamal encryption with respect to public key h,
						  and where Ea(x) does not need associated private key (some z with gz = h
						  mod p).
						- Computational concealing follows from semantic security of ElGamal en-
						  cryption, which depends on G satisfying the decisional Diffie-Hellman as-
						  sumption (we won’t cover this topic of “semantic security” here).
						- Since decryption is correct for ElGamal, and so a ciphertext has a unique
						  matching plaintext, the commitment scheme Ea(x) is information-theoretically
						  binding.
					- ((63fa92f9-7887-4f94-b5c6-d5769b7c5225)) Pedersen Commitment Scheme
					  collapsed:: true
						- $C(x, a) = B_a(x) = h^x · g^a$
						- This is the Pedersen Commitment Scheme:
						  • again, a is random – a blinding number
						  • x is committed value
						  • all exponentiations are modulo p
						  Commitment is revealed as (a, x), the same as for scheme Ea(x).
						- 这个commitment系统中, a可以蒙蔽住x, 即便拥有无限算力也无法得到确切的x, 不同于ElGamal的保证binding的绝对安全, 这个系统保证的是concealing的绝对安全, 相对的, binding就是computationally secure的 (如果不知道g的几次方是h, 就是计算上binding的, 很难被否认, 因此h是要被随机生成的, 不能通过g的几次方人工选定)
						- Information-theoretically concealing:
						  collapsed:: true
							- ((63fa94ae-17b4-46e8-a186-abae677539f0))
							- 因为是$h^x · g^a$ 会有很多很多x和a的组合得到相同的结果, 完全随机的结果就是非常保密, concealing
						- Computationally binding:
						  collapsed:: true
							- ((63fa95cc-827f-42ca-a4c1-c6a3ac570d1b))
							- 根据推导, 问题可以被简化为找这个z, 只要找到z就可以找b和y, 来让算式的值还是z了, 但是z的找是很难的一件事情, 所以说是计算上苦难
						- ((63fa9643-cd52-4a2f-9f5f-e9ecd9984616))
						  collapsed:: true
							- ((63fa9661-3f50-4cba-b7eb-1a7500c2950d))
							- 这个只是利用了很难在大数的情况下找到log值而已, g会生成一遍所有的域中的值, 所以要一个个算过去, 但是是唯一的, 所以依然binding; 只不过保密性没那么强
							- 为什么同样都是求discrete logarithm, 就不选用这个方案呢? 因为ElGamal 引入了随机的变量a, 能让这个算法不那么deterministic
		- Notes
		- 全局理解
			- Commitment用来发送方承诺其发送的信息不会变更, 同时保证接受方无法提前知晓发送的信息, 直到明文发出才能够验证
			- 两个概念, 两种程度
			  collapsed:: true
				- concealing: 保护发送者信息不被提前获取, 由算法空间的随机性决定, 或是基于的problem的困难程度决定
				- binding: 保护接收者得到的信息与承诺的一致: 由算法空间的随机性决定
				- information-theoretic: 当x的可能性无穷, 或是无法区分时, 就concealing了; 当x与承诺完全只有一个对应, 就binding了
				- computational: 完全binding的时候穷尽可以找到那个x, 完全concealing的时候, 可以找到替代的x; 因此只基于一定的算力下.
				- 两者永远无法同时满足完美secure
			- Hash-based commitment schemes
				- $C(x, r) = H(r || x)$
				- 相对 binding, 绝对 concealing.
				- 无穷算力的情况下, hash特性决定可以找到另外一组r, x得到一样的commitment. 也因此接收者即便找到一组也无法确定是不是发送者提供的那个, concealing了
			- ElGamal based commitment schemes
				- $C(x, a) = E_a(x) = (g^a, x · h^a)$
				- 相对concealing, 绝对binding
				- 提一句, ElGamal作为公钥加密算法时, h=g^b, 这里的指数b是私钥, g^-ab可以用来恢复x
				- 承诺系统中, (a,x)作为信息, 因为h的那个b难以得知(h是自动生成的), 因此无法得到x, 相对conceiling
				- x和a的范围都锁定在了q以后, 因此只有一个x和a可以得到这个结果, 绝对binding
				- 背景知识, g是2q+1大小的群的一个子循环群的generator, g写作f^2, 从而g^q一定可以模p为1 (因为f^2q = f^p-1 一定等于1), 因此得到的这个g一定是order 为q的循环群, 因为order为prime, 其中所有元素都是g, 所以g和h的选择可以做到随机
			- Pedersen Commitment Scheme with dual properties 与上边这个相反的properties
				- $C(x, a) = B_a(x) = h^x · g^a$
				- 相对binding, 绝对concealing
				- concealing: $h^x · g^a = g^{zx+a} = c$, for any x', some a' can be found; $g^{a'} = c·(h^{x'})^{-1}$
				- binding: suppose a pair(b, y), $z=\frac{a-b}{y-x}$, solving z is assumed to be very hard, so computationally binding
				- 实用的属性: $B_{a1} (x_1) · B_{a2} (x_2) = B_{a_1+a_2} (x_1 + x_2)$
			- Deterministic commitment scheme
				- $C(x, r) = B(x) = g^x$
				- 相对concealing, 绝对binding
				- concealing 依赖于Hardness of discrete logarithm problem in G
				- binding 源于每个g^x都不同
				- 没有用r来blinding, 会导致发送过的信息重新发送就失去了concealing
			- Commitment based on quadratic residues
				- $f(b,x)=m^b·x^2(mod\ N)$
				- 相对binding, 绝对concealing
				- 可以把$m^b·x^2$看作$y^2$这个要点在于很难算出y是什么, 即便能算出来, 因为有b的存在, 无法确定这个y是单纯的x还是x乘了根号m(r)的结果 (可以构造出相同结构$x=x*r^{-1}$). 因此是绝对concealing的
				- 如果能计算QR的话, 就可以选择任意一个b来修改答案了, 用的解法就是上面这个
	- Week9 Zero Knowledge proof
	  collapsed:: true
		- 68-69, 43, 71-74
		- Slides
		  collapsed:: true
			-
		- Notes
			- ![Xnip2023-03-06_09-50-40.png](../assets/Xnip2023-03-06_09-50-40_1678664503328_0.png)
			  collapsed:: true
				- 1. hash的方式证明我知道密码
				  2. 通过多次验证保证概率乘积很小
				  3. NP问题就可以用ZKP prove
				  4. V的honest是一个前提
				  5. 正经的ZKP过程是不会让V知道P的信息的
		- 宏观理解
			- NP问题, 可以被轻松verify的才可以, Everything in NP has zero knowledge proofs
			- Commitment -> Challenge -> Response -> Verify
			- Completeness,  Soundness, Zero-knowledge
			- Completeness 只要推一遍如果知道knowledge, 就一定能够在两种不同情况下response到能够被verify的答案即可
			- Soundness 是P只有很小的概率可以cheat, 可以由 special soundness 来imply, 具体是两个相同的commitment with 两个不同的challenge, 在honest V的情况下可以解出知识x; 普通情况下就是根据cheat的概率来求得多次的joint prob
			- Zero-knowledge 需要构建simulation, 具体流程是先设定好response的值和challenge的值 (例如随机的b = 0 or 1), 再根据commitment的算式算出commitment. 由于r和c都是随机的, 看不出来是假的
			- 案例有:
			  collapsed:: true
				- 3-colourability
				  collapsed:: true
					- P每次都生成一个新的permutation, 对原本的colour方案进行一个打乱, 一次性commit所有的vertex v, 以保密的形式, V选择2个vertex让P揭露, 验证是不是两个不同的颜色. 多次重复. 因为commit了所有vertex, 所以没法逃避 (1-1/E)^k
					- ZK的模拟, 因为commit保密性, 所以可以事先把需要的两个v给设定为不一样, 别的不管是啥都没事
				- graph isomorphism,
				  collapsed:: true
					- G1->G2->H
					- P用自己知道的pi随机选择一个G1或者G2生成H, V也随机选择一个G1 or G2 让P给出到达H的方案. V可以轻松认证, P一轮只有1/2概率逃避
					- ZK 的模拟, 自己选哪个G生成, 一开始H就从哪里生成, 那么$\psi$就一定可以
				- Schorr’s identification protocol
				  collapsed:: true
					- $g\ and\ y=g^x$ 是公共信息, $x$只有P知道, P 随机选k commit 一个$r=g^k$, V 发送$e$, P计算$s=k+x·e$, V 计算是否$r = g^s·y^{-e}$
					- ZK的模拟, 随机生成一个s和e, 让$r = g^s·y^{-e}$ 即可
				- Chaum-Pedersen protocol
				  collapsed:: true
					- 上面的扩展, (r1, r2) = (g^k, h^k), 使用相同的e对他俩challenge
				- Quadratic residue
				  collapsed:: true
					- P要证明他知道u, 这个 这个$u^2 = x$; $x$ 是公开的; P随机选了$v$ commit $y = v^2$, V随机选了0或者1, 来决定$u$餐不参与, P计算发送$z=u^b·v$, V验证$z^2=x^b·y$
					- ZK 模拟: 现决定$b$和$z$, 再计算$y=z^2·(x^b)^{-1}$
			- 疑问:
			  collapsed:: true
				- 平方剩余里还有必要b?
	- Revision
	  collapsed:: true
		- 数论
			- [[群]]是一种代数结构 algebraic structure, 满足非空集合, 二元运算和封闭性
				- ![image.png](../assets/image_1678320973751_0.png)
				- ![image.png](../assets/image_1678320995153_0.png)
				- ![image.png](../assets/image_1678321013903_0.png)
				- ![image.png](../assets/image_1678321028505_0.png)
				- ![image.png](../assets/image_1678320942145_0.png)
				- 最后的循环群是有生成元的阿贝尔群
			- 例如(G, *), G是一个集合, 但是为了方便起见, 之后也会把他称之为群
			- 群的大小叫做群的阶, 即order
			- 群要满足四个特性
				- 封闭性: 群内元素做任意操作还在群内
				- 结合性: 能结合率
				- 单位元: 有个e,
				- 逆元: 每个元素都有
			- [[乘法群]]: 由于每个元素都要有和它乘起来是1的东西, 因此需要1~ n-1中的所有元素都与n互素
			- [[子群]]: 子群也是群, 其集合是某个群的非空子集, 如果是有限集, 满足封闭性, 那就一定是子群 (子群不一定是有限集, 比如有理数加群就是实数加群的子群)
				- $G^m := \{a^m|a \in G\}, m \in Z$
					- $(Z_n^*)^m$是$Z_n^*$的子群 (m次剩余子群)
				- $mZ_n := \{mz\ mod\ n\ |\ z \in Z_n \}$
			- [[阿贝尔群]]是满足交换律的群, 因为并不是所有群都满足交换律的
			- [[陪集]](coset) 是G中的某个元素a, 乘上某个子群H得到的集合, 子集是划分的标准, 陪集写作$[a]_H$. 是以子群H为基准, 去观察大群中每一元素与子群的关系
				- 例如说3Z, 就是划分了Z中所有能被3整除的元素. 注意这是个加法群
					- $[2]_{3Z} = \{2, 2+-3, 2+-6, ...\}$
				- 在陪集的概念中, 子群就像是一个分类器, 把不同的主群G元素分类到一个个陪集中, 每个元素只存在于唯一的一个陪集中. 有一个特殊的陪集是子群H本身, 只有这个陪集拥有单位元, 因此才是一个群, 其他的都只是集合
					- ![image.png](../assets/image_1678315164396_0.png)
			- [[拉格朗日定理]]:
				- |H| | |G| . 如果G是有限群, H的阶一定是G的阶的因子. 比如G内有15个元素, 子群的阶只可能是1,3,5,15. 而素数阶的群, 就只能有e和它本身了
				- 在陪集概念中, G中的元素会根据与子群H的关系被分类到不同的陪集中, 这些陪集的并集就是整个群G, G的大小等于这些陪集的大小之和, 而H构造的陪集的大小都和H相等, 所以H的大小必定要能整除G的大小
					- ![image.png](../assets/image_1678315282969_0.png)
					- ![image.png](../assets/image_1678315309489_0.png)
			- 商群: G/N := $\{[a]_N|a\in G\}$ (G/N,*) 二元运算定义为$[a*b]_N$
				- G的子群N形成的陪集们的集合, 重点关注的是, G内除了N这个子群外的元素
					- ![image.png](../assets/image_1678315117705_0.png)
					- ![image.png](../assets/image_1678315064107_0.png)
			- [[循环群]]
				- 循环群G满足群G内的一个元素g, 可以与自身操作, 得到群内所有的元素.
				- 例如Z这个加法群中的+-1就是
				- Z5* 这个乘法群中的2和3都可以通过几次方得到1-4这四个数字
					- ![image.png](../assets/image_1678314953022_0.png)
				- 定理: 任意循环群都是阿贝尔群
				- 定理: 循环群的子群也是循环群
					- 例如我们有个循环群Z5*, 生成元是2, 他有个模5的子群{1,4}, 生成元是4, 也是循环群. 可以看到原生成元的幂也可以作为生成元生成子群循环群. g的幂生成的有可能是<g>G自己, 也可能是子群
			- 元素的阶 order:
				- 群有order, 群内的元素也有order
				- $a^n = e$, 最小的那个n就是元素a的阶
				- 元素的order可以理解为其离单位元的距离, 即经过几次与自己的运算可以到达e
				- 并不是所有元素都有order, 永远到不了的, 那就是无限阶的, 这种元素也成为不了gen
				- 但是对于有限循环群而言, G的阶就得等于其生成元的阶, 因为G就是g生成的
				- 性质2：有限循环群的阶是n，则生成元g的阶也是n， 且群里元素 g,g2,g3,…,gn (=g0 =e)各不相同。
					-
				- 性质 3: d是n的一个因子, n阶的有限循环群, 只有唯一一个子群的阶为d
					- 例如说, Z5* 这个乘法循环群的生成元是2或者3, 阶是4, 1-4这四个数字, 因子有1, 2, 4 所以刚好就是有三个循环子群, 生成元是G的生成元的幂
					- 再举个例子, commitment中我们取了个safe prime = 2q+1, 其乘法群的order是2q, 那么很显然其子群的阶必须为1, 2, q, 2q. 即只有四个子群
					-
				- 性质 4: n阶有限循环群<g>(指g生成的循环群), 对于所有整数k, $g^k$的阶(即$g^k$作为生成元生成的子群大小)为$n/gcd(n,k)$. 理解一下就是, 当k和n互素的时候, $g^k$的阶就和n一样大, 因为最小公因数是1, 其他情况下生成的子群都会比这个要小, 且子群的大小为n的因子
					- ![image.png](../assets/image_1678314676438_0.png)
				- 性质5: 生成元的个数与有限循环群的阶有关. n阶的有限循环群有$\phi (n)$个生成元. 这个就是上一个性质的理解推导结论, 公因数为1的情况生成的就是G本身, 即为n阶有限循环群的生成元. $\phi (n)$是欧拉函数, 计算的是1-n-1中有多少与n互素的数 (质数的欧拉函数就是质数减一)
					- ![image.png](../assets/image_1678306607952_0.png){:height 198, :width 623}
					- 还是以Z5* 这个乘法循环群为例, 其阶为4, $\phi(4) = 2$ (有两个与order 4 互素的数), 因此这个群有2个生成元. g的power k可以取1和3两个数字, 因为只有1, 3与4互素. 对于任意g, 例如2的1次和3次就是2和3; 而3的1次和3次也是3和2. 因此我们只要有一个g, 就可以得到这个循环群的所有生成元.
					- 非互素的power得到的g^k则是循环子群的生成元(根据上一条性质, 阶为4的循环群, 有1, 2, 4三个子群大小, 因此还有一个循环子群的大小为2, 可以由2^2=4生成
				- 性质 6: G是有限群, 元素a的阶是|G|的因子. 注意这里的群只需要是个有限群即可. 如果某个元素a有order, 则这个order是是|G|的因子. 这个也符合拉格朗日定理的直觉, 因为如果a能够生成子群的话, 子群的大小也一定是G的大小的因子.
					- 这里也推出了一个结论: 假设a的阶是k, 即a的k次会得到单位元, 因为k | |G|, 所以作为k的倍数的G次的a也能得到单位元, $a^{|G|}=e$, [[欧拉定理]]
					- ![image.png](../assets/image_1678314645055_0.png)
				- 性质7: 素数阶的群必然是有限循环群
					- 群的大小为素数的话, 那这个群一定是有限循环群. 平时我们遇到的单纯的模p的乘法群的阶是p-1, 并不一定是循环群. 但是如果群的阶为素数的话, 那就一定是了.
					- 原因在于假设G中元素a是有阶的, 那他的阶就只能是p的因子, 即1 或者 p, 如果a不是e的话, 那k就只能是p, 那就意味着, a一定是生成元了, 因为可以生成p个元素
						- ![image.png](../assets/image_1678314617363_0.png)
					- 拿commitment中的例子, safe prime (2q+1)的乘法群不一定是循环群, 因为其阶为2q, 但是根据拉格朗日定理, 它一定会有阶为1, 2, q, 2q的四个子群, 其中只有q是有用的. 而且这个子群的阶为一个prime q, 那么这个群就一定是个有限循环群, 不仅如此, 其Euler Totient为(q-1), 因为比质数小的数全都和他互素, 那么其生成元的个数也将会有q-1个, 也就是说除了单位元1以外其他的群内元素都可以作为生成元. 根据性质5, g的k=1~q-1都是生成元, 也就是所有非1都是生成元
						- 寻找能够生成q大小的子群的g的过程中, 目标是找到g^q = 1(意味着其能生成一个q大小的循环群$a^{|G|}=e$),  generator g的寻找, 也是充分利用了safe prime的特性: p = 2q+1, 2q = p-1, 而p-1恰好是p这个有限群的order, 任何群内元素(1~p-1)的(p-1)次方都等于单位元1(因为性质6及其推论), (p-1)次方又可以写作(2q)次方. 我们就想着去构造一个2q, 因此就把g写做了f^2, 那个g的q次方就可以表示为 $g^q = f^{2q} = f^{p-1}\ mod\ p = 1$ 随机数f的群order次方一定是1, 也就找到了那个g
						- 而h作为这个素阶群的一个元素, 肯定也是一个生成元, 也意味着它肯定是上面找到的g的某次方
				- [[原根]]
					- 如果g是模n下的原根, 则g这个元素的阶(即几次g操作可以得到e) 等于n的欧拉函数(即Zn\*的order, size, 因为是乘法群, 所以元素与n互素), 此时g是模n下的原根, 也是Zn\*的生成元
					- 即g能够在模n时生成的不同数的个数, 等于n以内与n互素的个数, 即g为原根
					- 但是原根并不一定存在, 也就是说$\phi(n)$阶的元素并不一定存在, 也就是说乘法群不一定会有生成元, 并不一定是循环群. 存在的条件是 #card
						- 设p是奇素数，e是正整数
						- $n=1,2,4,p^e,2p^{e}$时，模n下存在原根. 下表的周期, 即为Zn*和原根的order, 也是欧拉函数(n)
							- | n | 模n的原根(有*号的数没有原根，此时是有最大模n周期的数) | 周期  |
							  | 1 | 0 | 1 |
							  | 2 | 1 | 1 |
							  | 3 | 2 | 2 |
							  | 4 | 3 | 2 |
							  | 5 | 2, 3 | 4 |
							  | 6 | 5 | 2 |
							  | 7 | 3, 5 | 6 |
							  | 8* | 3, 5, 7 | 2 |
							  | 9 | 2, 5 | 6 |
							  | 10 | 3, 7 | 4 |
							  | 11 | 2, 6, 7, 8 | 10 |
							  | 12* | 5, 7, 11 | 2 |
							  | 13 | 2, 6, 7, 11 | 12 |
							  | 14 | 3, 5 | 6 |
							  | 15* | 2, 7, 8, 13 | 4 |
							  | 16* | 3, 5, 11, 13 | 4 |
							  | 17 | 3, 5, 6, 7, 10, 11, 12, 14 | 16 |
							  | 18 | 5, 11 | 6 |
							  | 19 | 2, 3, 10, 13, 14, 15 | 18 |
							  | 20* | 3, 7, 13, 17 | 4 |
							  | 21* | 2, 5, 10, 11, 17, 19 | 6 |
							  | 22 | 7, 13, 17, 19 | 10 |
							  | 23 | 5, 7, 10, 11, 14, 15, 17, 19, 20, 21 | 22 |
							  | 24* | 5, 7, 11, 13, 17, 19, 23 | 2 |
							  | 25 | 2, 3, 8, 12, 13, 17, 22, 23 | 20 |
							  | 26 | 7, 11, 15, 19 | 12 |
							  | 27 | 2, 5, 11, 14, 20, 23 | 18 |
							  | 28* | 3, 5, 11, 17, 19, 23 | 6 |
							  | 29 | 2, 3, 8, 10, 11, 14, 15, 18, 19, 21, 26, 27 | 28 |
							  | 30* | 7, 13, 17, 23 | 4 |
							  | 31 | 3, 11, 12, 13, 17, 21, 22, 24 | 30 |
							  | 32* | 3, 5, 11, 13, 19, 21, 27, 29 | 8 |
							  | 33* | 2, 5, 7, 8, 13, 14, 17, 19, 20, 26, 28, 29 | 10 |
							  | 34 | 3, 5, 7, 11, 23, 27, 29, 31 | 16 |
							  | 35* | 2, 3, 12, 17, 18, 23, 32, 33 | 12 |
							  | 36* | 5, 7, 11, 23, 29, 31 | 6 |
						- ![image.png](../assets/image_1678317151596_0.png)
					- 题外话, RSA中用到的n=p*q, 就不是一个循环群, 只是一个普通的阿贝尔群, 并没有生成元
					- 模n下构成的乘法群的阶是phi(n), 因为其元素必须要有逆元. 因为n阶有限循环群的生成元有phi(n)个, 所以模n的生成元, 即原根一共有$\phi(\phi(n))$个 (模n存在原根的情况下)
					- 性质都和上面元素的阶的性质一致, 只需要找到模n的阶即可
					- 有哪些循环群? #card
						- $Z^*_p$, where p is a prime
						- Order 为prime的都是循环群
				- [[离散对数]]
					- ![image.png](../assets/image_1678318731129_0.png)
					- 几个要点:
						- g必须是原根, 才叫离散对数
						- x范围为$[0,\phi(n))$
						- 由于x有固定范围, 在幂处可以进行mod phi n的操作, 但是最后的值还是mod n的
				- [[环]] (ring)
					- (R, +, ·) 是一个环 (+和乘只是抽象概念, 意思是需要有两个二元运算), 且满足
						- 加法阿贝尔群: (R, +)是阿贝尔群
						- (R,·) 满足封闭性、结合率 (乘法半群)
						- 分配律
					- 因为多项式这个就需要加法和乘法同时存在, 所以有了环
					- 例子
						- ![image.png](../assets/image_1678320827222_0.png)
						- ![image.png](../assets/image_1678320841297_0.png)
						- ![image.png](../assets/image_1678321181527_0.png)
					- [[零元]] theta (zero element)
						- 定义：*是定义在非空集合A (至少有两个元素)上的二元运算，theta E A, 如果对于A中所有元素，都有$\theta * a = a * \theta = \theta$ 也就是和零元操作会变成零元自己
						- 群中不可能有零元, 因为零元没有逆元, 群内唯一的元素可以是单位元
						- 加法单位元就是所谓的零元
						- 环里除了加法单位元以外的所有元素, 都统称为非零元素
						- 因为环中的加法必须是阿贝尔群, 所以零元是必不可少的
						- 平凡环(trivial ring) R只含有零元, $e = \theta$
						- 非平凡环(non-trivial ring): R 中不止有零元 $e \neq \theta$
					- [[零因子]] zero devisor
						- ![image.png](../assets/image_1678325112830_0.png)
						- 环里n个e可以等于theta, 如果这个n有质因数分解, 那就有零因子了
						- ![image.png](../assets/image_1678325183424_0.png)
						- ![image.png](../assets/image_1678325223976_0.png)
					- [[整环]]: Z, Q, R是个整环, 因为Z是个含幺交换环, 且没有零因子, 因为他是无限的, 没办法做到非零两束想乘等于0.
						- n是素数时, Zn是整环, 同时也是域, 因为n以下的数都与n互素, 任何两个数相乘模n都不会等于0
				- [[域]] (field)
					- 域是加法和乘法都能构成阿贝尔群的环
						- ![image.png](../assets/image_1678321998058_0.png)
					- ![image.png](../assets/image_1678321665629_0.png)
					- 本质上域也是一种整环, 域里也没有零因子, 即零元不能分解成两个非零元素的乘积
					- Q, R有理数和实数, 都是域, 是无限域; 但是无限整环不一定都是域, 比如说Z不是域, 因为尽管去掉了0, 但是Z中除了+-1以外的元素都没有乘法逆元, 所以不是阿贝尔群, 所以不是域
					- 但是有限整环都是有限域
					- 性质
						- 非平凡: 包涵至少两个元素, theta和e, 两个不相等
						  id:: 6409295e-d77e-485e-8525-27dcafcc0c4c
						- 没有零因子
				- [[特征]] Characteristic
					- 定义：R是环，如果存在最小正整数m，对于$\forall a \in R$, 使得$m a=\theta$，则称m 是环R的特征。如果这样的 m不存在，则称R的特征是0。记为 Char R.
						- 这个有点类似于乘法群里面元素的阶, 就是多少个a与自己操作能到e, 这里就变成了加法阶, 即多少个a与自己相加能到theta. 但是特征是对所有a而言, 使得环中的每一个元素都可以到达theta, 才叫做特征
						- 零元是0, 单位元是1
					- 定理1：如果环的特征不等于0，环里元素的加法阶就都是有限的，而且都是特征的因子。
					- 例子:
						- Z的char是0, 因为其无限
						- ![image.png](../assets/image_1678322671617_0.png)
						- 因为模n下Z_n里的所有元素自己相加n遍都能到达0, 例如Z6环的特征是6, 里面的元素的阶都是6的因子, 只能是1,2,3,6
					- 定理2: 含幺交换环的特征是0 或者其单位元的加法阶
						- ![image.png](../assets/image_1678323018873_0.png)
					- 定理3: 整环的特征等于0或是素数
						- 整环是非平凡环, 意味着整环里面至少包涵两个元素, theta 和e, 且两个不相等
						- ![image.png](../assets/image_1678323045361_0.png)
						- 设m为合数, 可以写作s和t的乘积, st都在(1,m)之间, 因为合数且乘积才能让e=theta
						- 因为me=theta, 根据结合率, me=ste=sete (e是单位元可以这么写), se和te因为s和t都在m以内, 是没办法得到theta的, 也就是说两者都不等于theta, 又因为整环没有零因子(即没有两个非零元的数的乘积等于theta), 所以上面这个式子不等于theta, 也就是me不等于theta. 这证明了m不能够写成两个非零数的乘积, 也就是说m必须是一个e乘上一个素数, 也就意味着m本身必须是素数. 这样才能实现至少包含两个元素.
						- 这也印证了为什么域的特征必须要是素数或者0
					- ![image.png](../assets/image_1678323682137_0.png)
		- 2122
		  collapsed:: true
			- Perfect security of crypto systems
				- simple cipher system and computations
				- perfectly secure
				- encryption modes
				- chosen-plaintext attacks
				- quantum random generator
				- XOR
			- Shamir secret sharing
			- Commitment schemes
				- design
				- Security analysis
		- 2021
		  collapsed:: true
			- Perfect security of crypto systems
				- one-time pad
				- pseudo-random number
			- Shamir Secret Sharing
			- commitment scheme
				- Security analysis
				- Elliptic curves
		- 1829
		  collapsed:: true
			- Symmetric Encryption
				- perfectly secure
			- Key Management
			- Public Key Cryptography
				- RSA
				- Elliptic Curve Public Key
			- Shamir Secret Sharing
			- Commitment Schemes
			- Zero-Knowledge Proofs
		- 总结一下
		  collapsed:: true
			- 固定的基础加密算法概率计算题, 分析简单加密系统的不安全性,
			- 对错题会cover很多内容
			- Shamir secret sharing 固定要考, 会计算, 证明其完备性
			- Commitment schemes 固定要考
			- 要会设计commitment protocol
			- 要会分析算法的security, 例如commitment的binding 和concealing
			- 公钥密码系统也要考, 要熟悉RSA和EC
			- ZKP要考的
		- 问答题模版
			- 并非所有commitment schemes 需要random input, deterministic的g^x就不需要
			- MAC可以accountability, commitment不可以
			-
			- w2
				- A crypto sys with |K|=|C|=|P|is perfectly secure, iff for all k in K, p(K=k)=1/|k| and for all m and c there is a unique k with e_k(m)=c. 因为p(C=c)>0就一定有k从m映射到c; m加密得到的一定在C中, c一定存在意味着一定有m到c, 所以c的解密也一定在P中, 所以是双向包含的injective关系. 所以一定是唯一的key在map. 1/K的概率则由贝叶斯分解p(P=m|C=c)和key的唯一性可以推出p(K=ki)=p(C=c), 各个key概率相同
			- w4
			- H=f(H||m_i) i=1...n的方式形成MAC的缺陷, 使用k作为初始H
				- h(k||m1) = c1 -> h(k||m1||m2) = f(c1||m2)
				- h(k||m1) = c1 -> h(k||m1||b||m2) = f(f(c1||b)||m2)
				- MAC的要求是 签名和验证只能通过MAC_k(m)进行! k后置可以用collision来找到两个相同m得到一样的MAC
			- w6
			- Why <t parties cannot recover secret
				- Parties 1 and 3 can determine a degree-1 polynomial g(x) by pooling their shares, but f(x) is degree-2. For any value z in Z31, there exists a degree-2 polynomial h(x) with h(0)=z, h(1)=g(1), and h(3)=g(3). Thus, parties 1 and 3 cannot learn anything about the secret s by combining their shares.
- ## Info
	- 8:2
	- 星期一 9:00 - 11:00
	- 这门课是关于开发保证integrity 和 confidentiality的密码工具, availability 不是重点
- ## Syllabus
	- Cryptographic primitives: pseudo-random number generators, block ciphers, pseudo-random functions, hash functions, message authentication codes, key derivation functions, public-key cryptography
	- Symmetric key cryptography: perfect secrecy and the one-time pad, modes of operation for semantic security and authenticated encryption (e.g. encrypt-then-MAC, OCB, GCM), message integrity (e.g. CMAC, HMAC)
	- Public key cryptography: trapdoor permutations (e.g. RSA), public key encryption (e.g. RSA, El Gamal), digital signatures, public-key infrastructures and certificates, hardness assumptions (e.g. integer factoring and Diffie-Hellmann), Elliptic Curve Cryptography
	- Authenticated key exchange protocols (e.g. TLS)
	- Quantum key exchange protocols
	- Cryptographic protocols: challenge-response authentication, zero-knowledge protocols, commitment schemes, oblivious transfer, secret sharing and applications, anonymity (may pick different protocols from that list in different instances of that module)
	- Security definitions and attacks on cryptographic primitives: goals (e.g. indistinguishability, unforgability, collison-resistance, cryptographic games, etc.) and attacker capabilities (e.g. chosen message attacks for signatures, birthday attacks, side channel attacks, fault injection attacks.
	- Advanced topics such as Secure Multi-Party Computations: secret sharing schemes and other techniques needed for defining such computations; presentation of one full scheme for secure two-party computations.
	- Cryptographic standards and references implementations
- ## Links
	- [Scientia](https://scientia.doc.ic.ac.uk/2223/modules/70009/materials)
	- [2021-2022](https://imperiallondon-my.sharepoint.com/personal/mrh_ic_ac_uk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fmrh%5Fic%5Fac%5Fuk%2FDocuments%2FC70009%2D2022)
	- [2022-2023](https://imperiallondon-my.sharepoint.com/personal/mrh_ic_ac_uk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fmrh%5Fic%5Fac%5Fuk%2FDocuments%2FC70009%2D2023)
		-
- ## PDF #PDF
	- ![0_CE_Slides.pdf](../assets/0_CE_Slides_1673716643789_0.pdf)
	- ![0_CE_Lecture_Notes.pdf](../assets/0_CE_Lecture_Notes_1673734798684_0.pdf)
		-
