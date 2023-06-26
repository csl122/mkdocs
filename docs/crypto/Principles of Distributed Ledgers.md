# Principles of Distributed Ledgers

tags:: IC, Security, Course, Block Chain, Uni-S10
alias:: PDL
Ratio: 7:3
Time: 星期五 11:00 - 13:00

- ## Notes
	- 术语表
		- ERC: Ethereum Request for Comment, 用来增加或修改以太坊的功能, 或是标准化以太坊features
		- EIP: Ethereal Improvement Proposals
		- Application Binary Interface (ABI)
		- Decentralised Finance ([[DeFi]])
		- Oracle
	- Week2 Intro to blockchain
	  collapsed:: true
		- novelty of bitcoin
		- smart contracts
		  collapsed:: true
			- a computerised transaction protocol that executes the terms of a contract.”
			- 智能合约等价于一段事先就被规定好逻辑和条款的计算机代码被激活运行的状态，同时，智能合约也提供了通用的用户接口，用户可以通过接口与用户交互。
			- 是一个由数字表单指定的承诺，这个承诺包含关系到多方执行的一组协议。
			- 智能合约是一个由计算机处理、可执行合约条款的交易协议，其总体目标是满足协议既定的条件，例如支付、抵押、保密协议。这可以降低合约欺诈造成的损失，降低仲裁和强制执行所产生的成本以及其他的交易成本。
			- 这里“既定的业务流程、机器人模样的人机交互界面、双方同意承诺”组成了智能合约的概念，它甚至具有一定的法律效力。即交易脚本, 定义了交易的过程
		- “Tokens” are smart contracts that contain the logic for storing and updating
		  balances of token holders
		- Decentralized Finance (DeFi) is a peer-to-peer powered financial system.
		  collapsed:: true
			- • Non-custodial
			  • Permissionless
			  • Openly auditable • Composable
		- Blockchain
		  collapsed:: true
			- • A data structure that stores information, such as transaction data
			  • Peer-to-peer network
			  • Data is recorded in multiple identical data stores (ledgers) that are collectively maintained by a distributed network of computers (nodes)
			  • Consensus algorithm (all nodes see the same data)
		- Block
		  collapsed:: true
			- A data structure that stores information, such as transaction data.
				- block header: Identifies a particular block of transaction
				- txn count: total number of transactions
				- txns: every transaction in the block
			- Proof of Work
				- 大家用算力solve puzzles, 类似于暴力搜索, 搜出来了就可以决定新的valid的transaction可以被加入到next block
					- 比如说通过解决一个puzzle, ‘sdhagisa’后面加什么数字可以让hash的前面几位
				- 所有人都可以取验证
				- target的可能性是2的256次分之一, 对于SHA-256
				- bitcoin用的是merckle tree 用来记录transactions 每一个区块还有一个Merkle哈希用来确保该区块的所有交易记录无法被篡改。
			- longest chain rule
				- 最长的会被认为是最靠谱的, 才会被include进去, 如果这次不够长, 就不会被认为valid, 会在池中等待下次, 如果mine了一个没用的, miner没办法得到reward, 大多数都没有
		-
		- ![image.png](../assets/image_1674217013609_0.png)
		- Forks:
		  collapsed:: true
			- soft fork: backward compatible protocol changes
			- hard fork: incur permanent  split of the blockchain; 例如新的currency出现的时候, 例如Bitcoin和Bitcoin cash, 这俩没有本质区别, 是block大小区别
		- ETH以太坊:
		  collapsed:: true
			- 通过 proof of stake 来决定谁有记账权, 再也没有mining, 只用简单计算hash
			- 拥有的以太坊越多, 拥有的时间越长, 越有可能有这个记账权, 获得奖励
			- 但有机制防止过分, 但是还是需要一点计算的
			- reward机制
				- • source vote: voted for the correct source checkpoint
				  • target vote: voted for the correct target checkpoint
				  • head vote: voted for the correct head block
				  • sync committee reward: participated in a sync committee • proposer reward: proposed a block in the correct slot
				  • block fees: fees associated with a particular block
			- Account
				- Externally owned accounts
					- 公钥和私钥,
					- 公钥是地址
					- 私钥用来签名
				- contract accounts
					- 部署为smart contracts
					- 没有相关联的私钥
			- 与比特币相比，以太坊首先不是一个单纯的数字货币项目，它可以提供全世界无差别的区块链智能合约应用平台，这个平台基于我们前面文章所介绍的区块链四大核心技术要素，即P2P网络、共识机制、账户模型、加密模块。
			- 除了以上的四个技术要素，以太坊还推出了EVM——以太坊智能合约虚拟机，并且，它还推出了自己的智能合约语言Solidity。
			- 于是，区块链的开发者因为智能合约的出现开始分为两类。第一类是公链底层开发者，主要是以C++和Go语言为主的全节点开发者，他们需要对区块链各个技术模块有很深的理解。
			- 第二类是智能合约开发者，也就是应用开发者，这类开发者对区块链的运行原理不需要理解很深，只需要会编写Solidity，了解规范即可。
		- Tutorial 模拟一下比特币的建立
			- LATER 记录一下tut学到的比特币知识
	- Week3 Ethereum and Smart Contracts
	  collapsed:: true
		- Lecture Ethereum and Smart Contracts
			- Ethereum: a transaction-based state machine.
				- 以太坊更像是一台世界计算机, world computer, truly global singleton, 因为智能合约的部署会让所有人都能接触到这台世界计算机的算力, 并且保证所有人接触到的统一
				- 以太坊的特点
					- • Allows you to run decentralized programs using the Ethereum Virtual Machine (EVM) 运行去中心化的程序
					  • Uses a blockchain as a means of storing state, but is much more than just a blockchain 区块链用于存储state
					  • Is maintained by a network of nodes, which store exact copies of this state 网络节点来存储copies
					  • Is non-custodial and permissionless and completely transparent (like other
					  blockchains) 完全透明, 啥都可以被所有人看到
				- ![image.png](../assets/image_1674829703037_0.png)
				- Blockchain
					- Stores a series of state transitions starting from a genesis state
					- The transition to the next state is determined by all transactions included in a block. 到下一个state的transition 是由block中的所有transaction决定的
					- transition is agreed upon by nodes
					- Blocks and states
					  collapsed:: true
						- ![image.png](../assets/image_1674830170394_0.png)
						- ![image.png](../assets/image_1674830201532_0.png)
						- 因此就没有比特币里面的UTXO, 不需要计算未花费余额了
					- Accounts as world state
						- Accounts are a mapping between addresses
						  and account state with the following information: accounts 是地址到地址state的mapping
							- ![image.png](../assets/image_1674830940676_0.png)
						- Externally owned accounts (EOA) 外部账户
						  • Account is created by generating private/public key pair
						  • Address is derived from the public key
						  • Can initiate transactions
						- Contract accounts (CA) 合约账户
							- • Deployed as smart contracts and controlled by their code • Do not have an associated private key
							  • Cannot initiate transactions
					- Transactions
						- A transaction is a cryptographically signed instruction (message call) sent by an actor external to Ethereum, i.e., an EOA 外部账户签名的指令
						- Basic components
							- • to: Address to receive the message call
							  • value: Amount of ETH to be transferred
							  • gasPrice and gasLimit: Maximum amount of ETH the sender is prepared to pay for the execution
							  • data: Byte array with the input data for the transaction
							  • signature: Signature of the sender to proof ownership
						- 以太坊中只有transactions 可以产生state change, transactions 可以做到的有
							- Transferring a balance to another account, e.g., a transfer of Ether 余额交易
							- Triggering the execution of smart contract code, which can cause more complex state transitions 智能合约的执行
						- Consensus + Chain Selection 如何达成共识
						  collapsed:: true
							- 目标 objective
								- • Agree on the current state of the system
								  • Find a way to transition to the next state
								  • Ensure that the state is valid (does not violate the rules of the system)
							- 向下一个state的转移 transitioning to the next state
								- Select a proposer for the next state, 并且确保没有太多proposers
								- agree that the proposed state should indeed be the next state
							- Selection and validation of the next block: proof-of-stake
								- Validators “stake” some ETH to participate in the network 根据持有以太币的量和时间来作为股权
								- Each “epoch”, a validator is randomly selected to create a new block 根据股权随机选择
								- 是否valid会被其他validators check
							- Chain selection rule
								- Different selection rules are possible
								  • Bitcoin uses the ”longest chain rule”
								  • Ethereum uses the ”heaviest chain rule”
								  • Allows nodes to reach agreement on which history is the right one
								  Heaviest Chain Rule: The version of the chain with the highest number of accumulated validator votes weighted by their staked balances
								- 以太坊用的是最肥链条法则
									- 拥有最多validator的链条会被选择
						- Ether
							- Ether: 原始基础货币
							  • Ether is the native cryptocurrency of Ethereum
							  • Balances for each account are stored directly as part of the world state 
							  • Ether is used to pay transaction fees (gas)
							- Tokens: 由智能合约生成的代币
							  • Tokens are implemented in smart contracts
							  • Balances are therefore stored in smart contracts rather than directly
							  • You will come across several standard interfaces for such tokens (ERC20 etc.)
							  • Anyone can create tokens on Ethereum
						- Ethereum Virtual Machine (EVM)
							- 如何在p2p网络中确定地跑代码
							- Objectives of the EVM
							  • Deterministic execution
							  • Verifiability of the execution and its outcome
							  • Atomicity of the execution (transactions are atomic)
							- ![image.png](../assets/image_1674835044644_0.png)
							- EVM 简介
							  collapsed:: true
								- Stack-based virtual machine
								- low-level, only jumps, no types
								- ADD, SUB, PUSH, POP
								- Has ephemeral and
								  permanent storage
								- 32bytes
								- nodes 会 reren the code executed by the proposer of the next block and verify that the outcome is correct
								  The code for execution is found in storage and its validity can be verified using the code hash of the contract account that is being called; code 用hash来验证对不对
								- 通常用Solidity编写, 被编译成字节码, 然后用transaction deploy, 并且保存在各个nodes中 (An EOA can issue a special transaction that contains the contract code to create a new contract address). CA会存储起来字节码和hash, 一旦这个transaction processed, contract is deployed
								- Executing code
									- An Ethereum client (node) chooses to include a transaction that makes a call to a contract account 客户端include一个transaction
									- After verifying the validity of the transaction, it executes the code locally using its EVM implementation 客户端验证, 然后本地执行code
									- This results in a state change that is broadcast to the network in the next block 状态转移后广播到网络里
								- Charge for compute resource (Gas)
									- why need gas?
										- • Prevent attacks on Ethereum (DoS)
										  • Prevent execution of infinite loops in smart contract code 
										  • Reward validators/miners
									- Gas Mechanism and Block Reward
										- 需要奖励计算, 提交transaction base fee 会被花掉, 还需要给别人priority fee, 当然也可以不给, 但是可能别人就很难从pool里把你挖出来了, 因为你不给人家奖励
										- ![image.png](../assets/image_1674835620328_0.png)
								- Submitting Transactions - Mempool
									- Node收到transaction后pending在mem pool中, 等待处理
									- ![image.png](../assets/image_1674836243186_0.png)
							- Smart Contracts
								- 不smart, 也不是contracts, 知识一堆部署在区块链的程序而已
								- Functions in smart contracts are called via transactions
								- ![image.png](../assets/image_1674836345346_0.png)
								- They can:
								  • Perform almost any computation (Turing complete) 
								  • Persist data
								  • Transfer money to other addresses or contracts
								  They cannot:
								  • Interact with anything outside of the blockchain: they are isolated (otherwise wouldn’t be deterministic)
								  • Be scheduled to do something periodically
							- Solidity
								- Strongly typed, fairly simple type system
								  • Looks vaguely like Javascript except it is statically typed like Java, C, Rust
								  • Supports multiple inheritance
								  • Solidity: just one example of a high-level programming language, compiles down into EVM bytecode. Any high level language that can compile down into EVM bytecode would work
								- LATER 补充Solidity的东西
							- Development flow for smart contracts
								- 1. Write high-level code
								  2. Test the code (using testing suite of choice, e.g. Hardhat, Brownie, Foundry) 
								  3. Optimise the code for gas e ciency
								  4. Compile the contract into Bytecode
								  5. Send a transaction to deploy the contract
								  6. Interact with the contract by sending transactions to the generated address
							- Inter-contract Communication
								- contract 间的通信用的是message calls
								- Message calls are the mechanism for inter contract communication
								  • Smart contracts are a bit like classes. Deploying a smart contract is a bit like creating an instance of that class
								  • One contract can call another via a message call • Every transaction wrapped in a message call
							- 危险
								- code is law - if the code allows it, it can be done
								- Therefore, if there is a bug and the code allows unintended behaviour,
								  programming intent can separate from reality
	- Week4 Smart Contract development
	  collapsed:: true
		- Development tools
			- we use CLI commands with Foundry
		- ERC-20 tokens: standard for fungible tokens, 比如ether, 两个可以是interchangeable, 联通的, 可以是表示一样的
		- ERC-721: standard for non-fungible tokens, 表示unique的things, not interchangeable, 比如NFT, 就是一种表示数字资产的唯一证明
	- Week5 Oracle, DeFi
		- centralised oracles are points of failure
		- 从很多exchange 源中获得并aggregate, 保证不受单一exchange dominant
		- 预言机, 就是用来给自成体系的区块链提供外部信息的, 例如外部的资金信息等等
		- 在ETH测试网络上面实验了从Chainlink VRF 获取可验证的随机数
		- Decentralised Finance ([[DeFi]])
		  collapsed:: true
			- Decentralized Finance (DeFi): a peer-to-peer powered financial system.
				- ![image.png](../assets/image_1676237354507_0.png)
			- Inter-Contract Communication
				- smart contract **support DeFi** needs support inter-contract communication
					- • be expressive enough to encode financial protocol rules
					  • allow conditional execution and bounded iteration
					  • feature atomic transactions so that no execution can result in an invalid state
			- Application Binary Interface (ABI)
			- Composability: Snapping Protocols Together Like Lego
				- Composability I: New Contract Instances
				- Composability II: Known Interface and Existing Instance
				- Composability III: Existing Instances and Low Level Calls
			- Properties of DeFi:
				- Non-custodial:
				  participants have full control over their funds at any point in time
				  总是有控制权
				- Permissionless:
				  anyone can interface with financial services without being censored or blocked by a third party
				  不会被第三方冻结
				- Openly auditable:
				  anyone can audit the state of the system
				  每个人都可以audit, check myself
				- Composable:
				  the financial services can be arbitrarily composed such that new financial products and services can be created
				  可以任意compose 产品
			- Point of DeFi
				- offering a new financial architecture that is non-custodial, permissionless, openly auditable, (pseudo)anonymous, and with potentially new capital e ciencies.
				- generalizes the promise at the heart of the original Bitcoin whitepaper, extending the innovation of non-custodial transactions to complex financial operations.
			- DeFi Primitives
				- Keepers, external agents who can trigger state updates.
				  用来发动state 更新的人, 因为只有非contract账户才可以发起transactions, 才可以更新状态
				- Oracles, a mechanism for importing o↵-chain data into the blockchain virtual machine
				  用来获取外部信息加载到链内的预言机
				- Governance, the process through which an on-chain system is able to change the terms of interaction
				  如何make decisions, 如何管理contract, 决定如何update, contract. 因为这些contract本身都是distributed 执行, 但是拥有者控制者是那个owner
			- Composability (还没写完)
			  SCHEDULED: <2023-02-14 Tue>
				- Protocols for Loanable Funds (PLFs)
					- 用来捡钱的一种东西, 可以抵押资产, 根据实时的汇率算得债的健康度, 健康度不行就自动执行清算程序, 可以有清算人来获得抵押资产
				- Flash loans
					- 仅发生在几分钟几秒钟的借款, 仅在一笔transaction中发生. 没有抵押, 只有算好的利息, 用完就得还
				- Yield Aggregators
					- 根据供需关系, 决定最佳的借款人和放贷人的利息
	- Week6
	- Week7
	- Week8
- ## Tutorial
	- Week 4 Smart Contract Development
		- [[Solidity]]
- ## Info
	- 8:2
	- CW: Week 6
	- Exam: 2H
- ## Syllabus
	- Hash Functions
	- Digital Signature
	- Decentralisation and permissionless/ permissioned ledgers
	- Wallets and transactions
	- Authenticated datastructures
	- Blocks and the blockchain
	- Proof of work and mining
	- Ethereum smart contracts
	- Smart contract security
	- Network layer propagation
	- Blockchain security and privacy
	- Building decentralised applications
	- Security and privacy of distributed ledgers
	- Scaling decentralised ledger and alternatives
	- Network and Hardware aspects of decentralised ledgers
- ## Links
	- [Scientia](https://scientia.doc.ic.ac.uk/2223/modules/70017/materials)
	- [Solidity by Example — Solidity 0.8.17 documentation](https://docs.soliditylang.org/en/v0.8.17/solidity-by-example.html)
	- [Solidity by Example](https://solidity-by-example.org/mapping/)
	- [GitHub - smartcontractkit/hardhat-starter-kit: A repo for boilerplate code for testing, deploying, and shipping chainlink solidity code.](https://github.com/smartcontractkit/hardhat-starter-kit)
	-
