# 🎨 Tarpan-nlp
European Wild Horse(Tarpan)，欧洲野马于1877年灭绝。 在这里是斯坦福的分词器程序。<br />

## 🤖 分词使用斯坦福NLP
1. CoreNLP主页：https://github.com/stanfordnlp/CoreNLP/<br />
2. 在线依存关系：http://nlp.stanford.edu:8080/parser/index.jsp<br />
3. 核心model-4.0.1下载：https://nlp.stanford.edu/software/stanford-corenlp-full-2020-07-09.zip<br />
4. 中文词包4.2.0下载：http://nlp.stanford.edu/software/stanford-corenlp-4.2.0-models-chinese.jar<br />
5. 也有集成的，在线的分词，例如例子中使用的，但是性能不怎么样。<br />

## 💻 应用
现在用在情感分析上，参考项目：https://github.com/sekift/tarpan 。<br />
### ⚡ 启动硬件要求
1. 内存6G+
2. 硬盘4G+
### 📰 输出结果展示
````text
"input": "酒店实在差，房间又小又脏，卫生间环境太差，整个酒店有点像马路边上的招待所。"
{
    "seged": "酒店 实在 差 ， 房间 又 小 又 脏 ， 卫生间 环境 太 差 ， 整 个 酒店 有点 像 马路 边上 的 招待所 。",
    "posed": "酒店#NN 实在#AD 差#VA ，#PU 房间#NN 又#AD 小#VA 又#AD 脏#VA ，#PU 卫生间#NN 环境#NN 太#AD 差#VA ，#PU 整#DT 个#M 酒店#NN 有点#AD 像#VV 马路#NN 边上#LC 的#DEG 招待所#NN 。#PU",
    "parsed": "root(ROOT-0, 差-3)   nsubj(差-3, 酒店-1)   advmod(差-3, 实在-2)   punct(差-3, ，-4)   nsubj(小-7, 房间-5)   advmod(小-7, 又-6)   conj(差-3, 小-7)   advmod(脏-9, 又-8)   conj(小-7, 脏-9)   punct(差-3, ，-10)   compound:nn(环境-12, 卫生间-11)   nsubj(差-14, 环境-12)   advmod(差-14, 太-13)   conj(差-3, 差-14)   punct(差-3, ，-15)   det(酒店-18, 整-16)   mark:clf(整-16, 个-17)   nsubj(像-20, 酒店-18)   advmod(像-20, 有点-19)   conj(差-3, 像-20)   nmod(招待所-24, 马路-21)   case(马路-21, 边上-22)   case(马路-21, 的-23)   dobj(像-20, 招待所-24)   punct(差-3, 。-25)"
}
````
 
## 📖 参考以下资料
### 🌰 宾州树《汉语词性标注规范》<br />
````text
词性标记	英文名称	中文名称	例子
AD	adverbs	副词	“还”
AS	Aspect marker	体标记	了，着，过
BA	in ba-const	把/将	把，将
CC	Coordinating conjunction	并列连词	“和”，“与”，“或”，“或者”
CD	Cardinal numbers	数词	“一百”
CS	Subordinating conj	从属连词	若，如果，如
DEC	for relative-clause etc	标句词，关系从句“的”	我买“的”书
DEG	Associative	所有格/联结作用“的”	我“的”书
DER	in V-de construction,and V-de-R	V得，表示结果补语的“得”	跑“得”气喘吁吁
DEV	before VP	表示方式状语的“地”	高兴/VA 地/DEV 说/VV
DT	Determiner	限定词	这
ETC	Tag for words in coordination phrase	"等”，“等等”	科技文教 等/ETC 领域
FW	Foreign words	外语词	ISO
IJ	interjection	感叹词	啊
IP  简单从句
JJ	Noun-modifier other than nouns	其他名词修饰语	共同/JJ 的/DEG 目的/NN 她/PN 是/VC 女/JJ 的/DEG
LB	in long bei-construction	长“被”	“被”他打了
LC	Localizer	方位词	桌子“上”
M	Measure word	量词	一“间”房子
MSP	Some particles	其他结构助词	他“所”需要的 所，而，以
NN	Common nouns	其他名词，普通名词	桌子
NR	Proper nouns	专有名词	北京
NT	Temporal nouns	时间名词	一月，汉朝
OD	Ordinal numbers	序数词	第一
ON	Onomatopoeia	拟声词	“哗啦啦”
P	Prepositions	介词	“在”
PN	pronouns	代词	“你”，“我”，“他”
PU	Punctuations	标点	，。
SB	in short bei-construction	短“被”	他“被”训了一顿
SP	Sentence-final particle	句末助词	他好 吧/SP
VA	Predicative adjective	谓语形容词	花很 红/VA 红彤彤 雪白 丰富
VC	Copula	系动词	“是”，“为”，“非”
VE	as the main verb	“有”作为主要动词	“有”，“无”
VV	Other verbs	其他动词，普通动词	走，可能，喜欢
````

### 🚩 依存关系含义
关系表示
````text
词性标记	英文名称	中文名称 
abbrev	abbreviation modifier	缩写
acomp	adjectival complement	形容词的补充
advcl 	adverbial clause modifier	状语从句修饰词
advmod	adverbial modifier	状语
agent	agent	代理,一般有by的时候会出现这个
amod	adjectival modifier	形容词
appos	appositional modifier	同位词
attr	attributive	属性
aux	auxiliary	非主要动词和助词	如BE,HAVE SHOULD/COULD等到
auxpass	passive auxiliary 被动词
cc	coordination	并列关系,一般取第一个词
ccomp	clausal complement	从句补语
complm	complementizer	引导从句的词好重聚中的主要动词
conj 	conjunct	连接两个并列的词
cop	copula	系动词（如be,seem,appear等）	（命题主词与谓词间的）连系
csubj 	clausal subject	从主关系
csubjpass	clausal passive subject 主从被动关系
dep	dependent	依赖关系
det	determiner	决定词	如冠词等
dobj 	direct object	直接宾语
expl	expletive	主要是抓取there
infmod	infinitival modifier	动词不定式
iobj 	indirect object	非直接宾语	也就是所以的间接宾语
mark	marker	主要出现在有“that” or “whether”“because”, “when”
mwe	multi-word expression	多个词的表示
neg	negation modifier	否定词
nn	noun compound modifier	名词组合形式
npadvmod	noun phrase as adverbial modifier	名词作状语
nsubj 	nominal subject	名词主语
nsubjpass	passive nominal subject	被动的名词主语
num	numeric modifier	数值修饰
number	element of compound number	组合数字
parataxis	parataxis	parataxis	并列关系
partmod	participial modifier	动词形式的修饰
pcomp	prepositional complement	介词补充
pobj 	object of a preposition	介词的宾语
poss	possession modifier	所有形式	所有格	所属
possessive	possessive modifier	这个表示所有者和那个’S的关系
preconj 	preconjunct	常常是出现在 “either”, “both”, “neither”的情况下
predet	predeterminer	前缀决定	常常是表示所有
prep	prepositional modifier
prepc	prepositional clausal modifier
prt	phrasal verb particle	动词短语
punct	punctuation 带有标点符号，一般无意义
purpcl 	purpose clause modifier	目的从句
quantmod	quantifier phrase modifier	数量短语
rcmod	relative clause modifier    相关关系
ref 	referent	指示物,指代
rel 	relative
root	root	最重要的词,从它开始,根节点
tmod	temporal modifier   时间短语
xcomp	open clausal complement 句子的补充
xsubj 	controlling subject 控制主体
````
中心语为谓词
````text
词性标记	英文名称	中文名称 
subj    subject    主语
nsubj   noun subject    名词性主语
top topic 主题
npsubj  -   被动型主语
csubj   clausal subject   从句主语
xsubj   x subject   x主语，一般一个主语下面含多个从句
````

中心语为谓词或介词
````text
词性标记	英文名称	中文名称
obj object  宾词
dobj    direct object   直接宾词
iobj    indirect object   间接宾语
range   -   间接宾语为数量词，又称为与格
pobj    -   介词宾语
lobj    -   时间介词
````

中心语为谓词
````text
词性标记	英文名称	中文名称
comp    complement   补语
ccomp   clausal complement   从句补语，一般由两个动词构成，中心语引导后一个动词所在的从句(IP)
xcomp   xclausal complement x从句补语
acomp   adjectival complement   形容词补语
tcomp   temporal complement 时间补语，例如遇到，以前
lccomp  localizer complement    位置补语，例如占，以上
````

中心语为名词
````text
词性标记	英文名称	中文名称
mod modifier    修饰语
pass	passive 被动修饰
tmod	time modifier   时间修饰
rcmod	relative clause modifier    关系从句修饰
nummod	number modifier 数量修饰
ornmod	numeric modifier    序数修饰
clf classifier modifier 类别修饰
nmod	noun modifier   复合名词修饰，如上海、浦东
amod	adjective modifier  形容词修饰
vmod	verbs modifier  动词修饰
prnmod	parenthetical modifier  插入词修饰
neg	negative modifier 不定修饰，如遇到、不
det	determiner modifier限定词修饰，如这些
possm	possessive marker   所属标记
poss	possessive modifier 所属修饰
dvpm	dvp marker  DVP标记
assm	associative marker  关联标记
assmod	associative modifier    关联修饰
prep	prepositional modifier  介词修饰
clmod	clause modifier 从句修饰，如因为、开始
plmod	prepositional localizer modifier    介词性地点修饰，如在、上
asp	aspect marker   时态标词
partmod	participial modifier    分词修饰，如不存在
etc	等关系
````

中心语为实词
````text
词性标记	英文名称	中文名称
conj 	conjunct    联合
cc 	coordination    连接，指中心词与连词，如开发，与
cop copula  系动，双指助动词 	  	 
````

其它
````text
词性标记	英文名称	中文名称
attr	attribute   属性关系，如 是
cordmod	coordinated verb compound   并列联合动词
mmod	modal verb  情态动词，如 能
ba	把字关系
tclaus	time clausal    时间从句
cpm	complementizer  补语化成分，一般指“的”引导的CP
````
新出现的待添加
````text
词性标记	英文名称	中文名称    例子
case    case    -   case(不错-2, 的-3)
topic   topic   -   nmod:topic(新-8, 酒店-4)
acl acl -   acl(酒气-31, 挥之不去-29)
discourse   discourse   论述   discourse(有-22, 了-24)
````

带冒号组合词
````text
词性标记	英文名称	中文名称    例子
nmod:topic  -   -   nmod:topic(新-8, 酒店-4)
compound:nn -   -   compound:nn(房-3, 商务-2)
nmod:assmod -   -   nmod:assmod(很脏-19, 酒店-16)
compound:nn -   -   compound:nn(很脏-19, 地毯-18)
aux:modal   -   -   aux:modal(见-24, 可-23)
mark:clf    -   -   mark:clf(一-4, 步-5)
nmod:prep   -   -   nmod:prep(摆-7, 房间里-9)
nmod:tmod   -   -   nmod:tmod(赶-24, 早上-23)
advmod:dvp  -   -   advmod:dvp(淋浴-43, 奇怪-40)
aux:ba  -   -   aux:ba(发现-60, 把-46)
aux:asp -   -   aux:asp(要-21, 了-22)
advcl:loc   -   -   advcl:loc(受到-9, 进入-2)

````
