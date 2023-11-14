<link rel="stylesheet" href="_static/css/main.css">

## 多码网开源导航

###  1. <a name='模型'></a>模型

#### 1.1 文本LLM模型

* ChatGLM：
  * 地址：https://github.com/THUDM/ChatGLM-6B
![](https://img.shields.io/github/stars/THUDM/ChatGLM-6B.svg)
  * 简介：中文领域效果最好的开源底座模型之一，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持

* ChatGLM2-6B
  * 地址：https://github.com/THUDM/ChatGLM2-6B
![](https://img.shields.io/github/stars/THUDM/ChatGLM2-6B.svg)
  * 简介：基于开源中英双语对话模型 ChatGLM-6B 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，引入了GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练；基座模型的上下文长度扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练；基于 Multi-Query Attention 技术实现更高效的推理速度和更低的显存占用；允许商业使用。

* ChatGLM3-6B
  * 地址：https://github.com/THUDM/ChatGLM3
![](https://img.shields.io/github/stars/THUDM/ChatGLM3.svg)
  * 简介：ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：更强大的基础模型： ChatGLM3-6B 的基础模型 ChatGLM3-6B-Base 采用了更多样的训练数据、更充分的训练步数和更合理的训练策略；更完整的功能支持： ChatGLM3-6B 采用了全新设计的 Prompt 格式，除正常的多轮对话外。同时原生支持工具调用（Function Call）、代码执行（Code Interpreter）和 Agent 任务等复杂场景；更全面的开源序列： 除了对话模型 ChatGLM3-6B 外，还开源了基础模型 ChatGLM3-6B-Base、长文本对话模型 ChatGLM3-6B-32K。以上所有权重对学术研究完全开放，在填写问卷进行登记后亦允许免费商业使用。

* Chinese-LLaMA-Alpaca：
  * 地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca
![](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca.svg)
  * 简介：中文LLaMA&Alpaca大语言模型+本地CPU/GPU部署，在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练

* Chinese-LLaMA-Alpaca-2：
  * 地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
![](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca-2.svg)
  * 简介：该项目将发布中文LLaMA-2 & Alpaca-2大语言模型，基于可商用的LLaMA-2进行二次开发。

* Chinese-LlaMA2：
  * 地址：https://github.com/michael-wzhu/Chinese-LlaMA2
![](https://img.shields.io/github/stars/michael-wzhu/Chinese-LlaMA2.svg)
  * 简介：该项目基于可商用的LLaMA-2进行二次开发决定在次开展Llama 2的中文汉化工作，包括Chinese-LlaMA2: 对Llama 2进行中文预训练；第一步：先在42G中文预料上进行训练；后续将会加大训练规模；Chinese-LlaMA2-chat: 对Chinese-LlaMA2进行指令微调和多轮对话微调，以适应各种应用场景和多轮对话交互。同时我们也考虑更为快速的中文适配方案：Chinese-LlaMA2-sft-v0: 采用现有的开源中文指令微调或者是对话数据，对LlaMA-2进行直接微调 (将于近期开源)。

* Llama2-Chinese：
  * 地址：https://github.com/FlagAlpha/Llama2-Chinese
![](https://img.shields.io/github/stars/FlagAlpha/Llama2-Chinese.svg)
  * 简介：该项目专注于Llama2模型在中文方面的优化和上层建设，基于大规模中文数据，从预训练开始对Llama2模型进行中文能力的持续迭代升级。

* OpenChineseLLaMA：
  * 地址：https://github.com/OpenLMLab/OpenChineseLLaMA
![](https://img.shields.io/github/stars/OpenLMLab/OpenChineseLLaMA.svg)
  * 简介：基于 LLaMA-7B 经过中文数据集增量预训练产生的中文大语言模型基座，对比原版 LLaMA，该模型在中文理解能力和生成能力方面均获得较大提升，在众多下游任务中均取得了突出的成绩。

* BELLE：
  * 地址：https://github.com/LianjiaTech/BELLE
![](https://img.shields.io/github/stars/LianjiaTech/BELLE.svg)
  * 简介：开源了基于BLOOMZ和LLaMA优化后的一系列模型，同时包括训练数据、相关模型、训练代码、应用场景等，也会持续评估不同训练数据、训练算法等对模型表现的影响。

* Panda：
  * 地址：https://github.com/dandelionsllm/pandallm
![](https://img.shields.io/github/stars/dandelionsllm/pandallm.svg)
  * 简介：开源了基于LLaMA-7B, -13B, -33B, -65B 进行中文领域上的持续预训练的语言模型, 使用了接近 15M 条数据进行二次预训练。

* Robin (罗宾):
  * 地址：https://github.com/OptimalScale/LMFlow
![](https://img.shields.io/github/stars/OptimalScale/LMFlow.svg)
  * 简介：Robin (罗宾)是香港科技大学LMFlow团队开发的中英双语大语言模型。仅使用180K条数据微调得到的Robin第二代模型，在Huggingface榜单上达到了第一名的成绩。LMFlow支持用户快速训练个性化模型，仅需单张3090和5个小时即可微调70亿参数定制化模型。
    
* Fengshenbang-LM：
  * 地址：https://github.com/IDEA-CCNL/Fengshenbang-LM
![](https://img.shields.io/github/stars/IDEA-CCNL/Fengshenbang-LM.svg)
  * 简介：Fengshenbang-LM(封神榜大模型)是IDEA研究院认知计算与自然语言研究中心主导的大模型开源体系，该项目开源了姜子牙通用大模型V1，是基于LLaMa的130亿参数的大规模预训练模型，具备翻译，编程，文本分类，信息抽取，摘要，文案生成，常识问答和数学计算等能力。除姜子牙系列模型之外，该项目还开源了太乙、二郎神系列等模型。

* BiLLa：
  * 地址：https://github.com/Neutralzz/BiLLa
![](https://img.shields.io/github/stars/Neutralzz/BiLLa.svg)
  * 简介：该项目开源了推理能力增强的中英双语LLaMA模型。模型的主要特性有：较大提升LLaMA的中文理解能力，并尽可能减少对原始LLaMA英文能力的损伤；训练过程增加较多的任务型数据，利用ChatGPT生成解析，强化模型理解任务求解逻辑；全量参数更新，追求更好的生成效果。

* Moss：
  * 地址：https://github.com/OpenLMLab/MOSS
![](https://img.shields.io/github/stars/OpenLMLab/MOSS.svg)
  * 简介：支持中英双语和多种插件的开源对话语言模型，MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。

* Luotuo-Chinese-LLM：
  * 地址：https://github.com/LC1332/Luotuo-Chinese-LLM
![](https://img.shields.io/github/stars/LC1332/Luotuo-Chinese-LLM.svg)
  * 简介：囊括了一系列中文大语言模型开源项目，包含了一系列基于已有开源模型（ChatGLM, MOSS, LLaMA）进行二次微调的语言模型，指令微调数据集等。

* Linly：
  * 地址：https://github.com/CVI-SZU/Linly
![](https://img.shields.io/github/stars/CVI-SZU/Linly.svg)
  * 简介：提供中文对话模型 Linly-ChatFlow 、中文基础模型 Linly-Chinese-LLaMA 及其训练数据。 中文基础模型以 LLaMA 为底座，利用中文和中英平行增量预训练。项目汇总了目前公开的多语言指令数据，对中文模型进行了大规模指令跟随训练，实现了 Linly-ChatFlow 对话模型。

* Firefly：
  * 地址：https://github.com/yangjianxin1/Firefly
![](https://img.shields.io/github/stars/yangjianxin1/Firefly.svg)
  * 简介：Firefly(流萤) 是一个开源的中文大语言模型项目，开源包括数据、微调代码、多个基于Bloom、baichuan等微调好的模型等；支持全量参数指令微调、QLoRA低成本高效指令微调、LoRA指令微调；支持绝大部分主流的开源大模型，如百川baichuan、Ziya、Bloom、LLaMA等。持lora与base model进行权重合并，推理更便捷。

* ChatYuan
  * 地址：https://github.com/clue-ai/ChatYuan
![](https://img.shields.io/github/stars/clue-ai/ChatYuan.svg)
  * 简介：元语智能发布的一系列支持中英双语的功能型对话语言大模型，在微调数据、人类反馈强化学习、思维链等方面进行了优化。

* ChatRWKV：
  * 地址：https://github.com/BlinkDL/ChatRWKV
![](https://img.shields.io/github/stars/BlinkDL/ChatRWKV.svg)
  * 简介：开源了一系列基于RWKV架构的Chat模型（包括英文和中文），发布了包括Raven，Novel-ChnEng，Novel-Ch与Novel-ChnEng-ChnPro等模型，可以直接闲聊及进行诗歌，小说等创作，包括7B和14B等规模的模型。

* CPM-Bee
  * 地址：https://github.com/OpenBMB/CPM-Bee
![](https://img.shields.io/github/stars/OpenBMB/CPM-Bee.svg)
  * 简介：一个完全开源、允许商用的百亿参数中英文基座模型。它采用Transformer自回归架构（auto-regressive），在超万亿（trillion）高质量语料上进行预训练，拥有强大的基础能力。开发者和研究者可以在CPM-Bee基座模型的基础上在各类场景进行适配来以创建特定领域的应用模型。

* TigerBot
  * 地址：https://github.com/TigerResearch/TigerBot
![](https://img.shields.io/github/stars/TigerResearch/TigerBot.svg)
  * 简介：一个多语言多任务的大规模语言模型(LLM)，开源了包括模型：TigerBot-7B, TigerBot-7B-base，TigerBot-180B，基本训练和推理代码，100G预训练数据，涵盖金融、法律、百科的领域数据以及API等。

* 书生·浦语
  * 地址：https://github.com/InternLM/InternLM-techreport
![](https://img.shields.io/github/stars/InternLM/InternLM-techreport.svg)
  * 简介：商汤科技、上海AI实验室联合香港中文大学、复旦大学和上海交通大学发布千亿级参数大语言模型“书生·浦语”（InternLM）。据悉，“书生·浦语”具有1040亿参数，基于“包含1.6万亿token的多语种高质量数据集”训练而成。

* Aquila
  * 地址：https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila
![](https://img.shields.io/github/stars/FlagAI-Open/FlagAI.svg)
  * 简介：由智源研究院发布，Aquila语言大模型在技术上继承了GPT-3、LLaMA等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的tokenizer，升级了BMTrain并行训练方法，是在中英文高质量语料基础上从０开始训练的，通过数据质量的控制、多种训练的优化方法，实现在更小的数据集、更短的训练时间，获得比其它开源模型更优的性能。也是首个支持中英双语知识、支持商用许可协议、符合国内数据合规需要的大规模开源语言模型。

* Baichuan-7B
  * 地址：https://github.com/baichuan-inc/baichuan-7B
![](https://img.shields.io/github/stars/baichuan-inc/baichuan-7B.svg)
  * 简介：Baichuan-13B 是由百川智能继 Baichuan-7B 之后开发的包含 130 亿参数的开源可商用的大规模语言模型，在权威的中文和英文 benchmark 上均取得同尺寸最好的效果。该项目发布包含有预训练 (Baichuan-13B-Base) 和对齐 (Baichuan-13B-Chat) 两个版本。
 
* Baichuan-13B
  * 地址：https://github.com/baichuan-inc/Baichuan-13B
![](https://img.shields.io/github/stars/baichuan-inc/baichuan-13B.svg)
  * 简介：由百川智能开发的一个开源可商用的大规模预训练语言模型。基于Transformer结构，在大约1.2万亿tokens上训练的70亿参数模型，支持中英双语，上下文窗口长度为4096。在标准的中文和英文权威benchmark（C-EVAL/MMLU）上均取得同尺寸最好的效果。

* Baichuan2
  * 地址：https://github.com/baichuan-inc/Baichuan2
![](https://img.shields.io/github/stars/baichuan-inc/Baichuan2.svg)
  * 简介：由百川智能推出的新一代开源大语言模型，采用 2.6 万亿 Tokens 的高质量语料训练，在多个权威的中文、英文和多语言的通用、领域 benchmark上取得同尺寸最佳的效果，发布包含有7B、13B的Base和经过PPO训练的Chat版本，并提供了Chat版本的4bits量化。

* Anima
  * 地址：https://github.com/lyogavin/Anima
![](https://img.shields.io/github/stars/lyogavin/Anima.svg)
  * 简介：由艾写科技开发的一个开源的基于QLoRA的33B中文大语言模型，该模型基于QLoRA的Guanaco 33B模型使用Chinese-Vicuna项目开放的训练数据集guanaco_belle_merge_v1.0进行finetune训练了10000个step，基于Elo rating tournament评估效果较好。

* KnowLM
  * 地址：https://github.com/zjunlp/KnowLM
![](https://img.shields.io/github/stars/zjunlp/KnowLM.svg)
  * 简介：KnowLM项目旨在发布开源大模型框架及相应模型权重以助力减轻知识谬误问题，包括大模型的知识难更新及存在潜在的错误和偏见等。该项目一期发布了基于Llama的抽取大模型智析，使用中英文语料对LLaMA（13B）进行进一步全量预训练，并基于知识图谱转换指令技术对知识抽取任务进行优化。

* BayLing
  * 地址：https://github.com/ictnlp/BayLing
![](https://img.shields.io/github/stars/ictnlp/BayLing.svg)
  * 简介：一个具有增强的跨语言对齐的通用大模型，由中国科学院计算技术研究所自然语言处理团队开发。百聆（BayLing）以LLaMA为基座模型，探索了以交互式翻译任务为核心进行指令微调的方法，旨在同时完成语言间对齐以及与人类意图对齐，将LLaMA的生成能力和指令跟随能力从英语迁移到其他语言（中文）。在多语言翻译、交互翻译、通用任务、标准化考试的测评中，百聆在中文/英语中均展现出更好的表现。百聆提供了在线的内测版demo，以供大家体验。

* YuLan-Chat
  * 地址：https://github.com/RUC-GSAI/YuLan-Chat
![](https://img.shields.io/github/stars/RUC-GSAI/YuLan-Chat.svg)
  * 简介：YuLan-Chat是中国人民大学GSAI研究人员开发的基于聊天的大语言模型。它是在LLaMA的基础上微调开发的，具有高质量的英文和中文指令。 YuLan-Chat可以与用户聊天，很好地遵循英文或中文指令，并且可以在量化后部署在GPU（A800-80G或RTX3090）上。

* PolyLM
  * 地址：https://github.com/DAMO-NLP-MT/PolyLM
![](https://img.shields.io/github/stars/DAMO-NLP-MT/PolyLM.svg)
  * 简介：一个在6400亿个词的数据上从头训练的多语言语言模型，包括两种模型大小(1.7B和13B)。PolyLM覆盖中、英、俄、西、法、葡、德、意、荷、波、阿、土、希伯来、日、韩、泰、越、印尼等语种，特别是对亚洲语种更友好。

* Qwen-7B
  * 地址：https://github.com/QwenLM/Qwen-7B
![](https://img.shields.io/github/stars/QwenLM/Qwen-7B.svg)
  * 简介：通义千问-7B（Qwen-7B） 是阿里云研发的通义千问大模型系列的70亿参数规模的模型，使用了超过2.2万亿token的自建大规模预训练数据集进行语言模型的预训练。数据集包括文本和代码等多种数据类型，覆盖通用领域和专业领域，能支持8K的上下文长度，针对插件调用相关的对齐数据做了特定优化，当前模型能有效调用插件以及升级为Agent。

* huozi
  * 地址：https://github.com/HIT-SCIR/huozi
![](https://img.shields.io/github/stars/HIT-SCIR/huozi.svg)
  * 简介：由哈工大自然语言处理研究所多位老师和学生参与开发的一个开源可商用的大规模预训练语言模型。 该模型基于 Bloom 结构的70 亿参数模型，支持中英双语，上下文窗口长度为 2048，同时还开源了基于RLHF训练的模型以及全人工标注的16.9K中文偏好数据集。

* YaYi
  * 地址：https://github.com/wenge-research/YaYi
![](https://img.shields.io/github/stars/wenge-research/YaYi.svg)
  * 简介：雅意大模型在百万级人工构造的高质量领域数据上进行指令微调得到，训练数据覆盖媒体宣传、舆情分析、公共安全、金融风控、城市治理等五大领域，上百种自然语言指令任务。雅意大模型从预训练初始化权重到领域模型的迭代过程中，我们逐步增强了它的中文基础能力和领域分析能力，并增加了多轮对话和部分插件能力。同时，经过数百名用户内测过程中持续不断的人工反馈优化，进一步提升了模型性能和安全性。已开源基于 LLaMA 2 的中文优化模型版本，探索适用于中文多领域任务的最新实践。。

* XVERSE-13B
  * 地址：https://github.com/xverse-ai/XVERSE-13B
![](https://img.shields.io/github/stars/xverse-ai/XVERSE-13B.svg)
  * 简介：由深圳元象科技自主研发的支持多语言的大语言模型，使用主流 Decoder-only 的标准Transformer网络结构，支持 8K 的上下文长度（Context Length），为同尺寸模型中最长，构建了 1.4 万亿 token 的高质量、多样化的数据对模型进行充分训练，包含中、英、俄、西等 40 多种语言，通过精细化设置不同类型数据的采样比例，使得中英两种语言表现优异，也能兼顾其他语言效果；基于BPE算法使用上百GB 语料训练了一个词表大小为100,278的分词器，能够同时支持多语言，而无需额外扩展词表。

* Skywork
  * 地址：https://github.com/SkyworkAI/Skywork
![](https://img.shields.io/github/stars/SkyworkAI/Skywork.svg)
  * 简介：该项目开源了天工系列模型，该系列模型在3.2TB高质量多语言和代码数据上进行预训练，开源了包括模型参数，训练数据，评估数据，评估方法。具体包括Skywork-13B-Base模型、Skywork-13B-Chat模型、Skywork-13B-Math模型和Skywork-13B-MM模型，以及每个模型的量化版模型，以支持用户在消费级显卡进行部署和推理。

* Yi
  * 地址：https://github.com/01-ai/Yi
![](https://img.shields.io/github/stars/01-ai/Yi.svg)
  * 简介：该项目开源了Yi-6B和Yi-34B等模型，该系列模型最长可支持200K的超长上下文窗口版本，可以处理约40万汉字超长文本输入，理解超过1000页的PDF文档。



