<p align="center">
  <a href="docs/README-en.md">English</a> •
  <a href="README.md">简体中文 (Simplified Chinese)</a>
</p>

# Fake News Detector

预训练数据集爬取自网络

## 大纲

* **自动化文本预处理**: 清洗原始新闻文本，去除噪声，标准化文本格式
* **TF-IDF 特征提取**: 将文本转化为机器学习模型可以理解的数值特征向量
* **多模型训练与评估**: 训练并比较逻辑回归、决策树、梯度提升和随机森林四种分类器的性能
* **模型持久化**: 保存训练好的向量化器和模型，实现训练与预测分离
* **增量适应 (再训练)**: 加载已有模型，利用新数据进行适应性训练
* **交互式预测**: 提供命令行工具，对用户输入的任意新闻文本进行实时真伪预测

## 技术细节与原理讲解

### 1. 数据预处理 (`utils.py: clean_text`)

将原始、混乱的文本数据转化为干净、一致的格式，以便后续的特征提取和模型训练

**步骤与原理**:

* **转小写 (`text.lower()`)**: 统一大小写，确保 "News" 和 "news" 被视为同一个词
* **移除方括号内容 (`re.sub(r'\[.*?\]', '', text)`)**: 通常用于移除编辑注记或引用来源标记，这些内容可能与新闻本身的真伪无关
* **移除特殊字符 (`re.sub(r'\W', ' ', text)`)**: `\W` 匹配任何非字母数字下划线的字符。将其替换为空格是为了分离单词，同时去除可能干扰分析的标点符号、特殊符号等。(*注：更精细的处理可能需要区分标点和真正无意义的符号*)
* **移除 URL (`re.sub(r'https?://\S+|www\.\S+', '', text)`)**: URL 本身通常不包含判断真伪的核心信息，且格式多样，易引入噪声
* **移除 HTML 标签 (`re.sub(r'<.*?>+', '', text)`)**: 清除网页抓取时可能残留的 HTML
* **移除标点 (`re.sub(r'[%s]' % re.escape(string.punctuation), '', text)`)**: 再次确认移除标点，虽然 `\W` 已处理大部分。
* **移除换行符 (`re.sub(r'\n', ' ', text)`)**: 将换行替换为空格，形成连续文本
* **移除含数字的单词 (`re.sub(r'\w*\d\w*', '', text)`)**: 移除像 "word123" 这样的词。这基于一个假设：含有数字的混合词对真伪判断贡献不大，有时可能是特殊 ID 或代码。(*注：这可能会误删有意义的数字信息，如日期年份，需要权衡*)
* **移除多余空格 (`re.sub(r'\s+', ' ', text).strip()`)**: 将连续的多个空格合并为一个，并移除首尾空格，使文本更规整

### 2. 特征提取: TF-IDF (`train.py: TfidfVectorizer`)

将清理后的文本数据转换为数值矩阵，作为输入

**原理**: TF-IDF (Term Frequency-Inverse Document Frequency) 是一种常用的文本表示方法，用于评估一个词语对于一个文件集或一个语料库中的其中一份文件的重要程度

* **TF (Term Frequency, 词频)**: 一个词语在单个文件中出现的频率。计算方式： `(词语 W 在文件 D 中出现的次数) / (文件 D 的总词数)`。TF 值越高，说明该词在该文件中越常见
* **IDF (Inverse Document Frequency, 逆文档频率)**: 衡量一个词语在整个语料库中的普遍程度。如果一个词在很多文件中都出现，那么它的 IDF 值会较低，说明它可能是一个通用词汇（如 "的", "是"），区分度不高。计算方式通常是： `log( (语料库总文件数 + 1) / (包含词语 W 的文件数 + 1) ) + 1` (加 1 是为了避免分母为 0 并进行平滑)。
* **TF-IDF**: 将 TF 和 IDF 相乘： `TF-IDF(W, D) = TF(W, D) * IDF(W)`
  * 如果一个词在特定文件中频繁出现 (TF 高)，但在整个语料库中很少见 (IDF 高)，那么它的 TF-IDF 值会很高，表明这个词对该文件具有很好的区分度，很可能是该文件的主题词或关键词
  * 反之，如果一个词在整个语料库中很常见 (IDF 低)，或者在当前文件中不常出现 (TF 低)，其 TF-IDF 值会较低

**实现**: scikit-learn 的 `TfidfVectorizer` 完成了以下工作：
    1.  **分词 (Tokenization)**: 将文本分割成单词（或称为 token）
    2.  **计数 (Counting)**: 计算每个词的词频
    3.  **IDF 计算**: 在整个训练集上计算每个词的 IDF 值
    4.  **TF-IDF 矩阵生成**: 计算每个词在每篇文档中的 TF-IDF 值，形成一个 `(文档数量) x (词汇表大小)` 的稀疏矩阵

**关键参数 (`config.py`)**:
    *   `MAX_FEATURES`: 限制最终特征矩阵的维度（词汇表大小）。只选择 TF-IDF 值最高（或文档频率最高，取决于实现）的前 `MAX_FEATURES` 个词作为特征。这有助于降维、减少噪声、防止过拟合
    *   `min_df` (`train.py` 中设置): 忽略那些在少于 `min_df` 个文档中出现的词。有助于过滤掉罕见词或拼写错误

### 3. 机器学习模型 (`train.py`)

本项目使用了四种不同类型的分类模型，以提供不同的视角和性能表现。所有模型都以 TF-IDF 特征矩阵作为输入，预测新闻的类别（0 代表虚假，1 代表真实）

* **逻辑回归 (Logistic Regression)**:
  * **原理**: 通过一个 Sigmoid 函数将线性组合的输入特征映射到 (0, 1) 区间，表示属于某个类别的概率。通过设定阈值（通常是 0.5）进行分类
  * **特点**: 简单、快速、易于解释（系数可以表示特征重要性，但在高维 TF-IDF 上解释性减弱），是很好的基线模型。对线性可分的数据效果好
* **决策树 (Decision Tree)**:
  * **原理**: 构建一个树状结构，每个内部节点代表一个特征（词的 TF-IDF 值）的测试，每个分支代表测试的结果，每个叶节点代表一个类别标签。通过从根节点开始一系列的决策来到达叶节点，从而进行分类
  * **特点**: 直观易懂，可以处理非线性关系。但容易过拟合（树长得过于复杂，完美拟合训练数据但在新数据上表现差），对数据的微小变动敏感
* **梯度提升分类器 (Gradient Boosting Classifier, GBC)**:
  * **原理**: 集成学习方法之一。它顺序地构建多棵决策树（通常是弱学习器），每一棵新树都试图纠正前面所有树的残差（预测误差）。最终的预测是所有树预测结果的加权组合
  * **特点**: 通常性能强大，能够发现复杂的非线性关系。但训练相对较慢，对参数敏感，也可能过拟合
* **随机森林 (Random Forest Classifier, RFC)**:
  * **原理**: 另一种集成学习方法。它构建大量的决策树，每棵树都在随机选择的数据子集（行抽样/bootstrap）和随机选择的特征子集（列抽样）上进行训练。最终的预测结果由所有树投票（分类问题）或平均（回归问题）决定
  * **特点**: 鲁棒性强，不易过拟合（相比单棵决策树），能够处理高维数据。训练可以并行化，速度较快。通常具有很好的泛化能力

### 4. 模型评估 (`train.py`)

**指标**:

* **准确率 (Accuracy)**: `(预测正确的样本数) / (总样本数)`。最直观的指标，但在类别不平衡（例如虚假新闻远少于真实新闻）时可能具有误导性
* **精确率 (Precision)**: `TP / (TP + FP)` (针对某一类别，如 "虚假新闻")。在所有被模型预测为 "虚假" 的新闻中，真正是 "虚假" 的比例。高精确率表示模型预测为 "虚假" 的结果比较可信，误报少
* **召回率 (Recall / Sensitivity)**: `TP / (TP + FN)` (针对某一类别)。在所有真正是 "虚假" 的新闻中，被模型成功找出来的比例。高召回率表示模型能尽可能多地找出 "虚假" 新闻，漏报少
* **F1 分数 (F1-Score)**: `2 * (Precision * Recall) / (Precision + Recall)`。精确率和召回率的调和平均数。用于综合评价模型的性能，特别是在 P 和 R 需要权衡时

**工具**: scikit-learn 的 `classification_report` 可以方便地计算并展示每个类别的 P, R, F1 以及总体准确率

### 5. 模型持久化 (`train.py`, `retrain.py`, `predict.py`)

将训练阶段得到的成果（向量化器和模型）保存到磁盘，以便未来可以快速加载并用于预测，无需重新训练

**实现**: 使用 `joblib` 库的 `dump` 和 `load` 函数。`joblib` 对包含大型 NumPy 数组的对象（如 scikit-learn 的模型和向量化器）进行了优化，比 Python 内置的 `pickle` 更高效

**关键点**: **必须同时保存和加载 TF-IDF 向量化器 (`vectorizer`) 和模型**。因为模型是基于特定向量化器产生的特征空间进行训练的。在预测或再训练时，必须使用**完全相同**的向量化器对象（通过加载 `tfidf_vectorizer.joblib`）来 `transform` 新的文本数据，以确保输入特征与训练时具有相同的维度和含义

### 6. 模型再训练 (`retrain.py`)

可利用新的数据更新已有的模型，使其能够适应数据分布的变化（概念漂移）或学习新的模式，而无需从头开始训练

**原理与实现**:

1. **加载**: 加载之前保存的向量化器和所有模型
2. **准备新数据**: 加载新的 CSV 文件 (`New_Fake.csv`, `New_True.csv`)，进行与初始训练时**完全相同**的文本预处理 (`clean_text`)
3. **转换新数据**: 使用**已加载的、未改变的**向量化器对象，调用其 `.transform()` 方法将预处理后的新文本数据转换为 TF-IDF 特征矩阵。**绝对不能在新数据上调用 `.fit()` 或 `.fit_transform()`**，否则会改变词汇表和 IDF 值，导致特征空间不一致
4. **继续训练**: 对每个加载成功的模型，调用其 `.fit(X_new_vec, y_new)` 方法
   * **重要说明**: 对于 scikit-learn 中的大多数标准分类器（包括本项目使用的 LR, DT, GBC, RFC），再次调用 `.fit()` 通常意味着**用新数据重新训练模型的主要参数**，而不是严格意义上的增量学习（即在原有知识基础上添加新知识）。它更像是一种**适应 (Adaptation)**，让模型根据最新的数据调整自己。对于支持 `warm_start=True` 的模型（如某些集成模型），配合调整 `n_estimators` 等参数可以实现更接近增量的效果（添加更多树），但本项目为保持一致性，统一使用 `fit` 进行适应
5. **保存**: 将适应了新数据的模型**覆盖**保存回原来的 `.joblib` 文件

## 项目结构

```
fake-news-detector/
│
├── data/                     # 存放数据集 (CSV 文件)
│   ├── Fake.csv              # 原始虚假新闻数据
│   ├── True.csv              # 原始真实新闻数据
│   ├── New_Fake.csv          # (可选) 用于再训练的新的虚假新闻数据
│   └── New_True.csv          # (可选) 用于再训练的新的真实新闻数据
│
├── models/                   # 存放训练好的模型和向量化器 (.joblib 文件)
│   └── (此目录在首次训练时自动创建)
│
├── config.py                 # 配置文件 (包含文件路径、训练参数等)
├── utils.py                  # 工具函数 (如文本清理函数 clean_text)
├── train.py                  # 初始模型训练脚本
├── retrain.py                # 模型增量训练/再训练脚本
├── predict.py                # 新闻真伪预测脚本
└── README.md                 # 本说明文件
```

## 环境要求

* Python 3.7+
* 库: `pandas`, `numpy`, `scikit-learn`, `joblib`

## 安装与设置

### 克隆项目

 ```bash
 git clone https://github.com/spawner1145/fake-news-detector
 ```

### 安装依赖

```bash
pip install -r requirements.txt
```

(推荐在虚拟环境中)

### 准备数据

   * 创建 `data/` 目录。
   * 将 `Fake.csv`, `True.csv` (及可选的 `New_*.csv`) 放入 `data/`。
   * 确保 CSV 文件包含 `text` 列，使用 UTF-8 编码。

### 数据集csv内部示例(总之就是title,text,subject,date这四列)(实际上只需要text这一列，别的三个其实可以不用):

   ```

   title,text,subject,date
   "As U.S. budget fight looms, Republicans flip their fiscal script","WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a “fiscal conservative” on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS’ “Face the Nation,” drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely to be linked to other issues, such as immigration policy, even as the November congressional election campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump and his Republicans want a big budget increase in military spending, while Democrats also want proportional increases for non-defense “discretionary” spending on programs that support education, scientific research, infrastructure, public health and environmental protection. “The (Trump) administration has already been willing to say: ‘We’re going to increase non-defense discretionary spending ... by about 7 percent,’” Meadows, chairman of the small but influential House Freedom Caucus, said on the program. “Now, Democrats are saying that’s not enough, we need to give the government a pay raise of 10 to 11 percent. For a fiscal conservative, I don’t see where the rationale is. ... Eventually you run out of other people’s money,” he said. Meadows was among Republicans who voted in late December for their party’s debt-financed tax overhaul, which is expected to balloon the federal budget deficit and add about $1.5 trillion over 10 years to the $20 trillion national debt. “It’s interesting to hear Mark talk about fiscal responsibility,” Democratic U.S. Representative Joseph Crowley said on CBS. Crowley said the Republican tax bill would require the  United States to borrow $1.5 trillion, to be paid off by future generations, to finance tax cuts for corporations and the rich. “This is one of the least ... fiscally responsible bills we’ve ever seen passed in the history of the House of Representatives. I think we’re going to be paying for this for many, many years to come,” Crowley said. Republicans insist the tax package, the biggest U.S. tax overhaul in more than 30 years,  will boost the economy and job growth. House Speaker Paul Ryan, who also supported the tax bill, recently went further than Meadows, making clear in a radio interview that welfare or “entitlement reform,” as the party often calls it, would be a top Republican priority in 2018. In Republican parlance, “entitlement” programs mean food stamps, housing assistance, Medicare and Medicaid health insurance for the elderly, poor and disabled, as well as other programs created by Washington to assist the needy. Democrats seized on Ryan’s early December remarks, saying they showed Republicans would try to pay for their tax overhaul by seeking spending cuts for social programs. But the goals of House Republicans may have to take a back seat to the Senate, where the votes of some Democrats will be needed to approve a budget and prevent a government shutdown. Democrats will use their leverage in the Senate, which Republicans narrowly control, to defend both discretionary non-defense programs and social spending, while tackling the issue of the “Dreamers,” people brought illegally to the country as children. Trump in September put a March 2018 expiration date on the Deferred Action for Childhood Arrivals, or DACA, program, which protects the young immigrants from deportation and provides them with work permits. The president has said in recent Twitter messages he wants funding for his proposed Mexican border wall and other immigration law changes in exchange for agreeing to help the Dreamers. Representative Debbie Dingell told CBS she did not favor linking that issue to other policy objectives, such as wall funding. “We need to do DACA clean,” she said.  On Wednesday, Trump aides will meet with congressional leaders to discuss those issues. That will be followed by a weekend of strategy sessions for Trump and Republican leaders on Jan. 6 and 7, the White House said. Trump was also scheduled to meet on Sunday with Florida Republican Governor Rick Scott, who wants more emergency aid. The House has passed an $81 billion aid package after hurricanes in Florida, Texas and Puerto Rico, and wildfires in California. The package far exceeded the $44 billion requested by the Trump administration. The Senate has not yet voted on the aid. ",politicsNews,"December 31, 2017 "
   U.S. military to accept transgender recruits on Monday: Pentagon,"WASHINGTON (Reuters) - Transgender people will be allowed for the first time to enlist in the U.S. military starting on Monday as ordered by federal courts, the Pentagon said on Friday, after President Donald Trump’s administration decided not to appeal rulings that blocked his transgender ban. Two federal appeals courts, one in Washington and one in Virginia, last week rejected the administration’s request to put on hold orders by lower court judges requiring the military to begin accepting transgender recruits on Jan. 1. A Justice Department official said the administration will not challenge those rulings. “The Department of Defense has announced that it will be releasing an independent study of these issues in the coming weeks. So rather than litigate this interim appeal before that occurs, the administration has decided to wait for DOD’s study and will continue to defend the president’s lawful authority in District Court in the meantime,” the official said, speaking on condition of anonymity. In September, the Pentagon said it had created a panel of senior officials to study how to implement a directive by Trump to prohibit transgender individuals from serving. The Defense Department has until Feb. 21 to submit a plan to Trump. Lawyers representing currently-serving transgender service members and aspiring recruits said they had expected the administration to appeal the rulings to the conservative-majority Supreme Court, but were hoping that would not happen. Pentagon spokeswoman Heather Babb said in a statement: “As mandated by court order, the Department of Defense is prepared to begin accessing transgender applicants for military service Jan. 1. All applicants must meet all accession standards.” Jennifer Levi, a lawyer with gay, lesbian and transgender advocacy group GLAD, called the decision not to appeal “great news.” “I’m hoping it means the government has come to see that there is no way to justify a ban and that it’s not good for the military or our country,” Levi said. Both GLAD and the American Civil Liberties Union represent plaintiffs in the lawsuits filed against the administration. In a move that appealed to his hard-line conservative supporters, Trump announced in July that he would prohibit transgender people from serving in the military, reversing Democratic President Barack Obama’s policy of accepting them. Trump said on Twitter at the time that the military “cannot be burdened with the tremendous medical costs and disruption that transgender in the military would entail.” Four federal judges - in Baltimore, Washington, D.C., Seattle and Riverside, California - have issued rulings blocking Trump’s ban while legal challenges to the Republican president’s policy proceed. The judges said the ban would likely violate the right under the U.S. Constitution to equal protection under the law. The Pentagon on Dec. 8 issued guidelines to recruitment personnel in order to enlist transgender applicants by Jan. 1. The memo outlined medical requirements and specified how the applicants’ sex would be identified and even which undergarments they would wear. The Trump administration previously said in legal papers that the armed forces were not prepared to train thousands of personnel on the medical standards needed to process transgender applicants and might have to accept “some individuals who are not medically fit for service.” The Obama administration had set a deadline of July 1, 2017, to begin accepting transgender recruits. But Trump’s defense secretary, James Mattis, postponed that date to Jan. 1, 2018, which the president’s ban then put off indefinitely. Trump has taken other steps aimed at rolling back transgender rights. In October, his administration said a federal law banning gender-based workplace discrimination does not protect transgender employees, reversing another Obama-era position. In February, Trump rescinded guidance issued by the Obama administration saying that public schools should allow transgender students to use the restroom that corresponds to their gender identity. ",politicsNews,"December 29, 2017 "
   ...(在后面继续加)
   ```

## 使用说明

在项目根目录下执行：

1. **初始训练**: `python train.py` (生成 `models/` 目录及内容)
2. **再训练 (可选)**: `python retrain.py` (更新 `models/` 中的模型，需先运行 train.py)
3. **预测**: `python predict.py` (加载模型进行交互式预测，需先运行 train.py 或 retrain.py)
   * 输入新闻文本，按回车预测
   * 输入 `退出` 或 `exit` 结束

## 配置

可在 `config.py` 中调整文件路径、TF-IDF 特征数 (`MAX_FEATURES`)、测试集比例 (`TEST_SET_SIZE`) 等参数

## 局限性与未来方向

* **特征单一**: 仅基于词频 (TF-IDF)，未考虑词语顺序、语义、上下文
* **静态模型**: 模型训练后是固定的，无法实时适应网络热点或新出现的造谣模式，需要定期再训练
* **依赖训练数据**: 模型的判断能力很大程度上取决于训练数据的质量和覆盖范围。可能存在偏见
* **预处理策略**: 当前的 `clean_text` 比较通用，可能需要针对特定数据集进行微调

**未来可探索的方向**:

* ***由于使用的数据集全是英文，模型不具备中文鉴别能力***
* **更先进的特征**: 使用词嵌入 (Word2Vec, GloVe, FastText) 或上下文相关的嵌入 (BERT, RoBERTa, XLNet) 来捕捉语义信息
* **深度学习模型**: 应用 CNN, RNN (LSTM/GRU), 或 Transformer 模型进行分类
* **元数据利用**: 结合新闻来源、作者、发布时间、评论等元数据进行综合判断
* **在线学习**: 实现模型能够持续地从新数据流中学习，而不是依赖批量的再训练
* **可解释性**: 使用 SHAP 或 LIME 等工具分析模型做出预测的原因
