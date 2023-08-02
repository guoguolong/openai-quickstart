# apikey: 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
# 导入 pandas 包。Pandas 是一个用于数据处理和分析的 Python 库
# 提供了 DataFrame 数据结构，方便进行数据的读取、处理、分析等操作。
import secret;
import pandas as pd
# 导入 tiktoken 库。Tiktoken 是 OpenAI 开发的一个库，用于从模型生成的文本中计算 token 数量。
import tiktoken
# 从 openai.embeddings_utils 包中导入 get_embedding 函数。
# 这个函数可以获取 GPT-3 模型生成的嵌入向量。
# 嵌入向量是模型内部用于表示输入数据的一种形式。
import openai
from openai.embeddings_utils import get_embedding

openai.api_key = secret.OPENAI_API_KEY
INPUT_DATAPATH = "./data/source.csv"
OUTPUT_DATAPATH = "data/fine_food_reviews_with_embeddings_1k_demo.csv"

df = pd.read_csv(INPUT_DATAPATH, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna() # 删除有空列值的行

# 增加一列：将 "Summary" 和 "Text" 字段组合成新的字段 "combined"
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)
# print(df.head(2))


# text-embedding-ada-002 模型支持的输入最大 Token 数是8191，向量维度 1536
EMBEDDING_MODEL = "text-embedding-ada-002" # 模型类型 - 建议使用官方推荐的第二代嵌入模型：text-embedding-ada-002
EMBEDDING_ENCODING = "cl100k_base" # text-embedding-ada-002 模型对应的分词器（TOKENIZER）
MAX_TOKENS = 250  # 在我们的 DEMO 中过滤 Token 超过 250 的文本
TOP_N = 2

# 首先将前 10 个条目进行初始筛选，假设不到一半会被过滤掉。
df = df.sort_values("Time").tail(TOP_N * 2) 
df.drop("Time", axis=1, inplace=True) # 删除 Time列

encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)

# 忽略太长无法嵌入的评论
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))

# 删除Token超长的样本(删除tokens大于 MAX_TOKENS)
df = df[df.n_tokens <= MAX_TOKENS].tail(TOP_N)

# 下面的语句，实际生成会耗时几分钟
# 提醒：非必须步骤，可直接复用项目中的嵌入文件 fine_food_reviews_with_embeddings_1k
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
df.to_csv(OUTPUT_DATAPATH)
