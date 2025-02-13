# NNScholar - 医学文献智能检索与分析系统

NNScholar是一个基于Python的医学文献智能检索与分析系统，它能够帮助研究者快速检索、分析和整理医学文献。

## 功能特点

- 智能文献检索：使用PubMed API进行文献检索
- 文献分析：自动分析文献的年份分布、研究热点等
- 智能评分：使用机器学习模型计算文献相关性
- 数据导出：支持Excel和文本格式的文献导出
- 期刊指标：自动获取期刊的影响因子、JCR分区等信息

## 安装说明

1. 安装Python 3.8或更高版本
2. 克隆或下载本项目
3. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 配置说明

1. 在项目根目录创建`.env`文件，添加以下配置：
```
DEEPSEEK_API_KEY=your_deepseek_api_key
PUBMED_API_KEY=your_pubmed_api_key
PUBMED_EMAIL=your_email
TOOL_NAME=nnscholar_pubmed
PUBMED_API_URL=https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
```

2. 替换以下配置项：
- `DEEPSEEK_API_KEY`：你的DeepSeek API密钥
- `PUBMED_API_KEY`：你的PubMed API密钥
- `PUBMED_EMAIL`：你的邮箱地址

## 使用说明

1. 启动应用：
```bash
python app.py
```

2. 访问应用：
- 打开浏览器访问 `http://localhost:5000`

3. 分析Excel文件：
```bash
python analyze_papers.py
```

## 文件结构

```
nnscholar-search-main/
├── app.py              # 主应用程序
├── analyze_papers.py   # 文献分析脚本
├── requirements.txt    # 依赖包列表
├── .env               # 环境配置文件
├── data/              # 数据文件目录
├── exports/           # 导出文件目录
├── logs/              # 日志文件目录
└── templates/         # 模板文件目录
```

## 注意事项

1. 首次运行时会自动下载NLTK数据
2. 确保有稳定的网络连接
3. API密钥请妥善保管，不要泄露

## 常见问题

1. 如果遇到NLTK数据下载问题，可以手动下载并放置在正确的目录
2. 如果遇到API限制，请检查API密钥是否正确
3. 如果需要更改端口，可以在app.py中修改

## 许可证

MIT License
