# PubMed文献智能检索系统

## 项目简介

这是一个基于Flask的智能文献检索系统，集成了PubMed API和DeepSeek AI，能够实现以下功能：

- 智能分析长文本并进行分句处理
- 自动生成PubMed检索策略
- 检索相关文献并提取关键信息
- 支持基于期刊影响因子、JCR分区和中科院分区的文献筛选
- 提供文献相关性评分和排序
- 生成带有文献引用的标注文本

## 环境要求

- Python 3.8+
- Flask >= 2.0.0
- NLTK >= 3.6.0
- BeautifulSoup4 >= 4.9.0
- Requests >= 2.25.0
- python-dotenv >= 0.19.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- lxml >= 4.6.0

## 安装步骤

1. 克隆项目到本地：
```bash
git clone https://github.com/luckylykkk/nnscholar-search.git
cd nnscholar-search
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境变量：
   - 复制`.env.example`为`.env`
   - 在`.env`文件中填入必要的API密钥：
     ```
     DEEPSEEK_API_KEY=your_deepseek_api_key
     PUBMED_API_KEY=your_pubmed_api_key
     ```

## 使用方法

1. 启动服务器：
```bash
python app.py
```

2. 访问Web界面：
   - 打开浏览器访问 `http://localhost:5000`
   - 在文本框中输入要检索的内容
   - 设置筛选条件（可选）：
     - 影响因子阈值
     - JCR分区要求
     - 中科院分区要求
     - 每句话的最大文献数量

3. API接口：
   - 搜索接口：`POST /api/search`
     ```json
     {
       "text": "要检索的文本内容",
       "filters": {
         "min_if": 3.0,
         "jcr_quartile": ["Q1", "Q2"],
         "cas_quartile": ["1", "2"],
         "papers_limit": 10
       }
     }
     ```
   - 筛选接口：`POST /api/filter`
     ```json
     {
       "papers": [...],
       "filters": {
         "min_if": 3.0,
         "jcr_quartile": ["Q1", "Q2"],
         "cas_quartile": ["1", "2"]
       }
     }
     ```

## 主要功能说明

1. 文本智能分析
   - 使用NLTK进行文本分句
   - 通过DeepSeek AI优化分句结果
   - 保持每个短句的完整语义

2. 文献检索
   - 智能生成PubMed检索策略
   - 支持MeSH术语和Title/Abstract检索
   - 自动扩展检索范围以提高召回率

3. 文献筛选
   - 支持基于影响因子的筛选
   - 支持基于期刊分区的筛选
   - 文献相关性评分和排序

4. 结果展示
   - 展示文献详细信息
   - 提供PubMed原文链接
   - 生成带有文献引用的标注文本

## 注意事项

1. API密钥安全
   - 请妥善保管API密钥
   - 不要将包含API密钥的配置文件提交到版本控制系统

2. 使用限制
   - 请遵守PubMed API的使用限制
   - 建议添加适当的请求延迟以避免触发限制

3. 错误处理
   - 系统会自动记录错误日志
   - 检查logs目录下的日志文件以排查问题

## 开发说明

1. 项目结构
```
.
├── app.py              # 主应用文件
├── templates/          # HTML模板
│   ├── index.html     # 主页面模板
│   └── results.html   # 结果页面模板
├── static/            # 静态文件
│   ├── css/          # CSS样式文件
│   ├── js/           # JavaScript文件
│   └── images/       # 图片资源
├── utils/            # 工具函数
│   ├── pubmed.py     # PubMed API相关函数
│   ├── deepseek.py   # DeepSeek AI相关函数
│   └── filters.py    # 文献筛选相关函数
├── data/              # 数据文件
│   ├── journal_if.csv # 期刊影响因子数据
│   └── jcr_cas.csv   # JCR和中科院分区数据
├── logs/              # 日志文件
└── requirements.txt   # 依赖清单
```

2. 日志记录
   - 使用Python的logging模块
   - 日志文件按日期命名
   - 包含INFO、WARNING和ERROR级别的日志

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用学术许可证（Academic License）。

Academic License

Copyright (c) 2024 NNScholar

本软件及其相关文档（以下简称"软件"）仅供学术研究和教育目的使用。通过使用本软件，您同意以下条款：

1. 使用限制
   - 本软件仅限用于非商业性的学术研究和教育目的
   - 禁止将本软件用于任何商业用途
   - 禁止将本软件或其衍生作品进行销售或获取商业利益

2. 引用要求
   如果您在学术论文、报告或其他出版物中使用了本软件，请按以下格式引用：
   ```
   NNScholar. (2024). PubMed文献智能检索系统 [Computer software]. 
   https://github.com/luckylykkk/nnscholar-search
   ```

3. 免责声明
   本软件按"原样"提供，不提供任何明示或暗示的保证，包括但不限于对适销性、特定用途适用性和非侵权性的保证。在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任负责。

4. 衍生作品
   - 允许修改和改进本软件
   - 衍生作品必须同样采用本学术许可证
   - 衍生作品必须明确标注基于本软件开发并注明原始出处

5. 终止条款
   如果您违反本许可证的任何条款，您使用本软件的权利将自动终止。

6. 其他
   - 本许可证的解释和执行受中华人民共和国法律管辖
   - 本许可证的任何修改或豁免必须以书面形式作出并经双方签署

如有任何问题或需要额外授权，请联系：
- GitHub: https://github.com/luckylykkk
- Email: 599298622@qq.com
