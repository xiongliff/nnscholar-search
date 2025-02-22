import requests
import json
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from bs4 import BeautifulSoup
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import traceback

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 在文件开头，全局初始化NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("正在下载NLTK停用词数据...")
    nltk.download('stopwords', quiet=True)

# 预先加载停用词
try:
    ENGLISH_STOP_WORDS = set(stopwords.words('english'))
    # 添加医学文献常见的停用词
    MEDICAL_STOP_WORDS = {
        'study', 'patient', 'result', 'method', 'conclusion',
        'background', 'objective', 'purpose', 'material', 'materials',
        'analysis', 'data', 'using', 'used', 'show', 'showed',
        'significant', 'significantly', 'found', 'may', 'also',
        'one', 'two', 'three', 'well', 'within', 'without'
    }
    ENGLISH_STOP_WORDS.update(MEDICAL_STOP_WORDS)
    logger.info(f"成功加载 {len(ENGLISH_STOP_WORDS)} 个停用词")
except Exception as e:
    logger.warning(f"加载NLTK停用词失败，使用基本停用词列表: {str(e)}")
    ENGLISH_STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over',
        'after'
    }

class JournalAnalyzer:
    def __init__(self):
        logger.info("开始初始化JournalAnalyzer...")
        
        self.pubmed_api_key = os.getenv('PUBMED_API_KEY')
        self.base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
        self.journals = {
            'radiology': '0033-8419',
            'circulation_imaging': '1941-9651'
        }
        
        # 直接使用全局停用词
        self.stop_words = ENGLISH_STOP_WORDS
        logger.info(f"使用 {len(self.stop_words)} 个停用词")
        
        logger.info("JournalAnalyzer初始化完成")
        
    def fetch_journal_articles(self, query):
        """获取指定检索策略的所有文章
        
        Args:
            query (str): 完整的PubMed检索策略
            
        Returns:
            list: 文章列表
        """
        logger.info(f"开始获取文章，检索策略: {query}")
        
        # 首先获取文章ID列表
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': '10000',  # 获取最大数量的结果
            'retmode': 'json',
            'api_key': self.pubmed_api_key
        }
        
        try:
            response = requests.get(f"{self.base_url}esearch.fcgi", params=search_params)
            response.raise_for_status()
            search_result = response.json()
            
            id_list = search_result['esearchresult']['idlist']
            logger.info(f"检索到 {len(id_list)} 篇文章")
            
            if not id_list:
                return []
                
            # 分批获取文章详情
            articles = []
            batch_size = 100
            
            for i in range(0, len(id_list), batch_size):
                batch_ids = id_list[i:i+batch_size]
                batch_articles = self._fetch_article_details(batch_ids)
                articles.extend(batch_articles)
                logger.info(f"已处理 {len(articles)}/{len(id_list)} 篇文章")
                time.sleep(0.5)  # 避免请求过于频繁
                
            return articles
            
        except Exception as e:
            logger.error(f"获取文章时出错: {str(e)}")
            return []
            
    def _fetch_article_details(self, id_list):
        """获取文章详细信息"""
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(id_list),
            'retmode': 'xml',
            'api_key': self.pubmed_api_key
        }
        
        try:
            response = requests.get(f"{self.base_url}efetch.fcgi", params=fetch_params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            articles = []
            
            for article in soup.find_all('PubmedArticle'):
                try:
                    # 提取文章信息
                    title = article.find('ArticleTitle').text if article.find('ArticleTitle') else ''
                    abstract = ' '.join(text.text for text in article.find_all('AbstractText')) if article.find('Abstract') else ''
                    
                    # 提取作者信息
                    authors = []
                    author_list = article.find('AuthorList')
                    if author_list:
                        for author in author_list.find_all('Author'):
                            last_name = author.find('LastName').text if author.find('LastName') else ''
                            fore_name = author.find('ForeName').text if author.find('ForeName') else ''
                            authors.append(f"{last_name} {fore_name}".strip())
                            
                    # 提取关键词
                    keywords = []
                    keyword_list = article.find('KeywordList')
                    if keyword_list:
                        keywords = [k.text for k in keyword_list.find_all('Keyword')]
                        
                    # 提取发表日期
                    pub_date = article.find('PubDate')
                    year = pub_date.find('Year').text if pub_date and pub_date.find('Year') else ''
                    
                    articles.append({
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'keywords': keywords,
                        'year': year
                    })
                    
                except Exception as e:
                    logger.error(f"解析文章时出错: {str(e)}")
                    continue
                    
            return articles
            
        except Exception as e:
            logger.error(f"获取文章详情时出错: {str(e)}")
            return []
            
    def save_to_file(self, articles, filename):
        """保存文章信息到JSON文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            logger.info(f"文章信息已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存文件时出错: {str(e)}")
            
    def analyze_hot_topics(self, articles):
        """分析热点主题，返回更详细的主题信息"""
        try:
            # 检查stop_words是否正确初始化
            if not hasattr(self, 'stop_words'):
                logger.error("stop_words属性未初始化")
                self.stop_words = set()
            elif not isinstance(self.stop_words, set):
                logger.error(f"stop_words类型错误: {type(self.stop_words)}")
                self.stop_words = set(self.stop_words)
            
            logger.info(f"使用 {len(self.stop_words)} 个停用词进行分析")
            
            # 收集所有文章的标题和摘要
            texts = []
            for article in articles:
                # 组合标题和摘要
                text_parts = []
                if article.get('title'):
                    text_parts.append(article['title'])
                if article.get('abstract'):
                    text_parts.append(article['abstract'])
                
                combined_text = ' '.join(text_parts)
                if combined_text.strip():
                    texts.append(combined_text.lower())
            
            if not texts:
                logger.warning("没有可分析的文本")
                return []
            
            # 文本预处理
            processed_texts = []
            for text in texts:
                # 移除标点符号和数字
                text = re.sub(r'[^\w\s-]', ' ', text)
                text = re.sub(r'\d+', ' ', text)
                # 移除停用词
                words = [word.strip() for word in text.split() 
                        if word.strip() and len(word) > 2 
                        and word not in self.stop_words]
                if words:
                    processed_texts.append(' '.join(words))
            
            if not processed_texts:
                logger.warning("预处理后没有可分析的文本")
                return []
            
            # 使用TF-IDF提取关键词
            vectorizer = TfidfVectorizer(
                max_features=500,  # 提取更多特征词
                ngram_range=(1, 4),  # 支持1-4个词的组合
                stop_words='english'
            )
            
            # 转换文本
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 计算TF-IDF得分
            tfidf_sums = tfidf_matrix.sum(axis=0).A1
            word_scores = list(zip(feature_names, tfidf_sums))
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 为每个主题构建详细信息
            enriched_topics = []
            for word, score in word_scores[:200]:  # 分析前200个主题
                # 基础统计
                topic_articles = []
                for idx, text in enumerate(processed_texts):
                    if word in text:
                        topic_articles.append(idx)
                
                article_count = len(topic_articles)
                if article_count < 2:  # 忽略出现次数过少的主题
                    continue
                    
                # 年份分布分析
                year_distribution = {}
                for idx in topic_articles:
                    year = articles[idx].get('year')
                    if year:
                        year_distribution[year] = year_distribution.get(year, 0) + 1
                
                # 计算年度趋势
                trend = 'stable'
                if year_distribution:
                    years_list = sorted(year_distribution.keys())
                    if len(years_list) > 1:
                        first_year_count = year_distribution[years_list[0]]
                        last_year_count = year_distribution[years_list[-1]]
                        if last_year_count > first_year_count * 1.5:
                            trend = 'rising'
                        elif last_year_count < first_year_count * 0.5:
                            trend = 'declining'
                
                # 构建主题详细信息
                topic_info = {
                    'topic': word,
                    'score': float(score),
                    'article_count': article_count,
                    'coverage_percentage': (article_count / len(articles)) * 100,
                    'year_distribution': dict(sorted(year_distribution.items())),
                    'trend': trend,
                    'example_articles': [
                        {
                            'title': articles[idx].get('title', ''),
                            'year': articles[idx].get('year'),
                            'journal': articles[idx].get('journal_info', {}).get('title')
                        }
                        for idx in topic_articles[:3]  # 只取前3个示例
                    ]
                }
                
                enriched_topics.append(topic_info)
            
            # 按重要性得分排序
            enriched_topics.sort(key=lambda x: x['score'], reverse=True)
            
            return enriched_topics
            
        except Exception as e:
            logger.error(f"分析热点主题时出错: {str(e)}\n{traceback.format_exc()}")
            return []
        
    def analyze_hot_authors(self, articles, top_n=10):
        """分析热点作者"""
        try:
            # 统计作者出现次数
            author_freq = Counter()
            author_articles_map = {}
            
            # 遍历所有文章
            for article in articles:
                # 获取作者列表
                authors = article.get('authors', [])
                if not authors:
                    continue
                    
                # 更新作者频率
                author_freq.update(authors)
                
                # 为每个作者收集文章信息
                for author in authors:
                    if author not in author_articles_map:
                        author_articles_map[author] = []
                    author_articles_map[author].append(article)
            
            # 获取出现次数最多的作者
            top_authors = author_freq.most_common(top_n)
            
            # 转换为更详细的格式
            author_stats = []
            for author, count in top_authors:
                # 获取该作者的所有文章
                author_articles = author_articles_map.get(author, [])
                
                # 计算作者的第一作者和通讯作者次数
                first_author_count = sum(1 for article in author_articles 
                                      if article.get('authors') and article['authors'][0] == author)
                corresponding_author_count = sum(1 for article in author_articles 
                                              if article.get('authors') and article['authors'][-1] == author)
                
                # 收集发表年份
                years = sorted(set(article.get('year') for article in author_articles 
                                 if article.get('year')))
                
                author_stats.append({
                    'name': author,
                    'total_papers': count,
                    'first_author': first_author_count,
                    'corresponding_author': corresponding_author_count,
                    'years': years
                })
            
            logger.info(f"分析完成，找到 {len(author_stats)} 位热点作者")
            return author_stats
            
        except Exception as e:
            logger.error(f"分析热点作者时出错: {str(e)}\n{traceback.format_exc()}")
            return []
            
    def generate_heatmap(self, top_words, filename):
        """生成热点方向热图"""
        try:
            # 使用 Agg backend 避免线程问题
            import matplotlib
            matplotlib.use('Agg')
            
            if not top_words:
                logger.warning("没有数据用于生成热力图")
                return False
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 准备数据
            topics = [item['topic'] for item in top_words]
            scores = [item['score'] for item in top_words]
            
            if not topics or not scores:
                logger.warning("热力图数据为空")
                return False
            
            # 创建方形矩阵
            size = int(np.ceil(np.sqrt(len(topics))))
            matrix = np.zeros((size, size))
            
            # 填充矩阵
            for i, score in enumerate(scores):
                if i >= size * size:  # 防止超出矩阵范围
                    break
                row = i // size
                col = i % size
                matrix[row][col] = score
            
            # 设置图形大小
            plt.figure(figsize=(12, 8))
            
            # 创建热图
            sns.heatmap(
                matrix,
                annot=[[topics[i*size + j] if i*size + j < len(topics) else '' 
                        for j in range(size)] for i in range(size)],
                fmt='',
                cmap='YlOrRd',
                cbar_kws={'label': 'Score'}
            )
            
            plt.title('Research Hot Topics Heatmap')
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"热图已保存到 {filename}")
            return True
            
        except Exception as e:
            logger.error(f"生成热图时出错: {str(e)}\n{traceback.format_exc()}")
            return False

def main():
    analyzer = JournalAnalyzer()
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 获取两个期刊的文章
    all_articles = []
    for journal in ['radiology', 'circulation_imaging']:
        articles = analyzer.fetch_journal_articles(f"{journal}[ta] AND (2024[pdat]:2025[pdat])")
        all_articles.extend(articles)
        
        # 保存原始数据
        analyzer.save_to_file(
            articles,
            f'output/{journal}_2024_2025.json'
        )
        
    # 分析热点方向
    hot_topics = analyzer.analyze_hot_topics(all_articles)
    
    # 分析热点作者
    hot_authors = analyzer.analyze_hot_authors(all_articles)
    
    # 生成热图
    analyzer.generate_heatmap(
        hot_topics,
        'output/hot_topics_heatmap.png'
    )
    
    # 打印热点方向
    print("\n热点研究方向 (按频率排序):")
    for topic in hot_topics:
        print(f"Topic: {topic['topic']}")
        print(f"Score: {topic['score']}")
        print(f"Article Count: {topic['article_count']}")
        print(f"Coverage Percentage: {topic['coverage_percentage']:.2f}%")
        print(f"Year Distribution: {topic['year_distribution']}")
        print(f"Trend: {topic['trend']}")
        print("Example Articles:")
        for example in topic['example_articles']:
            print(f"  - Title: {example['title']}, Year: {example['year']}, Journal: {example['journal']}")
        print()

if __name__ == '__main__':
    main() 