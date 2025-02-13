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

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class JournalAnalyzer:
    def __init__(self):
        self.pubmed_api_key = os.getenv('PUBMED_API_KEY')
        self.base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
        self.journals = {
            'radiology': '0033-8419',
            'circulation_imaging': '1941-9651'
        }
        
    def fetch_journal_articles(self, journal_name, start_year, end_year):
        """获取指定期刊在给定年份范围内的所有文章"""
        logger.info(f"开始获取 {journal_name} {start_year}-{end_year} 的文章")
        
        issn = self.journals.get(journal_name)
        if not issn:
            logger.error(f"未找到期刊 {journal_name} 的ISSN")
            return []
            
        # 构建搜索查询
        query = f"{issn}[ta] AND ({start_year}[pdat]:{end_year}[pdat])"
        
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
            logger.info(f"找到 {len(id_list)} 篇文章")
            
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
            
    def analyze_hot_topics(self, articles, top_n=10):
        """分析热点研究方向"""
        # 合并所有文本
        all_text = ' '.join([
            f"{article['title']} {article['abstract']} {' '.join(article['keywords'])}"
            for article in articles
        ])
        
        # 分词和预处理
        tokens = word_tokenize(all_text.lower())
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        # 添加领域特定的停用词
        domain_stop_words = {
            'study', 'patient', 'result', 'method', 'conclusion', 'background',
            'objective', 'purpose', 'aim', 'material', 'method', 'results',
            'conclusions', 'findings', 'using', 'used', 'may', 'also', 'one',
            'two', 'three', 'first', 'second', 'third'
        }
        stop_words.update(domain_stop_words)
        
        # 词形还原和过滤
        processed_tokens = []
        for token in tokens:
            if (
                token.isalnum() and
                len(token) > 2 and
                token not in stop_words
            ):
                token = lemmatizer.lemmatize(token)
                processed_tokens.append(token)
                
        # 统计词频
        word_freq = Counter(processed_tokens)
        top_words = word_freq.most_common(top_n)
        
        return top_words
        
    def generate_heatmap(self, top_words, filename):
        """生成热点方向热图"""
        # 准备数据
        words = [word for word, _ in top_words]
        frequencies = [freq for _, freq in top_words]
        
        # 创建方形矩阵
        size = int(np.ceil(np.sqrt(len(words))))
        matrix = np.zeros((size, size))
        
        # 填充矩阵
        for i, freq in enumerate(frequencies):
            row = i // size
            col = i % size
            if row < size and col < size:
                matrix[row][col] = freq
                
        # 设置图形大小
        plt.figure(figsize=(12, 8))
        
        # 创建热图
        sns.heatmap(
            matrix,
            annot=[[words[i*size + j] if i*size + j < len(words) else '' 
                    for j in range(size)] for i in range(size)],
            fmt='',
            cmap='YlOrRd',
            cbar_kws={'label': 'Frequency'}
        )
        
        plt.title('Research Hot Topics Heatmap')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"热图已保存到 {filename}")

def main():
    analyzer = JournalAnalyzer()
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 获取两个期刊的文章
    all_articles = []
    for journal in ['radiology', 'circulation_imaging']:
        articles = analyzer.fetch_journal_articles(journal, '2024', '2025')
        all_articles.extend(articles)
        
        # 保存原始数据
        analyzer.save_to_file(
            articles,
            f'output/{journal}_2024_2025.json'
        )
        
    # 分析热点方向
    hot_topics = analyzer.analyze_hot_topics(all_articles)
    
    # 生成热图
    analyzer.generate_heatmap(
        hot_topics,
        'output/hot_topics_heatmap.png'
    )
    
    # 打印热点方向
    print("\n热点研究方向 (按频率排序):")
    for word, freq in hot_topics:
        print(f"{word}: {freq}")

if __name__ == '__main__':
    main() 