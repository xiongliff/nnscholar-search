from flask import Flask, request, jsonify, render_template
import requests
import json
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor
import time
from bs4 import BeautifulSoup
import re
import logging
from datetime import datetime
import traceback
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# 加载环境变量
load_dotenv()

# 创建应用实例
app = Flask(__name__, template_folder='templates')

# 创建必要的目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'app_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
PUBMED_API_KEY = os.getenv('PUBMED_API_KEY')
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# 加载PubMed专家提示词模板
PROMPT_PATH = os.path.join(BASE_DIR, 'templates', 'pubmed_expert_prompt.md')
with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
    EXPERT_PROMPT = f.read()

# 加载期刊数据
def load_journal_data():
    """加载期刊相关数据"""
    data_dir = os.path.join(BASE_DIR, 'data', 'journal_metrics')
    journal_data = {}
    if_trend_data = {}
    
    try:
        # 加载JCR和中科院分区数据
        jcr_file = os.path.join(data_dir, 'jcr_cas_ifqb.json')
        if not os.path.exists(jcr_file):
            logger.error(f"期刊数据文件不存在: {jcr_file}")
            return {}, {}
            
        logger.info(f"开始加载期刊数据文件: {jcr_file}")
        with open(jcr_file, 'r', encoding='utf-8') as f:
            try:
                journal_list = json.load(f)
                logger.info(f"成功加载期刊数据，包含 {len(journal_list)} 条记录")
                
                # 记录一些原始数据示例
                if len(journal_list) > 0:
                    sample_raw = journal_list[:3]
                    logger.info(f"原始数据示例: {json.dumps(sample_raw, ensure_ascii=False)}")
                
                for journal in journal_list:
                    # 处理ISSN和eISSN
                    issn = journal.get('issn', '').strip()
                    eissn = journal.get('eissn', '').strip()
                    
                    # 标准化ISSN格式（移除连字符）
                    issn = issn.replace('-', '') if issn else None
                    eissn = eissn.replace('-', '') if eissn else None
                    
                    # 使用所有可能的ISSN作为键
                    issns = [i for i in [issn, eissn] if i]
                    
                    if issns:
                        # 处理影响因子，确保是数值类型
                        impact_factor = journal.get('IF', 'N/A')
                        try:
                            if impact_factor != 'N/A':
                                impact_factor = float(impact_factor)
                        except (ValueError, TypeError):
                            impact_factor = 'N/A'
                            logger.warning(f"无效的影响因子值: {journal.get('IF')} for {journal.get('journal')}")
                        
                        journal_info = {
                            'title': journal.get('journal', ''),
                            'if': impact_factor,
                            'jcr_quartile': journal.get('Q', 'N/A'),
                            'cas_quartile': journal.get('B', 'N/A')
                        }
                        
                        # 为每个ISSN都存储期刊信息
                        for issn_key in issns:
                            journal_data[issn_key] = journal_info
                
                logger.info(f"成功加载 {len(journal_data)} 条期刊数据")
                # 记录一些转换后的数据示例
                if journal_data:
                    sample_converted = {k: journal_data[k] for k in list(journal_data.keys())[:3]}
                    logger.info(f"转换后的数据示例: {json.dumps(sample_converted, ensure_ascii=False)}")
            except json.JSONDecodeError as e:
                logger.error(f"期刊数据文件格式错误: {str(e)}")
                return {}, {}
        
        # 加载五年影响因子趋势数据
        trend_file = os.path.join(data_dir, '5year.json')
        if os.path.exists(trend_file):
            logger.info(f"开始加载影响因子趋势数据: {trend_file}")
            with open(trend_file, 'r', encoding='utf-8') as f:
                try:
                    if_trend_data = json.load(f)
                    if not isinstance(if_trend_data, dict):
                        logger.error("影响因子趋势数据格式错误：应为字典类型")
                        if_trend_data = {}
                    else:
                        logger.info(f"成功加载影响因子趋势数据，包含 {len(if_trend_data)} 条记录")
                except json.JSONDecodeError as e:
                    logger.error(f"影响因子趋势数据文件格式错误: {str(e)}")
                    if_trend_data = {}
        else:
            logger.warning(f"影响因子趋势数据文件不存在: {trend_file}")
        
        return journal_data, if_trend_data
        
    except Exception as e:
        logger.error(f"加载期刊数据失败: {str(e)}\n{traceback.format_exc()}")
        return {}, {}

# 全局变量
try:
    JOURNAL_DATA, IF_TREND_DATA = load_journal_data()
except Exception as e:
    logger.error(f"加载期刊数据失败: {str(e)}")
    JOURNAL_DATA, IF_TREND_DATA = {}, {}

def get_journal_metrics(issn):
    """获取期刊指标数据"""
    try:
        if not issn:
            logger.warning("ISSN为空")
            return None
            
        logger.info(f"开始获取期刊指标，ISSN: {issn}")
        
        if not isinstance(JOURNAL_DATA, dict):
            logger.error(f"期刊数据格式错误: {type(JOURNAL_DATA)}")
            return None
            
        if len(JOURNAL_DATA) == 0:
            logger.warning("期刊数据为空，请检查数据文件是否正确加载")
            return None
            
        # 标准化ISSN格式（移除连字符）
        issn = issn.replace('-', '')
        
        # 尝试直接获取
        journal_info = JOURNAL_DATA.get(issn)
        
        if not journal_info:
            # 尝试其他格式的ISSN
            issn_with_hyphen = f"{issn[:4]}-{issn[4:]}"
            journal_info = JOURNAL_DATA.get(issn_with_hyphen)
        
        if not journal_info:
            logger.warning(f"未找到ISSN对应的期刊信息: {issn}")
            return None
            
        logger.info(f"获取到的原始期刊信息: {json.dumps(journal_info, ensure_ascii=False)}")
        
        # 处理影响因子的显示格式
        impact_factor = journal_info.get('if', 'N/A')
        if isinstance(impact_factor, (int, float)):
            impact_factor = f"{impact_factor:.3f}"  # 格式化为三位小数
        
        metrics = {
            'title': journal_info.get('title', ''),
            'impact_factor': impact_factor,
            'jcr_quartile': journal_info.get('jcr_quartile', 'N/A'),
            'cas_quartile': journal_info.get('cas_quartile', 'N/A')
        }
        
        logger.info(f"处理后的期刊指标: {json.dumps(metrics, ensure_ascii=False)}")
        return metrics
        
    except Exception as e:
        logger.error(f"获取期刊指标时发生错误: {str(e)}\n{traceback.format_exc()}")
        return None

def get_if_trend(issn):
    """获取期刊近五年影响因子趋势"""
    if not issn or issn not in IF_TREND_DATA:
        return None
    
    trend_data = IF_TREND_DATA[issn]
    years = list(trend_data.keys())[-5:]
    ifs = [trend_data[year] for year in years]
    
    # 生成趋势图
    plt.figure(figsize=(8, 4))
    plt.plot(years, ifs, marker='o')
    plt.title('Impact Factor Trend (5 Years)')
    plt.xlabel('Year')
    plt.ylabel('Impact Factor')
    plt.grid(True)
    
    # 转换为base64图片
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode()

def filter_papers_by_metrics(papers, filters):
    """根据期刊指标筛选文献"""
    try:
        logger.info(f"开始筛选文献，筛选条件: {filters}")
        logger.info(f"待筛选文献数量: {len(papers)}")
        
        # 初始化统计信息
        stats = {
            'total': len(papers),
            'year_filtered': 0,
            'if_filtered': 0,
            'jcr_filtered': 0,
            'cas_filtered': 0,
            'final': 0
        }
        
        # 1. 年份筛选
        year_filtered = []
        if 'year_range' in filters and filters['year_range']:
            year_range = filters['year_range']
            logger.debug(f"应用年份筛选，范围: {year_range}")
            for paper in papers:
                pub_year = paper.get('pub_year')
                try:
                    pub_year = int(pub_year) if pub_year else None
                    logger.debug(f"文献 {paper.get('pmid')} 发表年份: {pub_year}")
                    if pub_year and year_range[0] <= pub_year <= year_range[1]:
                        year_filtered.append(paper)
                        logger.debug(f"文献 {paper.get('pmid')} 通过年份筛选")
                    else:
                        logger.debug(f"文献 {paper.get('pmid')} 未通过年份筛选: {pub_year} not in {year_range}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"年份格式错误: {pub_year}, 错误信息: {str(e)}")
        else:
            year_filtered = papers.copy()
            logger.debug("未设置年份筛选条件，保留所有文献")
        stats['year_filtered'] = len(year_filtered)
        logger.info(f"1. 年份筛选 ({filters.get('year_range', '无限制')}): {len(papers)} -> {len(year_filtered)}")
        
        # 如果年份筛选后没有文献，记录所有文献的年份信息以便调试
        if len(year_filtered) == 0:
            logger.warning("年份筛选后没有文献，记录所有文献的年份信息：")
            for paper in papers:
                pmid = paper.get('pmid', 'Unknown')
                pub_year = paper.get('pub_year')
                title = paper.get('title', '')[:100]
                logger.warning(f"PMID: {pmid}, 年份: {pub_year}, 标题: {title}")
        
        # 2. 影响因子筛选
        if_filtered = []
        if 'min_if' in filters and filters['min_if']:
            min_if = float(filters['min_if'])
            for paper in year_filtered:
                journal_info = paper.get('journal_info', {})
                impact_factor = journal_info.get('impact_factor', 'N/A')
                try:
                    if impact_factor != 'N/A':
                        if isinstance(impact_factor, str):
                            impact_factor = float(impact_factor.replace(',', ''))
                        if float(impact_factor) >= min_if:
                            if_filtered.append(paper)
                            logger.debug(f"文献 {paper.get('pmid')} 通过影响因子筛选: {impact_factor} >= {min_if}")
                        else:
                            logger.debug(f"文献 {paper.get('pmid')} 未通过影响因子筛选: {impact_factor} < {min_if}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"影响因子格式错误: {impact_factor}, 错误信息: {str(e)}")
        else:
            if_filtered = year_filtered.copy()
        stats['if_filtered'] = len(if_filtered)
        logger.info(f"2. 影响因子筛选 (>= {filters.get('min_if', '无限制')}): {len(year_filtered)} -> {len(if_filtered)}")
        
        # 3. JCR分区筛选
        jcr_filtered = []
        if 'jcr_quartile' in filters and filters['jcr_quartile']:
            for paper in if_filtered:
                journal_info = paper.get('journal_info', {})
                jcr_q = journal_info.get('jcr_quartile', 'N/A')
                if jcr_q != 'N/A' and jcr_q in filters['jcr_quartile']:
                    jcr_filtered.append(paper)
                    logger.debug(f"文献 {paper.get('pmid')} 通过JCR分区筛选: {jcr_q} in {filters['jcr_quartile']}")
                else:
                    logger.debug(f"文献 {paper.get('pmid')} 未通过JCR分区筛选: {jcr_q} not in {filters['jcr_quartile']}")
        else:
            jcr_filtered = if_filtered.copy()
        stats['jcr_filtered'] = len(jcr_filtered)
        logger.info(f"3. JCR分区筛选 ({filters.get('jcr_quartile', '无限制')}): {len(if_filtered)} -> {len(jcr_filtered)}")
        
        # 4. CAS分区筛选
        cas_filtered = []
        if 'cas_quartile' in filters and filters['cas_quartile']:
            cas_filters = [str(q) for q in filters['cas_quartile']]
            for paper in jcr_filtered:
                journal_info = paper.get('journal_info', {})
                cas_q = journal_info.get('cas_quartile', 'N/A')
                if cas_q != 'N/A':
                    if isinstance(cas_q, str) and cas_q.startswith('B'):
                        cas_q = cas_q[1:]  # 移除'B'前缀
                    if cas_q in cas_filters:
                        cas_filtered.append(paper)
                        logger.debug(f"文献 {paper.get('pmid')} 通过CAS分区筛选: {cas_q} in {cas_filters}")
                    else:
                        logger.debug(f"文献 {paper.get('pmid')} 未通过CAS分区筛选: {cas_q} not in {cas_filters}")
        else:
            cas_filtered = jcr_filtered.copy()
        stats['cas_filtered'] = len(cas_filtered)
        logger.info(f"4. CAS分区筛选 ({filters.get('cas_quartile', '无限制')}): {len(jcr_filtered)} -> {len(cas_filtered)}")
        
        # 按影响因子排序
        filtered_papers = sorted(
            cas_filtered,
            key=lambda x: float(x.get('journal_info', {}).get('impact_factor', '0').replace(',', ''))
            if isinstance(x.get('journal_info', {}).get('impact_factor'), str)
            else float(x.get('journal_info', {}).get('impact_factor', '0')),
            reverse=True
        )
        
        # 限制每句话的文献数量
        if 'papers_limit' in filters and filters['papers_limit']:
            try:
                limit = int(filters['papers_limit'])
                filtered_papers = filtered_papers[:limit]
                logger.info(f"5. 应用数量限制 ({limit}): {len(cas_filtered)} -> {len(filtered_papers)}")
            except ValueError as e:
                logger.error(f"转换papers_limit时出错: {str(e)}")
        
        stats['final'] = len(filtered_papers)
        
        # 输出详细的筛选统计信息
        logger.info("\n筛选过程统计:")
        logger.info(f"初始文献数量: {stats['total']}")
        logger.info(f"1. 年份筛选后: {stats['year_filtered']} 篇")
        logger.info(f"2. 影响因子筛选后: {stats['if_filtered']} 篇")
        logger.info(f"3. JCR分区筛选后: {stats['jcr_filtered']} 篇")
        logger.info(f"4. CAS分区筛选后: {stats['cas_filtered']} 篇")
        logger.info(f"5. 最终结果: {stats['final']} 篇")
        
        return filtered_papers, stats
        
    except Exception as e:
        logger.error(f"筛选文献时发生错误: {str(e)}\n{traceback.format_exc()}")
        raise

def handle_api_error(func):
    """API错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求错误: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': '外部API请求失败，请稍后重试'
            }), 503
        except Exception as e:
            logger.error(f"未预期的错误: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'status': 'error',
                'message': '服务器内部错误'
            }), 500
    wrapper.__name__ = func.__name__  # 保留原函数名
    return wrapper

def call_deepseek_api(prompt):
    """调用DeepSeek API进行文本处理"""
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'deepseek-chat',
        'messages': [
            {'role': 'system', 'content': EXPERT_PROMPT},
            {'role': 'user', 'content': prompt}
        ]
    }
    
    response = requests.post(
        'https://api.deepseek.com/v1/chat/completions',
        headers=headers,
        json=data
    )
    
    return response.json()['choices'][0]['message']['content']

def parse_pubmed_xml(xml_content):
    """解析PubMed XML响应"""
    logger.info("开始解析PubMed XML响应")
    soup = BeautifulSoup(xml_content, 'lxml')
    articles = []
    article_count = len(soup.find_all('pubmedarticle'))
    logger.info(f"找到 {article_count} 篇文章记录")
    
    for i, article in enumerate(soup.find_all('pubmedarticle'), 1):
        try:
            logger.info(f"开始解析第 {i}/{article_count} 篇文章")
            
            # 提取文章标题
            title = article.find('articletitle')
            title = title.text if title else 'No title available'
            logger.info(f"文章标题: {title[:100]}...")
            
            # 提取发表年份
            pub_date = article.find('pubdate')
            pub_year = None
            if pub_date and pub_date.find('year'):
                pub_year = pub_date.find('year').text
                logger.info(f"发表年份: {pub_year}")
            
            # 提取期刊信息
            journal = article.find('journal')
            journal_info = {}
            if journal:
                # 提取ISSN
                issn_elem = journal.find('issn')
                if issn_elem:
                    issn = issn_elem.text
                    logger.info(f"找到ISSN: {issn}")
                else:
                    issn = None
                    logger.warning("未找到ISSN")
                journal_info['issn'] = issn
                
                # 提取期刊标题
                journal_title = journal.find('title')
                journal_info['title'] = journal_title.text if journal_title else ''
                logger.info(f"期刊标题: {journal_info['title']}")
                
                # 获取期刊指标
                if issn:
                    logger.info(f"开始获取期刊 {issn} 的指标信息")
                    metrics = get_journal_metrics(issn)
                    if metrics:
                        logger.info(f"成功获取期刊指标: {metrics}")
                        journal_info.update(metrics)
                    else:
                        logger.warning(f"未能获取期刊 {issn} 的指标信息")
                
            # 构建文章数据
            article_data = {
                'title': title,
                'abstract': article.find('abstract').text if article.find('abstract') else 'No abstract available',
                'authors': [f"{author.find('lastname').text} {author.find('forename').text}" 
                          for author in article.find('authorlist').find_all('author') 
                          if author.find('lastname') and author.find('forename')] if article.find('authorlist') else [],
                'pub_date': (lambda d: f"{d.find('year').text if d.find('year') else ''} {d.find('month').text if d.find('month') else ''}".strip())(article.find('pubdate')) if article.find('pubdate') else 'Date not available',
                'pub_year': pub_year,  # 添加发表年份字段
                'pmid': article.find('pmid').text if article.find('pmid') else '',
                'url': f'https://pubmed.ncbi.nlm.nih.gov/{article.find("pmid").text}/' if article.find('pmid') else '#',
                'journal_info': journal_info,
                'journal_issn': journal_info.get('issn', '')
            }
            
            logger.info(f"文章数据构建完成，期刊信息: {journal_info}")
            articles.append(article_data)
            
        except Exception as e:
            logger.error(f"解析第 {i} 篇文章时发生错误: {str(e)}\n{traceback.format_exc()}")
            continue
    
    logger.info(f"完成XML解析，成功解析 {len(articles)}/{article_count} 篇文章")
    return articles

def calculate_relevance(sentence, paper):
    """计算文献与句子的相关性得分"""
    # 这里可以使用更复杂的相关性计算方法
    # 当前使用简单的关键词匹配
    keywords = re.findall(r'\w+', sentence.lower())
    title_abstract = (paper['title'] + ' ' + paper['abstract']).lower()
    
    matches = sum(1 for keyword in keywords if keyword in title_abstract)
    relevance = min(100, int((matches / len(keywords)) * 100))
    
    return relevance

def search_pubmed(query, filters=None):
    """搜索PubMed文献"""
    try:
        logger.info(f"开始PubMed搜索，原始文本: {query}")
        
        # 使用DeepSeek生成搜索策略
        prompt = """作为医学文献检索专家，请为以下研究内容生成简洁的PubMed检索策略。
        
研究内容：{query}

要求：
1. 只提取最核心的2-3个医学概念
2. 使用标准的PubMed检索语法
3. 使用[Title/Abstract]和[Mesh]标签
4. 使用AND连接主要概念
5. 直接输出检索语法，不要其他解释

示例输出：
"Coronary CT Angiography"[Title/Abstract] AND "Atherosclerotic Plaque"[Mesh]"""

        logger.info("调用DeepSeek生成检索策略...")
        search_strategy = call_deepseek_api(prompt.format(query=query))
        logger.info(f"生成的检索策略: {search_strategy}")
        
        # 构建搜索请求
        search_params = {
            'db': 'pubmed',
            'term': search_strategy,
            'retmax': '100',
            'retmode': 'json',
            'api_key': PUBMED_API_KEY
        }
        
        # 发送搜索请求
        search_url = f"{PUBMED_BASE_URL}esearch.fcgi"
        response = requests.get(search_url, params=search_params)
        
        if response.status_code != 200:
            logger.error(f"PubMed搜索请求失败: HTTP {response.status_code}")
            return []
            
        search_result = response.json()
        id_list = search_result.get('esearchresult', {}).get('idlist', [])
        
        if not id_list:
            logger.warning("未找到文献，尝试更宽泛的搜索策略")
            # 使用更宽泛的搜索策略
            broader_strategy = search_strategy.replace('[Title/Abstract]', '[All Fields]').replace('[Mesh]', '[All Fields]')
            logger.info(f"更宽泛的检索策略: {broader_strategy}")
            
            search_params['term'] = broader_strategy
            response = requests.get(search_url, params=search_params)
            
            if response.status_code == 200:
                search_result = response.json()
                id_list = search_result.get('esearchresult', {}).get('idlist', [])
        
        if not id_list:
            # 如果仍然没有结果，使用最基本的关键词搜索
            basic_terms = extract_basic_terms(query)
            basic_query = ' AND '.join(f'"{term}"[All Fields]' for term in basic_terms)
            logger.info(f"使用基本关键词搜索: {basic_query}")
            
            search_params['term'] = basic_query
            response = requests.get(search_url, params=search_params)
            
            if response.status_code == 200:
                search_result = response.json()
                id_list = search_result.get('esearchresult', {}).get('idlist', [])
        
        if not id_list:
            logger.warning("所有搜索策略均未找到文献")
            return []
            
        logger.info(f"找到 {len(id_list)} 篇文献")
        return fetch_paper_details(id_list)
        
    except Exception as e:
        logger.error(f"PubMed搜索过程中发生错误: {str(e)}\n{traceback.format_exc()}")
        return []

def extract_basic_terms(text):
    """提取基本关键词"""
    # 提取括号中的缩写
    abbreviations = re.findall(r'\(([A-Z]+)\)', text)
    
    # 提取主要医学术语
    key_terms = []
    medical_terms = [
        'Coronary computed tomography angiography',
        'CCTA',
        'atherosclerotic plaque',
        'coronary'
    ]
    
    for term in medical_terms:
        if term.lower() in text.lower():
            key_terms.append(term)
    
    # 合并缩写和关键词，限制数量
    all_terms = key_terms + abbreviations
    return list(set(all_terms))[:3]  # 最多返回3个关键词

def fetch_paper_details(id_list):
    """获取文献详细信息"""
    try:
        if not id_list:
            return []
            
        # 构建请求参数
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(id_list),
            'retmode': 'xml',
            'api_key': PUBMED_API_KEY
        }
        
        # 发送请求获取详情
        logger.info(f"获取文献详情，ID列表: {id_list}")
        fetch_url = f"{PUBMED_BASE_URL}efetch.fcgi"
        response = requests.get(fetch_url, params=fetch_params)
        
        if response.status_code != 200:
            logger.error(f"获取文献详情失败: HTTP {response.status_code}\n响应内容: {response.text}")
            return []
            
        # 解析XML响应
        try:
            soup = BeautifulSoup(response.content, 'xml')
            articles = soup.find_all('PubmedArticle')
            logger.info(f"解析到 {len(articles)} 篇文献")
            
            papers = []
            for article in articles:
                try:
                    # 提取文献信息
                    paper = extract_paper_info(article)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.error(f"解析单篇文献时出错: {str(e)}\n{traceback.format_exc()}")
                    continue
            
            logger.info(f"成功解析 {len(papers)} 篇文献的详细信息")
            return papers
            
        except Exception as e:
            logger.error(f"解析XML响应时出错: {str(e)}\n{traceback.format_exc()}")
            return []
            
    except Exception as e:
        logger.error(f"获取文献详情过程中发生错误: {str(e)}\n{traceback.format_exc()}")
        return []

def extract_paper_info(article):
    """从XML中提取文献信息"""
    try:
        # 提取基本信息
        pmid = article.find('PMID').text if article.find('PMID') else None
        if not pmid:
            logger.warning("文献缺少PMID，跳过")
            return None
            
        logger.info(f"开始处理文献 PMID: {pmid}")
            
        # 提取标题
        title_element = article.find('ArticleTitle')
        title = title_element.text if title_element else None
        if not title:
            logger.warning(f"文献 {pmid} 缺少标题，跳过")
            return None
            
        logger.info(f"文献标题: {title[:100]}...")
            
        # 提取期刊信息
        journal_element = article.find('Journal')
        if not journal_element:
            logger.warning(f"文献 {pmid} 缺少期刊信息，跳过")
            return None
            
        # 提取期刊标题 - 优先使用Title，如果没有则使用ISOAbbreviation
        journal_title = None
        journal_full = journal_element.find('Title')
        journal_iso = journal_element.find('ISOAbbreviation')
        
        logger.info(f"期刊信息提取详情:")
        logger.info(f"- 完整标题: {journal_full.text if journal_full else 'N/A'}")
        logger.info(f"- ISO缩写: {journal_iso.text if journal_iso else 'N/A'}")
        
        if journal_full and journal_full.text:
            journal_title = journal_full.text
            logger.info(f"使用完整期刊标题: {journal_title}")
        elif journal_iso and journal_iso.text:
            journal_title = journal_iso.text
            logger.info(f"使用期刊ISO缩写: {journal_title}")
            
        if not journal_title:
            logger.warning(f"文献 {pmid} 缺少期刊标题")
            return None
            
        issn = journal_element.find('ISSN').text if journal_element.find('ISSN') else None
        
        logger.info(f"期刊信息 - 标题: {journal_title}, ISSN: {issn}")
        
        if not issn:
            logger.warning(f"文献 {pmid} 缺少ISSN，跳过")
            return None
            
        # 提取发表年份
        pub_year = None
        pub_date = article.find('PubDate')
        
        logger.info(f"开始提取发表年份，PubDate标签内容: {pub_date}")
        
        if pub_date:
            # 尝试从Year标签提取
            year_elem = pub_date.find('Year')
            if year_elem and year_elem.text:
                pub_year = year_elem.text
                logger.info(f"从Year标签提取的年份: {pub_year}")
            else:
                # 如果没有Year标签，尝试从MedlineDate中提取
                medline_date = pub_date.find('MedlineDate')
                if medline_date and medline_date.text:
                    logger.info(f"找到MedlineDate: {medline_date.text}")
                    try:
                        # MedlineDate可能的格式: "2020 Dec-2021 Jan"
                        pub_year = medline_date.text.split()[0]  # 取第一个年份
                        logger.info(f"从MedlineDate提取的年份: {pub_year}")
                    except (IndexError, ValueError):
                        logger.warning(f"无法从MedlineDate解析年份: {medline_date.text}")
                else:
                    logger.warning("未找到Year标签或MedlineDate标签")
        else:
            logger.warning(f"文献 {pmid} 缺少PubDate标签")
        
        if not pub_year:
            logger.warning(f"文献 {pmid} 未能提取到发表年份")
            return None
            
        try:
            pub_year = int(pub_year)
            logger.info(f"成功将年份转换为整数: {pub_year}")
        except ValueError:
            logger.warning(f"文献 {pmid} 的发表年份格式无效: {pub_year}")
            return None
            
        # 获取期刊指标
        journal_metrics = get_journal_metrics(issn)
        if not journal_metrics:
            logger.warning(f"未找到期刊 {journal_title} (ISSN: {issn}) 的指标信息")
            # 如果没有找到期刊指标，仍然保留期刊基本信息
            journal_metrics = {
                'title': journal_title,
                'issn': issn,
                'impact_factor': 'N/A',
                'jcr_quartile': 'N/A',
                'cas_quartile': 'N/A'
            }
            logger.info("使用默认期刊指标信息")
        else:
            # 确保期刊标题使用从PubMed获取的标题
            journal_metrics['title'] = journal_title
            logger.info(f"获取到期刊指标:")
            logger.info(f"- 期刊标题: {journal_metrics['title']}")
            logger.info(f"- 影响因子: {journal_metrics['impact_factor']}")
            logger.info(f"- JCR分区: {journal_metrics['jcr_quartile']}")
            logger.info(f"- CAS分区: {journal_metrics['cas_quartile']}")
            
        # 构建文献信息
        paper_info = {
            'pmid': pmid,
            'title': title,
            'journal_info': journal_metrics,
            'pub_year': pub_year
        }
        
        logger.info(f"文献 {pmid} 的完整信息:")
        logger.info(f"- 标题: {title}")
        logger.info(f"- 发表年份: {pub_year}")
        logger.info(f"- 期刊标题: {journal_metrics['title']}")
        logger.info(f"- ISSN: {issn}")
        logger.info(f"- 影响因子: {journal_metrics['impact_factor']}")
        logger.info(f"- JCR分区: {journal_metrics['jcr_quartile']}")
        logger.info(f"- CAS分区: {journal_metrics['cas_quartile']}")
        
        # 提取摘要
        abstract_element = article.find('Abstract')
        if abstract_element:
            abstract_text = ' '.join(text.text for text in abstract_element.find_all('AbstractText'))
            paper_info['abstract'] = abstract_text
            logger.info(f"成功提取摘要，长度: {len(abstract_text)}")
        else:
            paper_info['abstract'] = 'No abstract available'
            logger.warning("未找到摘要信息")
            
        # 提取作者信息
        author_list = article.find('AuthorList')
        if author_list:
            authors = []
            for author in author_list.find_all('Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name and fore_name:
                    authors.append(f"{last_name.text} {fore_name.text}")
            paper_info['authors'] = authors
            logger.info(f"成功提取作者信息，作者数量: {len(authors)}")
            if authors:
                logger.info(f"第一作者: {authors[0]}")
                if len(authors) > 1:
                    logger.info(f"通讯作者: {authors[-1]}")
        else:
            paper_info['authors'] = []
            logger.warning("未找到作者信息")
            
        return paper_info
            
    except Exception as e:
        logger.error(f"解析文献信息时出错: {str(e)}\n{traceback.format_exc()}")
        return None

def process_sentence(sentence, filters=None):
    """处理单个句子，分解为短句并添加相关文献序号"""
    logger.info(f"开始处理长句: {sentence[:100]}...")
    try:
        # 首先获取长句的所有相关文献
        logger.info("获取长句的所有相关文献...")
        all_papers = search_pubmed(sentence)
        
        # 为所有文献计算相关性得分并排序
        for paper in all_papers:
            paper['relevance'] = calculate_relevance(sentence, paper)
        all_papers.sort(key=lambda x: x['relevance'], reverse=True)
        
        # 创建文献ID到序号的映射
        paper_index_map = {paper['pmid']: idx for idx, paper in enumerate(all_papers, 1)}
        logger.info(f"长句共检索到 {len(all_papers)} 篇文献")

        # 使用文本分析专家提示词分解长句
        split_prompt = f"""作为语言逻辑分析专家和文本优化顾问，请将以下长句分解为多个独立短句：

长句：{sentence}

要求：
1. 每个短句必须结构完整，包含明确的主语、谓语和宾语
2. 避免使用代词（如"它"、"这"、"那"）指代前文内容
3. 保持原句的核心信息和逻辑关系
4. 确保每个短句都能独立表达完整的意思
5. 按序号列出短句"""

        logger.info("调用DeepSeek分解长句...")
        split_response = call_deepseek_api(split_prompt)
        logger.info(f"分解结果: {split_response}")

        # 解析短句列表
        short_sentences = []
        for line in split_response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                sentence = re.sub(r'^\d+\.\s*', '', line).strip()
                if not sentence.endswith('。'):
                    sentence += '。'
                if sentence:
                    short_sentences.append(sentence)

        logger.info(f"共分解出 {len(short_sentences)} 个短句")

        # 处理每个短句
        results = []
        for idx, short_sent in enumerate(short_sentences, 1):
            logger.info(f"处理第 {idx} 个短句: {short_sent}")

            # 生成检索策略
            search_prompt = f"""作为PubMed搜索专家，请为以下医学短句生成检索策略：

医学短句：{short_sent}

请按以下格式提供：

1. 核心医学概念：
- [列出2-3个最重要的医学概念，每个概念一行]

2. MeSH主题词：
- [列出对应的MeSH术语，每个术语一行]

3. 检索策略：
主策略：[使用MeSH Terms的精确检索策略]
补充策略：[使用Title/Abstract的扩展检索策略]"""

            search_strategy = call_deepseek_api(search_prompt)
            
            # 解析检索策略并搜索文献
            papers = search_pubmed(short_sent)
            
            # 计算相关性并排序
            for paper in papers:
                paper['relevance'] = calculate_relevance(short_sent, paper)
            papers.sort(key=lambda x: x['relevance'], reverse=True)
            
            # 获取文献在原始长句文献列表中的序号
            paper_indices = []
            for paper in papers[:10]:  # 只取相关性最高的前10篇
                if paper['pmid'] in paper_index_map:
                    paper_indices.append(paper_index_map[paper['pmid']])
            
            # 格式化文献序号列表
            indices_str = ', '.join(str(i) for i in sorted(paper_indices))
            
            # 构建带有文献序号的短句
            annotated_sentence = f"{short_sent} [相关文献序号：{indices_str}]"
            
            results.append({
                'short_sentence': short_sent,
                'annotated_sentence': annotated_sentence,
                'paper_indices': paper_indices,
                'papers': papers
            })

        final_result = {
            'original_sentence': sentence,
            'all_papers': all_papers,
            'short_sentences': results
        }

        # 生成格式化输出
        formatted_output = [
            "原始长句：",
            sentence,
            "\n相关文献列表：",
        ]
        
        # 添加排序后的文献列表
        for idx, paper in enumerate(all_papers, 1):
            formatted_output.append(f"{idx}. {paper.get('title', 'No title')} (PMID: {paper['pmid']})")
        
        formatted_output.append("\n分解后的短句及其相关文献：")
        for result in results:
            formatted_output.append(result['annotated_sentence'])

        final_result['formatted_output'] = '\n'.join(formatted_output)
        logger.info("长句处理完成")
        return final_result

    except Exception as e:
        logger.error(f"处理长句时发生错误: {str(e)}\n{traceback.format_exc()}")
        return {
            'original_sentence': sentence,
            'all_papers': [],
            'short_sentences': [],
            'formatted_output': f"处理过程中发生错误: {str(e)}"
        }

def generate_broader_query(query):
    """生成更宽泛的搜索策略"""
    try:
        # 移除一些限制性标签
        broader = query.replace('[Title/Abstract]', '[All Fields]')
        broader = broader.replace('[Mesh]', '[All Fields]')
        
        # 如果包含多个AND条件，只保留部分
        terms = broader.split(' AND ')
        if len(terms) > 2:
            # 保留最重要的2-3个术语
            terms = terms[:3]
            broader = ' AND '.join(terms)
        
        logger.info(f"原始查询: {query}")
        logger.info(f"更宽泛的查询: {broader}")
        return broader
        
    except Exception as e:
        logger.error(f"生成更宽泛查询时出错: {str(e)}")
        return query  # 出错时返回原始查询

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.json
        text = data.get('text', '').strip()
        filters = data.get('filters', {})
        
        if not text:
            logger.warning("接收到空文本输入")
            return jsonify({'error': '请输入要搜索的文本'}), 400
            
        try:
            # 分句
            sentences = split_sentences(text)
            logger.info(f"将文本分成{len(sentences)}句")
            
            all_results = []
            for i, sentence in enumerate(sentences, 1):
                try:
                    logger.info(f"开始处理第{i}句: {sentence}")
                    
                    # 搜索PubMed
                    papers = search_pubmed(sentence)
                    logger.info(f"第{i}句初始检索到 {len(papers)} 篇文献")
                    
                    # 检查年份提取情况
                    logger.info(f"检查文献年份提取情况:")
                    papers_with_year = []
                    for paper in papers:
                        pub_year = paper.get('pub_year')
                        pmid = paper.get('pmid', 'Unknown')
                        title = paper.get('title', '')[:100]
                        journal_info = paper.get('journal_info', {})
                        if pub_year:
                            logger.info(f"文献年份提取成功 - PMID: {pmid}, 年份: {pub_year}, 标题: {title}")
                            logger.info(f"期刊信息 - 影响因子: {journal_info.get('impact_factor', 'N/A')}, JCR: {journal_info.get('jcr_quartile', 'N/A')}, CAS: {journal_info.get('cas_quartile', 'N/A')}")
                            papers_with_year.append(paper)
                        else:
                            logger.warning(f"文献年份提取失败 - PMID: {pmid}, 标题: {title}")
                    
                    if not papers_with_year:
                        logger.warning(f"第{i}句所有文献都未提取到年份")
                        continue
                        
                    logger.info(f"第{i}句找到{len(papers_with_year)}篇有年份信息的文献")
                    
                    # 应用过滤条件
                    if filters:
                        logger.info(f"开始应用过滤条件: {filters}")
                        papers, stats = filter_papers_by_metrics(papers_with_year, filters)
                        logger.info(f"过滤后剩余{len(papers)}篇文献")
                        logger.info("过滤后的文献详细信息:")
                        for paper in papers:
                            pmid = paper.get('pmid', 'Unknown')
                            title = paper.get('title', '')[:100]
                            journal_info = paper.get('journal_info', {})
                            logger.info(f"- PMID: {pmid}")
                            logger.info(f"  标题: {title}")
                            logger.info(f"  发表年份: {paper.get('pub_year', 'N/A')}")
                            logger.info(f"  影响因子: {journal_info.get('impact_factor', 'N/A')}")
                            logger.info(f"  JCR分区: {journal_info.get('jcr_quartile', 'N/A')}")
                            logger.info(f"  CAS分区: {journal_info.get('cas_quartile', 'N/A')}")
                    else:
                        papers = papers_with_year
                        stats = {
                            'total': len(papers_with_year),
                            'year_filtered': len(papers_with_year),
                            'if_filtered': len(papers_with_year),
                            'jcr_filtered': len(papers_with_year),
                            'cas_filtered': len(papers_with_year),
                            'final': len(papers_with_year)
                        }
                        logger.info("未应用过滤条件，保留所有文献")
                    
                    # 添加到结果中
                    result = {
                        'sentence': sentence,
                        'papers': papers,
                        'stats': stats
                    }
                    all_results.append(result)
                    logger.info(f"第{i}句处理完成，添加到结果列表")
                    
                except Exception as e:
                    logger.error(f"处理第{i}句时出错: {str(e)}\n{traceback.format_exc()}")
                    continue
            
            if not all_results:
                logger.warning("未找到任何符合条件的文献")
                return jsonify({'error': '未找到符合条件的文献'}), 404
                
            logger.info(f"搜索完成，共处理{len(sentences)}句，找到结果的句子数：{len(all_results)}")
            logger.info("准备返回结果到前端")
            
            # 检查结果格式
            response_data = {'results': all_results}
            logger.info(f"响应数据大小: {len(str(response_data))} 字节")
            logger.info("响应数据示例:")
            if all_results:
                first_result = all_results[0]
                logger.info(f"第一句: {first_result['sentence']}")
                logger.info(f"文献数量: {len(first_result['papers'])}")
                if first_result['papers']:
                    first_paper = first_result['papers'][0]
                    logger.info(f"第一篇文献 - PMID: {first_paper.get('pmid')}, 标题: {first_paper.get('title', '')[:100]}")
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"处理文本时发生错误: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': '处理文本时发生错误'}), 500
            
    except Exception as e:
        logger.error(f"处理搜索请求时发生错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': '服务器内部错误'}), 500

@app.route('/api/filter', methods=['POST'])
@handle_api_error
def filter_papers():  # 修改函数名，避免与装饰器冲突
    """根据期刊指标筛选文献"""
    data = request.json
    papers = data.get('papers', [])
    filters = data.get('filters', {})
    
    if not papers:
        return jsonify({
            'status': 'error',
            'message': '请提供要筛选的文献列表'
        }), 400
    
    try:
        filtered_papers, stats = filter_papers_by_metrics(papers, filters)
        
        return jsonify({
            'status': 'success',
            'papers': filtered_papers,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"筛选文献时发生错误: {str(e)}\n{traceback.format_exc()}")
        raise

def split_sentences(text):
    """将文本分割成句子"""
    try:
        sentences = []
        
        # 处理英文句子
        # 1. 先将句号+空格的模式替换为特殊标记，避免被括号内的句号干扰
        text = re.sub(r'\.\s+([A-Z])', r'SENT_BREAK\1', text)
        
        # 2. 按特殊标记分割
        temp_sentences = text.split('SENT_BREAK')
        
        # 3. 处理每个分割后的句子
        for sent in temp_sentences:
            sent = sent.strip()
            if sent:
                # 如果句子末尾没有标点，添加句号
                if not sent[-1] in '.!?。！？':
                    sent += '.'
                sentences.append(sent)
        
        # 如果没有找到句子，将整个文本作为一个句子
        if not sentences and text.strip():
            sentences = [text.strip()]
        
        logger.info(f"文本分句结果: {len(sentences)} 句")
        for i, s in enumerate(sentences, 1):
            logger.debug(f"第 {i} 句: {s}")
        
        return sentences
        
    except Exception as e:
        logger.error(f"文本分句时出错: {str(e)}\n{traceback.format_exc()}")
        return [text.strip()] if text.strip() else []

if __name__ == '__main__':
    try:
        # 下载NLTK数据
        print("正在下载NLTK数据...")
        nltk.download('punkt', quiet=True)
        print("NLTK数据下载完成")
        
        # 检查必要的配置
        if not DEEPSEEK_API_KEY:
            raise ValueError("未设置DEEPSEEK_API_KEY")
        if not PUBMED_API_KEY:
            raise ValueError("未设置PUBMED_API_KEY")
        
        print("正在启动应用服务器...")
        # 启动应用
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"启动失败: {str(e)}")
        logger.error(f"启动失败: {str(e)}\n{traceback.format_exc()}")
        raise 