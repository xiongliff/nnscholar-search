// 执行搜索
async function executeSearch() {
    try {
        const query = document.getElementById('searchInput').value.trim();
        if (!query) {
            showToast('warning', '请输入搜索关键词');
            return;
        }

        // 显示加载动画
        showLoading();
        
        // 获取筛选条件
        const filters = getFilters();
        
        // 发送搜索请求
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                filters: filters,
                execute_search: true
            })
        });

        const result = await response.json();
        
        if (result.success) {
            // 显示结果
            displayResults(result.data);
            
            // 显示导出文件链接
            document.getElementById('exportLinks').style.display = 'block';
            document.getElementById('excelLink').href = `/exports/${result.export_path.split('/').pop()}`;
            document.getElementById('excelLink').textContent = result.export_path.split('/').pop();
            document.getElementById('reportLink').href = `/exports/${result.report_path.split('/').pop()}`;
            document.getElementById('reportLink').textContent = result.report_path.split('/').pop();
            document.getElementById('listLink').href = `/exports/${result.text_path.split('/').pop()}`;
            document.getElementById('listLink').textContent = result.text_path.split('/').pop();
            
            showToast('success', `找到 ${result.filtered_count} 篇相关文献，已自动导出并分析`);
        } else {
            showToast('error', result.error || '搜索失败');
        }
    } catch (error) {
        console.error('搜索出错:', error);
        showToast('error', '搜索过程中出现错误');
    } finally {
        hideLoading();
    }
}

// 在导出成功后保存文件路径
function saveExportPath(filePath) {
    localStorage.setItem('lastExportPath', filePath);
}

// 修改现有的导出函数，添加自动分析功能
async function exportToExcel() {
    try {
        showToast('info', '正在导出数据...');
        
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                papers: window.searchResults,
                query: document.getElementById('search-input').value
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            saveExportPath(result.data.file_path); // 保存导出文件路径
            
            // 自动开始分析
            showToast('info', '正在分析论文数据...');
            
            const analysisResponse = await fetch('/api/analyze-papers', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    file_path: result.data.file_path
                })
            });
            
            const analysisResult = await analysisResponse.json();
            
            if (analysisResult.success) {
                showToast('success', '导出和分析完成！');
                
                // 更新模态框内容
                document.getElementById('reportContent').textContent = analysisResult.data.report;
                document.getElementById('listContent').textContent = analysisResult.data.text_list;
                
                // 显示模态框
                const modal = new bootstrap.Modal(document.getElementById('analysisModal'));
                modal.show();
            } else {
                showToast('warning', '导出成功，但分析失败：' + analysisResult.error);
            }
        } else {
            showToast('error', '导出失败：' + result.error);
        }
    } catch (error) {
        console.error('导出失败:', error);
        showToast('error', `导出失败: ${error.message}`);
    }
} 