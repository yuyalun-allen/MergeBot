# Use this script with scrapy framework
# Putting it to scrapy_project/scrapy_project/spiders/merge_nature.py

import scrapy

class MergeNatureSpider(scrapy.Spider):
    name = "merge_nature"
    start_urls = ['https://merge-nature.netlify.app/']

    def parse(self, response):
        # 提取项目名称和链接
        projects = response.css('table tbody tr')
        for project in projects:
            project_name = project.css('td a::text').get()
            project_url = project.css('td a::attr(href)').get()
            if project_name and project_url:
                yield response.follow(project_url, self.parse_commits, meta={'project_name': project_name})

    def parse_commits(self, response):
        project_name = response.meta['project_name']
        # 提取 commit SHA 和链接
        commits = response.css('table tbody tr')
        for commit in commits:
            commit_sha = commit.css('td a::text').get()
            commit_url = commit.css('td a::attr(href)').get()
            if commit_sha and commit_url:
                yield response.follow(commit_url, self.parse_chunks, meta={'project_name': project_name, 'commit_sha': commit_sha})

    def parse_chunks(self, response):
        project_name = response.meta['project_name']
        commit_sha = response.meta['commit_sha']
        chunks_data = []

        # 找到所有 chunk 表格的索引
        chunk_tables = response.css('table[id="table"]')
        
        for index, _ in enumerate(chunk_tables):
            # 找到对应的 conflicting content 和 solution content 表格
            conflicting_table = response.css(f'table#tableConflict')[index]
            solution_table = response.css(f'table#tableSolution')[index]

            # 提取冲突内容
            conflicting_content = conflicting_table.css('tbody tr td[data-title="ConflictingContent"] pre::text').get()
            
            # 提取解决方案内容
            solution_content = solution_table.css('tbody tr td[data-title="SolutionContent"] pre::text').get()

            if conflicting_content and solution_content:
                chunks_data.append({
                    'conflict_content': conflicting_content,
                    'solution_content': solution_content
                })
        if len(chunks_data) != 0:
            data = {
                'project_name': project_name,
                'commit_sha': commit_sha,
                'chunk': chunks_data
            }
            yield data
