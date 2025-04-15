PATH_TO_DATASETS    = 'Datasets/'
PATH_TO_GRAPHS      = 'Graphs/'
PATH_TO_MODELS      = 'Output/Models/'
PATH_TO_RANKINGS    = 'Output/MyRankings/'
PATH_TO_REPORTS     = 'Output/MyReports/'
PATH_TO_IMAGES      = 'Output/MyImages/'
PATH_TO_METRICS     = 'Output/MyMetrics/'

import os

# 自动创建输出目录
for path in [PATH_TO_MODELS, PATH_TO_RANKINGS, PATH_TO_REPORTS, PATH_TO_IMAGES, PATH_TO_METRICS]:
    os.makedirs(path, exist_ok=True)