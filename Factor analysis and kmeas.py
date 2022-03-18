import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
#1.数据读取，筛选需要的字段
data_analysis=pd.read_excel('新能源- 给淘宝0730.xlsx',engine='openpyxl')
data=data_analysis[data_analysis.columns[1:]]
data=data.set_index('证券简称')
data.dropna(inplace=True)
#2.相关性分析
#查看各字段数据类型，缺失值
data.info()
#描述统计
data.describe()
#计算协方差
data.cov()
#计算相关系数
data.corr()
#画热力图，数值为两个变量之间的相关系数
plt.figure(figsize=(15,15))
sns.heatmap(data.corr())
plt.savefig('corr.png')
# 数据归一化
from sklearn.preprocessing import StandardScaler,MinMaxScaler  # 数据归一化
datas = StandardScaler().fit_transform(data)  # 训练模型并转换数据
datas = pd.DataFrame(datas,columns=data.columns)
datas.index = data.index
#########################相关性检验
#3.充分性检验和相关性检验
df_model = datas.copy()
#充分性检测 p值要小于0.05
print('巴特利球形度检验')
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df_model)
print('卡方值：',chi_square_value,'P值', p_value)
#相关性检验 kmo要大于0.6
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df_model)
print('KMO检验：',kmo_model)

########################因子分析
#4.做碎石图，确定因子个数
fa = FactorAnalyzer()
# fa = FactorAnalyzer(rotation='varimax',method='principal',impute='mean')
fa.fit(df_model)
ev, v = fa.get_eigenvalues()
print('相关矩阵特征值：',ev)
#Create scree plot using matplotlib
plt.figure(figsize=(8, 6.5))
plt.scatter(range(1,datas.shape[1]+1),ev)
plt.plot(range(1,datas.shape[1]+1),ev)
plt.title('碎石图',fontdict={'weight':'normal','size': 25})
plt.xlabel('因子',fontdict={'weight':'normal','size': 15})
plt.ylabel('特征值',fontdict={'weight':'normal','size': 15})
plt.grid()
plt.savefig('碎石图.png')
plt.show()
#5.根据确定的因子个数进行5个因子分析，获取方差贡献率和因子得分
#取旋转后的结果
n = 5
# fa2 = FactorAnalyzer(n,method='principal')
fa2 = FactorAnalyzer(n,rotation='varimax',method='principal')
fa2.fit(df_model)
#给出贡献率
var = fa2.get_factor_variance()
#计算因子得分
fa2_score = fa2.transform(datas)
#得分表
column_list = ['F'+str(i) for i in np.arange(n)+1]
fa_score = pd.DataFrame(fa2_score,columns=column_list)
fa_score.index = data.index
print("\n各因子得分:\n",fa_score)
#旋转后因子载荷矩阵
fa2_loadings = pd.DataFrame(fa2.loadings_,index=df_model.columns,columns=column_list)
print("\n特殊因子方差:\n", fa.get_communalities())
fa2_loadings['特殊方差'] = pd.Series(data=fa.get_communalities(), index=df_model.columns)
fa2_loadings.to_csv('因子载荷矩阵.csv')
j = pd.read_csv('因子载荷矩阵.csv')
# plt.figure(figsize=(15,15))
# sns.heatmap(fa2_loadings)
#方差贡献表
df_fv = pd.DataFrame()
df_fv['因子'] = column_list
df_fv['方差贡献'] = var[1]
df_fv['累计方差贡献'] = var[2]
df_fv['累计方差贡献占比'] = var[1]/var[1].sum()
print("\n方差贡献表:\n",df_fv)
#6.计算综合得分及排名(综合得分=累计方差贡献占比1 * 因子得分1 + 累计方差贡献占比2 * 因子得分2 + …)
datas['factor_score'] = ((var[1]/var[1].sum())*fa2_score).sum(axis=1)
datas = datas.sort_values(by='factor_score',ascending=False)
datas['rank'] = range(1,len(datas)+1)

#########################kmeans聚类
###############################手肘法核心思想
#'利用SSE选择k'
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans #导入kmeans算法
SSE = []  # 存放每次结果的误差平方和
for k in range(1, 10):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(fa_score)
    SSE.append(estimator.inertia_)  # estimator.inertia_获取聚类准则的总和
X = range(1, 10)
plt.figure(figsize=(8, 6.5))
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('手肘法')
plt.plot(X, SSE, 'o-')
# plt.savefig('手肘图')
plt.show()
'显然，肘部对于的k值为4(曲率最高)，故对于这个数据集的聚类而言，最佳聚类数应该选4。'
###############################确定K值，开始kmeans++聚类
# kmeans++聚类
k = 6                        # 确定聚类中心数
kmeans_model = KMeans(n_clusters = k,init='k-means++',n_jobs=4,random_state=123)   # 构建模型 默认init='k-means++'
kmeans_model.fit(fa_score)        # 模型训练
# 查看聚类结果
kmeans_cc = kmeans_model.cluster_centers_  # 聚类中心
print('kmeans_各类聚类中心为：\n',kmeans_cc)
kmeans_labels = kmeans_model.labels_       # 样本的类别标签
print('kmeans_各样本的类别标签为：\n',kmeans_labels)
r1 = pd.Series(kmeans_model.labels_).value_counts()  # 统计不同类别样本的数目
print('kmeans_最终每个类别的数目为：\n',r1)

# 添加排名以及聚类结果
fa_score['factor_score'] = datas['factor_score']
fa_score['rank'] = datas['rank']
fa_score['label']=kmeans_labels
fa_score=fa_score.sort_values(by='factor_score',ascending=False)
fa_score.to_csv('kmeans_result.csv')



