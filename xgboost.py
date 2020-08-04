import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn import metrics
from sklearn import GridSearchCV
data_pre=data.copy()
'''数据预处理'''
'''
特征必须为客户预定时就能获得的数据
因此排除booking_changes，reservation_status，assigned_room_type等特征
2）考虑一定的信息脱敏和通用性，排除country（国籍）,arrival_date_year（入住年份）等特征
'''
data_pre.drop(labels=['country','reservation_status','reservation_status_date','arrival_date_year','assigned_room_type'],axis=1,inplace=True)
#one-hot编码
predict_origin = pd.get_dummies(data_pre)

'''决策树_五折交叉检验'''
predict_origin_class = predict_origin['is_canceled']
predict_origin = predict_origin.drop('is_canceled', axis=1)
print('*******Decision Tree——five folds*******')


cv = KFold(n_splits=5, random_state=100, shuffle=True)
for train_index, test_index in cv.split(predict_origin_class):
    x_train, x_test, y_train, y_test = predict_origin.iloc[train_index], predict_origin.iloc[test_index], predict_origin_class.iloc[train_index], predict_origin_class.iloc[test_index]
        
dctree = DecisionTreeClassifier()
dctree.fit(x_train,y_train)
y_predtree = dctree.predict(x_test)
cross = accuracy_score(y_test,y_predtree)
f1_macro = f1_score(y_test,y_predtree,average='macro')  
print('f1_macro: {0}'.format(f1_macro)) 
print("K-Fold: %.3f" %cross)


'''决策树_五折交叉检验'''

cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
 
model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(x_train, y_train)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))