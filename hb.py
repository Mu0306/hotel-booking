import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_origin = pd.read_csv('D:\kaggle\hotel booknig\hotel_bookings.csv', encoding='gbk')

'''1.数据清洗'''

'''缺失值处理'''
data = data_origin.copy()
missing = data.isnull().sum(axis=0)  #列相同的相加
missing[missing!=0]

data.children.fillna(data.children.mode()[0],inplace=True)
data.country.fillna(data.country.mode()[0],inplace=True)
data.agent.fillna(0, inplace=True)
data.drop('company',inplace=True, axis=1)#删除列axis=1,删除行axis=0

'''异常值处理'''
# 入住人数为0
zero_guest=data[data[['adults', 'children', 'babies']].sum(axis=1)==0]
data.drop(zero_guest.index, inplace=True)

# 入住天数为0
zero_days = data[data[['stays_in_weekend_nights',
                       'stays_in_week_nights']].sum(axis=1) == 0]
data.drop(zero_days.index, inplace=True)

# 餐食类型Undefined/SC合并
data.meal.replace("Undefined", "SC", inplace=True)
#data.info()

'''2.数据可视化分析'''

'''客房信息分析'''
sns.countplot(x='hotel', hue='is_canceled', color='red',data=data)
#城市酒店订单量明显超过度假酒店，但同时预订取消的可能性也远远高于度假酒店
plt.show()
''''''
index = 1
for room_type in ['reserved_room_type', 'assigned_room_type']:
    # plt.figure(figsize=(6,8))
    ax1 = plt.subplot(2, 1, index)
    index += 1
    ax2 = ax1.twinx()#共享x轴
    ax1.bar(
        data.groupby(room_type).size().index,
        data.groupby(room_type).size(),color='red')
    ax1.set_xlabel(room_type)
    ax1.set_ylabel('Number')
    ax2.plot(
        data.groupby(room_type)['is_canceled'].mean(), color='coral',linestyle='-.')
    ax2.set_ylabel('Cancellation rate')
    #订单预定和分配的房间类型多数集中在A/D/E/F四类，其中A类房型取消率高出其余三类约7-8个百分点，值得关注。
    plt.show()

# 房间类型变更对取消预定的影响
data['room_chaged']=data['reserved_room_type']!=data['assigned_room_type']
sns.countplot(x='room_chaged',hue='is_canceled',color='red',data=data)
#房型变更过的客户取消预定的概率远远小于未变更过的客户，可能有以下原因：
#客户到达酒店后临时更改房型，多数客户会选择不取消预定，直接入住；
#客户自行更改房型，相对取消预定而言，这类客户更愿意更改房间类型而保证正常入住。

''' 客户画像分析'''

# 入住人数模式分析
# 单人
single = (data.adults == 1) & (data.children == 0) & (data.babies == 0)
# 双人
couple = (data.adults == 2) & (data.children == 0) & (data.babies == 0)
# 家庭
family = (data.adults >= 2) & (data.children > 0) | (data.babies > 0)

data['people_mode'] = single.astype(int) + couple.astype(int) * 2 + family.astype(int) * 3
plt.figure(figsize=(10,6))
index=1
for hotel_kind in ['City Hotel','Resort Hotel']:
    plt.subplot(1,2,index)
    index+=1
    sns.countplot(x='people_mode',
              hue='is_canceled',
              data=data[data.hotel == hotel_kind],color='blue')
    plt.xticks([0, 1, 2, 3], ['Others', 'Single', 'Couple', 'Family'])
    plt.title(hotel_kind)
plt.tight_layout()
plt.show()

# 查看餐食类型与取消预订的关系
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.pie(data[data['is_canceled'] == 1].meal.value_counts(), 
        labels=data[data['is_canceled'] == 1].meal.value_counts().index,  
    colors=('oldlace', 'wheat', 'moccasin', 'orange' ), autopct="%.2f%%") 
plt.legend(loc=1)
plt.title('Canceled')
plt.subplot(122)
plt.pie(data[data['is_canceled'] == 0].meal.value_counts(),
        colors=('oldlace', 'wheat', 'moccasin', 'orange' ), 
        autopct="%.2f%%",
        labels=data[data['is_canceled'] == 0].meal.value_counts().index)
plt.legend(loc=1)
plt.title('Uncanceled')
plt.show()
# 查看不同国家订单取消率
# 选取预定数前25的国家/地区
countries_25 = list(
    data.groupby('country').size().sort_values(ascending=False).head(25).index)
#data[data.country.isin(countries_25)].shape[0] / data.shape[0]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
plt.xticks(range(25), countries_25)
ax1.bar(
    range(25), data[data.country.isin(countries_25)].groupby('country').size().sort_values(ascending=False),color='lightsteelblue')
ax1.set_xlabel('Country')
ax1.set_ylabel('Total Number of Booking')
ax2.plot(
    range(25),
    data[data.country.isin(countries_25)].groupby('country')['is_canceled'].mean().loc[countries_25], 'bo-')
ax2.set_ylabel('Cancellation rate')
plt.show()

# 查看客户预定历史与取消订单的关系
# 是否回头客
plt.figure(figsize=(10,6))
index=1
for hotel_kind in ['City Hotel','Resort Hotel']:
    plt.subplot(1,2,index)
    index+=1
    sns.countplot(x='is_repeated_guest',
              hue='is_canceled',
              data=data[data.hotel == hotel_kind],color='green')
    plt.xticks([0, 1], ['New Guest', 'Repeated Guest'])
    plt.title(hotel_kind)
plt.tight_layout()
plt.show()

# 之前取消预定次数
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(data.groupby('previous_cancellations')['is_canceled'].mean(),
         'r+')
plt.xlabel('Previous Cancellations')
# 之前未取消预定次数
plt.subplot(122)
plt.plot(data.groupby('previous_bookings_not_canceled')['is_canceled'].mean(),
         'b+')
plt.ylim(0, 1)
plt.xlabel('Previous Un-Cancellations')
plt.show()


