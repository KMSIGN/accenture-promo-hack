from django.http import HttpResponse
from django.template import loader
from typing import Any, Dict, List
from pandas.core.frame import DataFrame
from sklearn.linear_model import Lasso
import random
from numpy.random import multinomial
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from plotly.offline import plot
import math
import numpy as np
import pandas as pd

sold_added_med = None
soldpieces_med = None
normal_sold = None
weeks = None
items = None
lenc = None
reg = None
soldpieces_median = None
max_sale = None
item_avalible_sales = None

def startup():
    global soldpieces_med, sold_added_med, normal_sold, weeks, items, lenc, reg, soldpieces_median, max_sale, item_avalible_sales
    sales_path: str = "../data/sales_history.csv"
    promo_path: str = "../data/promo_history.xlsx"

    promo = pd.read_excel(promo_path, index_col=0)
    sales = pd.read_csv(sales_path, index_col=0)

    promo = promo.drop(columns=0)
    sales['sale_dt'] = pd.to_datetime(sales['sale_dt'])

    # этот код ужасен, если у вас не как минимум 21:9 монитор можете не пытатся его читать
    # обработка продаж
    sales_sum = sales.drop(['posid'], axis=1).groupby(
        ['skutertiaryid', 'sale_dt']).sum().reset_index()

    sales_sum['week'] = sales_sum['sale_dt'].dt.isocalendar().week
    sales_sum['year'] = sales_sum['sale_dt'].dt.isocalendar().year
    sales_sum = sales_sum.drop('sale_dt', axis=1)

    # обработка промо
    promo_sum = promo.copy(deep=True)
    promo_sum['start_year'] = promo_sum['start_dttm'].dt.isocalendar().year
    promo_sum['start_week'] = promo_sum['start_dttm'].dt.isocalendar().week
    promo_sum['end_year'] = promo_sum['end_dttm'].dt.isocalendar().year
    promo_sum['end_week'] = promo_sum['end_dttm'].dt.isocalendar().week
    promo_sum = promo_sum.drop(
        ['start_dttm', 'end_dttm', 'promotypeid'], axis=1).sort_values('skutertiaryid')

    prices = {}
    for itemid in sales['skutertiaryid'].unique():
        item_sales = sales[sales['skutertiaryid'] == itemid]
        prices[itemid] = (item_sales['salerevenuerub'] / item_sales['soldpieces']
                          ).replace([np.inf, -np.inf], np.nan).max()

    promo_revenue = []
    promo_soldpieces = []

    for i, row in promo_sum.iterrows():
        promo_sales = sales_sum[(sales_sum['week'] >= row['start_week']) & (sales_sum['week'] <= row['end_week']) & (
                (sales_sum['year'] == row['start_year']) | (sales_sum['year'] == row['end_year']))]
        promo_revenue.append(prices[row['skutertiaryid']] *
                             row['chaindiscountvalue'] * (promo_sales['soldpieces'].sum()))
        promo_soldpieces.append(promo_sales['soldpieces'].sum())

    promo_sum['revenue'] = promo_revenue
    promo_sum['soldpieces'] = promo_soldpieces
    promo_sum = promo_sum[promo_sum['soldpieces'] != 0]

    normal_sold = {}  # нужно в глобал
    for itemid in sales_sum['skutertiaryid'].unique():
        normal_sold[itemid] = sales_sum[sales_sum['skutertiaryid']
                                        == itemid]['salerevenuerub'].median()

    sold_added = []
    for i, row in promo_sum.iterrows():
        sold_added.append((row['soldpieces'] / normal_sold[row['skutertiaryid']]))
    promo_sum['sold_added'] = sold_added

    sold_added_med = {}  # нужно в глобал
    soldpieces_med = {}  # нужно в глобал
    for itemid in promo_sum['skutertiaryid'].unique():
        sold_added_med[itemid] = promo_sum[promo_sum['skutertiaryid']
                                           == itemid]['sold_added'].median()
        soldpieces_med[itemid] = promo_sum[promo_sum['skutertiaryid']
                                           == itemid]['soldpieces'].median()

    promo_sum = promo_sum.drop(['end_year', 'end_week'], axis=1)

    train_data_list = []
    for itemid in promo_sum['skutertiaryid'].unique():
        promo_item = promo_sum[promo_sum['skutertiaryid']
                               == itemid].sort_values('start_week')
        last_week: int = 0
        for index, row in promo_item.iterrows():
            if row['start_week'] - last_week > 1:
                for i in range(last_week + 1, int(row['start_week'])):
                    week_sales = sales_sum[(sales_sum['year'] == row['start_year']) & (
                            sales_sum['week'] == i) & (sales_sum['skutertiaryid'] == row['skutertiaryid'])]
                    if len(week_sales) > 0:
                        train_data_list.append([
                            int(row['skutertiaryid']),
                            0,
                            int(row['start_year']),
                            int(i),
                            0,
                            week_sales.iloc[0]['soldpieces'],
                            week_sales.iloc[0]['soldpieces'] /
                            normal_sold[row['skutertiaryid']]
                        ])
                    else:
                        train_data_list.append([int(row['skutertiaryid']), 0, int(row['start_year']),
                                                int(i), 0, np.nan, np.nan])
            last_week = int(row['start_week'])
            train_data_list.append(row.values)
        if last_week < 53:
            for i in range(last_week + 1, 53 + 1):
                train_data_list.append([
                    int(row['skutertiaryid']),
                    0,
                    int(row['start_year']),
                    int(i),
                    0,
                    week_sales.iloc[0]['soldpieces'],
                    week_sales.iloc[0]['soldpieces'] /
                    normal_sold[row['skutertiaryid']]
                ])
    train_data: DataFrame = pd.DataFrame(
        train_data_list, columns=promo_sum.columns)

    for i, row in train_data.iterrows():
        if np.isnan(row['soldpieces']):
            row['soldpieces'] = soldpieces_med[row['skutertiaryid']]
        if np.isnan(row['sold_added']):
            row['sold_added'] = sold_added_med[row['skutertiaryid']]

    train_data = train_data.dropna()
    weeks = train_data.sort_values('start_week')[
        'start_week'].unique()  # нужно в глобал
    items = train_data['skutertiaryid'].unique()  # нужно в глобал

    lenc = LabelEncoder()  # нужно в глобал

    y = list(train_data['sold_added'])
    X = list(zip(train_data['chaindiscountvalue'], train_data['soldpieces'], train_data['revenue'],
                 train_data['start_week'], lenc.fit_transform(train_data['skutertiaryid'])))

    reg = Lasso(alpha=0.1, positive=True)  # нужно в глобал
    reg.fit(X, y)

    soldpieces_median = {}  # нужно в глобал
    for itemid in items:
        sales_item = sales_sum[sales_sum['skutertiaryid'] == itemid]
        soldpieces_med[itemid] = {}
        for week in weeks:
            soldpieces_med[itemid][week] = sales_item[sales_item['week']
                                                      == week]['soldpieces'].median()

    max_sale = {}  # нужно в глобал
    for itemid in items:
        max_sale[itemid] = promo_sum[promo_sum['skutertiaryid']
                                     == itemid]['chaindiscountvalue'].max()

    item_avalible_sales = {}  # нужно в глобал
    for itemid in items:
        item_avalible_sales[itemid] = [
            np.around(i, decimals=2) for i in np.arange(0, max_sale[itemid], 0.05)]


def get_random_sale_for_item(itemid: Any) -> List[float]:
    return item_avalible_sales[itemid][random.randint(0, len(item_avalible_sales[itemid])-1)]


# нужно в глобал
def get_random_distribution(n: int, all_sum: int, s: float = random.random()) -> List:
    return np.random.multinomial(all_sum, np.random.dirichlet(np.ones(n)*s))


# нужно в глобал
def predict(itemid: int, next_revenue: int, seed: float = random.random(), tries: int = 10) -> pd.DataFrame:
    test_data_list: List[List[Any]] = []
    for week in weeks:
        test_data_list.append([
            soldpieces_med[itemid][week],
            week,
            itemid
        ])
    test_data = pd.DataFrame(test_data_list,
                             columns=['soldpieces', 'start_week', 'skutertiaryid'])

    tries_dict = {}
    for i in range(tries):
        test_data['revenue'] = get_random_distribution(
            len(weeks), next_revenue, seed)
        test_data['chaindiscountvalue'] = [get_random_sale_for_item(itemid)
                                           for itemid in test_data['skutertiaryid']]
        X = list(zip(test_data['chaindiscountvalue'], test_data['soldpieces'],
                     test_data['revenue'], test_data['start_week'], lenc.transform(test_data['skutertiaryid'])))
        predict_result = reg.predict(X)
        tries_dict[np.sum(predict_result)] = predict_result

    test_data['sold_added'] = tries_dict[np.max(list(tries_dict.keys()))]
    test_data['revenue'] = test_data['revenue'].apply(
        lambda x: np.around(x, decimals=math.floor(-1 * len(str(next_revenue))/3)))
    test_data['soldpieces'] = (
        test_data['soldpieces'] * (1 + test_data['sold_added'])).astype(int)

    return test_data


def index(request):
    global lenc
    if request.method == "GET":
        templ = loader.get_template('predictor/predictor_index.html')
        return HttpResponse(templ.render(None, request))
    elif request.method == "POST":
        req = request.POST.get("budget", "0")
        distrN = float(request.POST.get("distr", 0.1))
        if req == "":
            templ = loader.get_template('predictor/predictor_index.html')
            return HttpResponse(templ.render(None, request))

        budget = int(req)
        distrN = pow(distrN, 4)


        stats = {}
        gen_df = pd.DataFrame()

        for item in lenc.classes_:
            d = predict(item, int(budget/len(lenc.classes_)), tries=100)
            gen_df = gen_df.append(d)
            stats[item] = d
        gen_stat = pd.DataFrame()
        for df in stats.values():
            new_r = df[["chaindiscountvalue"]].T
            new_r.index = [int(df.iloc[0]['skutertiaryid']),]
            gen_stat = gen_stat.append(new_r)

        out_tbs = {}
        for i, v in stats.items():
            out_tbs[i] = v.to_html(classes=['dataset'])

        x = [f'Product №{i}' for i in gen_df['skutertiaryid'].unique()]
        gen_df['item_def'] = gen_df['soldpieces'] - gen_df['soldpieces'] * gen_df['sold_added']
        gen_df['item_inc'] = gen_df['soldpieces'] * gen_df['sold_added']

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=gen_df.groupby('skutertiaryid')["item_def"].sum(), name="Default trend"))
        fig.add_trace(go.Bar(x=x, y=gen_df.groupby('skutertiaryid')["item_inc"].sum(), name="Promotion impact"))
        fig.update_layout(
            barmode='stack',
            xaxis_title="Sold pieces",
            yaxis_title="Products",
            font=dict(
                size=16,
            )
        )
        plt_div = plot(fig, output_type='div')

        va = {
            "gen_stat": gen_stat.to_html(classes=['dataset']),
            "tables": out_tbs,
            "plot": plt_div,
        }

        templ = loader.get_template('predictor/prediction.html')
        return HttpResponse(templ.render(va, request))
