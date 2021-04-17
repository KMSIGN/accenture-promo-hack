from django.http import HttpResponse
from django.template import loader
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

promo_sum = None
sales_sum = None
sales_df = None


def startup():
    global promo_sum, sales_sum, sales_df
    promo_sum = pd.read_csv('../data/promo_sum_history.csv')
    sales_sum = pd.read_csv('../data/sales_sum_history.csv')
    sales_df = pd.read_csv("../data/sales_history.csv")


def general_plot(request):
    templ = loader.get_template('plotting/plotslist.html')
    va = {
        "chart_list": [
            {"name": "Scu revenue chart", "url": "/plot/revenue", "desc": "Explanation of chart"},
            {"name": "Revenue for sold items", "url": "/plot/sales_sum", "desc": "Statistic of every sold item by time"},
            {"name": "Count of sold items", "url": "/plot/sales_count/", "desc": "Count of every sold item"}
        ]
    }
    return HttpResponse(templ.render(va, request))


def revenue_from_items(request):
    templ = loader.get_template('plotting/plot.html')
    seles_uniq = sales_sum['skutertiaryid'].unique()
    fig = go.Figure()

    for i in seles_uniq:
        df = sales_sum[sales_sum['skutertiaryid'] == i]
        df = df.sort_values(by='week')
        trace1 = go.Scatter(x=df['week'], y=df['salerevenuerub'], name=f'Product №{i}')
        fig.add_trace(trace1)

    fig.update_layout(
        title="Revenue stats by weeks of 2019",
        xaxis_title="Date",
        yaxis_title="Sale revenue rub",
        legend_title="Products",
        font=dict(
            size=16,
        )
    )

    plt_div = plot(fig, output_type='div')
    va = {
        "title": "Product stats",
        "plot": plt_div
    }
    return HttpResponse(templ.render(va, request))


def selling_stat_plot(request):
    templ = loader.get_template('plotting/plot.html')
    promo_uniq = promo_sum['skutertiaryid'].unique()

    fig = go.Figure()
    for i in promo_uniq:
        df = promo_sum[promo_sum['skutertiaryid'] == i]
        df = df.sort_values(by='start_week')

        trace1 = go.Scatter(x=df['start_week'], y=df['soldpieces'], name=f'товар №{i}')
        fig.add_trace(trace1)

    fig.update_layout(
        xaxis_title="start_week",
        yaxis_title="sold pieces",
        font=dict(
            size=16,
        )
    )

    plt_div = plot(fig, output_type='div')

    va = {
        "title": "Product stats",
        "plot": plt_div
    }
    return HttpResponse(templ.render(va, request))


def sales_stat(request):
    templ = loader.get_template('plotting/plot.html')

    df = sales_df.groupby('skutertiaryid').sum()
    x = [f'Product №{i}' for i in range(10)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=df['soldpieces']))
    fig.update_layout(
        xaxis_title="Sold pieces",
        yaxis_title="Products",
        font=dict(
            size=16,
        )
    )

    plt_div = plot(fig, output_type='div')

    va = {
        "title": "Product sales stats",
        "plot": plt_div
    }
    return HttpResponse(templ.render(va, request))
