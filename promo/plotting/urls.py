from django.urls import path
from .views import general_plot, revenue_from_items, selling_stat_plot, sales_stat, startup


urlpatterns = [
    path("", general_plot, name="plot"),
    path("revenue/", revenue_from_items, name="rev_plot"),
    path("sales_sum/", selling_stat_plot, name="sal_plot"),
    path("sales_count/", sales_stat, name="sal_stat")
]

startup()
