from django.urls import path
from .views import general_plot, plot_details


urlpatterns = [
    path("", general_plot, name="plot_list"),
    path("/<string:plot_nm>", )
]
