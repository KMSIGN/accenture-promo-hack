from django.urls import path
from .views import general_plot


urlpatterns = [
    path("", general_plot, name="plot")
]
