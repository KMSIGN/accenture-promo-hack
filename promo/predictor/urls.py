from django.urls import path
from .views import index, startup

urlpatterns = [
    path("", index, name="predictor"),
]

startup()
