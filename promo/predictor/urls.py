from django.urls import path
from .views import index, downloader, startup

urlpatterns = [
    path("", index, name="predictor"),
    path("<str:key>/", downloader)
]

startup()
