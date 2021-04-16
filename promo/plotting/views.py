from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .plotter import get_plot


def general_plot(request):
    templ = loader.get_template('plotting/plot.html')

    va = {
        "title": "test plot",
        "plot": get_plot()
    }

    return HttpResponse(templ.render(va, request))
