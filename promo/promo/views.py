from django.template import loader
from django.http import HttpResponse


def index(request):
    templ = loader.get_template('promo/index.html')
    return HttpResponse(templ.render(None, request))


def features(request):
    return HttpResponse("Its all O K")


def prediction(request):
    return HttpResponse("Not implemented yet")
