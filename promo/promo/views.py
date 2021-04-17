from django.template import loader
from django.http import HttpResponse


def index(request):
    templ = loader.get_template('promo/index.html')
    return HttpResponse(templ.render(None, request))


def features(request):
    return HttpResponse("Its all O K")


def uploader(request):
    if request.method == "GET":
        templ = loader.get_template("promo/uploader.html")

        return HttpResponse(templ.render(None, request))
    elif request.method == "POST":
        templ = loader.get_template("promo/index.html")
        va = {
            "show_popup": True,
            "modal_text": "Successfully uploaded"
        }
        return HttpResponse(templ.render(va, request))

