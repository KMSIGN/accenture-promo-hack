from django.template import loader
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage


def index(request):
    templ = loader.get_template('promo/index.html')
    return HttpResponse(templ.render(None, request))


def features(request):
    return HttpResponse("Its all O K")


def uploader(request):
    if request.method == "GET":
        templ = loader.get_template("promo/uploader.html")
        return HttpResponse(templ.render({}, request))
    elif request.method == "POST" and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage("../data/")
        filename = fs.save(myfile.name, myfile)
        templ = loader.get_template("promo/index.html")
        va = {
            "show_popup": True,
            "modal_text": "Successfully uploaded"
        }
        return HttpResponse(templ.render(va, request))

