import pandas as pd
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
        type_u = request.POST.get("type", "Nothing")
        fs = FileSystemStorage("../data/")
        filename = fs.save(myfile.name, myfile)
        try:
            df = pd.read_csv("../data/" + filename)
        except Exception as ex:
            df = pd.read_excel("../data/" + filename)

        if check_df(df, type_u):
            va = {
                "show_popup": True,
                "modal_text": "Successfully uploaded"
            }
        else:
            va = {
                "show_popup": True,
                "modal_text": "An error occurred"
            }
        templ = loader.get_template("promo/index.html")
        return HttpResponse(templ.render(va, request))


def check_df(df, type_d):
    if type_d == "Promo history":
        if set(["start_dttm", "end_dttm", "skutertiaryid", "promotypeid", "chaindiscountvalue"]).issubset(df.columns):
            return True
        else:
            return False
    elif type_d == "Sales history":
        if set(["salerevenuerub", "soldpieces", "skutertiaryid", "posid", "sale_dt"]).issubset(df.columns):
            return True
        else:
            return False
    return False
