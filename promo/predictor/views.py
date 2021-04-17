from django.http import HttpResponse
from django.template import loader
import pandas as pd
import numpy as np


def index(request):
    if request.method == "GET":
        templ = loader.get_template('predictor/predictor_index.html')
        return HttpResponse(templ.render(None, request))
    elif request.method == "POST":
        # get inuts

        # make predictions

        budget = request.POST.get("budget", 0)


        # draw chrats and tables

        df = pd.DataFrame(np.random.randint(0, 100, size=(53, 6)), columns=list('ABCDEF'))

        va = {
            "table": df.to_html(classes=['dataset']),
            "chart": budget,
        }

        templ = loader.get_template('predictor/prediction.html')
        return HttpResponse(templ.render(va, request))
