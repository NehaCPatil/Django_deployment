from django.shortcuts import render
import pickle
import numpy as np
import importlib.util

from django.http import HttpResponse

spec = importlib.util.spec_from_file_location("function", "polls/model/linear_regression_model.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)


def index(request):
    return render(request, 'index.html')


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    # print(loaded_model)
    result = loaded_model.predict(to_predict)
    print(result[0])
    return result[0]


def result(request):
    if request.method == 'POST':
        exp_years = int(request.POST.get('YearsExperience'))
        result = ValuePredictor(exp_years)
        return HttpResponse('Salary is {}'.format(int(result)))

