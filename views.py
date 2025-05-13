from django.shortcuts import render
import numpy as np
import joblib
from django.conf import settings
import os
def index(request):
    prediction=None
    if request.method=="POST":
        Avg_Area_Income=request.POST.get("Avg_Area_Income")
        Avg_Area_House_Age=request.POST.get("Avg_Area_House_Age")
        Avg_Area_Number_of_Rooms=request.POST.get("Avg_Area_Number_of_Rooms")
        Avg_Area_Number_of_Bedrooms=request.POST.get("Avg_Area_Number_of_Bedrooms")
        Area_population=request.POST.get("Area_population")
        data=[[Avg_Area_Income,Avg_Area_House_Age,Avg_Area_Number_of_Rooms,Avg_Area_Number_of_Bedrooms,Area_population]]
        ext_data=np.array([
            float(Avg_Area_Income),
            float(Avg_Area_House_Age),
            float(Avg_Area_Number_of_Rooms),
            float(Avg_Area_Number_of_Bedrooms),
            float(Area_population)
        ]).reshape(1,-1)
        model_path = os.path.join(settings.BASE_DIR,"homecost","model","homecost_againtry160file.pkl")
        loaded_model=joblib.load(model_path)
        prediction=loaded_model.predict(ext_data)
    return render(request,"index.html",{'price':prediction})