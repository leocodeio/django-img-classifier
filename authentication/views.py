from django.shortcuts import render
from django.http import HttpResponse
from . import util
from PIL import Image
import os

classification_result = {
    'score_lionel_messi': 0.8,
    'score_maria_sharapova': 0.7,
    'score_roger_federer': 0.6,
    'score_virat_kohli': 0.5,
    'score_serena_williams': 0.4,
}

def handle_image(img_path):
    response = util.classify_image(img_path)
    print(response)
    return response

def home(request):
    if request.method == 'POST':
        if 'image_data' in request.FILES:
            image_file = request.FILES['image_data']
            with open('sample.jpg', 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)
            print("Image saved as sample.jpg")

            util.load_saved_artifacts()

            img_path = os.path.join(os.getcwd(), 'sample.jpg')
            global classification_result
            output = handle_image(img_path)
            k=0
            for i in classification_result:
                classification_result[i]=output[k]
                k+=1
            print(classification_result)
        else:
            print("No image file found in the request")
        return render(request, 'classify.html', {'classification_result': classification_result})

    else:
        return render(request, 'home.html')

def classify(request):
    return render(request, 'classify.html', {'classification_result': classification_result})
